"""
IMU-Enhanced Event Camera Training Script.

This is the ONLY file you edit during experiments.
Everything else (data loading, evaluation) is fixed in prepare_data.py.

Usage:
    uv run train.py
    python train.py  # If using pip instead of uv
"""

import gc
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import (
    DATA_DIR,
    EVAL_SAMPLES,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    evaluate_combined_metric,
    make_dataloader,
)
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Model Architecture (EDIT THIS)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model configuration."""

    image_size: tuple[int, int] = IMAGE_SIZE
    imu_seq_len: int = MAX_SEQ_LEN
    imu_hidden_dim: int = 128
    rgb_channels: int = 2  # frame pair: (current, previous) stacked channel-wise
    base_channels: int = 32
    event_channels: int = 2  # positive and negative
    use_imu: bool = True  # set False for image-only ablation


class IMUEncoder(nn.Module):  # type: ignore[misc]
    """Encode IMU sequences into feature vectors."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(imu_seq)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        imu_features = self.fc(h_cat)
        return self.norm(imu_features)


class RGBEncoder(nn.Module):  # type: ignore[misc]
    """Encode grayscale images into feature maps with skip connections."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(inplace=True),
        )
        self.out_channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3


class MultiScaleFiLM(nn.Module):  # type: ignore[misc]
    """Multi-scale FiLM modulation with IMU conditioning."""

    def __init__(self, base_channels: int, imu_dim: int = 128) -> None:
        super().__init__()
        # FiLM parameters for each scale
        scales = [base_channels, base_channels * 2, base_channels * 4]

        # Shared IMU feature projection
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, imu_dim * 2),
            nn.SiLU(),
            nn.Linear(imu_dim * 2, imu_dim * 2),
        )

        # FiLM scale and bias for each resolution.
        # Scale initialized to 1 (identity) so features flow through at step 0.
        # Kaiming default (scale≈0) causes near-zero activations early in training.
        self.film_params = nn.ModuleList([nn.Linear(imu_dim * 2, c) for c in scales])
        self.film_bias = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(imu_dim * 2, c),
                    nn.Tanh(),  # bias in (-1, 1)
                )
                for c in scales
            ]
        )
        for linear in self.film_params:
            nn.init.zeros_(linear.weight)
            nn.init.ones_(linear.bias)

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], imu_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project IMU features once
        imu_proj = self.imu_proj(imu_features)

        # Apply FiLM at each scale
        modulated = []
        for f, scale_fn, bias_fn in zip(features, self.film_params, self.film_bias, strict=True):
            scale = scale_fn(imu_proj).view(f.shape[0], f.shape[1], 1, 1)
            bias = bias_fn(imu_proj).view(f.shape[0], f.shape[1], 1, 1)
            modulated.append(f * scale + bias)

        return tuple(modulated)


class EventPredictionHead(nn.Module):  # type: ignore[misc]
    """Predict ON/OFF event probability maps from decoder features."""

    def __init__(self, in_channels: int, out_size: tuple[int, int]) -> None:
        super().__init__()
        self.out_size = out_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logits → sigmoid probabilities in [0, 1]
        logit = self.decoder(x)
        prob = torch.sigmoid(logit)
        if prob.shape[2:] != self.out_size:
            # Interpolate logits before sigmoid for better gradient flow
            logit_up = F.interpolate(
                logit, size=self.out_size, mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(logit_up)
        return prob


class EventPredictor(nn.Module):  # type: ignore[misc]
    """RGB + IMU → ON/OFF event probability maps.

    When config.use_imu=False the IMU encoder and FiLM fusion are bypassed
    (image-only ablation): features pass straight from encoder to decoder.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rgb_encoder = RGBEncoder(
            in_channels=config.rgb_channels, base_channels=config.base_channels
        )
        if config.use_imu:
            self.imu_encoder = IMUEncoder(input_dim=6, hidden_dim=config.imu_hidden_dim)
            self.fusion = MultiScaleFiLM(
                base_channels=config.base_channels, imu_dim=config.imu_hidden_dim
            )
        base = config.base_channels
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
        )
        self.event_head = EventPredictionHead(in_channels=base, out_size=config.image_size)

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        x1, x2, x3 = self.rgb_encoder(image)

        if self.config.use_imu:
            imu_features = self.imu_encoder(imu_seq)
            x1, x2, x3 = self.fusion((x1, x2, x3), imu_features)

        # U-Net decoder with skip connections
        d1 = self.up1(x3)
        if d1.shape[2:] != x2.shape[2:]:
            d1 = F.interpolate(d1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x2], dim=1)
        d2 = self.up2(d1)
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, x1], dim=1)
        d3 = self.up3(d2)

        return self.event_head(d3)


# ---------------------------------------------------------------------------
# Optimizer and Hyperparameters (EDIT THIS)
# ---------------------------------------------------------------------------

# Model architecture
BASE_CHANNELS = 32  # Reduced for faster iteration with synthetic data
IMU_HIDDEN_DIM = 128  # Reduced for faster iteration

# Knowledge distillation (optional)
USE_DISTILLATION = False  # Set to True to distill from Multimodal spatiotemporal teacher
DISTILLATION_WEIGHT = 0.5  # Balance between task loss and distillation loss

# Training
TOTAL_BATCH_SIZE = 32
DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0  # Experiment: no weight decay
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.01

# Evaluation
FINAL_EVAL_BATCH_SIZE = 16


def focal_bce_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal BCE loss for sparse event prediction.

    Handles class imbalance (~4.6% positive pixels) by down-weighting easy
    negatives via the (1-pt)^gamma factor.  Alpha up-weights the rare positive
    (event) class.

    Alpha convention: alpha is the weight on POSITIVES.
      alpha > 0.5  → upweight positives (correct for positive-rare tasks)
      alpha = 0.75 → positives:negatives = 0.75:0.25 = 3:1 per-pixel weight
      With 4.6% positives: effective ratio = 0.046*0.75 : 0.954*0.25 = 0.126:0.124 ≈ 1:1
      This roughly equalises positive/negative gradient contributions.

    Args:
        pred:  Predicted probabilities (B, 2, H, W) in [0, 1].
        gt:    Ground-truth event maps  (B, 2, H, W) in [0, 1].
        gamma: Focusing parameter (2.0 is standard from RetinaNet).
        alpha: Weight for positive class (must be > 0.5 for positive-rare tasks).
    """
    # Threshold GT to binary targets.
    # GT is normalized by max_events=100, so ≥0.5 events/33ms → GT ≈ 0.005.
    gt_bin = (gt > 0.005).float()
    # Clamp predictions for numerical stability
    pred_c = pred.clamp(1e-6, 1.0 - 1e-6)
    bce = -(gt_bin * torch.log(pred_c) + (1.0 - gt_bin) * torch.log(1.0 - pred_c))
    # p_t: model confidence in the correct class
    pt = gt_bin * pred_c + (1.0 - gt_bin) * (1.0 - pred_c)
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = gt_bin * alpha + (1.0 - gt_bin) * (1.0 - alpha)
    return (alpha_weight * focal_weight * bce).mean()


def get_lr_multiplier(progress: float) -> float:
    """Learning rate schedule with warmup and cooldown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training Loop (DO NOT EDIT BELOW THIS LINE)
# ---------------------------------------------------------------------------


def get_peak_memory_mb() -> float:
    """Get peak GPU memory in MB."""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return 0.0


def setup_device() -> str:
    """Setup and return the device with optimal settings (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        print("CUDA GPU acceleration enabled")
        return "cuda"
    if torch.backends.mps.is_available():
        # Enable MPS graph fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("MPS GPU acceleration enabled")
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        return "mps"
    print("WARNING: No GPU available, using CPU (slow)")
    return "cpu"


def create_model(device: str) -> tuple[EventPredictor, int]:
    """Create model and return it with parameter count."""
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def augment_batch(
    images: torch.Tensor,
    imu_seq: torch.Tensor,
    gt_events: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physics-consistent augmentation for event prediction.

    Only transforms that do NOT change which events the DVS would fire are
    applied.  Specifically:
    - Uniform brightness scaling cancels in log(I_t) - log(I_{t-1}), so events
      are unchanged (valid).
    - Gaussian noise (small) slightly perturbs log-intensities but rarely crosses
      the ~0.2 ln-unit threshold; kept for regularization.
    - IMU noise: small additive perturbation on normalized IMU.

    Excluded (physics-inconsistent, corrupt GT labels):
    - Contrast jitter: changes I_t/I_{t-1} → changes which events fire.
    - Occlusion rectangles: sets pixels to 0 but does not zero GT events there.
    - Event label dropout: randomly mislabels positive pixels as background.
    """
    # 1. Uniform brightness scaling: log(k*I_t) - log(k*I_{t-1}) = log(I_t/I_{t-1})
    #    The k cancels exactly, so DVS events are unaffected.
    if torch.rand(1).item() > 0.5:
        brightness_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7–1.3
        images = (images * brightness_factor).clamp(0, 1)

    # 2. Small Gaussian noise: std kept ≤ 0.02 so log-intensity perturbation
    #    rarely exceeds the ~0.2 ln-unit DVS threshold.
    if torch.rand(1).item() > 0.5:
        images = (images + torch.randn_like(images) * 0.02).clamp(0, 1)

    # 3. IMU noise: small fractional perturbation on already-normalized values.
    if torch.rand(1).item() > 0.5:
        imu_seq = imu_seq + torch.randn_like(imu_seq) * 0.02

    return images, imu_seq, gt_events


def print_progress(
    step: int,
    progress: float,
    loss: float,
    event_loss: float,
    rate_loss: float,
    flow_loss: float,
    lrm: float,
    dt: float,
    remaining: float,
) -> None:
    """Print training progress with per-component loss breakdown."""
    pct_done = 100 * progress
    sps = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | "
        f"loss: {loss:.4f} evt: {event_loss:.4f} rate: {rate_loss:.4f} flow: {flow_loss:.4f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | sps: {sps} | rem: {remaining:.0f}s ",
        end="",
        flush=True,
    )


def _accumulate_step(
    model: EventPredictor,
    batch: dict[str, torch.Tensor],
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum_steps: int,
) -> tuple[float, float, float, float]:
    """Run one micro-batch forward+backward; return (total, event, rate, flow) losses."""
    images = batch["image"].to(device)
    imu_seq = batch["imu_seq"].to(device)
    gt_events = batch["events"].to(device)

    if model.training:
        images, imu_seq, gt_events = augment_batch(images, imu_seq, gt_events)

    pred_prob = model(images, imu_seq)

    # Focal BCE — primary loss (~4.6% positive pixels, alpha=0.75 upweights events)
    event_loss = focal_bce_loss(pred_prob, gt_events) / grad_accum_steps

    # Rate regularisation — keep mean prediction near 5% (measured event sparsity)
    rate_reg = (
        F.mse_loss(pred_prob.mean().unsqueeze(0), torch.tensor([0.05], device=device))
        / grad_accum_steps
    )

    # Flow consistency — high-flow frames should predict more events globally
    flow_tensor = batch.get("flow", None)
    if flow_tensor is not None:
        flow_tensor = flow_tensor.to(device)
        flow_mag = torch.sqrt(flow_tensor[:, 0] ** 2 + flow_tensor[:, 1] ** 2)
        FLOW_SCALE = 5.0
        flow_mag_n = (flow_mag.mean(dim=(1, 2)) / FLOW_SCALE).clamp(0.0, 1.0)
        flow_loss = (
            F.mse_loss(pred_prob.mean(dim=(1, 2, 3)), flow_mag_n.detach()) / grad_accum_steps
        )
    else:
        flow_loss = torch.tensor(0.0, device=device)

    loss = event_loss + 0.01 * rate_reg + 0.1 * flow_loss

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item(), event_loss.item(), rate_reg.item(), flow_loss.item()


def run_training_loop(
    model: EventPredictor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: str,
    grad_accum_steps: int = 8,
) -> tuple[float, int]:
    """Run the training loop."""
    total_training_time = 0.0
    step = 0
    smooth: dict[str, float] = {"total": 0.0, "evt": 0.0, "rate": 0.0, "flow": 0.0}

    scaler = torch.amp.GradScaler(device="cuda") if device == "cuda" else None
    model.train()
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()
        optimizer.zero_grad()
        acc = {"total": 0.0, "evt": 0.0, "rate": 0.0, "flow": 0.0}

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            total, evt, rate, flow = _accumulate_step(
                model, batch, device, scaler, grad_accum_steps
            )
            acc["total"] += total
            acc["evt"] += evt
            acc["rate"] += rate
            acc["flow"] += flow

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        dt = time.time() - t0
        total_training_time += dt

        ema = 0.9
        debias = 1 - ema ** (step + 1)
        for k in smooth:
            smooth[k] = ema * smooth[k] + (1 - ema) * acc[k]

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        print_progress(
            step,
            progress,
            smooth["total"] / debias,
            smooth["evt"] / debias,
            smooth["rate"] / debias,
            smooth["flow"] / debias,
            lrm,
            dt,
            max(0.0, TIME_BUDGET - total_training_time),
        )

        step += 1
        if total_training_time >= TIME_BUDGET:
            break
        if step % 1000 == 0:
            gc.collect()

    print()
    return total_training_time, step


def print_results(
    eval_metrics: dict[str, float],
    total_training_time: float,
    total_time: float,
    num_steps: int,
    num_params: int,
    peak_vram_mb: float,
) -> None:
    """Print final results."""
    print("---")
    # Primary metric: F1 (robust to class imbalance, meaningful for sparse events)
    print(f"f1_score:    {eval_metrics['f1_score']:.4f}")
    print(f"f1_on:       {eval_metrics['f1_on']:.4f}")
    print(f"f1_off:      {eval_metrics['f1_off']:.4f}")
    print(f"precision:   {eval_metrics['precision']:.4f}")
    print(f"recall:      {eval_metrics['recall']:.4f}")
    # Secondary metric
    print(f"event_mse:   {eval_metrics['event_mse']:.6f}")
    # Training stats
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds: {total_time:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"samples_per_sec: {eval_metrics['samples_per_sec']:.1f}")
    print(f"num_steps: {num_steps}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"base_channels: {BASE_CHANNELS}")
    print(f"imu_hidden_dim: {IMU_HIDDEN_DIM}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": {
            "base_channels": BASE_CHANNELS,
            "imu_hidden_dim": IMU_HIDDEN_DIM,
        },
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint saved to: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, torch.optim.Optimizer | None, int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"✅ Checkpoint loaded from: {filepath}")
    print(f"   Epoch: {epoch}, Loss: {loss:.6f}")

    return model, optimizer, epoch, loss


def train(resume_from: str | None = None) -> None:
    """Main training function.

    Args:
        resume_from: Path to checkpoint to resume from (optional)
    """
    device = setup_device()
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")

    model, num_params = create_model(device)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create dataloaders — val uses train IMU stats to avoid leakage
    from prepare_data import FPVDataset

    _train_ds = FPVDataset(DATA_DIR, split="train")
    train_imu_stats = (_train_ds._imu_mean, _train_ds._imu_std)
    del _train_ds

    train_loader = make_dataloader(DATA_DIR, "train", DEVICE_BATCH_SIZE, MAX_SEQ_LEN, IMAGE_SIZE)
    val_loader = make_dataloader(
        DATA_DIR,
        "val",
        FINAL_EVAL_BATCH_SIZE,
        MAX_SEQ_LEN,
        IMAGE_SIZE,
        imu_stats=train_imu_stats,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        model, optimizer, start_epoch, _ = load_checkpoint(resume_from, model, optimizer, device)
        print(f"Resuming from epoch {start_epoch}")

    t_start = time.time()
    total_training_time, num_steps = run_training_loop(
        model, optimizer, train_loader, device, grad_accum_steps
    )

    t_train = time.time()
    print(f"Training completed in {t_train - t_start:.1f}s")

    # Save checkpoint
    checkpoint_path = "3d_aware_model_checkpoint.pt"
    save_checkpoint(model, optimizer, num_steps, total_training_time, checkpoint_path)

    del train_loader

    print("Starting final eval...")
    eval_metrics = evaluate_combined_metric(model, val_loader, device, EVAL_SAMPLES)
    t_eval = time.time()
    print(f"Final eval completed in {t_eval - t_train:.1f}s")

    del val_loader

    peak_vram_mb = get_peak_memory_mb()
    print_results(
        eval_metrics, total_training_time, t_eval - t_start, num_steps, num_params, peak_vram_mb
    )


if __name__ == "__main__":
    train()
