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
    rgb_channels: int = 1
    base_channels: int = 32
    event_channels: int = 2  # positive and negative


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

        # Scale and bias for each resolution
        self.film_params = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(imu_dim * 2, c),
                    nn.Sigmoid(),  # scale in (0, 1)
                )
                for c in scales
            ]
        )
        self.film_bias = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(imu_dim * 2, c),
                    nn.Tanh(),  # bias in (-1, 1)
                )
                for c in scales
            ]
        )

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


class AdaINFusion(nn.Module):  # type: ignore[misc]
    """AdaIN-style fusion: normalize RGB, then modulate with IMU statistics."""

    def __init__(self, base_channels: int, imu_dim: int = 128) -> None:
        super().__init__()
        self.base_channels = base_channels
        # IMU predicts style statistics for each scale
        scales = [base_channels, base_channels * 2, base_channels * 4]
        total_params = sum(c * 2 for c in scales)  # gamma + beta for each

        self.imu_to_style = nn.Sequential(
            nn.Linear(imu_dim, imu_dim * 2),
            nn.SiLU(),
            nn.Linear(imu_dim * 2, total_params),
        )
        self.eps = 1e-5

    def forward(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor], imu_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get style parameters from IMU
        style = self.imu_to_style(imu_features)

        # Normalize each feature map, then apply IMU style
        modulated = []
        param_idx = 0

        for f in features:
            c = f.shape[1]
            # Instance normalization
            mean = f.mean(dim=(2, 3), keepdim=True)
            std = f.std(dim=(2, 3), keepdim=True) + self.eps
            f_norm = (f - mean) / std

            # Get gamma and beta for this scale
            gamma = style[:, param_idx : param_idx + c].view(f.shape[0], c, 1, 1)
            beta = style[:, param_idx + c : param_idx + c * 2].view(f.shape[0], c, 1, 1)
            param_idx += c * 2

            modulated.append(f_norm * gamma + beta)

        return tuple(modulated)


class EventPredictionHead(nn.Module):  # type: ignore[misc]
    """Predict events from features."""

    def __init__(self, in_channels: int, out_size: tuple[int, int]) -> None:
        super().__init__()
        self.out_size = out_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        events = self.decoder(x)
        if events.shape[2:] != self.out_size:
            events = F.interpolate(events, size=self.out_size, mode="bilinear", align_corners=False)
        return events


class EventPredictor(nn.Module):  # type: ignore[misc]
    """Main model: RGB + IMU -> Events with Multi-scale FiLM."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.imu_encoder = IMUEncoder(input_dim=6, hidden_dim=config.imu_hidden_dim)
        self.rgb_encoder = RGBEncoder(
            in_channels=config.rgb_channels, base_channels=config.base_channels
        )
        # Multi-scale FiLM modulation (applies at ALL encoder layers)
        self.fusion = MultiScaleFiLM(
            base_channels=config.base_channels, imu_dim=config.imu_hidden_dim
        )
        # Decoder with skip connections
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
        self.final = nn.Conv2d(base, 2, 3, padding=1)
        self.out_size = config.image_size

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        imu_features = self.imu_encoder(imu_seq)
        x1, x2, x3 = self.rgb_encoder(image)
        # Apply multi-scale FiLM
        x1_mod, x2_mod, x3_mod = self.fusion((x1, x2, x3), imu_features)

        # Decoder with skip connections using modulated features
        d1 = self.up1(x3_mod)
        if d1.shape[2:] != x2_mod.shape[2:]:
            d1 = F.interpolate(d1, size=x2_mod.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x2_mod], dim=1)
        d2 = self.up2(d1)
        if d2.shape[2:] != x1_mod.shape[2:]:
            d2 = F.interpolate(d2, size=x1_mod.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, x1_mod], dim=1)
        d3 = self.up3(d2)
        events = self.final(d3)

        if events.shape[2:] != self.out_size:
            events = F.interpolate(events, size=self.out_size, mode="bilinear", align_corners=False)
        return events


# ---------------------------------------------------------------------------
# Optimizer and Hyperparameters (EDIT THIS)
# ---------------------------------------------------------------------------

# Model architecture
BASE_CHANNELS = 32  # Reduced for faster iteration with synthetic data
IMU_HIDDEN_DIM = 128  # Reduced for faster iteration

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
    """Setup and return the device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
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
    """Apply RGB-only data augmentation (no geometric transforms).

    Augmentations:
    - Gaussian noise
    - Brightness jitter
    - Contrast jitter
    - Random occlusions (dropout rectangles)
    - IMU noise

    Note: No geometric transforms (flip/rotate) to avoid modifying event labels.
    """
    B, C, H, W = images.shape

    # 1. Gaussian noise (additive)
    if torch.rand(1).item() > 0.5:
        noise_std = torch.rand(1).item() * 0.1  # 0 to 0.1 std
        images = images + torch.randn_like(images) * noise_std
        images = images.clamp(0, 1)

    # 2. Brightness jitter (multiply by factor)
    if torch.rand(1).item() > 0.5:
        brightness_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
        images = images * brightness_factor
        images = images.clamp(0, 1)

    # 3. Contrast jitter (scale around mean)
    if torch.rand(1).item() > 0.5:
        contrast_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7 to 1.3
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - mean) * contrast_factor + mean
        images = images.clamp(0, 1)

    # 4. Random occlusions (rectangle dropout)
    if torch.rand(1).item() > 0.5:
        # Number of occlusions
        num_occlusions = torch.randint(1, 4, (1,)).item()
        for _ in range(num_occlusions):
            # Random occlusion size (5% to 20% of image)
            occ_h = int(H * (0.05 + torch.rand(1).item() * 0.15))
            occ_w = int(W * (0.05 + torch.rand(1).item() * 0.15))

            # Random position
            y = torch.randint(0, H - occ_h, (1,)).item()
            x = torch.randint(0, W - occ_w, (1,)).item()

            # Apply occlusion (set to 0 or random value)
            if torch.rand(1).item() > 0.5:
                images[:, :, y : y + occ_h, x : x + occ_w] = 0  # Black occlusion
            else:
                images[:, :, y : y + occ_h, x : x + occ_w] = torch.rand(1).item()  # Random gray

    # 5. IMU noise
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(imu_seq) * 0.01
        imu_seq = imu_seq + noise

    return images, imu_seq, gt_events


def training_step(
    model: EventPredictor,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: str,
) -> float:
    """Perform one training step and return loss."""
    images = batch["image"].to(device)
    imu_seq = batch["imu_seq"].to(device)
    gt_events = batch["events"].to(device)

    # Apply augmentation during training
    if model.training:
        images, imu_seq, gt_events = augment_batch(images, imu_seq, gt_events)

    optimizer.zero_grad()
    pred_events = model(images, imu_seq)
    loss = F.mse_loss(pred_events, gt_events)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def print_progress(
    step: int, progress: float, loss: float, lrm: float, dt: float, remaining: float
) -> None:
    """Print training progress."""
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
        f"remaining: {remaining:.0f}s ",
        end="",
        flush=True,
    )


def run_training_loop(
    model: EventPredictor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: str,
) -> tuple[float, int]:
    """Run the training loop and return total time and step count."""
    total_training_time = 0.0
    step = 0
    smooth_train_loss = 0.0
    grad_accum_steps = 2  # Experiment: gradient accumulation

    model.train()
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            if model.training:
                images, imu_seq, gt_events = augment_batch(images, imu_seq, gt_events)

            pred_events = model(images, imu_seq)
            # Use MSE loss for stability
            loss = F.mse_loss(pred_events, gt_events) / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        optimizer.step()
        loss_val = accumulated_loss

        dt = time.time() - t0
        total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_val
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        remaining = max(0.0, TIME_BUDGET - total_training_time)
        print_progress(step, progress, debiased_smooth_loss, lrm, dt, remaining)

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
    """Print final results in standard format."""
    print("---")
    print(f"event_bpb: {eval_metrics['event_bpb']:.6f}")
    print(f"event_mse: {eval_metrics['event_mse']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds: {total_time:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"samples_per_sec: {eval_metrics['samples_per_sec']:.1f}")
    print(f"num_steps: {num_steps}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"base_channels: {BASE_CHANNELS}")
    print(f"imu_hidden_dim: {IMU_HIDDEN_DIM}")


def train() -> None:
    """Main training function."""
    device = setup_device()
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")

    model, num_params = create_model(device)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    train_loader = make_dataloader(DATA_DIR, "train", DEVICE_BATCH_SIZE, MAX_SEQ_LEN, IMAGE_SIZE)
    val_loader = make_dataloader(DATA_DIR, "val", FINAL_EVAL_BATCH_SIZE, MAX_SEQ_LEN, IMAGE_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    t_start = time.time()
    total_training_time, num_steps = run_training_loop(model, optimizer, train_loader, device)

    t_train = time.time()
    print(f"Training completed in {t_train - t_start:.1f}s")

    print("Starting final eval...")
    eval_metrics = evaluate_combined_metric(model, val_loader, device, EVAL_SAMPLES)
    t_eval = time.time()
    print(f"Final eval completed in {t_eval - t_train:.1f}s")

    peak_vram_mb = get_peak_memory_mb()
    print_results(
        eval_metrics, total_training_time, t_eval - t_start, num_steps, num_params, peak_vram_mb
    )


if __name__ == "__main__":
    train()
