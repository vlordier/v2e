"""
IMU-Enhanced Event Camera Training Script.

This is the ONLY file you edit during experiments.
Everything else (data loading, evaluation) is fixed in prepare_data.py.

Usage:
    uv run train.py
    python train.py  # If using pip instead of uv
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fno_event_predictor import FNOEventPredictor, FourierLayer
from mlflow_utils import MlflowRunManager
from prepare_data import (
    DATA_DIR,
    EVAL_SAMPLES,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    evaluate_combined_metric,
    evaluate_physics_baseline,
    make_dataloader,
)
from torch.utils.data import DataLoader


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with the largest power-of-2 group count ≤ 8 that divides channels.

    Safe for any channel count — avoids GroupNorm(8, c) crashing when c % 8 != 0,
    which would otherwise happen if the autoresearch agent tries BASE_CHANNELS not
    divisible by 8 (e.g., 20, 12, 6).
    """
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


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
    fusion_type: str = "film"  # film | gated | additive | none
    model_family: str = "balanced"  # light | balanced | heavy
    unet_depth: int = 3


class IMUEncoder(nn.Module):  # type: ignore[misc]
    """Encode IMU sequences into feature vectors via attention-pooled bidirectional LSTM.

    Uses the full LSTM output (all T timesteps) with a learned temporal attention
    to produce a context vector.  This lets the model learn WHICH part of the IMU
    window matters most: early timesteps → scene geometry; late timesteps → current
    instantaneous motion.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        # Temporal attention: score each timestep → soft-weighted average
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(imu_seq)  # (B, T, 2*H)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (B, 2*H)
        return self.norm(self.fc(context))


def build_channel_schedule(base_channels: int, depth: int, model_family: str) -> list[int]:
    """Construct a channel pyramid for light/balanced/heavy U-Net branches."""
    family = model_family.strip().lower()
    templates = {
        "light": (1, 2, 2, 4),
        "balanced": (1, 2, 4, 4),
        "heavy": (1, 2, 4, 8),
    }
    multipliers = templates.get(family, templates["balanced"])
    depth = max(2, min(int(depth), len(multipliers)))
    return [max(8, base_channels) * mult for mult in multipliers[:depth]]


class RGBEncoder(nn.Module):  # type: ignore[misc]
    """Encode stacked grayscale frame pairs into a configurable feature pyramid."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        model_family: str = "balanced",
    ) -> None:
        super().__init__()
        channel_schedule = build_channel_schedule(base_channels, depth, model_family)
        self.blocks = nn.ModuleList()
        prev_channels = in_channels
        for out_channels in channel_schedule:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, out_channels, 3, stride=2, padding=1),
                    _gn(out_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    _gn(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            prev_channels = out_channels
        self.channel_schedule = channel_schedule
        self.out_channels = channel_schedule[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return tuple(features)


class MultiScaleFiLM(nn.Module):  # type: ignore[misc]
    """Multi-scale IMU conditioning supporting FiLM, gated, and additive fusion."""

    def __init__(
        self, channel_schedule: list[int], imu_dim: int = 128, fusion_type: str = "film"
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type.strip().lower()

        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, imu_dim * 2),
            nn.SiLU(),
            nn.Linear(imu_dim * 2, imu_dim * 2),
        )

        self.film_scale = nn.ModuleList([nn.Linear(imu_dim * 2, c) for c in channel_schedule])
        self.film_bias = nn.ModuleList([nn.Linear(imu_dim * 2, c) for c in channel_schedule])
        for linear in [*self.film_scale, *self.film_bias]:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self, features: tuple[torch.Tensor, ...], imu_features: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        imu_proj = self.imu_proj(imu_features)

        modulated = []
        for f, scale_fn, bias_fn in zip(features, self.film_scale, self.film_bias, strict=True):
            scale = scale_fn(imu_proj).view(f.shape[0], f.shape[1], 1, 1)
            bias = bias_fn(imu_proj).view(f.shape[0], f.shape[1], 1, 1)

            if self.fusion_type == "none":
                modulated.append(f)
            elif self.fusion_type == "additive":
                modulated.append(f + bias)
            elif self.fusion_type == "gated":
                gate = torch.sigmoid(scale + 4.0)
                modulated.append(f * gate + bias)
            else:
                film_scale = 1.0 + 0.5 * torch.tanh(scale)
                modulated.append(f * film_scale + bias)

        return tuple(modulated)


class EventPredictionHead(nn.Module):  # type: ignore[misc]
    """Predict ON/OFF event probability maps from full-resolution decoder features.

    Takes feature maps already at the output spatial resolution (after U-Net decoding)
    and applies a lightweight 2-layer conv projection.  No stride, no upsampling —
    resolution management is done in EventPredictor.forward() via input padding.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


class EventPredictor(nn.Module):  # type: ignore[misc]
    """RGB + IMU → ON/OFF event probability maps with search-ready architecture knobs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rgb_encoder = RGBEncoder(
            in_channels=config.rgb_channels,
            base_channels=config.base_channels,
            depth=config.unet_depth,
            model_family=config.model_family,
        )
        self.channel_schedule = self.rgb_encoder.channel_schedule
        self.use_conditioning = config.use_imu and config.fusion_type.lower() != "none"
        if self.use_conditioning:
            self.imu_encoder = IMUEncoder(input_dim=6, hidden_dim=config.imu_hidden_dim)
            self.fusion = MultiScaleFiLM(
                self.channel_schedule,
                imu_dim=config.imu_hidden_dim,
                fusion_type=config.fusion_type,
            )

        self.up_blocks = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        for in_channels, skip_channels in zip(
            self.channel_schedule[:0:-1],
            reversed(self.channel_schedule[:-1]),
            strict=True,
        ):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, skip_channels, 4, stride=2, padding=1),
                    _gn(skip_channels),
                    nn.SiLU(inplace=True),
                )
            )
            self.merge_blocks.append(
                nn.Sequential(
                    nn.Conv2d(skip_channels * 2, skip_channels, 3, padding=1),
                    _gn(skip_channels),
                    nn.SiLU(inplace=True),
                )
            )

        head_channels = self.channel_schedule[0]
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(head_channels, head_channels, 4, stride=2, padding=1),
            _gn(head_channels),
            nn.SiLU(inplace=True),
        )
        self.event_head = EventPredictionHead(in_channels=head_channels)

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        _, _, H, W = image.shape

        pad_factor = 2 ** max(1, self.config.unet_depth)
        pad_h = (pad_factor - H % pad_factor) % pad_factor
        pad_w = (pad_factor - W % pad_factor) % pad_factor
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))

        features = list(self.rgb_encoder(image))

        if self.use_conditioning:
            imu_features = self.imu_encoder(imu_seq)
            features = list(self.fusion(tuple(features), imu_features))

        x = features[-1]
        for up_block, merge_block, skip in zip(
            self.up_blocks,
            self.merge_blocks,
            reversed(features[:-1]),
            strict=True,
        ):
            x = up_block(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = merge_block(torch.cat([x, skip], dim=1))

        x = self.final_up(x)
        return self.event_head(x)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# Optimizer and Hyperparameters (EDIT THIS)
# ---------------------------------------------------------------------------

# Model selection: "unet" (default) or "fno"
MODEL_TYPE = "unet"

# UNet / fusion architecture search knobs
BASE_CHANNELS = 32
IMU_HIDDEN_DIM = 128
FUSION_TYPE = "film"  # film | gated | additive | none
MODEL_FAMILY = "light"  # light | balanced | heavy
UNET_DEPTH = 3

# FNO hyperparameters (only used when MODEL_TYPE="fno")
FNO_MODES = 8  # Fourier modes per spatial dim
FNO_LAYERS = 4  # Number of FNO+FiLM blocks
FNO_CHANNELS = 128  # Feature channels in FNO trunk

# Training
TOTAL_BATCH_SIZE = 4
DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.01
RATE_REG_WEIGHT = 0.01

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
    # Compute the loss in float32 for stable AMP/CUDA backward passes.
    pred_c = pred.float().clamp(1e-6, 1.0 - 1e-6)

    # Threshold GT to binary targets.
    # GT is normalized by max_events=100, so ≥0.5 events/33ms → GT ≈ 0.005.
    gt_bin = (gt > 0.005).to(dtype=pred_c.dtype)
    bce = -(gt_bin * torch.log(pred_c) + (1.0 - gt_bin) * torch.log(1.0 - pred_c))
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


def compute_autoresearch_score(
    eval_metrics: dict[str, float],
    peak_vram_mb: float,
    total_seconds: float,
) -> dict[str, float]:
    """Scalarize quality + efficiency for Optuna/autoresearch decisions.

    Accuracy remains dominant, but similar-quality models are nudged toward
    lower latency, lower VRAM, and faster end-to-end completion.
    """
    samples_per_sec = max(eval_metrics.get("samples_per_sec", 0.0), 1e-6)
    latency_ms = 1000.0 / samples_per_sec
    vram_gb = max(peak_vram_mb, 0.0) / 1024.0

    quality_score = eval_metrics.get(
        "quality_score",
        0.60 * eval_metrics.get("val_ap", 0.0)
        + 0.25 * eval_metrics.get("f1_score", 0.0)
        + 0.15 * (0.5 * (eval_metrics.get("f1_on", 0.0) + eval_metrics.get("f1_off", 0.0))),
    )
    latency_score = min(1.0, 120.0 / max(latency_ms, 1.0))
    vram_score = min(1.0, 4.0 / max(vram_gb, 0.25))
    runtime_score = min(1.0, TIME_BUDGET / max(total_seconds, 1.0))
    efficiency_score = 0.45 * latency_score + 0.35 * vram_score + 0.20 * runtime_score
    autoresearch_score = 0.85 * quality_score + 0.15 * efficiency_score

    return {
        "quality_score": float(quality_score),
        "efficiency_score": float(efficiency_score),
        "latency_ms": float(latency_ms),
        "autoresearch_score": float(autoresearch_score),
    }


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
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("CUDA GPU acceleration enabled (TF32 + cudnn benchmark)")
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("MPS GPU acceleration enabled")
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        return "mps"
    print("WARNING: No GPU available, using CPU (slow)")
    return "cpu"


def create_model(device: str) -> tuple[nn.Module, int]:
    """Create model (UNet or FNO) and return it with parameter count."""
    if MODEL_TYPE == "fno":
        model: nn.Module = FNOEventPredictor(
            modes=FNO_MODES,
            fno_layers=FNO_LAYERS,
            channels=FNO_CHANNELS,
            imu_hidden_dim=IMU_HIDDEN_DIM,
        ).to(device)
    else:
        config = ModelConfig(
            base_channels=BASE_CHANNELS,
            imu_hidden_dim=IMU_HIDDEN_DIM,
            fusion_type=FUSION_TYPE,
            model_family=MODEL_FAMILY,
            unet_depth=UNET_DEPTH,
        )
        model = EventPredictor(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    return model, num_params


def optimize_model_for_device(model: nn.Module, device: str) -> nn.Module:
    """Enable optional runtime optimizations for CUDA training."""
    compile_enabled = (
        device == "cuda"
        and hasattr(torch, "compile")
        and os.getenv("V2E_TORCH_COMPILE", "1").strip().lower() not in {"0", "false", "no"}
    )
    if not compile_enabled:
        return model

    if any(isinstance(module, FourierLayer) for module in model.modules()):
        print(
            "torch.compile disabled for Fourier/FNO models: "
            "current PyTorch inductor crashes on complex FFT backward"
        )
        return model

    compile_mode = os.getenv("V2E_TORCH_COMPILE_MODE", "reduce-overhead")
    try:
        compiled_model = torch.compile(model, mode=compile_mode)
        print(f"torch.compile enabled (mode={compile_mode})")
        return compiled_model
    except Exception as exc:
        print(f"WARNING: torch.compile unavailable, continuing without it: {exc}")
        return model


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
    if torch.rand(1).item() > 0.5:
        brightness_factor = 0.7 + torch.rand(1).item() * 0.6  # 0.7–1.3
        images = (images * brightness_factor).clamp(0, 1)

    if torch.rand(1).item() > 0.5:
        images = (images + torch.randn_like(images) * 0.02).clamp(0, 1)

    if torch.rand(1).item() > 0.5:
        imu_seq = imu_seq + torch.randn_like(imu_seq) * 0.02

    return images, imu_seq, gt_events


def print_progress(
    step: int,
    progress: float,
    loss: float,
    event_loss: float,
    rate_loss: float,
    lrm: float,
    dt: float,
    remaining: float,
) -> None:
    """Print training progress with per-component loss breakdown."""
    pct_done = 100 * progress
    sps = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | "
        f"loss: {loss:.4f} evt: {event_loss:.4f} rate: {rate_loss:.4f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | sps: {sps} | rem: {remaining:.0f}s ",
        end="",
        flush=True,
    )


def _all_finite_tensors(*values: torch.Tensor) -> bool:
    """Return True when every tensor contains only finite values."""
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def update_smoothed_losses(
    smooth: dict[str, float],
    acc: dict[str, float],
    step: int,
    ema: float = 0.9,
) -> tuple[dict[str, float], float]:
    """Update EMA loss display without letting one NaN poison later logs."""
    debias = 1 - ema ** (step + 1)
    if not all(math.isfinite(acc[name]) for name in smooth):
        return smooth, debias

    for name in smooth:
        smooth[name] = ema * smooth[name] + (1 - ema) * acc[name]
    return smooth, debias


def _accumulate_step(
    model: nn.Module,
    batch: dict[str, Any],
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_accum_steps: int,
    rate_target: torch.Tensor,
) -> tuple[float, float, float] | None:
    """Run one micro-batch forward+backward; return losses or None if non-finite."""
    non_blocking = device != "cpu"
    images = batch["image"].to(device, non_blocking=non_blocking)
    imu_seq = batch["imu_seq"].to(device, non_blocking=non_blocking)
    gt_events = batch["events"].to(device, non_blocking=non_blocking)

    if model.training:
        images, imu_seq, gt_events = augment_batch(images, imu_seq, gt_events)

    # autocast: fp16 on CUDA for ~2× throughput; no-op on MPS/CPU
    autocast_type = "cuda" if device == "cuda" else "cpu"
    with torch.amp.autocast(device_type=autocast_type, enabled=(device == "cuda")):
        pred_prob = model(images, imu_seq)

    pred_for_loss = pred_prob.float()
    if not _all_finite_tensors(pred_for_loss, gt_events):
        return None

    # Focal BCE — primary loss (~4.6% positive pixels, alpha=0.75 upweights events)
    event_loss = focal_bce_loss(pred_for_loss, gt_events) / grad_accum_steps

    # Rate regularisation — per channel (ON, OFF separately) to avoid a model that
    # puts all mass into one channel satisfying the combined-mean target.
    per_ch_mean = pred_for_loss.mean(dim=(0, 2, 3))  # (2,) mean over B, H, W
    rate_reg = F.mse_loss(per_ch_mean, rate_target.to(dtype=per_ch_mean.dtype)) / grad_accum_steps

    loss = event_loss + RATE_REG_WEIGHT * rate_reg
    if not _all_finite_tensors(event_loss, rate_reg, loss):
        return None

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item(), event_loss.item(), rate_reg.item()


def run_training_loop(  # noqa: C901, PLR0912, PLR0915
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: str,
    grad_accum_steps: int = 8,
) -> tuple[float, int, int, int]:
    """Run the training loop."""
    total_training_time = 0.0
    step = 0
    smooth: dict[str, float] = {"total": 0.0, "evt": 0.0, "rate": 0.0}
    skipped_micro_batches = 0
    skipped_optimizer_steps = 0

    scaler = torch.amp.GradScaler(device="cuda") if device == "cuda" else None
    # Pre-allocate rate target once to avoid per-step tensor creation
    rate_target = torch.full((2,), 0.05, device=device)
    model.train()
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        acc = {"total": 0.0, "evt": 0.0, "rate": 0.0}
        valid_micro_batches = 0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            losses = _accumulate_step(model, batch, device, scaler, grad_accum_steps, rate_target)
            if losses is None:
                skipped_micro_batches += 1
                if skipped_micro_batches <= 3 or skipped_micro_batches % 10 == 0:
                    print(
                        f"\nWARNING: skipped non-finite micro-batch at step {step} "
                        f"(count={skipped_micro_batches})",
                        flush=True,
                    )
                continue

            total, evt, rate = losses
            valid_micro_batches += 1
            acc["total"] += total
            acc["evt"] += evt
            acc["rate"] += rate

        if valid_micro_batches > 0:
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=False
            )
            if math.isfinite(float(grad_norm)):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                skipped_optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.update()
                if skipped_optimizer_steps <= 3 or skipped_optimizer_steps % 10 == 0:
                    print(
                        f"\nWARNING: skipped optimizer step {step} because gradients became "
                        "non-finite",
                        flush=True,
                    )
        else:
            skipped_optimizer_steps += 1

        dt = time.time() - t0
        total_training_time += dt

        smooth, debias = update_smoothed_losses(
            smooth,
            acc if valid_micro_batches > 0 else {k: float("nan") for k in smooth},
            step,
        )

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
            lrm,
            dt,
            max(0.0, TIME_BUDGET - total_training_time),
        )

        step += 1
        if total_training_time >= TIME_BUDGET:
            break

    print()
    return total_training_time, step, skipped_micro_batches, skipped_optimizer_steps


def print_results(
    eval_metrics: dict[str, float],
    total_training_time: float,
    total_time: float,
    num_steps: int,
    num_params: int,
    peak_vram_mb: float,
    physics_metrics: dict[str, float] | None = None,
    skipped_micro_batches: int = 0,
    skipped_optimizer_steps: int = 0,
) -> dict[str, float]:
    """Print final results and return the scalarized autoresearch summary."""
    score_metrics = compute_autoresearch_score(eval_metrics, peak_vram_mb, total_time)

    print("---")
    # Primary metric: Average Precision (threshold-free, smooth signal for autoresearch)
    print(f"val_ap: {eval_metrics['val_ap']:.4f}")
    # Secondary metrics
    print(f"f1_score: {eval_metrics['f1_score']:.4f}")
    if physics_metrics is not None:
        delta = eval_metrics["f1_score"] - physics_metrics["f1"]
        sign = "+" if delta >= 0 else ""
        print(f"physics_f1: {physics_metrics['f1']:.4f}  (model delta: {sign}{delta:.4f})")
    print(f"f1_on: {eval_metrics['f1_on']:.4f}")
    print(f"f1_off: {eval_metrics['f1_off']:.4f}")
    print(f"precision: {eval_metrics['precision']:.4f}")
    print(f"recall: {eval_metrics['recall']:.4f}")
    print(f"event_mse: {eval_metrics['event_mse']:.6f}")
    print(f"quality_score: {score_metrics['quality_score']:.4f}")
    print(f"efficiency_score: {score_metrics['efficiency_score']:.4f}")
    # Training stats
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds: {total_time:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"samples_per_sec: {eval_metrics['samples_per_sec']:.1f}")
    print(f"latency_ms: {score_metrics['latency_ms']:.2f}")
    print(f"autoresearch_score: {score_metrics['autoresearch_score']:.4f}")
    print(f"num_steps: {num_steps}")
    print(f"skipped_micro_batches: {skipped_micro_batches}")
    print(f"skipped_optimizer_steps: {skipped_optimizer_steps}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"model_type: {MODEL_TYPE}")
    print(f"base_channels: {BASE_CHANNELS}")
    print(f"imu_hidden_dim: {IMU_HIDDEN_DIM}")
    print(f"fusion_type: {FUSION_TYPE}")
    print(f"model_family: {MODEL_FAMILY}")
    print(f"unet_depth: {UNET_DEPTH}")
    print(f"fno_modes: {FNO_MODES}")
    print(f"fno_layers: {FNO_LAYERS}")
    print(f"fno_channels: {FNO_CHANNELS}")
    return score_metrics


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
            "model_type": MODEL_TYPE,
            "base_channels": BASE_CHANNELS,
            "imu_hidden_dim": IMU_HIDDEN_DIM,
            "fusion_type": FUSION_TYPE,
            "model_family": MODEL_FAMILY,
            "unet_depth": UNET_DEPTH,
            "fno_modes": FNO_MODES,
            "fno_layers": FNO_LAYERS,
            "fno_channels": FNO_CHANNELS,
        },
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to: {filepath}")


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

    print(f"Checkpoint loaded from: {filepath} (epoch {epoch}, loss {loss:.6f})")
    return model, optimizer, epoch, loss


def train(resume_from: str | None = None) -> None:  # noqa: PLR0915
    """Main training function."""
    mlflow_run = MlflowRunManager(run_name=os.getenv("MLFLOW_RUN_NAME"))
    status = "FINISHED"

    try:
        device = setup_device()
        print(f"Device: {device}")
        print(f"Time budget: {TIME_BUDGET}s")

        model, num_params = create_model(device)
        model = optimize_model_for_device(model, device)
        print(f"Model parameters: {num_params / 1e6:.2f}M")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        compile_requested = device == "cuda" and os.getenv(
            "V2E_TORCH_COMPILE", "1"
        ).strip().lower() not in {"0", "false", "no"}
        mlflow_tags = {"device": device}
        for env_name, tag_name in {
            "AUTORESEARCH_SEARCH_KIND": "autoresearch.strategy",
            "AUTORESEARCH_EXPERIMENT_NAME": "autoresearch.experiment",
            "AUTORESEARCH_TRIAL_NUMBER": "autoresearch.trial_number",
        }.items():
            value = os.getenv(env_name, "").strip()
            if value:
                mlflow_tags[tag_name] = value

        mlflow_run.start(
            params={
                "model_type": MODEL_TYPE,
                "base_channels": BASE_CHANNELS,
                "imu_hidden_dim": IMU_HIDDEN_DIM,
                "fusion_type": FUSION_TYPE,
                "model_family": MODEL_FAMILY,
                "unet_depth": UNET_DEPTH,
                "fno_modes": FNO_MODES,
                "fno_layers": FNO_LAYERS,
                "fno_channels": FNO_CHANNELS,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "device_batch_size": DEVICE_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "warmdown_ratio": WARMDOWN_RATIO,
                "final_lr_frac": FINAL_LR_FRAC,
                "final_eval_batch_size": FINAL_EVAL_BATCH_SIZE,
                "time_budget_s": TIME_BUDGET,
                "resume_from": resume_from or "",
                "torch_compile_requested": compile_requested,
                "torch_compile_mode": os.getenv("V2E_TORCH_COMPILE_MODE", "reduce-overhead"),
                "dataloader_workers": os.getenv("DATALOADER_WORKERS", "auto"),
            },
            tags=mlflow_tags,
        )

        # Create dataloaders — val uses train IMU stats to avoid leakage
        train_loader = make_dataloader(
            DATA_DIR, "train", DEVICE_BATCH_SIZE, MAX_SEQ_LEN, IMAGE_SIZE
        )
        train_imu_stats = (
            train_loader.dataset._imu_mean,
            train_loader.dataset._imu_std,
        )
        val_loader = make_dataloader(
            DATA_DIR,
            "val",
            FINAL_EVAL_BATCH_SIZE,
            MAX_SEQ_LEN,
            IMAGE_SIZE,
            imu_stats=train_imu_stats,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        grad_accum_steps = max(1, math.ceil(TOTAL_BATCH_SIZE / DEVICE_BATCH_SIZE))
        effective_batch_size = grad_accum_steps * DEVICE_BATCH_SIZE
        print(
            f"Gradient accumulation steps: {grad_accum_steps} (effective batch size {effective_batch_size})"
        )

        if resume_from and os.path.exists(resume_from):
            model, optimizer, start_epoch, _ = load_checkpoint(
                resume_from, model, optimizer, device
            )
            print(f"Resuming from epoch {start_epoch}")

        t_start = time.time()
        (
            total_training_time,
            num_steps,
            skipped_micro_batches,
            skipped_optimizer_steps,
        ) = run_training_loop(model, optimizer, train_loader, device, grad_accum_steps)

        t_train = time.time()
        print(f"Training completed in {t_train - t_start:.1f}s")

        checkpoint_path = Path("event_predictor_checkpoint.pt")
        save_checkpoint(model, optimizer, num_steps, total_training_time, str(checkpoint_path))

        del train_loader

        print("Starting final eval...")
        physics_metrics = evaluate_physics_baseline(val_loader, device, EVAL_SAMPLES)
        print(
            f"Physics baseline: F1={physics_metrics['f1']:.4f} "
            f"P={physics_metrics['precision']:.4f} R={physics_metrics['recall']:.4f}"
        )
        if physics_metrics["f1"] < 0.01:
            print(
                "WARNING: physics baseline F1≈0 — image files may be missing. "
                "Run: python download_fpv.py --sequence indoor_forward_3"
            )

        eval_metrics = evaluate_combined_metric(model, val_loader, device, EVAL_SAMPLES)
        t_eval = time.time()
        print(f"Final eval completed in {t_eval - t_train:.1f}s")

        del val_loader

        peak_vram_mb = get_peak_memory_mb()
        score_metrics = print_results(
            eval_metrics,
            total_training_time,
            t_eval - t_start,
            num_steps,
            num_params,
            peak_vram_mb,
            physics_metrics,
            skipped_micro_batches,
            skipped_optimizer_steps,
        )

        mlflow_run.set_tags({"run.status": "finished"})
        mlflow_run.log_metrics(
            {
                "val_ap": eval_metrics["val_ap"],
                "f1_score": eval_metrics["f1_score"],
                "f1_on": eval_metrics["f1_on"],
                "f1_off": eval_metrics["f1_off"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "event_mse": eval_metrics["event_mse"],
                "quality_score": score_metrics["quality_score"],
                "efficiency_score": score_metrics["efficiency_score"],
                "autoresearch_score": score_metrics["autoresearch_score"],
                "samples_per_sec": eval_metrics["samples_per_sec"],
                "latency_ms": score_metrics["latency_ms"],
                "physics_f1": physics_metrics["f1"],
                "training_seconds": total_training_time,
                "total_seconds": t_eval - t_start,
                "peak_vram_mb": peak_vram_mb,
                "num_steps": float(num_steps),
                "num_params_m": num_params / 1e6,
                "skipped_micro_batches": float(skipped_micro_batches),
                "skipped_optimizer_steps": float(skipped_optimizer_steps),
            },
            step=num_steps,
        )
        mlflow_run.log_artifacts(
            [
                checkpoint_path,
                Path("run.log"),
                Path(__file__),
                Path("prepare_data.py"),
            ]
        )
    except Exception as exc:
        status = "FAILED"
        mlflow_run.set_tags(
            {
                "run.status": "failed",
                "exception_type": type(exc).__name__,
            }
        )
        mlflow_run.log_text(f"{type(exc).__name__}: {exc}\n", "exception.txt")
        raise
    finally:
        mlflow_run.end(status=status)


if __name__ == "__main__":
    train()
