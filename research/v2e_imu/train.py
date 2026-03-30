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


class IMUEncoder(nn.Module):
    """Encode IMU sequences into feature vectors."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        # imu_seq: (B, seq_len, 6)
        lstm_out, (h_n, _) = self.lstm(imu_seq)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=-1)
        imu_features = self.fc(h_cat)
        imu_features = self.norm(imu_features)
        return imu_features


class RGBEncoder(nn.Module):
    """Encode grayscale images into feature maps."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class IMUConditionedFusion(nn.Module):
    """Modulate RGB features with IMU information."""

    def __init__(self, rgb_channels: int, imu_dim: int = 128):
        super().__init__()
        self.imu_to_scale = nn.Sequential(nn.Linear(imu_dim, rgb_channels), nn.Sigmoid())
        self.imu_to_bias = nn.Sequential(nn.Linear(imu_dim, rgb_channels), nn.Tanh())

    def forward(self, rgb_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_features.shape
        scale = self.imu_to_scale(imu_features).view(B, C, 1, 1)
        bias = self.imu_to_bias(imu_features).view(B, C, 1, 1)
        return rgb_features * scale + bias


class EventPredictionHead(nn.Module):
    """Predict events from features."""

    def __init__(self, in_channels: int, out_size: tuple[int, int]):
        super().__init__()
        self.out_size = out_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        events = self.decoder(x)
        if events.shape[2:] != self.out_size:
            events = F.interpolate(events, size=self.out_size, mode="bilinear", align_corners=False)
        return events


class EventPredictor(nn.Module):
    """
    Main model: RGB + IMU -> Events

    IMU conditions RGB processing.
    Events are predicted as 2-channel map (positive/negative).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.imu_encoder = IMUEncoder(input_dim=6, hidden_dim=config.imu_hidden_dim)
        self.rgb_encoder = RGBEncoder(
            in_channels=config.rgb_channels, base_channels=config.base_channels
        )
        self.fusion = IMUConditionedFusion(
            rgb_channels=self.rgb_encoder.out_channels, imu_dim=config.imu_hidden_dim
        )
        self.event_head = EventPredictionHead(
            in_channels=self.rgb_encoder.out_channels, out_size=config.image_size
        )

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        # Encode
        imu_features = self.imu_encoder(imu_seq)
        rgb_features = self.rgb_encoder(image)

        # Fuse
        fused_features = self.fusion(rgb_features, imu_features)

        # Predict events
        events = self.event_head(fused_features)

        return events


# ---------------------------------------------------------------------------
# Optimizer and Hyperparameters (EDIT THIS)
# ---------------------------------------------------------------------------

# Model architecture
BASE_CHANNELS = 32
IMU_HIDDEN_DIM = 128

# Training
TOTAL_BATCH_SIZE = 32
DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
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
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def train():
    """Main training function."""
    # Setup device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")

    # Create model
    config = ModelConfig(
        base_channels=BASE_CHANNELS,
        imu_hidden_dim=IMU_HIDDEN_DIM,
    )
    model = EventPredictor(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create dataloaders
    train_loader = make_dataloader(
        data_dir=DATA_DIR,
        split="train",
        batch_size=DEVICE_BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        image_size=IMAGE_SIZE,
    )

    val_loader = make_dataloader(
        data_dir=DATA_DIR,
        split="val",
        batch_size=FINAL_EVAL_BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
        image_size=IMAGE_SIZE,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Gradient accumulation
    grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Training loop
    t_start = time.time()
    total_training_time = 0.0
    step = 0
    smooth_train_loss = 0.0

    model.train()
    train_iter = iter(train_loader)

    while True:
        t0 = time.time()

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["image"].to(device)
        imu_seq = batch["imu_seq"].to(device)
        gt_events = batch["events"].to(device)

        # Forward pass
        pred_events = model(images, imu_seq)
        loss = F.mse_loss(pred_events, gt_events)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update timing
        dt = time.time() - t0
        total_training_time += dt

        # Update smooth loss
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item()
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        # Learning rate schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for param_group in optimizer.param_groups:
            param_group["lr"] = LEARNING_RATE * lrm

        # Print progress
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
        remaining = max(0.0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"remaining: {remaining:.0f}s ",
            end="",
            flush=True,
        )

        step += 1

        # Check if done
        if total_training_time >= TIME_BUDGET:
            break

        # Garbage collection
        if step % 1000 == 0:
            gc.collect()

    print()
    t_train = time.time()
    print(f"Training completed in {t_train - t_start:.1f}s")

    # Final evaluation
    print("Starting final eval...")
    eval_metrics = evaluate_combined_metric(model, val_loader, device, EVAL_SAMPLES)
    t_eval = time.time()
    print(f"Final eval completed in {t_eval - t_train:.1f}s")

    # Get memory usage
    peak_vram_mb = get_peak_memory_mb()

    # Print results in standard format
    print("---")
    print(f"event_bpb: {eval_metrics['event_bpb']:.6f}")
    print(f"event_mse: {eval_metrics['event_mse']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds: {t_eval - t_start:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"samples_per_sec: {eval_metrics['samples_per_sec']:.1f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    print(f"base_channels: {BASE_CHANNELS}")
    print(f"imu_hidden_dim: {IMU_HIDDEN_DIM}")


if __name__ == "__main__":
    train()
