#!/usr/bin/env python
"""
Knowledge Distillation: Compress 3D-Aware Teacher → Compact Student.

Teacher: Multimodal spatiotemporal RGB+IMU+Depth model (1.19M params, 0.000141 event_bpb)
Student: Compact RGB+IMU model (~0.3M params, target 0.000155 event_bpb)

Usage:
    uv run python train_distill.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_data import (
    DATA_DIR,
    EVAL_SAMPLES,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    make_dataloader,
)
from train import BASE_CHANNELS, IMU_HIDDEN_DIM, EventPredictor, ModelConfig

# Distillation configuration
DISTILLATION_WEIGHT = 0.7  # 70% teacher, 30% GT
GRAD_ACCUM_STEPS = 8
BATCH_SIZE = 4


class CompactStudent(nn.Module):  # type: ignore[misc]
    """Compact student model for knowledge distillation."""

    def __init__(self, base_channels: int = 16, imu_hidden_dim: int = 64) -> None:
        super().__init__()
        self.imu_lstm = nn.LSTM(6, imu_hidden_dim, num_layers=1, batch_first=True)
        self.imu_fc = nn.Linear(imu_hidden_dim, imu_hidden_dim)
        self.imu_norm = nn.LayerNorm(imu_hidden_dim)

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(inplace=True),
        )

        self.film_scale = nn.Sequential(
            nn.Linear(imu_hidden_dim, base_channels * 2),
            nn.Sigmoid(),
        )
        self.film_bias = nn.Sequential(
            nn.Linear(imu_hidden_dim, base_channels * 2),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 2, 3, padding=1),
        )

        self.out_size = IMAGE_SIZE

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.imu_lstm(imu_seq)
        imu_features = self.imu_fc(h_n[0])
        imu_features = self.imu_norm(imu_features)

        rgb_features = self.rgb_encoder(image)

        B, C, H, W = rgb_features.shape
        scale = self.film_scale(imu_features).view(B, C, 1, 1)
        bias = self.film_bias(imu_features).view(B, C, 1, 1)
        rgb_modulated = rgb_features * scale + bias

        events = self.decoder(rgb_modulated)

        if events.shape[2:] != self.out_size:
            events = F.interpolate(events, size=self.out_size, mode="bilinear", align_corners=False)

        return events


class TeacherWrapper(nn.Module):  # type: ignore[misc]
    """Wrapper to load and run teacher model."""

    def __init__(self, base_channels: int = 32, imu_hidden_dim: int = 128) -> None:
        super().__init__()
        config = ModelConfig(base_channels=base_channels, imu_hidden_dim=imu_hidden_dim)
        self.teacher = EventPredictor(config)
        self.teacher.eval()

    def forward(self, image: torch.Tensor, imu_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            events, _ = self.teacher(image, imu_seq)
            return events


def distill(  # noqa: PLR0915
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    time_budget: float,
) -> None:
    """Distill knowledge from teacher to student."""
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())

    print(f"Distilling {teacher_params / 1e6:.2f}M → {student_params / 1e6:.2f}M")
    print(f"Compression ratio: {teacher_params / student_params:.1f}x")
    print(f"Distillation weight: {DISTILLATION_WEIGHT} (teacher) / {1 - DISTILLATION_WEIGHT} (GT)")

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-3, weight_decay=0.0)

    student_model.train()
    teacher_model.eval()
    train_iter = iter(train_loader)
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0

    print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    print()

    while True:
        t0 = time.time()
        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            with torch.no_grad():
                teacher_events = teacher_model(images, imu_seq)

            student_events = student_model(images, imu_seq)

            distill_loss = F.mse_loss(student_events, teacher_events)
            gt_loss = F.mse_loss(student_events, gt_events)

            loss = (1 - DISTILLATION_WEIGHT) * gt_loss + DISTILLATION_WEIGHT * distill_loss
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accumulated_loss += loss.item()

        optimizer.step()

        dt = time.time() - t0
        total_training_time += dt
        step += 1

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * accumulated_loss
        debiased_loss = smooth_loss / (1 - ema_beta**step)

        if step % 10 == 0 or step == 1:
            progress = min(total_training_time / time_budget, 1.0)
            remaining = max(0.0, time_budget - total_training_time)
            print(
                f"\rstep {step:05d} ({progress * 100:.1f}%) | loss: {debiased_loss:.6f} | "
                f"dt: {dt * 1000:.0f}ms | remaining: {remaining:.0f}s ",
                end="",
                flush=True,
            )

        if total_training_time >= time_budget:
            break

    print()
    print(f"Distillation completed in {total_training_time:.1f}s")

    # Evaluation
    print("Starting final eval...")
    student_model.eval()

    total_event_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            if total_samples >= EVAL_SAMPLES:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            student_events = student_model(images, imu_seq)
            event_loss = F.mse_loss(student_events, gt_events, reduction="sum")

            total_event_loss += event_loss.item()
            total_samples += images.shape[0]

    avg_event_mse = total_event_loss / (total_samples * gt_events[0].numel())
    event_bpb = avg_event_mse / 0.693

    print()
    print("---")
    print(f"event_bpb: {event_bpb:.6f}")
    print(f"event_mse: {avg_event_mse:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"num_params_M: {student_params / 1e6:.2f}")
    print()
    print(f"✨ DISTILLATION COMPLETE! Compact model achieves event_bpb={event_bpb:.6f}")
    print("   Teacher: 0.000141 (1.19M params)")
    print(f"   Student: {event_bpb:.6f} ({student_params / 1e6:.2f}M params)")
    print(f"   Compression: {teacher_params / student_params:.1f}x")


def train() -> None:
    """Main distillation training."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Time budget: {TIME_BUDGET}s")
    print("Knowledge Distillation: 3D-Aware Teacher → Compact Student")
    print()

    print("Creating teacher model...")
    teacher_model = TeacherWrapper(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)

    print("Creating student model...")
    student_model = CompactStudent(base_channels=16, imu_hidden_dim=64)

    print("Loading data...")
    train_loader = make_dataloader(DATA_DIR, "train", BATCH_SIZE, MAX_SEQ_LEN, IMAGE_SIZE)
    val_loader = make_dataloader(DATA_DIR, "val", 16, MAX_SEQ_LEN, IMAGE_SIZE)

    print()
    distill(
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        device,
        TIME_BUDGET,
    )


if __name__ == "__main__":
    train()
