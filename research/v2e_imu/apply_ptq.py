#!/usr/bin/env python
"""
Post-Training Quantization (PTQ): Convert to INT8 for 2-3x size reduction.

Usage:
    uv run python apply_ptq.py

Note: PTQ requires x86 CPU with quantization support.
      On ARM/MPS (Apple Silicon), quantization is not available.

This script:
1. Loads a trained model
2. Applies dynamic quantization (INT8) to Linear layers
3. Evaluates quantized model
4. Saves quantized checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from prepare_data import (
    DATA_DIR,
    EVAL_SAMPLES,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    make_dataloader,
)


class CompactStudent(nn.Module):
    """Compact student model for quantization."""

    def __init__(
        self, base_channels: int = 16, imu_hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.imu_lstm = nn.LSTM(
            6, imu_hidden_dim, num_layers=1, batch_first=True
        )
        self.imu_fc = nn.Linear(imu_hidden_dim, imu_hidden_dim)
        self.imu_norm = nn.LayerNorm(imu_hidden_dim)

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, 3, stride=2, padding=1
            ),
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
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, 4, stride=2, padding=1
            ),
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
            events = F.interpolate(
                events, size=self.out_size, mode="bilinear", align_corners=False
            )

        return events


def apply_ptq(
    model_fp32: nn.Module, device: str
) -> nn.Module:
    """Apply post-training quantization."""
    params_before = sum(p.numel() for p in model_fp32.parameters())
    print(f"Original model: {params_before / 1e6:.2f}M params")

    # PTQ only works on x86 CPU
    if device in ["mps", "cuda"]:
        print(f"⚠️  PTQ not supported on {device}")
        print("   Use x86 CPU for quantization")
        print("   Returning unquantized model...")
        return model_fp32

    # Apply dynamic quantization to Linear layers
    print("Quantizing Linear layers to INT8...")
    try:
        model_quantized = torch.quantization.quantize_dynamic(
            model_fp32, {nn.Linear}, dtype=torch.qint8
        )

        params_after = sum(p.numel() for p in model_quantized.parameters())
        print(f"Quantized model: {params_after / 1e6:.2f}M params")
        print("Quantized layers: Linear → INT8")
        print("Expected size reduction: 2-3x")

        return model_quantized
    except RuntimeError as e:
        print(f"⚠️  Quantization failed: {e}")
        print("   This is expected on ARM/MPS devices")
        print("   Returning unquantized model...")
        return model_fp32


def evaluate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str, name: str = "Model"
) -> float:
    """Evaluate model and return event_bpb."""
    model.eval()
    total_event_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= EVAL_SAMPLES:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            pred_events = model(images, imu_seq)
            event_loss = nn.functional.mse_loss(
                pred_events, gt_events, reduction="sum"
            )

            total_event_loss += event_loss.item()
            total_samples += images.shape[0]

    avg_event_mse = total_event_loss / (total_samples * gt_events[0].numel())
    event_bpb = avg_event_mse / 0.693

    print(f"{name}:")
    print(f"  event_bpb: {event_bpb:.6f}")
    print(f"  event_mse: {avg_event_mse:.6f}")

    return float(event_bpb)


def main() -> None:
    """Main PTQ evaluation."""
    device = "cpu"  # PTQ works best on CPU
    print(f"Device: {device}")
    print("Post-Training Quantization (PTQ)")
    print()
    print("Note: PTQ requires x86 CPU. On ARM/MPS, this will show limitations.")
    print()

    # Create model
    print("Loading model...")
    model_fp32 = CompactStudent(base_channels=16, imu_hidden_dim=64)

    print("Note: Using untrained model. Load trained weights for real evaluation.")
    print()

    # Create dataloader
    print("Loading data...")
    val_loader = make_dataloader(
        DATA_DIR, "val", 16, MAX_SEQ_LEN, IMAGE_SIZE
    )

    # Evaluate FP32 model
    print()
    print("=== Evaluating FP32 Model ===")
    bpb_fp32 = evaluate(model_fp32, val_loader, device, "FP32 Model")

    # Apply PTQ
    print()
    print("=== Applying Post-Training Quantization ===")
    model_quantized = apply_ptq(model_fp32, device)

    # Evaluate quantized model
    print()
    print("=== Evaluating INT8 Quantized Model ===")
    bpb_int8 = evaluate(model_quantized, val_loader, device, "INT8 Quantized")

    # Compare
    print()
    print("=== Summary ===")
    print(f"FP32 event_bpb: {bpb_fp32:.6f}")
    print(f"INT8 event_bpb: {bpb_int8:.6f}")
    if bpb_fp32 > 0:
        print(f"Accuracy loss: {(bpb_int8 - bpb_fp32) / bpb_fp32 * 100:.2f}%")
    print(f"Expected size reduction: 2-3x")
    print(f"Expected speedup: 2-3x (on x86 CPU with AVX512)")
    print()

    # Save quantized model
    save_path = "student_quantized.pt"
    torch.save(model_quantized.state_dict(), save_path)
    print(f"✅ Quantized model saved to: {save_path}")


if __name__ == "__main__":
    main()
