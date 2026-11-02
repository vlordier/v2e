#!/usr/bin/env python
"""
Structured Pruning: Remove redundant filters based on L1 importance.

Usage:
    uv run python apply_pruning.py --prune-ratio 0.5

This script:
1. Loads a trained model
2. Computes filter importance (L1 norm)
3. Prunes bottom X% of filters
4. Evaluates pruned model
"""

import argparse

import torch
import torch.nn as nn
from prepare_data import (
    DATA_DIR,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    make_dataloader,
)


class CompactStudent(nn.Module):  # type: ignore[misc]
    """Compact student model for pruning."""

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
            events = nn.functional.interpolate(
                events, size=self.out_size, mode="bilinear", align_corners=False
            )

        return events


def compute_importance(model: nn.Module) -> dict[str, torch.Tensor]:
    """Compute L1 importance score for each conv filter."""
    importance: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            weight = module.weight.data
            l1_norm = weight.abs().sum(dim=(1, 2, 3))
            importance[name] = l1_norm

    return importance


def prune_model(
    model: nn.Module, importance: dict[str, torch.Tensor], prune_ratio: float = 0.5
) -> tuple[nn.Module, int, int]:
    """Prune bottom X% of filters."""
    print(f"Pruning ratio: {prune_ratio * 100:.0f}%")

    total_filters = 0
    pruned_filters = 0

    for name, l1_norm in importance.items():
        module = dict(model.named_modules())[name]

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            num_filters = l1_norm.numel()
            total_filters += num_filters

            k = int(num_filters * prune_ratio)
            if k == 0:
                continue

            threshold = l1_norm.kthvalue(k).values
            mask = (l1_norm > threshold).float()

            module.weight.data *= mask.view(-1, 1, 1, 1)
            pruned_filters += int((1 - mask).sum().item())

    print(f"Total filters: {total_filters}")
    print(f"Pruned filters: {pruned_filters} ({pruned_filters / total_filters * 100:.1f}%)")

    return model, total_filters, pruned_filters


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> float:
    """Quick evaluation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= 20:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            pred_events = model(images, imu_seq)
            loss = nn.functional.mse_loss(pred_events, gt_events)

            total_loss += loss.item()
            total_samples += images.shape[0]

    return total_loss / total_samples


def main() -> None:
    """Main pruning evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-ratio", type=float, default=0.5, help="Pruning ratio (0.5 = 50%)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print("Structured Pruning (L1-based)")
    print()

    # Create model
    print("Creating model...")
    model = CompactStudent(base_channels=16, imu_hidden_dim=64)
    model = model.to(device)

    print("Note: Using untrained model. Load trained weights for real pruning.")
    print()

    # Evaluate before pruning
    print("=== Before Pruning ===")
    val_loader = make_dataloader(DATA_DIR, "val", 16, MAX_SEQ_LEN, IMAGE_SIZE)
    loss_before = evaluate(model, val_loader, device)
    print(f"Loss: {loss_before:.6f}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # Compute importance
    print("=== Computing Filter Importance ===")
    importance = compute_importance(model)
    for name, scores in importance.items():
        print(f"  {name}: {scores.numel()} filters, L1 mean={scores.mean().item():.4f}")
    print()

    # Prune
    print("=== Pruning ===")
    model_pruned, total, pruned = prune_model(model, importance, args.prune_ratio)
    print()

    # Evaluate after pruning
    print("=== After Pruning ===")
    loss_after = evaluate(model_pruned, val_loader, device)
    print(f"Loss: {loss_after:.6f}")
    print(f"Params: {sum(p.numel() for p in model_pruned.parameters()) / 1e6:.2f}M")
    print()

    # Summary
    print("=== Summary ===")
    if loss_before > 0:
        print(f"Loss change: {(loss_after - loss_before) / loss_before * 100:.2f}%")
    print(f"Filters pruned: {pruned}/{total} ({pruned / total * 100:.1f}%)")
    print()
    print("Next step: Fine-tune pruned model to recover accuracy")


if __name__ == "__main__":
    main()
