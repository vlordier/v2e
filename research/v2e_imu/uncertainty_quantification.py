#!/usr/bin/env python
"""
Uncertainty Quantification via MC Dropout.

Usage:
    uv run python uncertainty_quantification.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from train import BASE_CHANNELS, IMU_HIDDEN_DIM, EventPredictor, ModelConfig


def enable_dropout(model):
    """Enable dropout during inference for MC Dropout."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active


def mc_dropout_uncertainty(model, rgb, imu, num_samples=10):
    """
    Estimate uncertainty using MC Dropout.
    
    Args:
        model: PyTorch model with dropout
        rgb: Input RGB tensor (B, 1, H, W)
        imu: Input IMU tensor (B, T, 6)
        num_samples: Number of Monte Carlo samples
    
    Returns:
        mean: Mean prediction (B, 2, H, W)
        variance: Predictive variance (B, 2, H, W)
        std: Predictive standard deviation (B, 2, H, W)
    """
    model.train()  # Enable dropout
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred_events, pred_depth = model(rgb, imu)
            predictions.append(pred_events)
    
    model.eval()
    
    # Stack predictions
    predictions = torch.stack(predictions, dim=0)  # (N, B, 2, H, W)
    
    # Compute statistics
    mean = predictions.mean(dim=0)  # (B, 2, H, W)
    variance = predictions.var(dim=0)  # (B, 2, H, W)
    std = torch.sqrt(variance)  # (B, 2, H, W)
    
    return mean, variance, std


def quantify_uncertainty():
    """Run uncertainty quantification on sample data."""
    print("="*70)
    print("UNCERTAINTY QUANTIFICATION (MC Dropout)")
    print("="*70)
    print()
    
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config).to(device)
    
    # Load checkpoint
    checkpoint_path = Path("3d_aware_model_checkpoint.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✅ Checkpoint loaded")
    print()
    
    # Create sample input
    batch_size = 2
    rgb = torch.randn(batch_size, 1, 260, 346).to(device)
    imu = torch.randn(batch_size, 50, 6).to(device)
    
    print(f"Input shape: RGB {rgb.shape}, IMU {imu.shape}")
    print()
    
    # Run MC Dropout
    print("Running MC Dropout (10 samples)...")
    mean, variance, std = mc_dropout_uncertainty(model, rgb, imu, num_samples=10)
    
    print("✅ Uncertainty estimated")
    print()
    
    # Print statistics
    print("="*70)
    print("UNCERTAINTY STATISTICS")
    print("="*70)
    print(f"Prediction mean:   {mean.mean().item():.6f} ± {mean.std().item():.6f}")
    print(f"Prediction std:    {std.mean().item():.6f} (epistemic uncertainty)")
    print(f"Variance (mean):   {variance.mean().item():.8f}")
    print(f"Variance (max):    {variance.max().item():.8f}")
    print()
    
    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Coefficient of variation (CV)
    cv = std / (mean + 1e-6)
    print(f"Coefficient of Variation (CV): {cv.mean().item():.4f}")
    
    if cv.mean().item() < 0.1:
        print("✅ Low uncertainty - Model is confident")
    elif cv.mean().item() < 0.3:
        print("⚠️  Moderate uncertainty - Some regions uncertain")
    else:
        print("❌ High uncertainty - Model is uncertain")
    
    # Spatial uncertainty map
    spatial_uncertainty = std.mean(dim=1)  # Average over channels
    print(f"\nSpatial uncertainty map shape: {spatial_uncertainty.shape}")
    print(f"  Min: {spatial_uncertainty.min().item():.6f}")
    print(f"  Max: {spatial_uncertainty.max().item():.6f}")
    print(f"  Mean: {spatial_uncertainty.mean().item():.6f}")
    
    # High uncertainty regions
    high_uncertainty_mask = spatial_uncertainty > spatial_uncertainty.mean() + 2 * spatial_uncertainty.std()
    high_uncertainty_ratio = high_uncertainty_mask.float().mean().item() * 100
    print(f"\nHigh uncertainty regions (>2σ): {high_uncertainty_ratio:.2f}% of pixels")
    
    print()
    print("="*70)
    print("USAGE EXAMPLE")
    print("="*70)
    print("""
# For safety-critical applications:
mean, variance, std = mc_dropout_uncertainty(model, rgb, imu)

# Use mean for prediction
events = mean

# Use std for confidence
confidence_mask = std < threshold  # Regions where model is confident

# Reject uncertain predictions
safe_events = events * (std < threshold).float()
    """)
    
    # Save results
    import json
    results = {
        "prediction_mean": float(mean.mean().item()),
        "prediction_std": float(std.mean().item()),
        "variance_mean": float(variance.mean().item()),
        "coefficient_of_variation": float(cv.mean().item()),
        "high_uncertainty_ratio": high_uncertainty_ratio,
        "interpretation": "Low uncertainty" if cv.mean().item() < 0.1 else "Moderate uncertainty" if cv.mean().item() < 0.3 else "High uncertainty",
    }
    
    with open("uncertainty_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to uncertainty_results.json")
    
    return mean, variance, std


if __name__ == "__main__":
    quantify_uncertainty()
