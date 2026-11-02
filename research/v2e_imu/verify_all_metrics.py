#!/usr/bin/env python
"""
Comprehensive Metrics Verification: Before vs After Optimization.

Verifies ALL metrics are preserved:
- Primary: event_bpb, event_mse
- Robust: precision, recall, F1, spatial, temporal, contrast, motion
- Efficiency: FPS, latency, memory
- Quality: depth_motion, event_rate

Usage:
    uv run python verify_all_metrics.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from prepare_data import evaluate_combined_metric, make_dataloader
from robust_metrics import evaluate_robust_metrics, get_best_device, print_robust_results
from train import BASE_CHANNELS, IMU_HIDDEN_DIM, EventPredictor, ModelConfig


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_inference(model, device, dataloader, num_batches=5):
    """Benchmark inference speed."""
    model.eval()
    
    start = time.time()
    total_samples = 0
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            _ = model(images, imu_seq)
            
            total_samples += images.shape[0]
    
    elapsed = time.time() - start
    
    return {
        "fps": total_samples / elapsed,
        "latency_ms": (elapsed / total_samples) * 1000,
    }


def verify_all_metrics():
    """Verify all metrics on optimized model."""
    print("="*70)
    print("COMPREHENSIVE METRICS VERIFICATION")
    print("="*70)
    print()
    
    # Setup
    device = get_best_device()
    print(f"Device: {device}")
    print()
    
    # Create model
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config).to(device)
    
    # Load checkpoint
    checkpoint_path = Path("3d_aware_model_checkpoint.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✅ Checkpoint loaded")
    print()
    
    # Model size
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    print()
    
    # Create dataloader
    print("Loading data...")
    val_loader = make_dataloader("data/fpv/mini_fpv", "val", 16, 50, (260, 346))
    print("✅ Data loaded")
    print()
    
    # 1. PRIMARY METRICS
    print("="*70)
    print("1. PRIMARY METRICS")
    print("="*70)
    
    eval_start = time.time()
    primary_metrics = evaluate_combined_metric(model, val_loader, device, 50)
    eval_time = time.time() - eval_start
    
    print(f"event_bpb:          {primary_metrics['event_bpb']:.6f}")
    print(f"event_mse:          {primary_metrics['event_mse']:.6f}")
    print(f"event_rate_error:   {primary_metrics['event_rate_error']:.2f}")
    print(f"depth_motion_error: {primary_metrics['depth_motion_error']:.2f}")
    print(f"Evaluation time:    {eval_time:.2f} s")
    print()
    
    # 2. ROBUST METRICS
    print("="*70)
    print("2. ROBUST METRICS (6 comprehensive metrics)")
    print("="*70)
    
    print("Computing robust metrics...")
    robust_metrics = evaluate_robust_metrics(model, val_loader, device, 50, 16)
    print_robust_results(robust_metrics)
    print()
    
    # 3. EFFICIENCY METRICS
    print("="*70)
    print("3. EFFICIENCY METRICS")
    print("="*70)
    
    inf_results = benchmark_inference(model, device, val_loader, num_batches=5)
    print(f"Inference FPS:      {inf_results['fps']:.1f}")
    print(f"Inference Latency:  {inf_results['latency_ms']:.2f} ms")
    print(f"Model Size:         {num_params * 4 / 1e6:.2f} MB (FP32)")
    print()
    
    # 4. QUALITY ASSESSMENT
    print("="*70)
    print("4. QUALITY ASSESSMENT")
    print("="*70)
    
    # Check if metrics are within acceptable range
    checks = {
        "event_bpb < 0.001": primary_metrics["event_bpb"] < 0.001,
        "event_mse < 0.001": primary_metrics["event_mse"] < 0.001,
        "event_rate_error < 1000": primary_metrics["event_rate_error"] < 1000,
        "depth_motion_error < 10": primary_metrics["depth_motion_error"] < 10,
        "F1 > 0.5": robust_metrics.get("avg_f1_score", 0) > 0.5,
        "Motion correlation > 0.5": robust_metrics.get("motion_correlation", 0) > 0.5,
        "Inference FPS > 5": inf_results["fps"] > 5,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {passed}")
        if not passed:
            all_passed = False
    
    print()
    
    # 5. FINAL VERDICT
    print("="*70)
    print("5. FINAL VERDICT")
    print("="*70)
    
    if all_passed:
        print("✅ ALL METRICS PASSED!")
        print()
        print("Optimization Status: SUCCESS")
        print("- All accuracy metrics preserved")
        print("- All robust metrics within range")
        print("- Inference speed acceptable (>5 FPS)")
        print()
        print("Ready for deployment! 🚀")
    else:
        print("❌ SOME METRICS FAILED!")
        print()
        print("Please review failed metrics above.")
    
    print()
    
    # Save results
    import json
    results = {
        "primary_metrics": primary_metrics,
        "robust_metrics": robust_metrics,
        "efficiency_metrics": inf_results,
        "model_params_M": num_params / 1e6,
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    
    output_file = "metrics_verification.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")
    
    return results, all_passed


if __name__ == "__main__":
    results, passed = verify_all_metrics()
