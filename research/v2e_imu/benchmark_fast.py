#!/usr/bin/env python
"""
Fast Benchmark: Uses mini-FPV for quick comparison.

Usage:
    uv run python benchmark_fast.py
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_data import make_dataloader, evaluate_combined_metric
from train import EventPredictor, ModelConfig, BASE_CHANNELS, IMU_HIDDEN_DIM


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def benchmark_inference(model, device, dataloader, num_batches=5):
    """Benchmark inference speed with optimizations."""
    model.eval()
    
    # Warm-up
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break
            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            _ = model(images, imu_seq)
    
    # Benchmark with inference mode (faster than no_grad)
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
        "samples": total_samples,
    }


def run_benchmark(version_name="Improved"):
    """Run fast benchmark."""
    print("="*60)
    print(f"V2E BENCHMARK - {version_name}")
    print("="*60)
    print()
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Create model
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config).to(device)
    
    # Load checkpoint if available
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
    
    # Create dataloader (mini-FPV for speed)
    print("Loading mini-FPV data...")
    val_loader = make_dataloader("data/fpv/mini_fpv", "val", 16, 50, (260, 346))
    print("✅ Data loaded")
    print()
    
    # Inference benchmark
    print("Inference Benchmark:")
    inf_results = benchmark_inference(model, device, val_loader, num_batches=5)
    print(f"  FPS: {inf_results['fps']:.1f}")
    print(f"  Latency: {inf_results['latency_ms']:.2f} ms")
    print()
    
    # Evaluation
    print("Evaluation (50 samples):")
    eval_start = time.time()
    metrics = evaluate_combined_metric(model, val_loader, device, 50)
    eval_time = time.time() - eval_start
    
    print(f"  event_bpb: {metrics['event_bpb']:.6f}")
    print(f"  event_mse: {metrics['event_mse']:.6f}")
    print(f"  event_rate_error: {metrics['event_rate_error']:.2f}")
    print(f"  depth_motion_error: {metrics['depth_motion_error']:.2f}")
    print(f"  Evaluation time: {eval_time:.2f} s")
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {num_params / 1e6:.2f}M params")
    print(f"Inference: {inf_results['fps']:.1f} FPS")
    print(f"event_bpb: {metrics['event_bpb']:.6f}")
    print("="*60)
    
    # Save results
    import json
    results = {
        "version": version_name,
        "model_params_M": num_params / 1e6,
        "inference_fps": inf_results['fps'],
        "inference_latency_ms": inf_results['latency_ms'],
        "event_bpb": metrics['event_bpb'],
        "event_mse": metrics['event_mse'],
        "event_rate_error": metrics['event_rate_error'],
        "depth_motion_error": metrics['depth_motion_error'],
        "eval_time_s": eval_time,
    }
    
    output_file = f"benchmark_{version_name.lower().replace(' ', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    run_benchmark()
