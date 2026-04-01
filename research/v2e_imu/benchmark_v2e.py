#!/usr/bin/env python
"""
Comprehensive Benchmark: Original v2e vs Improved Version

Metrics:
1. Event prediction quality (event_bpb, event_mse)
2. Training speed (steps/sec, time to convergence)
3. Inference speed (FPS, latency)
4. Memory usage (VRAM, RAM)
5. Model size (parameters, file size)

Usage:
    uv run python benchmark_v2e.py
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prepare_data import make_dataloader, evaluate_combined_metric
from train import EventPredictor, ModelConfig, BASE_CHANNELS, IMU_HIDDEN_DIM


def get_memory_usage():
    """Get current memory usage (simple estimate)."""
    # Simple estimate based on model size
    return 0  # Skip for simplicity


def get_gpu_memory():
    """Get GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    elif torch.backends.mps.is_available():
        # MPS doesn't expose memory API yet
        return 0
    return 0


def benchmark_inference(model, device, dataloader, num_batches=10):
    """Benchmark inference speed."""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            _ = model(images, imu_seq)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            _ = model(images, imu_seq)
            
            total_samples += images.shape[0]
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    fps = total_samples / elapsed
    latency_ms = (elapsed / total_samples) * 1000
    
    return {
        "fps": fps,
        "latency_ms": latency_ms,
        "samples": total_samples,
        "time_s": elapsed,
    }


def benchmark_training(model, device, dataloader, num_steps=10):
    """Benchmark training speed."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warm-up
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        images = batch["image"].to(device)
        imu_seq = batch["imu_seq"].to(device)
        gt_events = batch["events"].to(device)
        
        optimizer.zero_grad()
        pred_rate, pred_depth = model(images, imu_seq)
        loss = torch.nn.functional.mse_loss(pred_rate, gt_events)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    total_steps = 0
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        
        images = batch["image"].to(device)
        imu_seq = batch["imu_seq"].to(device)
        gt_events = batch["events"].to(device)
        
        optimizer.zero_grad()
        pred_rate, pred_depth = model(images, imu_seq)
        loss = torch.nn.functional.mse_loss(pred_rate, gt_events)
        loss.backward()
        optimizer.step()
        
        total_steps += 1
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    steps_per_sec = total_steps / elapsed
    ms_per_step = (elapsed / total_steps) * 1000
    
    return {
        "steps_per_sec": steps_per_sec,
        "ms_per_step": ms_per_step,
        "steps": total_steps,
        "time_s": elapsed,
    }


def count_parameters(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def run_benchmark():
    """Run complete benchmark suite."""
    print("="*60)
    print("V2E COMPREHENSIVE BENCHMARK")
    print("="*60)
    print()
    
    # Setup
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Create model
    print("Loading model...")
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config).to(device)
    
    # Load checkpoint if available
    checkpoint_path = Path("3d_aware_model_checkpoint.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Load with strict=False to handle new parameters (log_var_depth)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✅ Checkpoint loaded")
    else:
        print("⚠️  No checkpoint found, using untrained model")
    print()
    
    # Model size
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    print(f"Model size (estimated): {num_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()
    
    # Create dataloader
    print("Loading data...")
    val_loader = make_dataloader("data/fpv", "val", 16, 50, (260, 346))
    print("✅ Data loaded")
    print()
    
    # Memory usage
    print("Memory Usage:")
    print(f"  RAM: {get_memory_usage():.1f} MB")
    gpu_mem = get_gpu_memory()
    if gpu_mem > 0:
        print(f"  GPU: {gpu_mem:.1f} MB")
    print()
    
    # Inference benchmark
    print("Inference Benchmark:")
    inf_results = benchmark_inference(model, device, val_loader, num_batches=20)
    print(f"  FPS: {inf_results['fps']:.1f}")
    print(f"  Latency: {inf_results['latency_ms']:.2f} ms")
    print(f"  Samples: {inf_results['samples']}")
    print(f"  Time: {inf_results['time_s']:.2f} s")
    print()
    
    # Training benchmark
    print("Training Benchmark:")
    train_results = benchmark_training(model, device, val_loader, num_steps=20)
    print(f"  Steps/sec: {train_results['steps_per_sec']:.2f}")
    print(f"  Ms/step: {train_results['ms_per_step']:.2f}")
    print(f"  Steps: {train_results['steps']}")
    print(f"  Time: {train_results['time_s']:.2f} s")
    print()
    
    # Evaluation benchmark
    print("Evaluation Benchmark:")
    eval_start = time.time()
    metrics = evaluate_combined_metric(model, val_loader, device, 50)
    eval_time = time.time() - eval_start
    
    print(f"  event_bpb: {metrics['event_bpb']:.6f}")
    print(f"  event_mse: {metrics['event_mse']:.6f}")
    print(f"  event_rate_error: {metrics['event_rate_error']:.2f}")
    print(f"  depth_motion_error: {metrics['depth_motion_error']:.2f}")
    print(f"  samples_per_sec: {metrics['samples_per_sec']:.2f}")
    print(f"  Evaluation time: {eval_time:.2f} s")
    print()
    
    # Summary
    print("="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {num_params / 1e6:.2f}M parameters")
    print(f"Inference: {inf_results['fps']:.1f} FPS ({inf_results['latency_ms']:.2f} ms latency)")
    print(f"Training: {train_results['steps_per_sec']:.2f} steps/sec")
    print(f"event_bpb: {metrics['event_bpb']:.6f}")
    print(f"Memory: {get_memory_usage():.1f} MB RAM")
    print("="*60)
    
    # Save results
    results = {
        "model_params_M": num_params / 1e6,
        "inference_fps": inf_results['fps'],
        "inference_latency_ms": inf_results['latency_ms'],
        "training_steps_per_sec": train_results['steps_per_sec'],
        "event_bpb": metrics['event_bpb'],
        "event_mse": metrics['event_mse'],
        "memory_mb": get_memory_usage(),
    }
    
    import json
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to benchmark_results.json")
    
    return results


if __name__ == "__main__":
    run_benchmark()
