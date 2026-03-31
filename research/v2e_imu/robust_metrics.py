#!/usr/bin/env python
"""
Robust Event Prediction Metrics.

Comprehensive evaluation beyond simple MSE:
1. Event sparsity analysis
2. Temporal consistency
3. Spatial coherence
4. Precision/Recall
5. Contrast sensitivity
6. Rate-motion correlation

Usage:
    uv run python robust_metrics.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from prepare_data import (
    DATA_DIR,
    EVAL_SAMPLES,
    IMAGE_SIZE,
    MAX_SEQ_LEN,
    make_dataloader,
)
from train import EventPredictor, ModelConfig, BASE_CHANNELS, IMU_HIDDEN_DIM


def get_best_device() -> str:
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        print("Using CUDA GPU acceleration")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) acceleration")
        return "mps"
    print("Using CPU (no GPU available)")
    return "cpu"


def compute_event_sparsity(
    pred_events: torch.Tensor, gt_events: torch.Tensor
) -> Dict[str, float]:
    """Analyze event sparsity - are we generating realistic event counts?"""
    pred_count = (pred_events.abs() > 0.01).sum().item()
    gt_count = (gt_events.abs() > 0.01).sum().item()

    pred_sparsity = 1.0 - pred_count / pred_events.numel()
    gt_sparsity = 1.0 - gt_count / gt_events.numel()
    count_ratio = pred_count / (gt_count + 1e-6)

    return {
        "pred_event_count": float(pred_count),
        "gt_event_count": float(gt_count),
        "count_ratio": float(count_ratio),
        "pred_sparsity": float(pred_sparsity),
        "gt_sparsity": float(gt_sparsity),
        "sparsity_diff": float(abs(pred_sparsity - gt_sparsity)),
    }


def compute_temporal_consistency(
    pred_events: torch.Tensor, gt_events: torch.Tensor
) -> Dict[str, float]:
    """Check if events are temporally consistent (smooth over time)."""
    pred_rate = pred_events.abs().mean(dim=(1, 2, 3))
    gt_rate = gt_events.abs().mean(dim=(1, 2, 3))

    pred_cv = pred_rate.std() / (pred_rate.mean() + 1e-6)
    gt_cv = gt_rate.std() / (gt_rate.mean() + 1e-6)

    return {
        "pred_temporal_cv": float(pred_cv.item()),
        "gt_temporal_cv": float(gt_cv.item()),
        "temporal_cv_diff": float(abs(pred_cv.item() - gt_cv.item())),
    }


def compute_spatial_coherence(
    pred_events: torch.Tensor, gt_events: torch.Tensor
) -> Dict[str, float]:
    """Check if events are spatially coherent (clustered properly)."""
    # Events shape: (B, 2, H, W) -> mean over channels -> (B, H, W)
    pred_spatial = pred_events.mean(dim=1)
    gt_spatial = gt_events.mean(dim=1)

    # Spatial gradients (handle small tensors)
    if pred_spatial.shape[2] > 1:
        pred_grad_x = pred_spatial[:, :, :-1].diff(dim=2).abs().mean()
    else:
        pred_grad_x = torch.tensor(0.0, device=pred_spatial.device)
    
    if pred_spatial.shape[1] > 1:
        pred_grad_y = pred_spatial[:, :-1, :].diff(dim=1).abs().mean()
    else:
        pred_grad_y = torch.tensor(0.0, device=pred_spatial.device)
    
    if gt_spatial.shape[2] > 1:
        gt_grad_x = gt_spatial[:, :, :-1].diff(dim=2).abs().mean()
    else:
        gt_grad_x = torch.tensor(0.0, device=gt_spatial.device)
    
    if gt_spatial.shape[1] > 1:
        gt_grad_y = gt_spatial[:, :-1, :].diff(dim=1).abs().mean()
    else:
        gt_grad_y = torch.tensor(0.0, device=gt_spatial.device)

    pred_spatial_var = (pred_grad_x + pred_grad_y).item() / 2
    gt_spatial_var = (gt_grad_x + gt_grad_y).item() / 2

    return {
        "pred_spatial_variance": float(pred_spatial_var),
        "gt_spatial_variance": float(gt_spatial_var),
        "spatial_variance_diff": float(abs(pred_spatial_var - gt_spatial_var)),
    }


def compute_precision_recall(
    pred_events: torch.Tensor, gt_events: torch.Tensor, threshold: float = 0.1
) -> Dict[str, float]:
    """Compute precision and recall for event detection."""
    pred_binary = pred_events.abs() > threshold
    gt_binary = gt_events.abs() > threshold

    tp = (pred_binary & gt_binary).sum().item()
    fp = (pred_binary & ~gt_binary).sum().item()
    fn = (~pred_binary & gt_binary).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_contrast_sensitivity(
    pred_events: torch.Tensor, gt_events: torch.Tensor
) -> Dict[str, float]:
    """Check if model detects both high and low contrast events."""
    high_contrast_mask = gt_events.abs() > 0.5
    low_contrast_mask = (gt_events.abs() > 0.1) & (gt_events.abs() <= 0.5)

    if high_contrast_mask.sum() > 0:
        high_pred = pred_events[high_contrast_mask]
        high_gt = gt_events[high_contrast_mask]
        high_mse = F.mse_loss(high_pred, high_gt).item()
    else:
        high_mse = 0.0

    if low_contrast_mask.sum() > 0:
        low_pred = pred_events[low_contrast_mask]
        low_gt = gt_events[low_contrast_mask]
        low_mse = F.mse_loss(low_pred, low_gt).item()
    else:
        low_mse = 0.0

    return {
        "high_contrast_mse": float(high_mse),
        "low_contrast_mse": float(low_mse),
        "contrast_ratio": float(low_mse / (high_mse + 1e-6)),
    }


def compute_rate_motion_correlation(
    pred_events: torch.Tensor, imu_seq: torch.Tensor
) -> Dict[str, float]:
    """Correlate event rate with IMU motion magnitude."""
    pred_rate = pred_events.abs().mean(dim=(1, 2, 3))
    imu_motion = imu_seq[:, :, :3].norm(dim=-1).mean(dim=1)

    if len(pred_rate) > 1:
        correlation = torch.corrcoef(torch.stack([pred_rate, imu_motion]))[0, 1]
    else:
        correlation = torch.tensor(0.0)

    return {
        "rate_motion_correlation": float(correlation.item()),
    }


def evaluate_robust_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_samples: int = EVAL_SAMPLES,
    eval_batch_size: int = 8,
) -> Dict[str, float]:
    """Comprehensive robust evaluation with proper batching."""
    model.eval()

    all_sparsity: List[Dict[str, float]] = []
    all_temporal: List[Dict[str, float]] = []
    all_spatial: List[Dict[str, float]] = []
    all_precision_recall: List[Dict[str, float]] = []
    all_contrast: List[Dict[str, float]] = []
    all_rate_motion: List[Dict[str, float]] = []

    total_samples = 0
    batches_processed = 0

    print(f"Evaluating with batch size {eval_batch_size}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= num_samples:
                break

            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)

            # Get predictions
            output = model(images, imu_seq)
            if isinstance(output, tuple):
                pred_events, _ = output
            else:
                pred_events = output

            # Compute all metrics
            all_sparsity.append(compute_event_sparsity(pred_events, gt_events))
            all_temporal.append(compute_temporal_consistency(pred_events, gt_events))
            all_spatial.append(compute_spatial_coherence(pred_events, gt_events))
            all_precision_recall.append(compute_precision_recall(pred_events, gt_events))
            all_contrast.append(compute_contrast_sensitivity(pred_events, gt_events))
            all_rate_motion.append(
                compute_rate_motion_correlation(pred_events, imu_seq)
            )

            total_samples += images.shape[0]
            batches_processed += 1

            if batches_processed % 10 == 0:
                print(f"  Processed {batches_processed} batches ({total_samples} samples)")

    print(f"  Total: {batches_processed} batches, {total_samples} samples")

    # Average all metrics
    def average_list(list_of_dicts: List[Dict[str, float]], key: str) -> float:
        return sum(d[key] for d in list_of_dicts) / len(list_of_dicts)

    robust_metrics = {
        # Sparsity
        "avg_pred_event_count": average_list(all_sparsity, "pred_event_count"),
        "avg_gt_event_count": average_list(all_sparsity, "gt_event_count"),
        "avg_count_ratio": average_list(all_sparsity, "count_ratio"),
        "avg_sparsity_diff": average_list(all_sparsity, "sparsity_diff"),
        # Temporal
        "avg_pred_temporal_cv": average_list(all_temporal, "pred_temporal_cv"),
        "avg_gt_temporal_cv": average_list(all_temporal, "gt_temporal_cv"),
        "avg_temporal_cv_diff": average_list(all_temporal, "temporal_cv_diff"),
        # Spatial
        "avg_pred_spatial_variance": average_list(all_spatial, "pred_spatial_variance"),
        "avg_gt_spatial_variance": average_list(all_spatial, "gt_spatial_variance"),
        "avg_spatial_variance_diff": average_list(all_spatial, "spatial_variance_diff"),
        # Precision/Recall
        "avg_precision": average_list(all_precision_recall, "precision"),
        "avg_recall": average_list(all_precision_recall, "recall"),
        "avg_f1_score": average_list(all_precision_recall, "f1_score"),
        # Contrast
        "avg_high_contrast_mse": average_list(all_contrast, "high_contrast_mse"),
        "avg_low_contrast_mse": average_list(all_contrast, "low_contrast_mse"),
        "avg_contrast_ratio": average_list(all_contrast, "contrast_ratio"),
        # Rate-Motion
        "avg_rate_motion_correlation": average_list(
            all_rate_motion, "rate_motion_correlation"
        ),
    }

    return robust_metrics


def print_robust_results(metrics: Dict[str, float]) -> None:
    """Print robust metrics in readable format."""
    print("\n" + "=" * 60)
    print("ROBUST METRICS SUMMARY")
    print("=" * 60)

    print("\n📊 EVENT SPARSITY")
    print(f"  Predicted events: {metrics['avg_pred_event_count']:.0f}")
    print(f"  Ground truth events: {metrics['avg_gt_event_count']:.0f}")
    print(f"  Count ratio: {metrics['avg_count_ratio']:.2f} (ideal: 1.0)")
    print(f"  Sparsity difference: {metrics['avg_sparsity_diff']:.4f} (lower=better)")

    print("\n⏱️  TEMPORAL CONSISTENCY")
    print(f"  Pred temporal CV: {metrics['avg_pred_temporal_cv']:.4f}")
    print(f"  GT temporal CV: {metrics['avg_gt_temporal_cv']:.4f}")
    print(f"  CV difference: {metrics['avg_temporal_cv_diff']:.4f} (lower=better)")

    print("\n🗺️  SPATIAL COHERENCE")
    print(f"  Pred spatial variance: {metrics['avg_pred_spatial_variance']:.4f}")
    print(f"  GT spatial variance: {metrics['avg_gt_spatial_variance']:.4f}")
    print(f"  Variance difference: {metrics['avg_spatial_variance_diff']:.4f} (lower=better)")

    print("\n🎯 PRECISION & RECALL")
    print(f"  Precision: {metrics['avg_precision']:.4f}")
    print(f"  Recall: {metrics['avg_recall']:.4f}")
    print(f"  F1 Score: {metrics['avg_f1_score']:.4f}")

    print("\n🔆 CONTRAST SENSITIVITY")
    print(f"  High contrast MSE: {metrics['avg_high_contrast_mse']:.6f}")
    print(f"  Low contrast MSE: {metrics['avg_low_contrast_mse']:.6f}")
    print(f"  Contrast ratio: {metrics['avg_contrast_ratio']:.2f} (ideal: 1.0)")

    print("\n📈 RATE-MOTION CORRELATION")
    print(f"  Correlation: {metrics['avg_rate_motion_correlation']:.4f} (higher=better)")

    print("\n" + "=" * 60)

    # Overall assessment
    print("\n📋 OVERALL ASSESSMENT")

    good = []
    if 0.8 <= metrics["avg_count_ratio"] <= 1.2:
        good.append("✅ Event count realistic")
    if metrics["avg_f1_score"] > 0.7:
        good.append("✅ Good precision/recall")
    if metrics["avg_rate_motion_correlation"] > 0.5:
        good.append("✅ Good motion correlation")
    if metrics["avg_sparsity_diff"] < 0.1:
        good.append("✅ Realistic sparsity")

    bad = []
    if metrics["avg_count_ratio"] > 1.5:
        bad.append("❌ Generating too many events (hallucination)")
    if metrics["avg_count_ratio"] < 0.5:
        bad.append("❌ Generating too few events")
    if metrics["avg_f1_score"] < 0.5:
        bad.append("❌ Poor precision/recall")
    if metrics["avg_rate_motion_correlation"] < 0.3:
        bad.append("❌ Weak motion correlation")

    for item in good:
        print(f"  {item}")
    for item in bad:
        print(f"  {item}")

    if not good and not bad:
        print("  ⚠️  Run full evaluation for accurate assessment")

    print()


def main() -> None:
    """Main robust evaluation."""
    # Get best available device
    device = get_best_device()
    print("Robust Event Prediction Metrics")
    print()

    # Create model
    print("Loading model...")
    config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
    model = EventPredictor(config)
    model = model.to(device)

    print("Note: Using untrained model. Load trained weights for real evaluation.")
    print()

    # Create dataloader with appropriate batch size
    print("Loading data...")
    eval_batch_size = 8 if device == "cpu" else 16
    val_loader = make_dataloader(
        DATA_DIR, "val", eval_batch_size, MAX_SEQ_LEN, IMAGE_SIZE
    )

    # Evaluate
    print()
    print("Computing robust metrics...")
    metrics = evaluate_robust_metrics(
        model, val_loader, device, EVAL_SAMPLES, eval_batch_size
    )

    # Print results
    print_robust_results(metrics)

    # Save to file
    output_file = "robust_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
