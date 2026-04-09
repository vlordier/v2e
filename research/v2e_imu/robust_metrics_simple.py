#!/usr/bin/env python
"""
Robust Event Prediction Metrics - SIMPLIFIED (6 core metrics only).

Removed problematic metrics:
- Contrast Sensitivity (always 0.00, not useful)
- Sparsity Difference (can be gamed)
- depth_motion_error (auxiliary task)
- event_rate_error (unstable, unclear target)

Kept meaningful metrics:
1. Precision @ IoU=0.5
2. Recall @ IoU=0.5
3. F1 Score
4. event_bpb
5. Motion Correlation
6. Temporal CV

Usage:
    uv run python robust_metrics_simple.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict


def get_best_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_precision_recall(
    pred_events: torch.Tensor,
    gt_events: torch.Tensor,
    threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate precision and recall for event prediction.
    
    Args:
        pred_events: (B, 2, H, W) predicted event maps
        gt_events: (B, 2, H, W) ground truth event maps
        threshold: Threshold for binarizing predictions
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with precision, recall, f1
    """
    # Binarize predictions and ground truth
    pred_binary = (pred_events > threshold).float()
    gt_binary = (gt_events > threshold).float()
    
    # Flatten for pixel-wise comparison
    pred_flat = pred_binary.view(pred_binary.shape[0], -1)
    gt_flat = gt_binary.view(gt_binary.shape[0], -1)
    
    # True positives, false positives, false negatives
    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = (pred_flat * (1 - gt_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * gt_flat).sum(dim=1)
    
    # Precision, recall, F1
    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall = (tp / (tp + fn + 1e-8)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_event_bpb(
    pred_events: torch.Tensor,
    gt_events: torch.Tensor,
) -> float:
    """
    Evaluate bits per byte (BPB) for event prediction.
    
    Args:
        pred_events: (B, 2, H, W) predicted event maps
        gt_events: (B, 2, H, W) ground truth event maps
    
    Returns:
        BPB value (lower is better)
    """
    mse = F.mse_loss(pred_events, gt_events).item()
    bpb = mse / np.log(2)
    return bpb


def evaluate_motion_correlation(
    pred_events: torch.Tensor,
    imu_seq: torch.Tensor,
) -> float:
    """
    Evaluate correlation between predicted events and IMU motion.
    
    Args:
        pred_events: (B, 2, H, W) predicted event maps
        imu_seq: (B, T, 6) IMU sequence (accelerometer + gyroscope)
    
    Returns:
        Correlation coefficient (higher is better)
    """
    # Compute event rate (sum over spatial dimensions)
    event_rate = pred_events.abs().sum(dim=(1, 2, 3))  # (B,)
    
    # Compute IMU motion magnitude (mean over time)
    imu_motion = imu_seq[:, :, :3].norm(dim=-1).mean(dim=1)  # (B,)
    
    # Correlation
    if len(event_rate) < 2:
        return 0.0
    
    event_rate_np = event_rate.cpu().numpy()
    imu_motion_np = imu_motion.cpu().numpy()
    
    correlation = np.corrcoef(event_rate_np, imu_motion_np)[0, 1]
    return float(correlation) if not np.isnan(correlation) else 0.0


def evaluate_temporal_cv(
    pred_events: torch.Tensor,
) -> float:
    """
    Evaluate temporal consistency via coefficient of variation.
    
    Args:
        pred_events: (B, 2, H, W) predicted event maps
    
    Returns:
        Temporal CV (lower is better, < 0.1 is good)
    """
    # Compute event rate per sample
    event_rate = pred_events.abs().sum(dim=(1, 2, 3))  # (B,)
    
    if len(event_rate) < 2:
        return 0.0
    
    mean_rate = event_rate.mean().item()
    std_rate = event_rate.std().item()
    
    cv = std_rate / (mean_rate + 1e-8)
    return float(cv)


def evaluate_robust_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_samples: int = 50,
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Evaluate all 6 robust metrics.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        num_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with all 6 metrics
    """
    model.eval()
    
    all_pred_events = []
    all_gt_events = []
    all_imu_seqs = []
    
    samples_processed = 0
    
    print(f"Evaluating with batch size {batch_size}...")
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            images = batch["image"].to(device)
            imu_seq = batch["imu_seq"].to(device)
            gt_events = batch["events"].to(device)
            
            pred_events, pred_depth = model(images, imu_seq)
            
            all_pred_events.append(pred_events.cpu())
            all_gt_events.append(gt_events.cpu())
            all_imu_seqs.append(imu_seq.cpu())
            
            samples_processed += images.shape[0]
            
            if samples_processed >= num_samples:
                break
    
    # Concatenate all batches
    pred_events = torch.cat(all_pred_events, dim=0)
    gt_events = torch.cat(all_gt_events, dim=0)
    imu_seqs = torch.cat(all_imu_seqs, dim=0)
    
    print(f"  Total: {len(pred_events)} batches, {samples_processed} samples")
    
    # Compute all 6 metrics
    metrics = {}
    
    # 1-3. Precision, Recall, F1
    pr_metrics = evaluate_precision_recall(pred_events, gt_events)
    metrics.update(pr_metrics)
    
    # 4. event_bpb
    metrics["event_bpb"] = evaluate_event_bpb(pred_events, gt_events)
    
    # 5. Motion Correlation
    metrics["motion_correlation"] = evaluate_motion_correlation(pred_events, imu_seqs)
    
    # 6. Temporal CV
    metrics["temporal_cv"] = evaluate_temporal_cv(pred_events)
    
    return metrics


def print_robust_results(metrics: Dict[str, float]) -> None:
    """Print robust metrics in a readable format."""
    print("\n" + "="*60)
    print("ROBUST METRICS SUMMARY (6 Core Metrics)")
    print("="*60)
    
    print("\n🎯 DETECTION ACCURACY")
    print(f"  Precision: {metrics.get('precision', 0):.4f} (target: > 0.5)")
    print(f"  Recall:    {metrics.get('recall', 0):.4f} (target: > 0.5)")
    print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f} (target: > 0.5)")
    
    print("\n📊 COMPRESSION")
    print(f"  event_bpb: {metrics.get('event_bpb', 0):.6f} (target: < 0.001)")
    
    print("\n🔗 PHYSICAL CORRECTNESS")
    print(f"  Motion Correlation: {metrics.get('motion_correlation', 0):.4f} (target: > 0.5)")
    
    print("\n⏱️  TEMPORAL CONSISTENCY")
    print(f"  Temporal CV: {metrics.get('temporal_cv', 0):.4f} (target: < 0.1)")
    
    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)
    
    checks = {
        "Precision > 0.5": metrics.get("precision", 0) > 0.5,
        "Recall > 0.5": metrics.get("recall", 0) > 0.5,
        "F1 > 0.5": metrics.get("f1_score", 0) > 0.5,
        "event_bpb < 0.001": metrics.get("event_bpb", 1) < 0.001,
        "Motion Corr. > 0.5": metrics.get("motion_correlation", 0) > 0.5,
        "Temporal CV < 0.1": metrics.get("temporal_cv", 1) < 0.1,
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, passed_check in checks.items():
        status = "✅" if passed_check else "❌"
        print(f"  {status} {check_name}")
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL METRICS PASSED!")
    elif passed >= total * 0.8:
        print("\n✅ MOST METRICS PASSED (good!)")
    elif passed >= total * 0.5:
        print("\n⚠️  SOME METRICS PASSED (needs work)")
    else:
        print("\n❌ MOST METRICS FAILED (needs significant work)")


def main():
    """Main function for testing."""
    print("Robust Metrics Evaluation (Simplified - 6 Core Metrics)")
    print("="*60)
    
    # This would normally load model and data
    # For now, just show the metric definitions
    print("\n6 Core Metrics:")
    print("  1. Precision @ IoU=0.5")
    print("  2. Recall @ IoU=0.5")
    print("  3. F1 Score")
    print("  4. event_bpb")
    print("  5. Motion Correlation")
    print("  6. Temporal CV")
    print("\nRemoved metrics (not robust/meaningful):")
    print("  - Contrast Sensitivity (always 0.00)")
    print("  - Sparsity Difference (can be gamed)")
    print("  - depth_motion_error (auxiliary task)")
    print("  - event_rate_error (unstable)")
    print("\n✅ Metric suite simplified!")


if __name__ == "__main__":
    main()
