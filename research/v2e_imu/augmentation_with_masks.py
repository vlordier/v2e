#!/usr/bin/env python
"""
Occlusion-Aware Augmentation with Mask Tracking.

Key insight: When we add occlusions, we should verify the model
does NOT generate events in occluded regions.

This module:
1. Applies augmentations (occlusions, noise, etc.)
2. Returns occlusion masks
3. Enables occlusion-aware evaluation

Usage:
    images, occlusion_mask = augment_with_mask(images, imu_seq, gt_events)
    # Train model...
    # Evaluate: check pred_events[occlusion_mask] ≈ 0
"""


import torch
import torch.nn.functional as F


def augment_with_occlusion_mask(
    images: torch.Tensor,
    imu_seq: torch.Tensor,
    gt_events: torch.Tensor,
    return_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply RGB augmentations and return occlusion mask.
    
    Args:
        images: (B, C, H, W) input images
        imu_seq: (B, T, 6) IMU sequences
        gt_events: (B, 2, H, W) ground truth events
        return_mask: Whether to return occlusion mask
    
    Returns:
        images: Augmented images
        imu_seq: Augmented IMU (with noise)
        gt_events: Ground truth events (unchanged)
        occlusion_mask: (B, H, W) boolean mask where True = occluded
    """
    B, C, H, W = images.shape
    
    # Initialize occlusion mask (all False = no occlusion)
    occlusion_mask = torch.zeros((B, H, W), dtype=torch.bool, device=images.device)
    
    # 1. Gaussian noise
    if torch.rand(1).item() > 0.5:
        noise_std = torch.rand(1).item() * 0.1
        images = images + torch.randn_like(images) * noise_std
        images = images.clamp(0, 1)
    
    # 2. Brightness jitter
    if torch.rand(1).item() > 0.5:
        brightness_factor = 0.7 + torch.rand(1).item() * 0.6
        images = images * brightness_factor
        images = images.clamp(0, 1)
    
    # 3. Contrast jitter
    if torch.rand(1).item() > 0.5:
        contrast_factor = 0.7 + torch.rand(1).item() * 0.6
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - mean) * contrast_factor + mean
        images = images.clamp(0, 1)
    
    # 4. Random occlusions (WITH MASK TRACKING)
    if torch.rand(1).item() > 0.5:
        num_occlusions = torch.randint(1, 4, (1,)).item()
        for _ in range(num_occlusions):
            # Random occlusion size (5% to 20% of image)
            occ_h = int(H * (0.05 + torch.rand(1).item() * 0.15))
            occ_w = int(W * (0.05 + torch.rand(1).item() * 0.15))
            
            # Ensure valid dimensions
            occ_h = max(1, min(occ_h, H))
            occ_w = max(1, min(occ_w, W))
            
            # Random position
            max_y = max(0, H - occ_h)
            max_x = max(0, W - occ_w)
            y = torch.randint(0, max_y + 1, (1,)).item()
            x = torch.randint(0, max_x + 1, (1,)).item()
            
            # Apply occlusion to images
            occlusion_value = torch.rand(1).item()
            images[:, :, y:y+occ_h, x:x+occ_w] = occlusion_value
            
            # Mark occluded region in mask
            if return_mask:
                occlusion_mask[:, y:y+occ_h, x:x+occ_w] = True
    
    # 5. IMU noise
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(imu_seq) * 0.01
        imu_seq = imu_seq + noise
    
    return images, imu_seq, gt_events, occlusion_mask


def compute_occlusion_aware_metrics(
    pred_events: torch.Tensor,
    gt_events: torch.Tensor,
    occlusion_mask: torch.Tensor,
) -> dict[str, float]:
    """
    Compute metrics that check if model generates events in occluded regions.
    
    Key insight: Model should NOT generate events where image is occluded,
    because occlusions are artificial and don't correspond to real motion.
    
    Args:
        pred_events: (B, 2, H, W) predicted events
        gt_events: (B, 2, H, W) ground truth events
        occlusion_mask: (B, H, W) boolean mask where True = occluded
    
    Returns:
        Dictionary of occlusion-aware metrics
    """
    B, _, H, W = pred_events.shape
    
    # Expand mask to match event dimensions (B, 2, H, W)
    mask_expanded = occlusion_mask.unsqueeze(1).expand(-1, 2, -1, -1)
    
    # Events in occluded regions (should be ZERO)
    if occlusion_mask.sum() > 0:
        pred_events_occluded = pred_events[mask_expanded]
        occluded_event_magnitude = pred_events_occluded.abs().mean().item()
        occluded_event_count = (pred_events_occluded.abs() > 0.01).sum().item()
    else:
        occluded_event_magnitude = 0.0
        occluded_event_count = 0
    
    # Events in visible regions (should match ground truth)
    visible_mask = ~mask_expanded
    if visible_mask.sum() > 0:
        pred_events_visible = pred_events[visible_mask]
        gt_events_visible = gt_events[visible_mask]
        visible_mse = F.mse_loss(pred_events_visible, gt_events_visible).item()
    else:
        visible_mse = 0.0
    
    # Occlusion violation score (lower is better)
    occlusion_violation = occluded_event_magnitude
    
    # Visible region precision/recall
    if visible_mask.sum() > 0:
        pred_visible = pred_events[visible_mask].abs() > 0.1
        gt_visible = gt_events[visible_mask].abs() > 0.1
        
        tp = (pred_visible & gt_visible).sum().item()
        fp = (pred_visible & ~gt_visible).sum().item()
        fn = (~pred_visible & gt_visible).sum().item()
        
        visible_precision = tp / (tp + fp + 1e-6)
        visible_recall = tp / (tp + fn + 1e-6)
    else:
        visible_precision = 0.0
        visible_recall = 0.0
    
    total_pixels = B * H * W
    occluded_pixels = occlusion_mask.sum().item()
    
    return {
        "occluded_event_magnitude": float(occluded_event_magnitude),
        "occluded_event_count": int(occluded_event_count),
        "occlusion_violation_score": float(occlusion_violation),
        "visible_region_mse": float(visible_mse),
        "visible_precision": float(visible_precision),
        "visible_recall": float(visible_recall),
        "occlusion_area_ratio": float(occluded_pixels / total_pixels),
    }


def print_occlusion_metrics(metrics: dict[str, float]) -> None:
    """Print occlusion-aware metrics."""
    print("\n" + "=" * 60)
    print("OCCLUSION-AWARE METRICS")
    print("=" * 60)
    
    print(f"\n🎭 Occlusion Area: {metrics['occlusion_area_ratio']*100:.1f}% of image")
    
    print("\n🚫 Events in Occluded Regions (should be ZERO)")
    print(f"  Event magnitude: {metrics['occluded_event_magnitude']:.6f}")
    print(f"  Event count: {metrics['occluded_event_count']}")
    print(f"  Violation score: {metrics['occlusion_violation_score']:.6f} (lower=better)")
    
    print("\n✅ Events in Visible Regions")
    print(f"  MSE: {metrics['visible_region_mse']:.6f}")
    print(f"  Precision: {metrics['visible_precision']:.4f}")
    print(f"  Recall: {metrics['visible_recall']:.4f}")
    
    # Assessment
    print("\n📋 ASSESSMENT")
    if metrics["occlusion_violation_score"] < 0.01:
        print("  ✅ Model correctly suppresses events in occluded regions")
    elif metrics["occlusion_violation_score"] < 0.1:
        print("  ⚠️  Model generates some events in occluded regions")
    else:
        print("  ❌ Model heavily hallucinates in occluded regions!")
    
    if metrics["visible_precision"] > 0.7 and metrics["visible_recall"] > 0.7:
        print("  ✅ Good performance in visible regions")
    elif metrics["visible_precision"] > 0.5 or metrics["visible_recall"] > 0.5:
        print("  ⚠️  Moderate performance in visible regions")
    else:
        print("  ❌ Poor performance in visible regions")
    
    print()


# Example usage in training
if __name__ == "__main__":
    # Test the augmentation
    B, C, H, W = 2, 1, 260, 346
    T = 50
    
    images = torch.rand(B, C, H, W)
    imu_seq = torch.randn(B, T, 6)
    gt_events = torch.zeros(B, 2, H, W)
    
    # Force occlusion for testing
    torch.manual_seed(42)
    
    # Apply augmentation with mask tracking
    images_aug, imu_aug, gt_events, occ_mask = augment_with_occlusion_mask(
        images, imu_seq, gt_events, return_mask=True
    )
    
    print(f"Original image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Augmented image range: [{images_aug.min():.3f}, {images_aug.max():.3f}]")
    print(f"Occlusion mask: {occ_mask.sum().item()} pixels occluded ({occ_mask.sum().item() / (B*H*W) * 100:.1f}%)")
    
    # Simulate model predictions (random noise = bad model)
    pred_events = torch.randn_like(gt_events) * 0.5
    
    # Compute occlusion-aware metrics
    metrics = compute_occlusion_aware_metrics(pred_events, gt_events, occ_mask)
    print_occlusion_metrics(metrics)
    
    # Now test with a "good" model that doesn't generate events in occluded regions
    print("\n" + "="*60)
    print("TESTING 'GOOD' MODEL (no events in occluded regions)")
    print("="*60)
    
    pred_events_good = torch.randn_like(gt_events) * 0.5
    # Zero out events in occluded regions
    mask_expanded = occ_mask.unsqueeze(1).expand(-1, 2, -1, -1)
    pred_events_good[mask_expanded] = 0
    
    metrics_good = compute_occlusion_aware_metrics(pred_events_good, gt_events, occ_mask)
    print_occlusion_metrics(metrics_good)
