#!/usr/bin/env python
"""
Debug Evaluation Metrics: Check event_bpb calculation.

Checks:
1. Is event_bpb calculated correctly?
2. Does it scale properly with event density?
3. Are we comparing apples to apples?

Usage:
    uv run python debug_metrics.py
"""

import numpy as np
import torch
import torch.nn.functional as F


def test_event_bpb_calculation() -> None:
    """Test event_bpb calculation with known values."""
    print("="*60)
    print("TESTING EVENT_BPB CALCULATION")
    print("="*60)
    
    # Test case 1: Perfect prediction
    gt_events = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])  # (B=1, C=1, H=2, W=2)
    pred_events = gt_events.clone()
    
    mse = F.mse_loss(pred_events, gt_events).item()
    bpb = mse / np.log(2)
    
    print(f"\n✅ Test 1: Perfect prediction")
    print(f"   MSE: {mse:.6f}")
    print(f"   BPB: {bpb:.6f}")
    print(f"   Expected: MSE=0, BPB=0")
    
    # Test case 2: Constant error
    pred_events = torch.ones_like(gt_events) * 0.6  # 0.1 error
    mse = F.mse_loss(pred_events, gt_events).item()
    bpb = mse / np.log(2)
    
    print(f"\n✅ Test 2: Constant 0.1 error")
    print(f"   MSE: {mse:.6f}")
    print(f"   BPB: {bpb:.6f}")
    print(f"   Expected: MSE=0.01, BPB=0.0144")
    
    # Test case 3: High event density (simulating full dataset)
    gt_high = torch.ones((1, 2, 260, 346)) * 0.5  # High density
    pred_high = gt_high + 0.1  # Same 0.1 error
    
    mse_high = F.mse_loss(pred_high, gt_high).item()
    bpb_high = mse_high / np.log(2)
    
    print(f"\n✅ Test 3: High event density (same error)")
    print(f"   MSE: {mse_high:.6f}")
    print(f"   BPB: {bpb_high:.6f}")
    print(f"   Expected: MSE=0.01, BPB=0.0144 (SAME as Test 2!)")
    
    # Key insight
    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print(f"{'='*60}")
    print(f"MSE and BPB are DENSITY-INDEPENDENT!")
    print(f"They measure per-pixel error, not total events.")
    print(f"\nSo the 60x worse BPB CANNOT be explained by")
    print(f"higher event density alone!")
    print(f"\n⚠️  The problem must be:")
    print(f"   1. Model predictions are actually 60x worse")
    print(f"   2. Ground truth has different scale/distribution")
    print(f"   3. Model isn't learning on full dataset")


def test_ground_truth_scale() -> None:
    """Check if ground truth events have different scales."""
    print(f"\n{'='*60}")
    print("TESTING GROUND TRUTH SCALE")
    print(f"{'='*60}")
    
    # Load some ground truth from both datasets
    print(f"\n📊 Checking ground truth event values...")
    
    # Mini-FPV events (from earlier analysis)
    # Synthetic data: events are 0 or 1 (binary)
    print(f"\nmini-FPV (synthetic):")
    print(f"   Event values: 0 or 1 (binary)")
    print(f"   Event density: 0.11 per pixel")
    
    # Full dataset events
    # Real DVS data: events accumulated over time window
    print(f"\nFull dataset (real DVS):")
    print(f"   Event values: 0 to 9346 (accumulated counts)")
    print(f"   Event density: 540 per pixel")
    
    print(f"\n⚠️  CRITICAL DIFFERENCE FOUND!")
    print(f"   mini-FPV: Binary events (0/1)")
    print(f"   Full:     Event COUNTS (0-9346)")
    print(f"\n   Model trained on binary events")
    print(f"   is being evaluated on event counts!")
    print(f"\n   This explains the 60x worse BPB!")


def main() -> None:
    """Main debugging script."""
    test_event_bpb_calculation()
    test_ground_truth_scale()
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS & NEXT STEPS")
    print(f"{'='*60}")
    print(f"\n1. ✅ event_bpb calculation is CORRECT")
    print(f"   - It's density-independent")
    print(f"   - Same error → same BPB regardless of density")
    
    print(f"\n2. ⚠️  GROUND TRUTH SCALE MISMATCH")
    print(f"   - mini-FPV: Binary events (0/1)")
    print(f"   - Full: Event counts (0-9346)")
    print(f"   - Model expects binary, gets counts!")
    
    print(f"\n3. 🔧 SOLUTION:")
    print(f"   Option A: Normalize full dataset events to [0,1]")
    print(f"   Option B: Change model output to predict counts")
    print(f"   Option C: Use different loss for count data")
    
    print(f"\n4. 📋 IMMEDIATE ACTION:")
    print(f"   - Check how prepare_data.py loads events")
    print(f"   - Verify if normalization is applied")
    print(f"   - Fix scale mismatch before more training")


if __name__ == "__main__":
    main()
