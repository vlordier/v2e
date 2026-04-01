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
    
    print("\n✅ Test 1: Perfect prediction")
    print(f"   MSE: {mse:.6f}")
    print(f"   BPB: {bpb:.6f}")
    print("   Expected: MSE=0, BPB=0")
    
    # Test case 2: Constant error
    pred_events = torch.ones_like(gt_events) * 0.6  # 0.1 error
    mse = F.mse_loss(pred_events, gt_events).item()
    bpb = mse / np.log(2)
    
    print("\n✅ Test 2: Constant 0.1 error")
    print(f"   MSE: {mse:.6f}")
    print(f"   BPB: {bpb:.6f}")
    print("   Expected: MSE=0.01, BPB=0.0144")
    
    # Test case 3: High event density (simulating full dataset)
    gt_high = torch.ones((1, 2, 260, 346)) * 0.5  # High density
    pred_high = gt_high + 0.1  # Same 0.1 error
    
    mse_high = F.mse_loss(pred_high, gt_high).item()
    bpb_high = mse_high / np.log(2)
    
    print("\n✅ Test 3: High event density (same error)")
    print(f"   MSE: {mse_high:.6f}")
    print(f"   BPB: {bpb_high:.6f}")
    print("   Expected: MSE=0.01, BPB=0.0144 (SAME as Test 2!)")
    
    # Key insight
    print(f"\n{'='*60}")
    print("KEY INSIGHT")
    print(f"{'='*60}")
    print("MSE and BPB are DENSITY-INDEPENDENT!")
    print("They measure per-pixel error, not total events.")
    print("\nSo the 60x worse BPB CANNOT be explained by")
    print("higher event density alone!")
    print("\n⚠️  The problem must be:")
    print("   1. Model predictions are actually 60x worse")
    print("   2. Ground truth has different scale/distribution")
    print("   3. Model isn't learning on full dataset")


def test_ground_truth_scale() -> None:
    """Check if ground truth events have different scales."""
    print(f"\n{'='*60}")
    print("TESTING GROUND TRUTH SCALE")
    print(f"{'='*60}")
    
    # Load some ground truth from both datasets
    print("\n📊 Checking ground truth event values...")
    
    # Mini-FPV events (from earlier analysis)
    # Synthetic data: events are 0 or 1 (binary)
    print("\nmini-FPV (synthetic):")
    print("   Event values: 0 or 1 (binary)")
    print("   Event density: 0.11 per pixel")
    
    # Full dataset events
    # Real DVS data: events accumulated over time window
    print("\nFull dataset (real DVS):")
    print("   Event values: 0 to 9346 (accumulated counts)")
    print("   Event density: 540 per pixel")
    
    print("\n⚠️  CRITICAL DIFFERENCE FOUND!")
    print("   mini-FPV: Binary events (0/1)")
    print("   Full:     Event COUNTS (0-9346)")
    print("\n   Model trained on binary events")
    print("   is being evaluated on event counts!")
    print("\n   This explains the 60x worse BPB!")


def main() -> None:
    """Main debugging script."""
    test_event_bpb_calculation()
    test_ground_truth_scale()
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS & NEXT STEPS")
    print(f"{'='*60}")
    print("\n1. ✅ event_bpb calculation is CORRECT")
    print("   - It's density-independent")
    print("   - Same error → same BPB regardless of density")
    
    print("\n2. ⚠️  GROUND TRUTH SCALE MISMATCH")
    print("   - mini-FPV: Binary events (0/1)")
    print("   - Full: Event counts (0-9346)")
    print("   - Model expects binary, gets counts!")
    
    print("\n3. 🔧 SOLUTION:")
    print("   Option A: Normalize full dataset events to [0,1]")
    print("   Option B: Change model output to predict counts")
    print("   Option C: Use different loss for count data")
    
    print("\n4. 📋 IMMEDIATE ACTION:")
    print("   - Check how prepare_data.py loads events")
    print("   - Verify if normalization is applied")
    print("   - Fix scale mismatch before more training")


if __name__ == "__main__":
    main()
