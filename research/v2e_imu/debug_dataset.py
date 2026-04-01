#!/usr/bin/env python
"""
Debug Dataset Quality: Compare mini-FPV vs Full Dataset.

Checks:
1. Event density distribution
2. Event polarity balance (pos vs neg)
3. Temporal event rate
4. Spatial event distribution
5. Ground truth quality issues

Usage:
    uv run python debug_dataset.py
"""

from pathlib import Path

import numpy as np


def load_events_file(filepath: Path) -> dict[str, np.ndarray]:
    """Load events from text file."""
    print(f"Loading {filepath}...")
    
    timestamps = []
    x_coords = []
    y_coords = []
    polarities = []
    
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamps.append(float(parts[0]))
                x_coords.append(int(parts[1]))
                y_coords.append(int(parts[2]))
                polarities.append(int(parts[3]))
    
    return {
        "timestamps": np.array(timestamps, dtype=np.float64),
        "x": np.array(x_coords, dtype=np.int32),
        "y": np.array(y_coords, dtype=np.int32),
        "polarity": np.array(polarities, dtype=np.int8),
    }


def analyze_events(events: dict[str, np.ndarray], name: str) -> dict:
    """Analyze event statistics."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    total_events = len(events["timestamps"])
    duration = events["timestamps"][-1] - events["timestamps"][0]
    event_rate = total_events / duration if duration > 0 else 0
    
    # Polarity balance
    pos_events = (events["polarity"] == 1).sum()
    neg_events = (events["polarity"] == 0).sum()
    pos_ratio = pos_events / total_events if total_events > 0 else 0
    
    # Spatial distribution
    H, W = 260, 346  # DAVIS346 resolution
    event_map = np.zeros((H, W), dtype=np.int32)
    np.add.at(event_map, (events["y"], events["x"]), 1)
    
    spatial_stats = {
        "mean": float(event_map.mean()),
        "std": float(event_map.std()),
        "max": int(event_map.max()),
        "min": int(event_map.min()),
        "nonzero_ratio": float((event_map > 0).sum() / (H * W)),
    }
    
    # Temporal distribution (events per second)
    time_bins = np.arange(0, duration + 1, 1.0)  # 1-second bins
    events_per_sec = np.histogram(events["timestamps"], bins=time_bins)[0]
    
    temporal_stats = {
        "mean_rate": float(events_per_sec.mean()),
        "std_rate": float(events_per_sec.std()),
        "max_rate": int(events_per_sec.max()),
        "min_rate": int(events_per_sec.min()),
    }
    
    # Print results
    print("\n📊 BASIC STATISTICS")
    print(f"  Total events:      {total_events:,}")
    print(f"  Duration:          {duration:.1f}s")
    print(f"  Event rate:        {event_rate:.0f} events/sec")
    
    print("\n🔃 POLARITY BALANCE")
    print(f"  Positive events:   {pos_events:,} ({pos_ratio*100:.1f}%)")
    print(f"  Negative events:   {neg_events:,} ({(1-pos_ratio)*100:.1f}%)")
    print(f"  Balance ratio:     {pos_ratio/(1-pos_ratio) if pos_ratio < 1 else float('inf'):.2f}")
    
    print("\n🗺️  SPATIAL DISTRIBUTION")
    print(f"  Mean per pixel:    {spatial_stats['mean']:.2f}")
    print(f"  Std per pixel:     {spatial_stats['std']:.2f}")
    print(f"  Max per pixel:     {spatial_stats['max']:,}")
    print(f"  Min per pixel:     {spatial_stats['min']}")
    print(f"  Non-zero pixels:   {spatial_stats['nonzero_ratio']*100:.1f}%")
    
    print("\n⏱️  TEMPORAL DISTRIBUTION")
    print(f"  Mean rate/sec:     {temporal_stats['mean_rate']:.0f}")
    print(f"  Std rate/sec:      {temporal_stats['std_rate']:.0f}")
    print(f"  Max rate/sec:      {temporal_stats['max_rate']:,}")
    print(f"  Min rate/sec:      {temporal_stats['min_rate']}")
    
    return {
        "name": name,
        "total_events": total_events,
        "duration": duration,
        "event_rate": event_rate,
        "pos_ratio": pos_ratio,
        "spatial": spatial_stats,
        "temporal": temporal_stats,
    }


def compare_datasets(stats_list: list[dict]) -> None:
    """Compare statistics across datasets."""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    if len(stats_list) < 2:
        print("Need at least 2 datasets to compare")
        return
    
    # Normalize to first dataset (mini-FPV)
    base = stats_list[0]
    
    print(f"\n{'Metric':<25} {base['name']:<15}", end="")
    for stats in stats_list[1:]:
        ratio = stats["total_events"] / base["total_events"]
        print(f" {stats['name']:<15} (x{ratio:.0f})", end="")
    print()
    
    print(f"{'-'*80}")
    print(f"{'Total events:':<25} {base['total_events']:<15,}", end="")
    for stats in stats_list[1:]:
        print(f" {stats['total_events']:<15,}", end="")
    print()
    
    print(f"{'Event rate (evt/s):':<25} {base['event_rate']:<15,.0f}", end="")
    for stats in stats_list[1:]:
        ratio = stats["event_rate"] / base["event_rate"]
        print(f" {stats['event_rate']:<15,.0f} (x{ratio:.1f})", end="")
    print()
    
    print(f"{'Positive ratio:':<25} {base['pos_ratio']:<15.3f}", end="")
    for stats in stats_list[1:]:
        print(f" {stats['pos_ratio']:<15.3f}", end="")
    print()
    
    print(f"{'Spatial mean/pixel:':<25} {base['spatial']['mean']:<15.2f}", end="")
    for stats in stats_list[1:]:
        ratio = stats["spatial"]["mean"] / base["spatial"]["mean"]
        print(f" {stats['spatial']['mean']:<15.2f} (x{ratio:.0f})", end="")
    print()
    
    print(f"{'Temporal mean rate:':<25} {base['temporal']['mean_rate']:<15,.0f}", end="")
    for stats in stats_list[1:]:
        ratio = stats["temporal"]["mean_rate"] / base["temporal"]["mean_rate"]
        print(f" {stats['temporal']['mean_rate']:<15,.0f} (x{ratio:.1f})", end="")
    print()


def main() -> None:
    """Main debugging script."""
    print("="*60)
    print("DATASET QUALITY DEBUGGING")
    print("="*60)
    
    # Define datasets to compare
    datasets = {
        "mini-FPV": Path("data/fpv/synthetic_fpv/events.txt"),
        "indoor_forward_3": Path("data/fpv/indoor_forward_3/events.txt"),
        "indoor_forward_9": Path("data/fpv/indoor_forward_9/events.txt"),
        "indoor_forward_10": Path("data/fpv/indoor_forward_10/events.txt"),
    }
    
    all_stats = []
    
    # Analyze each dataset
    for name, filepath in datasets.items():
        if filepath.exists():
            events = load_events_file(filepath)
            stats = analyze_events(events, name)
            all_stats.append(stats)
        else:
            print(f"\n⚠️  {name} not found at {filepath}")
    
    # Compare datasets
    if all_stats:
        compare_datasets(all_stats)
        
        # Key insights
        print(f"\n{'='*60}")
        print("KEY INSIGHTS")
        print(f"{'='*60}")
        
        if len(all_stats) >= 2:
            mini = all_stats[0]
            full = all_stats[1]
            
            event_rate_ratio = full["event_rate"] / mini["event_rate"]
            spatial_ratio = full["spatial"]["mean"] / mini["spatial"]["mean"]
            
            print(f"\n1. Event density: Full dataset has {event_rate_ratio:.0f}x higher event rate")
            print(f"   → This makes prediction {event_rate_ratio:.0f}x harder!")
            
            print(f"\n2. Spatial density: Full dataset has {spatial_ratio:.0f}x more events per pixel")
            print(f"   → Model needs to predict {spatial_ratio:.0f}x more events")
            
            print("\n3. Polarity balance:")
            print(f"   Mini-FPV: {mini['pos_ratio']:.3f} positive")
            print(f"   Full:     {full['pos_ratio']:.3f} positive")
            if abs(mini["pos_ratio"] - full["pos_ratio"]) > 0.05:
                print("   ⚠️  SIGNIFICANT DIFFERENCE - may affect learning!")
            
            print("\n4. Temporal variance:")
            print(f"   Mini-FPV: {mini['temporal']['std_rate']:.0f} std")
            print(f"   Full:     {full['temporal']['std_rate']:.0f} std")
            if full["temporal"]["std_rate"] > mini["temporal"]["std_rate"] * 2:
                print("   ⚠️  Full dataset has much more variable event rate!")
        
        print("\n✅ Dataset analysis complete!")
        print("   Next: Check evaluation metric calculation")


if __name__ == "__main__":
    main()
