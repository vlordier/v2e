#!/usr/bin/env python
"""
Fast Hyperparameter Optimization for Rate Penalty Weight.

Focus: Find optimal rate_weight (depth_weight fixed at 0.1)
Search: [0.001, 0.005, 0.01, 0.02, 0.05]
Time: 3 minutes per configuration

Usage:
    uv run python hpo_fast.py
"""

import json
import subprocess
import time
from pathlib import Path

# Focused search space (rate_weight only)
RATE_WEIGHTS = [0.001, 0.005, 0.01, 0.02, 0.05]
DEPTH_WEIGHT = 0.1  # Fixed

# Fast HPO configuration
TIME_BUDGET = 180  # 3 minutes per run
EVAL_SAMPLES = 30  # Minimal for speed


def run_single_config(rate_weight: float) -> dict:
    """Run training with specific rate_weight."""
    print(f"\n{'=' * 60}")
    print(f"Testing rate_weight={rate_weight}")
    print(f"{'=' * 60}")

    train_script = Path("train.py")
    original = train_script.read_text()

    # Modify hyperparameters
    modified = (
        original.replace(
            "depth_weight = 0.1  # Auxiliary task weight",
            f"depth_weight = {DEPTH_WEIGHT}  # HPO fixed",
        )
        .replace(
            "rate_weight = 0.5  # INCREASED: Strong event hallucination prevention (was 0.05)",
            f"rate_weight = {rate_weight}  # HPO test",
        )
        .replace(
            "TIME_BUDGET = 900  # 15 minutes per experiment (increased for proper evaluation)",
            f"TIME_BUDGET = {TIME_BUDGET}  # HPO fast",
        )
        .replace(
            "EVAL_SAMPLES = 100  # Number of samples for evaluation (reduced for speed)",
            f"EVAL_SAMPLES = {EVAL_SAMPLES}  # HPO fast",
        )
    )

    try:
        train_script.write_text(modified)

        start = time.time()
        result = subprocess.run(
            ["uv", "run", "python", "train.py"],
            capture_output=True,
            text=True,
            timeout=TIME_BUDGET + 30,
        )
        elapsed = time.time() - start

        # Parse metrics
        output = result.stdout + result.stderr
        metrics = {
            "rate_weight": rate_weight,
            "depth_weight": DEPTH_WEIGHT,
            "training_time": elapsed,
        }

        for line in output.split("\n"):
            if "event_bpb:" in line and "event_mse:" not in line:
                try:
                    metrics["event_bpb"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "event_mse:" in line:
                try:
                    metrics["event_mse"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "depth_motion_error:" in line:
                try:
                    metrics["depth_motion_error"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        print(f"✅ Completed in {elapsed:.1f}s")
        if "event_bpb" in metrics:
            print(f"   event_bpb: {metrics['event_bpb']:.6f}")

        return metrics

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"rate_weight": rate_weight, "error": str(e)}
    finally:
        train_script.write_text(original)


def main() -> None:
    """Run fast HPO search."""
    print("=" * 60)
    print("FAST HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Testing {len(RATE_WEIGHTS)} rate_weight values")
    print(f"depth_weight fixed at {DEPTH_WEIGHT}")
    print(f"Time per config: {TIME_BUDGET}s")
    print(f"Estimated total: {len(RATE_WEIGHTS) * TIME_BUDGET / 60:.0f} minutes")
    print()

    results = []
    start_time = time.time()

    for rate_weight in RATE_WEIGHTS:
        metrics = run_single_config(rate_weight)
        results.append(metrics)

        # Save intermediate
        with open("hpo_fast_results.json", "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - start_time

    # Analyze results
    print("\n" + "=" * 60)
    print("HPO RESULTS SUMMARY")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        # Sort by event_bpb
        best = min(valid_results, key=lambda x: x.get("event_bpb", float("inf")))

        print("\n🏆 BEST CONFIGURATION:")
        print(f"   rate_weight: {best['rate_weight']}")
        print(f"   depth_weight: {DEPTH_WEIGHT}")
        print(f"   event_bpb: {best.get('event_bpb', 'N/A'):.6f}")
        print(f"   event_mse: {best.get('event_mse', 'N/A'):.6f}")
        print(f"   depth_motion: {best.get('depth_motion_error', 'N/A'):.2f}")
        print(f"   Training time: {best.get('training_time', 0):.1f}s")

        # Save best config
        with open("best_config.json", "w") as f:
            json.dump(
                {
                    "rate_weight": best["rate_weight"],
                    "depth_weight": DEPTH_WEIGHT,
                    "event_bpb": best.get("event_bpb"),
                    "event_mse": best.get("event_mse"),
                    "depth_motion_error": best.get("depth_motion_error"),
                },
                f,
                indent=2,
            )
        print("\n✅ Saved to: best_config.json")

        # Show all results
        print("\n📊 ALL RESULTS:")
        print(f"   {'rate_weight':<12} {'event_bpb':<12} {'event_mse':<12} {'depth_motion':<12}")
        print(f"   {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
        for r in sorted(valid_results, key=lambda x: x.get("event_bpb", float("inf"))):
            print(
                f"   {r['rate_weight']:<12.4f} {r.get('event_bpb', 0):<12.6f} {r.get('event_mse', 0):<12.6f} {r.get('depth_motion_error', 0):<12.2f}"
            )

    print(f"\n⏱️  Total HPO time: {total_time / 60:.1f} minutes")
    print("\n✅ HPO complete! Run full training with best configuration.")


if __name__ == "__main__":
    main()
