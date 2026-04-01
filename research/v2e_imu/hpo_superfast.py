#!/usr/bin/env python
"""
Super-Fast HPO using mini_fpv dataset.

Time: 90 seconds per configuration
Search: rate_weight in [0.001, 0.005, 0.01, 0.02, 0.05]
Total: ~8 minutes for all 5 configs

Usage:
    cd data/fpv && mv indoor_forward_* backup_temp/  # Use mini_fpv
    uv run python hpo_superfast.py
    cd data/fpv && mv backup_temp/* .  # Restore
"""

import json
import subprocess
import time
from pathlib import Path

RATE_WEIGHTS = [0.001, 0.005, 0.01, 0.02, 0.05]
DEPTH_WEIGHT = 0.1

TIME_BUDGET = 90  # 90 seconds
EVAL_SAMPLES = 20


def run_config(rate_weight: float) -> dict:
    """Run single configuration."""
    print(f"\n{'=' * 50}")
    print(f"Testing rate_weight={rate_weight}")
    print(f"{'=' * 50}")

    train_script = Path("train.py")
    original = train_script.read_text()

    modified = (
        original.replace(
            "depth_weight = 0.1  # Auxiliary task weight", f"depth_weight = {DEPTH_WEIGHT}"
        )
        .replace(
            "rate_weight = 0.5  # INCREASED: Strong event hallucination prevention (was 0.05)",
            f"rate_weight = {rate_weight}  # HPO",
        )
        .replace(
            "TIME_BUDGET = 900  # 15 minutes per experiment (increased for proper evaluation)",
            f"TIME_BUDGET = {TIME_BUDGET}  # HPO",
        )
        .replace(
            "EVAL_SAMPLES = 100  # Number of samples for evaluation (reduced for speed)",
            f"EVAL_SAMPLES = {EVAL_SAMPLES}  # HPO",
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

        print(f"✅ {elapsed:.0f}s | event_bpb: {metrics.get('event_bpb', 'N/A')}")
        return metrics

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {"rate_weight": rate_weight, "error": str(e)}
    finally:
        train_script.write_text(original)


def main() -> None:
    """Run super-fast HPO."""
    print("=" * 50)
    print("SUPER-FAST HPO (mini_fpv dataset)")
    print("=" * 50)
    print(f"Configs: {len(RATE_WEIGHTS)}")
    print(f"Time each: ~{TIME_BUDGET}s")
    print(f"Total: ~{len(RATE_WEIGHTS) * TIME_BUDGET / 60:.0f} minutes")
    print()

    results = []
    start = time.time()

    for rate_weight in RATE_WEIGHTS:
        metrics = run_config(rate_weight)
        results.append(metrics)

        with open("hpo_superfast.json", "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - start

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    valid = [r for r in results if "error" not in r]

    if valid:
        best = min(valid, key=lambda x: x.get("event_bpb", float("inf")))

        print(f"\n🏆 BEST: rate_weight={best['rate_weight']}")
        print(f"   event_bpb: {best.get('event_bpb', 0):.6f}")
        print(f"   event_mse: {best.get('event_mse', 0):.6f}")

        with open("best_config.json", "w") as f:
            json.dump(
                {
                    "rate_weight": best["rate_weight"],
                    "depth_weight": DEPTH_WEIGHT,
                    "event_bpb": best.get("event_bpb"),
                    "event_mse": best.get("event_mse"),
                },
                f,
                indent=2,
            )

        print("\n📊 ALL:")
        for r in sorted(valid, key=lambda x: x.get("event_bpb", float("inf"))):
            print(
                f"   {r['rate_weight']:.3f}: bpb={r.get('event_bpb', 0):.6f}, mse={r.get('event_mse', 0):.6f}"
            )

    print(f"\n⏱️  Total: {total / 60:.1f} minutes")


if __name__ == "__main__":
    main()
