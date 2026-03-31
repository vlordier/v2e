#!/usr/bin/env python
"""
Hyperparameter Optimization for 3D-Aware Event Prediction.

Searches for optimal loss weights:
- rate_weight: [0.001, 0.005, 0.01, 0.05, 0.1]
- depth_weight: [0.05, 0.1, 0.2]

Usage:
    uv run python hpo_search.py
    
Runs multiple short training sessions and finds best configuration.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# Hyperparameter search space
RATE_WEIGHTS = [0.001, 0.005, 0.01, 0.05, 0.1]
DEPTH_WEIGHTS = [0.05, 0.1, 0.2]

# Training configuration for HPO
HPO_TIME_BUDGET = 300  # 5 minutes per run (short for quick iteration)
HPO_EVAL_SAMPLES = 50  # Fewer samples for faster evaluation


def run_training(
    rate_weight: float,
    depth_weight: float,
    time_budget: int = HPO_TIME_BUDGET,
) -> Tuple[float, Dict[str, float]]:
    """
    Run training with specific hyperparameters.
    
    Returns:
        final_loss: Training loss at end
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training: rate_weight={rate_weight}, depth_weight={depth_weight}")
    print(f"{'='*60}")
    
    # Modify train.py temporarily with these hyperparameters
    train_script = Path("train.py")
    original_content = train_script.read_text()
    
    # Replace hyperparameters in training_step function
    modified_content = original_content.replace(
        "depth_weight = 0.1  # Auxiliary task weight",
        f"depth_weight = {depth_weight}  # HPO: Auxiliary task weight"
    ).replace(
        "rate_weight = 0.5  # INCREASED: Strong event hallucination prevention (was 0.05)",
        f"rate_weight = {rate_weight}  # HPO: Event hallucination prevention"
    )
    
    # Also update TIME_BUDGET and EVAL_SAMPLES for faster HPO
    modified_content = modified_content.replace(
        "TIME_BUDGET = 900  # 15 minutes per experiment (increased for proper evaluation)",
        f"TIME_BUDGET = {time_budget}  # HPO: {time_budget/60:.0f} minutes"
    ).replace(
        "EVAL_SAMPLES = 100  # Number of samples for evaluation (reduced for speed)",
        f"EVAL_SAMPLES = {HPO_EVAL_SAMPLES}  # HPO: Faster evaluation"
    )
    
    try:
        # Write modified script
        train_script.write_text(modified_content)
        
        # Run training
        start_time = time.time()
        result = subprocess.run(
            ["uv", "run", "python", "train.py"],
            capture_output=True,
            text=True,
            timeout=time_budget + 60,  # Add 1 minute buffer
        )
        training_time = time.time() - start_time
        
        # Parse output for metrics
        output = result.stdout + result.stderr
        metrics = parse_training_output(output)
        metrics["training_time"] = training_time
        metrics["rate_weight"] = rate_weight
        metrics["depth_weight"] = depth_weight
        
        # Get final loss
        final_loss = metrics.get("event_bpb", float("inf"))
        
        print(f"Training completed in {training_time:.1f}s")
        print(f"event_bpb: {final_loss:.6f}")
        
        return final_loss, metrics
        
    except subprocess.TimeoutExpired:
        print(f"Training timed out after {time_budget + 60}s")
        return float("inf"), {"error": "timeout"}
    except Exception as e:
        print(f"Training failed: {e}")
        return float("inf"), {"error": str(e)}
    finally:
        # Restore original script
        train_script.write_text(original_content)


def parse_training_output(output: str) -> Dict[str, float]:
    """Parse training output for metrics."""
    metrics = {}
    
    # Look for metric lines
    for line in output.split("\n"):
        if "event_bpb:" in line:
            try:
                metrics["event_bpb"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "event_mse:" in line:
            try:
                metrics["event_mse"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "event_rate_error:" in line:
            try:
                metrics["event_rate_error"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "depth_motion_error:" in line:
            try:
                metrics["depth_motion_error"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
    
    return metrics


def run_hpo_search() -> List[Dict]:
    """Run complete HPO search."""
    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Rate weights: {RATE_WEIGHTS}")
    print(f"Depth weights: {DEPTH_WEIGHTS}")
    print(f"Time budget per run: {HPO_TIME_BUDGET}s")
    print(f"Total configurations: {len(RATE_WEIGHTS) * len(DEPTH_WEIGHTS)}")
    print(f"Estimated total time: {(len(RATE_WEIGHTS) * len(DEPTH_WEIGHTS) * HPO_TIME_BUDGET) / 3600:.1f} hours")
    
    all_results = []
    start_time = time.time()
    
    for rate_weight in RATE_WEIGHTS:
        for depth_weight in DEPTH_WEIGHTS:
            config_start = time.time()
            
            final_loss, metrics = run_training(rate_weight, depth_weight)
            
            metrics["final_loss"] = final_loss
            metrics["config_time"] = time.time() - config_start
            all_results.append(metrics)
            
            # Save intermediate results
            save_hpo_results(all_results)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"HPO completed in {total_time/3600:.2f} hours")
    print(f"{'='*60}")
    
    return all_results


def save_hpo_results(results: List[Dict]) -> None:
    """Save HPO results to file."""
    output_file = f"hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Find most recent results file
    existing_files = list(Path(".").glob("hpo_results_*.json"))
    if existing_files:
        output_file = sorted(existing_files)[-1]
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def print_best_results(results: List[Dict]) -> None:
    """Print and analyze best results."""
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)
    
    # Sort by event_bpb (lower is better)
    sorted_results = sorted(results, key=lambda x: x.get("event_bpb", float("inf")))
    
    print("\nTop 5 by event_bpb:")
    for i, result in enumerate(sorted_results[:5], 1):
        if "error" not in result:
            print(f"\n{i}. rate_weight={result['rate_weight']}, depth_weight={result['depth_weight']}")
            print(f"   event_bpb: {result.get('event_bpb', 'N/A'):.6f}")
            print(f"   event_mse: {result.get('event_mse', 'N/A'):.6f}")
            print(f"   Training time: {result.get('training_time', 0):.1f}s")
    
    # Find best configuration
    if sorted_results and "error" not in sorted_results[0]:
        best = sorted_results[0]
        print("\n" + "="*60)
        print("RECOMMENDED CONFIGURATION")
        print("="*60)
        print(f"rate_weight: {best['rate_weight']}")
        print(f"depth_weight: {best['depth_weight']}")
        print(f"event_bpb: {best.get('event_bpb', 'N/A'):.6f}")
        print(f"event_mse: {best.get('event_mse', 'N/A'):.6f}")
        
        # Save recommended config
        with open("best_hpo_config.json", "w") as f:
            json.dump({
                "rate_weight": best["rate_weight"],
                "depth_weight": best["depth_weight"],
                "event_bpb": best.get("event_bpb"),
                "event_mse": best.get("event_mse"),
            }, f, indent=2)
        print("\n✅ Saved to: best_hpo_config.json")


def main() -> None:
    """Main HPO search."""
    print("Starting Hyperparameter Optimization...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run HPO search
    results = run_hpo_search()
    
    # Analyze and print best results
    print_best_results(results)
    
    print("\n✅ HPO complete!")
    print("Next step: Run full training with best configuration")


if __name__ == "__main__":
    main()
