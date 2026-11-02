#!/usr/bin/env python3
"""Quick synthetic sweep for the experimental v2e realism controls.

This script runs a few small, reproducible frame sequences through
`EventEmulator` and reports how the new knobs affect event statistics.
It is meant as a lightweight sanity-check and comparison aid rather than
an exhaustive benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v2ecore.emulator import EventEmulator


@dataclass
class SweepResult:
    scenario: str
    variant: str
    total_events: int
    on_events: int
    off_events: int


DEFAULTS = dict(
    pos_thres=0.05,
    neg_thres=0.05,
    sigma_thres=0.0,
    cutoff_hz=0,
    leak_rate_hz=0,
    shot_noise_rate_hz=0,
    device="cpu",
)


def count_events(events: np.ndarray | None) -> tuple[int, int, int]:
    if events is None or len(events) == 0:
        return 0, 0, 0
    on_events = int((events[:, 3] > 0).sum())
    off_events = int((events[:, 3] < 0).sum())
    return int(len(events)), on_events, off_events


def run_sequence(
    scenario: str,
    variant: str,
    frames: list[np.ndarray],
    timestamps: list[float],
    **config,
) -> SweepResult:
    emulator = EventEmulator(**DEFAULTS, **config)
    total_events = 0
    on_events = 0
    off_events = 0

    for frame, timestamp in zip(frames, timestamps, strict=False):
        events = emulator.generate_events(frame, timestamp)
        total, on_count, off_count = count_events(events)
        total_events += total
        on_events += on_count
        off_events += off_count

    return SweepResult(
        scenario=scenario,
        variant=variant,
        total_events=total_events,
        on_events=on_events,
        off_events=off_events,
    )


def build_scenarios() -> list[SweepResult]:
    results: list[SweepResult] = []

    rapid_frames = [
        np.ones((20, 20), dtype=np.uint8) * 10,
        np.ones((20, 20), dtype=np.uint8) * 80,
        np.ones((20, 20), dtype=np.uint8) * 255,
    ]
    rapid_times = [0.001, 0.002, 0.0022]
    results.append(run_sequence("rapid-ramp", "baseline", rapid_frames, rapid_times, seed=1))
    results.append(
        run_sequence(
            "rapid-ramp",
            "soft-refractory",
            rapid_frames,
            rapid_times,
            seed=1,
            refractory_period_s=0.01,
            refractory_mode="soft",
            refractory_tau_s=0.002,
        )
    )
    results.append(
        run_sequence(
            "rapid-ramp",
            "threshold-adapt",
            rapid_frames,
            rapid_times,
            seed=2,
            threshold_adaptation_gain=0.1,
            threshold_adaptation_tau_s=0.05,
        )
    )

    static_frames = [np.ones((8, 8), dtype=np.uint8) * 64 for _ in range(3)]
    static_times = [0.001, 0.002, 0.003]
    results.append(run_sequence("static-hold", "baseline", static_frames, static_times, seed=3))
    results.append(
        run_sequence(
            "static-hold",
            "hot-pixels",
            static_frames,
            static_times,
            seed=3,
            hot_pixel_fraction=0.05,
            hot_pixel_rate_hz=1000.0,
            bursty_pixel_fraction=0.05,
            bursty_pixel_rate_hz=400.0,
            row_noise_rate_hz=30.0,
        )
    )

    cut_frames = [
        np.ones((12, 12), dtype=np.uint8) * 10,
        np.ones((12, 12), dtype=np.uint8) * 240,
    ]
    cut_times = [0.001, 0.002]
    results.append(run_sequence("scene-cut", "baseline", cut_frames, cut_times, seed=4))
    results.append(
        run_sequence(
            "scene-cut",
            "cut-reset",
            cut_frames,
            cut_times,
            seed=4,
            scene_cut_policy="reset",
            scene_cut_threshold=0.2,
        )
    )

    return results


def print_results(results: list[SweepResult]) -> None:
    print("scenario       variant            total  on   off")
    print("-------------  -----------------  -----  ---  ---")
    for result in results:
        print(
            f"{result.scenario:<13}  {result.variant:<17}  {result.total_events:>5}  "
            f"{result.on_events:>3}  {result.off_events:>3}"
        )

    def lookup(scenario: str, variant: str) -> SweepResult:
        return next(r for r in results if r.scenario == scenario and r.variant == variant)

    rapid_baseline = lookup("rapid-ramp", "baseline").total_events
    rapid_soft = lookup("rapid-ramp", "soft-refractory").total_events
    rapid_adapt = lookup("rapid-ramp", "threshold-adapt").total_events
    static_base = lookup("static-hold", "baseline").total_events
    static_hot = lookup("static-hold", "hot-pixels").total_events
    cut_base = lookup("scene-cut", "baseline").total_events
    cut_reset = lookup("scene-cut", "cut-reset").total_events

    print("\nSummary:")
    if rapid_baseline:
        print(
            f"- Soft refractory reduced rapid-ramp activity by "
            f"{100 * (rapid_baseline - rapid_soft) / rapid_baseline:.1f}% "
            f"({rapid_baseline} -> {rapid_soft} events)."
        )
        print(
            f"- Threshold adaptation reduced rapid-ramp activity by "
            f"{100 * (rapid_baseline - rapid_adapt) / rapid_baseline:.1f}% "
            f"({rapid_baseline} -> {rapid_adapt} events)."
        )
    print(
        f"- Defect controls added {static_hot - static_base} extra events on a static scene "
        f"({static_base} -> {static_hot})."
    )
    print(
        f"- Scene-cut reset suppressed the cut burst from {cut_base} to {cut_reset} events."
    )


if __name__ == "__main__":
    print_results(build_scenarios())
