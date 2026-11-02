"""A proper search function for threshold estimation.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import logging
import os
import subprocess

import h5py
import numpy as np

from v2ecore.v2e_utils import select_events_in_roi

logger = logging.getLogger(__name__)


def evaluate_threshold(cfg, idx, threshold, event_count_diffs, ref_event_count=0):

    # v2e commands
    v2e_command = [
        "v2e.py",
        "-i",
        cfg.input,
        "-o",
        cfg.output_folder,
        "--dvs_emulator_seed",
        "42",
        "--start_time",
        f"{cfg.start}",
        "--stop_time",
        f"{cfg.stop}",
        "--overwrite",
        "--unique_output_folder",
        "false",
        "--no_preview",
        "--skip_video_output",
        "--pos_thres",
        f"{threshold}",
        "--neg_thres",
        f"{threshold}",
        "--sigma_thres",
        "0",
        "--cutoff_hz",
        "0",
        "--leak_rate_hz",
        "0",
        "--shot_noise_rate_hz",
        "0",
        "--input_frame_rate",
        f"{cfg.input_frame_rate}",
        #  "--input_slowmotion_factor", "17.866666708",
        "--dvs_h5",
        "dvs_events.h5",
        "--dvs_aedat2",
        "None",
        "--dvs_text",
        "None",
        "--dvs_exposure",
        "duration",
        "0.001",
        #  "--disable_slomo",
        "--auto_timestamp_resolution",
        "True",
        "--slomo_model",
        cfg.slomo_model,
    ]

    # Note: Currently the slomo model must be re-run for each threshold evaluation.
    # Future enhancement: Check for existing video_slomo.avi and reuse it with --disable_slomo
    # to avoid repeated slomo computation.

    if event_count_diffs[idx] is None:
        # evaluate and assign
        subprocess.run(v2e_command)

        # load events
        data = h5py.File(os.path.join(cfg.output_folder, "dvs_events.h5"), "r")
        curr_events = data["events"][()]
        curr_events = select_events_in_roi(curr_events, cfg.x, cfg.y)
        curr_event_count = curr_events.shape[0]

        # assign event counts
        event_count_diffs[idx] = abs(curr_event_count - ref_event_count)

    return event_count_diffs[idx], event_count_diffs


def threshold_estimator(
    cfg, low_thres_idx, high_thres_idx, threshold_range, event_count_diffs, ref_event_count=0
):
    """Binary search for estimating threshold.

    Assume only single minimum.

    # Arguments
    cfg: a result of parsing arguments, NameSpace that contains
        all necessary arguments. such as IO, V2E settings, etc
    low_thres: float
        threshold boundary (low)
    high_thres: float
        threshold boundary (high)
    threshold_range: list
        a list of thresholds, all values larger than 0.
        assume small to large
    event_count_diffs: list
        a list of event count absolute differences
    v2e_commands: for subprocess to call
        should be able to modify threshold parameter.
    ref_event_count: int
        should be the reference event count.
    """

    # boundary
    start_idx, end_idx = 0, len(threshold_range) - 1

    curr_mid_idx = (high_thres_idx + low_thres_idx) // 2
    while low_thres_idx < high_thres_idx:
        logger.info(
            "current range: %s-%s",
            threshold_range[low_thres_idx],
            threshold_range[high_thres_idx],
        )

        # terminate condition, if neighbors are all higher than current,
        # then terminate
        # left neighbor
        if curr_mid_idx > 5:
            left_count, event_count_diffs = evaluate_threshold(
                cfg,
                curr_mid_idx - 5,
                threshold_range[curr_mid_idx - 5],
                event_count_diffs,
                ref_event_count,
            )
        else:
            left_count = np.inf

        # right neighbor
        if curr_mid_idx < end_idx - 5:
            right_count, event_count_diffs = evaluate_threshold(
                cfg,
                curr_mid_idx + 5,
                threshold_range[curr_mid_idx + 5],
                event_count_diffs,
                ref_event_count,
            )
        else:
            right_count = np.inf

        if (curr_mid_idx == start_idx + 5 or left_count >= curr_mid_count) and (
            curr_mid_idx == end_idx - 5 or right_count >= curr_mid_count
        ):
            return curr_mid, event_count_diffs
        elif curr_mid_idx > start_idx and left_count < curr_mid_count:
            high_thres_idx = curr_mid_idx
        elif curr_mid_idx < end_idx and right_count < curr_mid_count:
            low_thres_idx = curr_mid_idx + 1

    return curr_mid, event_count_diffs
