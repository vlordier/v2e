import cv2
import numpy as np

from v2ecore.calibration import calibrate_thresholds
from v2ecore.emulator import EventEmulator


def _write_video(frames, path, fps):
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), True)
    assert writer.isOpened(), f"Could not open writer for {path}"
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _events_to_numpy(events):
    if events is None:
        return np.empty((0, 4), dtype=np.float32)
    if hasattr(events, "detach"):
        return events.detach().cpu().numpy()
    return np.asarray(events)


def _write_real_events_from_emulator(frames, fps, path, pos_thres, neg_thres):
    emu = EventEmulator(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        device="cpu",
    )
    with open(path, "w", encoding="utf-8") as fh:
        for i, frame in enumerate(frames):
            t = i / fps
            events = _events_to_numpy(emu.generate_events(frame, t))
            for row in events:
                fh.write(f"{float(row[0]):.6f} {int(row[1])} {int(row[2])} {int(row[3])}\n")


def _count_events_txt(path):
    on = off = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = float(line.split()[3])
            if p > 0:
                on += 1
            else:
                off += 1
    return on, off


def _simulate_counts(frames, fps, pos_thres, neg_thres):
    emu = EventEmulator(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        device="cpu",
    )
    on = off = 0
    for i, frame in enumerate(frames):
        t = i / fps
        events = _events_to_numpy(emu.generate_events(frame, t))
        if events.size == 0:
            continue
        on += int(np.sum(events[:, 3] > 0))
        off += int(np.sum(events[:, 3] <= 0))
    return on, off


def test_calibrate_thresholds_match_emulator_counts_on_oscillatory_sequence(tmp_path):
    """Returned thresholds should reproduce real counts on a sign-changing sequence.

    This specifically guards against analytical over-counting on low-intensity,
    oscillatory inputs where summing positive/negative diffs is too optimistic.
    """
    fps = 30.0
    h = w = 24

    base = np.tile(np.linspace(0, 24, w, dtype=np.float32), (h, 1))
    frames = []
    deltas = [0, 5, -4, 6, -5, 7, -6, 8, -7, 32, -18, 40, -22, 52]
    curr = base.copy()
    for i, delta in enumerate(deltas):
        curr = np.clip(curr + delta, 0, 255)
        frame = curr.copy()
        # Spatial asymmetry makes ON/OFF counts differ and exercises both searches.
        frame[:, : w // 2] *= 0.45
        if i % 3 == 0:
            frame[h // 3 : 2 * h // 3, :] += 6
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    video_path = tmp_path / "oscillatory.avi"
    events_path = tmp_path / "real_events.txt"
    _write_video(frames, video_path, fps)
    _write_real_events_from_emulator(
        frames, fps, events_path, pos_thres=0.22, neg_thres=0.26
    )

    real_on, real_off = _count_events_txt(events_path)
    assert real_on > 0 and real_off > 0

    pos_thres, neg_thres = calibrate_thresholds(
        video_path=video_path,
        real_events_path=events_path,
        start_time=0.0,
        stop_time=(len(frames) - 1) / fps + 1e-6,
    )

    pred_on, pred_off = _simulate_counts(frames, fps, pos_thres, neg_thres)

    on_err = abs(pred_on - real_on) / real_on
    off_err = abs(pred_off - real_off) / real_off

    assert on_err < 0.25, f"ON error too high: {on_err:.1%} ({pred_on=} vs {real_on=})"
    assert off_err < 0.25, f"OFF error too high: {off_err:.1%} ({pred_off=} vs {real_off=})"
