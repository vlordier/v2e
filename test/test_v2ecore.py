from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from v2ecore.emulator import EventEmulator
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.v2e_utils import check_lowpass, video_writer


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def mock_cv2():
    with patch("cv2.imread") as mock:
        mock.return_value = np.zeros((260, 346), dtype=np.uint8)
        yield mock


@pytest.fixture
def emulator(device):
    torch.manual_seed(0)
    return EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        device=device,
        output_folder="/tmp",
        output_width=346,
        output_height=260,
    )


def test_emulator_init(device):
    emu = EventEmulator(pos_thres=0.1, neg_thres=0.15, device=device)
    assert emu.pos_thres_nominal == 0.1
    assert emu.neg_thres_nominal == 0.15
    assert emu.cutoff_hz == 0
    assert emu.leak_rate_hz == 0.1


@pytest.mark.parametrize(
    "pos_thres,neg_thres,sigma_thres",
    [
        (0.2, 0.2, 0.0),
        (0.2, 0.2, 0.03),
        (0.01, 0.01, 0.001),
    ],
)
def test_emulator_thresholds(pos_thres, neg_thres, sigma_thres, device, mock_cv2):
    frame = np.ones((10, 10), dtype=np.uint8) * 128
    emu = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres, device=device
    )
    emu.generate_events(frame, 1 / 500.0)
    if sigma_thres > 0:
        assert torch.std(emu.pos_thres) > 0
    else:
        if isinstance(emu.pos_thres, torch.Tensor):
            assert torch.all(emu.pos_thres == pos_thres)
        else:
            assert emu.pos_thres == pos_thres


def test_emulator_generate_events_first_frame_returns_none(emulator, mock_cv2):
    frame = np.ones((260, 346), dtype=np.uint8)
    events = emulator.generate_events(frame, 0.002)
    assert events is None
    assert emulator.base_log_frame is not None


@pytest.mark.parametrize(
    "cutoff_hz,leak_rate_hz,shot_noise_rate_hz",
    [(0, 0, 0), (200, 0.2, 10), (0, 0.1, 0)],
)
def test_emulator_generate_constant_frame_no_events(
    cutoff_hz, leak_rate_hz, shot_noise_rate_hz, device
):
    emu = EventEmulator(
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        device=device,
    )
    frame = np.ones((10, 10), dtype=np.uint8) * 128
    emu.generate_events(frame, 0.002)
    events = emu.generate_events(frame, 0.004)
    if leak_rate_hz == 0 and shot_noise_rate_hz == 0:
        assert events is None or len(events) == 0


def test_renderer_init_exposure_modes():
    for mode in ExposureMode:
        renderer = EventRenderer(exposure_mode=mode, exposure_value=1.0)
        assert renderer.exposure_mode == mode


@pytest.mark.parametrize(
    "mode,value,expected",
    [
        (ExposureMode.DURATION, 1 / 30.0, 30.0),
        (ExposureMode.COUNT, 100, 100),
        (ExposureMode.AREA_COUNT, 50, 50),
    ],
)
def test_renderer_exposure(mode, value, expected):
    renderer = EventRenderer(exposure_mode=mode, exposure_value=value)
    if mode == ExposureMode.DURATION:
        assert renderer.frame_rate_hz == expected
    else:
        assert (
            getattr(renderer, "event_count" if mode == ExposureMode.COUNT else "area_count")
            == expected
        )


@patch("v2ecore.v2e_utils.cv2.VideoWriter")
def test_video_writer(mock_writer):
    writer = video_writer("/tmp/test.avi", 260, 346, 30)
    mock_writer.assert_called_once()
    assert writer is not None


@pytest.mark.parametrize(
    "cutoffhz,fs,warning",
    [
        (0, 500, False),
        (0, 0, False),
        (10, 500, False),
        (300, 500, True),  # eps > 0.3
    ],
)
def test_check_lowpass(cutoffhz, fs, warning, capsys):
    mock_logger = Mock()
    with capsys.disabled():
        check_lowpass(cutoffhz, fs, mock_logger)
    if warning:
        mock_logger.warning.assert_called_once()
    else:
        mock_logger.warning.assert_not_called()


def test_emulator_leak_events(device):
    emu = EventEmulator(
        leak_rate_hz=1000,
        cutoff_hz=0,
        shot_noise_rate_hz=0,
        sigma_thres=0,
        seed=4,
        device=device,
    )
    frame = np.ones((4, 4), dtype=np.uint8) * 128
    emu.generate_events(frame, 0.1)
    emu.generate_events(frame, 0.2)
    events = emu.generate_events(frame, 0.3)
    assert events is not None and len(events) > 0


def test_soft_refractory_reduces_rapid_retriggering(device):
    common = dict(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        seed=1,
        device=device,
    )
    frame_0 = np.ones((20, 20), dtype=np.uint8) * 10
    frame_1 = np.ones((20, 20), dtype=np.uint8) * 80
    frame_2 = np.ones((20, 20), dtype=np.uint8) * 255

    emu_none = EventEmulator(**common, refractory_period_s=0.0)
    emu_soft = EventEmulator(
        **common,
        refractory_period_s=0.01,
        refractory_mode="soft",
        refractory_tau_s=0.002,
    )

    for emu in (emu_none, emu_soft):
        emu.generate_events(frame_0, 0.001)
        emu.generate_events(frame_1, 0.002)

    events_none = emu_none.generate_events(frame_2, 0.0022)
    events_soft = emu_soft.generate_events(frame_2, 0.0022)

    assert events_none is not None and len(events_none) > 0
    soft_count = 0 if events_soft is None else len(events_soft)
    assert 0 < soft_count < len(events_none)


def test_threshold_adaptation_reduces_immediate_followup_activity(device):
    common = dict(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        seed=2,
        device=device,
    )
    frame_0 = np.ones((20, 20), dtype=np.uint8) * 10
    frame_1 = np.ones((20, 20), dtype=np.uint8) * 80
    frame_2 = np.ones((20, 20), dtype=np.uint8) * 255

    emu_plain = EventEmulator(**common)
    emu_adapt = EventEmulator(
        **common,
        threshold_adaptation_gain=0.1,
        threshold_adaptation_tau_s=0.05,
    )

    for emu in (emu_plain, emu_adapt):
        emu.generate_events(frame_0, 0.001)
        emu.generate_events(frame_1, 0.002)

    events_plain = emu_plain.generate_events(frame_2, 0.0025)
    events_adapt = emu_adapt.generate_events(frame_2, 0.0025)

    assert events_plain is not None and len(events_plain) > 0
    adapt_count = 0 if events_adapt is None else len(events_adapt)
    assert 0 < adapt_count < len(events_plain)


def test_hot_pixels_emit_events_on_static_frame(device):
    emu = EventEmulator(
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        hot_pixel_fraction=1.0,
        hot_pixel_rate_hz=1000.0,
        seed=3,
        device=device,
    )
    frame = np.ones((8, 8), dtype=np.uint8) * 64
    emu.generate_events(frame, 0.001)
    events = emu.generate_events(frame, 0.002)
    assert events is not None and len(events) > 0


def test_scene_cut_reset_policy_suppresses_cut_burst(device):
    common = dict(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0,
        leak_rate_hz=0,
        shot_noise_rate_hz=0,
        device=device,
    )
    dark = np.ones((12, 12), dtype=np.uint8) * 10
    bright = np.ones((12, 12), dtype=np.uint8) * 240

    emu_none = EventEmulator(**common, scene_cut_policy="none")
    emu_reset = EventEmulator(**common, scene_cut_policy="reset", scene_cut_threshold=0.2)

    emu_none.generate_events(dark, 0.001)
    emu_reset.generate_events(dark, 0.001)

    events_none = emu_none.generate_events(bright, 0.002)
    events_reset = emu_reset.generate_events(bright, 0.002)

    assert events_none is not None and len(events_none) > 0
    assert events_reset is None or len(events_reset) == 0
