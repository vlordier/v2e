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
    pytest.raises(AttributeError, lambda: emu.pos_thres_nominal)
    pytest.raises(AttributeError, lambda: emu.neg_thres_nominal)
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
        assert torch.all(emu.pos_thres == pos_thres)


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
            getattr(
                renderer, "event_count" if mode == ExposureMode.COUNT else "area_count"
            )
            == expected
        )


@patch("v2ecore.v2e_utils.video_writer")
def test_video_writer(mock_writer):
    writer = video_writer("/tmp/test.avi", 260, 346, 30)
    mock_writer.assert_called_once()
    assert writer is not None


@pytest.mark.parametrize(
    "cutoffhz,fs,warning",
    [
        (0, 500, False),
        (0, 0, False),
        (100, 500, False),
        (300, 500, True),  # eps > 0.3
    ],
)
def test_check_lowpass(cutoffhz, fs, warning, capsys):
    with capsys.disabled():
        check_lowpass(cutoffhz, fs, Mock())
    captured = capsys.readouterr()
    if warning:
        assert "warning" in captured.err.lower()
    else:
        assert "warning" not in captured.err.lower()


@pytest.mark.parametrize(
    "dt,cutoff_hz,expected_events",
    [
        (1 / 500.0, 0, 10),  # expect leak events
    ],
)
def test_emulator_leak_events(dt, cutoff_hz, expected_events, device):
    emu = EventEmulator(leak_rate_hz=0.1, cutoff_hz=cutoff_hz, device=device)
    frame = np.zeros((20, 20), dtype=np.uint8)
    emu.generate_events(frame, dt)
    events = emu.generate_events(frame, 2 * dt)
    assert len(events) > expected_events - 5  # approximate
