from unittest.mock import patch

import numpy as np
import pytest

from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.v2e_utils import video_writer


@pytest.fixture
def mock_cv2():
    with patch("cv2.VideoWriter") as mock:
        yield mock


def test_renderer_render_empty_events(mock_cv2):
    renderer = EventRenderer(exposure_mode=ExposureMode.DURATION, exposure_value=1 / 30)
    renderer.height = 260
    renderer.width = 346
    renderer.output_path = "/tmp"
    renderer.video_output_file_name = "test.avi"
    frames = renderer.render_events_to_frames(np.array([]), 260, 346)
    assert frames is None


@pytest.mark.parametrize(
    "mode", [ExposureMode.COUNT, ExposureMode.AREA_COUNT, ExposureMode.SOURCE]
)
def test_renderer_modes(mode, mock_cv2):
    renderer = EventRenderer(exposure_mode=mode, exposure_value=100)
    renderer.height = 260
    renderer.width = 346
    events = np.array([[0.0, 170, 130, 1.0]])
    frames = renderer.render_events_to_frames(events, 260, 346, return_frames=True)
    assert frames is not None
    assert frames.shape[1:] == (260, 346)


def test_video_writer_path_error():
    with pytest.raises(FileNotFoundError):
        video_writer("/nonexistent/dir/test.avi", 260, 346, 30)
