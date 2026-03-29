from pathlib import Path
from unittest.mock import Mock

import pytest

from v2ecore.v2e_utils import check_lowpass, video_writer


def test_video_writer_path_error(caplog):
    Path("/tmp/nonexistent").mkdir(exist_ok=True)  # ensure dir exists for test
    with pytest.raises(OSError, match="No such file"):
        video_writer("/tmp/nonexistent/test.avi", 260, 346)


def test_check_lowpass(caplog):
    mock_logger = Mock()
    check_lowpass(300, 500, mock_logger)
    assert "warning" in caplog.text.lower()
