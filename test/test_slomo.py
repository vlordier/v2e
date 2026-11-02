from unittest.mock import Mock, patch

from v2ecore.slomo import SuperSloMo


@patch("v2ecore.model.UNet")
@patch("v2ecore.dataloader.FramesDirectory")
def test_slomo_interpolate_mock(mock_frames, mock_unet):
    mock_model = Mock()
    mock_model.load_state_dict = Mock()
    mock_unet.return_value = mock_model
    slomo = SuperSloMo(
        model="/fake/model.pth", auto_upsample=False, upsampling_factor=2
    )
    times, avg = slomo.interpolate("/fake/frames", "/fake/output", (346, 260))
    assert len(times) > 0
    assert avg > 0
