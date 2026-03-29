"""Shared fixtures for v2e tests."""

import pytest
import torch
import numpy as np


@pytest.fixture(params=["cpu"], ids=["cpu"])
def device(request):
    """Tensor device for testing."""
    return torch.device(request.param)


@pytest.fixture(params=[(64, 64), (260, 346)], ids=["small", "dvs346"])
def frame_size(request):
    """Frame dimensions (H, W)."""
    return request.param


@pytest.fixture
def pos_thres(frame_size, device):
    """Positive threshold tensor."""
    H, W = frame_size
    return torch.full((H, W), 0.2, dtype=torch.float32, device=device)


@pytest.fixture
def neg_thres(frame_size, device):
    """Negative threshold tensor."""
    H, W = frame_size
    return torch.full((H, W), 0.2, dtype=torch.float32, device=device)


@pytest.fixture
def noise_rate(frame_size, device):
    """Noise rate array (uniform)."""
    H, W = frame_size
    return torch.ones(H, W, dtype=torch.float32, device=device)


@pytest.fixture
def seeded_frames(frame_size):
    """Generate deterministic test frames."""
    H, W = frame_size

    def _make_frames(n, seed=42):
        frames = []
        for i in range(n):
            torch.manual_seed(seed + i)
            frames.append(torch.randint(0, 256, (H, W), dtype=torch.float32))
        return frames

    return _make_frames


@pytest.fixture
def master_reference():
    """Master branch reference values for 64x64, 10 frames, cutoff=200Hz, leak=0.1."""
    return {
        "lp_vals": [
            1.1982928514, 4.5538768768, 3.3053162098, 2.4699878693, 4.2341065407,
            5.2574954033, 4.7004804611, 4.1271343231, 4.5951199532, 3.2776193619,
        ],
        "tolerance": 1e-6,
    }
