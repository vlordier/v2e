import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
IMU_DIR = ROOT / "research" / "v2e_imu"
sys.path.insert(0, str(IMU_DIR))

import train  # noqa: E402
from fno_event_predictor import FNOEventPredictor, FourierLayer  # noqa: E402


class UpdateSmoothedLossesTests(unittest.TestCase):
    def test_ignores_non_finite_accumulator(self) -> None:
        smooth = {"total": 0.9, "evt": 0.4, "rate": 0.1}

        updated, debias = train.update_smoothed_losses(
            smooth.copy(),
            {"total": float("nan"), "evt": float("inf"), "rate": 0.1},
            step=4,
        )

        self.assertGreater(debias, 0)
        self.assertEqual(updated, smooth)
        self.assertTrue(all(math.isfinite(value) for value in updated.values()))

    def test_applies_finite_values(self) -> None:
        updated, debias = train.update_smoothed_losses(
            {"total": 0.0, "evt": 0.0, "rate": 0.0},
            {"total": 1.0, "evt": 0.5, "rate": 0.25},
            step=0,
            ema=0.9,
        )

        self.assertGreater(debias, 0)
        self.assertGreater(updated["total"], 0)
        self.assertGreater(updated["evt"], 0)
        self.assertGreater(updated["rate"], 0)


class FNOHardeningTests(unittest.TestCase):
    def test_fourier_layer_handles_small_feature_maps(self) -> None:
        layer = FourierLayer(channels=16, modes=10)
        x = torch.randn(2, 16, 8, 8)

        y = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())

    def test_optimize_model_skips_torch_compile_for_fno(self) -> None:
        model = FNOEventPredictor(modes=6, fno_layers=2, channels=32)

        with mock.patch.dict(os.environ, {"V2E_TORCH_COMPILE": "1"}, clear=False):
            with mock.patch.object(
                torch,
                "compile",
                side_effect=AssertionError("FNO should not be compiled"),
            ) as mocked_compile:
                optimized = train.optimize_model_for_device(model, "cuda")

        self.assertIs(optimized, model)
        mocked_compile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
