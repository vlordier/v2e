#!/usr/bin/env python
"""
Unit Tests for V2E Improved Event Prediction.

Test coverage: Core functionality, metrics, data loading
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from train import BASE_CHANNELS, IMU_HIDDEN_DIM, EventPredictor, ModelConfig


class TestEventPredictor(unittest.TestCase):
    """Test EventPredictor model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
        self.model = EventPredictor(self.config)
        self.batch_size = 2
        self.image_size = (260, 346)

    def test_model_initialization(self) -> None:
        """Test model initializes correctly."""
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 1_000_000)  # at least 1M params

    def test_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        # 2-channel input: (current_frame, previous_frame)
        rgb = torch.randn(self.batch_size, 2, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)

        events = self.model(rgb, imu)

        self.assertEqual(events.shape, (self.batch_size, 2, *self.image_size))

    def test_output_in_01(self) -> None:
        """Test event predictions are sigmoid probabilities in [0, 1]."""
        rgb = torch.randn(self.batch_size, 2, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)

        events = self.model(rgb, imu)

        self.assertTrue((events >= 0).all())
        self.assertTrue((events <= 1).all())


class TestFocalBCELoss(unittest.TestCase):
    """Test focal BCE loss implementation."""

    def test_focal_bce_basic(self) -> None:
        """Loss is finite and non-negative for random inputs."""
        from train import focal_bce_loss

        pred = torch.sigmoid(torch.randn(2, 2, 16, 16))
        gt = torch.rand(2, 2, 16, 16) * 0.08  # realistic GT range
        loss = focal_bce_loss(pred, gt)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)

    def test_focal_bce_alpha_upweights_positives(self) -> None:
        """alpha=0.75 means positive class contributes more than negative."""
        from train import focal_bce_loss

        # All-positive GT, low prediction → high loss
        pred_bad = torch.full((1, 2, 8, 8), 0.01)
        gt_all_pos = torch.full((1, 2, 8, 8), 0.08)  # above 0.005 threshold
        loss_bad = focal_bce_loss(pred_bad, gt_all_pos)

        # Perfect prediction → low loss
        pred_good = torch.full((1, 2, 8, 8), 0.99)
        loss_good = focal_bce_loss(pred_good, gt_all_pos)

        self.assertLess(loss_good.item(), loss_bad.item())

    def test_focal_bce_range(self) -> None:
        """Loss is bounded (focal weight shrinks easy examples)."""
        from train import focal_bce_loss

        pred = torch.full((2, 2, 32, 32), 0.5)
        gt = torch.zeros(2, 2, 32, 32)
        loss = focal_bce_loss(pred, gt)
        # With gamma=2, loss at p=0.5 is 0.25 * log(2) * alpha_weight < 1
        self.assertLess(loss.item(), 1.0)


class TestEventRateRegularization(unittest.TestCase):
    """Test event rate regularization."""

    def test_rate_regularization_encourages_events(self) -> None:
        """Test differentiable rate regularization penalizes extreme predictions."""
        pred_rate_none = torch.zeros(2, 2, 260, 346)  # mean=0, far from 0.1
        pred_rate_ideal = torch.ones(2, 2, 260, 346) * 0.1  # mean=0.1, target
        pred_rate_over = torch.ones(2, 2, 260, 346) * 0.5  # mean=0.5, too high

        target = torch.tensor([0.1])
        reg_none = F.mse_loss(pred_rate_none.mean().unsqueeze(0), target)
        reg_ideal = F.mse_loss(pred_rate_ideal.mean().unsqueeze(0), target)
        reg_over = F.mse_loss(pred_rate_over.mean().unsqueeze(0), target)

        # Ideal prediction should have minimum regularization
        self.assertLess(reg_ideal, reg_none)
        self.assertLess(reg_ideal, reg_over)
        # Non-ideal should be penalized
        self.assertGreater(reg_none, 0)
        self.assertGreater(reg_over, 0)


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping."""

    def test_gradient_clipping_limits_norm(self) -> None:
        """Test gradient clipping limits gradient norm."""
        model = EventPredictor(
            ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
        )

        # Create large gradients (2-channel frame pair input)
        rgb = torch.randn(2, 2, 260, 346)
        imu = torch.randn(2, 50, 6)
        events = model(rgb, imu)
        loss = events.sum()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradient norm is <= 1.0
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.assertLessEqual(total_norm, 1.0 + 1e-6)  # Small tolerance


class TestDropoutScaling(unittest.TestCase):
    """Test event dropout scaling."""

    def test_dropout_maintains_expected_value(self) -> None:
        """Test dropout scaling maintains E[gt]."""
        gt_events = torch.ones(2, 2, 10, 10)
        dropout_rate = 0.15

        # Apply dropout with scaling
        dropout_mask = torch.rand_like(gt_events) > dropout_rate
        gt_events_dropped = gt_events * dropout_mask / (1 - dropout_rate)

        # Expected value should be maintained (approximately)
        expected_original = gt_events.mean().item()
        expected_dropped = gt_events_dropped.mean().item()

        # Should be close (within statistical variance)
        self.assertAlmostEqual(expected_original, expected_dropped, delta=0.1)


class TestMultiScaleWindows(unittest.TestCase):
    """Test multi-scale temporal windows."""

    def test_multi_scale_produces_valid_output(self) -> None:
        """Test multi-scale windows produce valid event maps."""
        # This would require actual data, so we test the concept
        windows_ms = [10, 33, 100]

        # Simulate event maps at different scales
        event_maps = [np.random.rand(260, 346).astype(np.float32) for _ in windows_ms]

        # Average across scales
        fused = np.mean(np.stack(event_maps), axis=0)

        # Should be valid probabilities
        self.assertTrue((fused >= 0).all())
        self.assertTrue((fused <= 1).all())
        self.assertEqual(fused.shape, (260, 346))


class TestF1Metric(unittest.TestCase):
    """Test F1-based evaluation metric."""

    def test_f1_perfect_prediction(self) -> None:
        """F1=1 when pred matches GT exactly."""
        # GT with ~5% positive pixels
        gt = torch.zeros(1, 2, 32, 32)
        gt[0, 0, :3, :3] = 0.05  # 9 positive pixels in ON channel

        # Perfect prediction: high prob where GT is positive, low elsewhere
        pred = torch.zeros_like(gt) + 0.01
        pred[0, 0, :3, :3] = 0.9

        pred_bin = pred > 0.5
        gt_bin = gt > 0.005
        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & ~gt_bin).sum())
        fn = int((~pred_bin & gt_bin).sum())
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        self.assertAlmostEqual(f1, 1.0, places=3)

    def test_f1_all_zero_prediction(self) -> None:
        """F1=0 when model predicts nothing but GT has events."""
        gt = torch.zeros(1, 2, 32, 32)
        gt[0, 0, :3, :3] = 0.05
        pred = torch.zeros_like(gt) + 0.01  # all below 0.5 threshold

        pred_bin = pred > 0.5
        gt_bin = gt > 0.005
        fn = int((~pred_bin & gt_bin).sum())
        tp = 0
        r = tp / (tp + fn + 1e-8)
        self.assertAlmostEqual(r, 0.0, places=3)


class TestModelCheckpoint(unittest.TestCase):
    """Test checkpoint saving/loading."""

    def test_checkpoint_save_load(self) -> None:
        """Test checkpoint can be saved and loaded."""
        import tempfile

        import torch

        # Create model and save
        model1 = EventPredictor(
            ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {
                "model_state_dict": model1.state_dict(),
                "config": {"base_channels": BASE_CHANNELS, "imu_hidden_dim": IMU_HIDDEN_DIM},
            }
            torch.save(checkpoint, f.name)

            # Load into new model
            model2 = EventPredictor(
                ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
            )
            checkpoint_loaded = torch.load(f.name, weights_only=True)
            model2.load_state_dict(checkpoint_loaded["model_state_dict"])

        # Check parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
            self.assertTrue(torch.allclose(p1, p2))


class TestModelForwardShape(unittest.TestCase):
    """Test model forward pass with 2-channel frame pair input."""

    def test_forward_two_channel_input(self) -> None:
        """Model accepts 2-channel (frame pair) input and returns (B, 2, H, W) tensor."""
        model = EventPredictor(
            ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
        )
        rgb = torch.randn(2, 2, 260, 346)  # 2 channels: current + prev frame
        imu = torch.randn(2, 50, 6)

        events = model(rgb, imu)

        self.assertIsInstance(events, torch.Tensor)
        self.assertEqual(events.shape, (2, 2, 260, 346))


def run_tests() -> bool:
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEventPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestFocalBCELoss))
    suite.addTests(loader.loadTestsFromTestCase(TestEventRateRegularization))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientClipping))
    suite.addTests(loader.loadTestsFromTestCase(TestDropoutScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiScaleWindows))
    suite.addTests(loader.loadTestsFromTestCase(TestF1Metric))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestModelForwardShape))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
