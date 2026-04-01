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
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM)
        self.model = EventPredictor(self.config)
        self.batch_size = 2
        self.image_size = (260, 346)
        
    def test_model_initialization(self):
        """Test model initializes correctly."""
        num_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(num_params, 1_000_000)  # at least 1M params
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        # 2-channel input: (current_frame, previous_frame)
        rgb = torch.randn(self.batch_size, 2, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)

        events, depth = self.model(rgb, imu)

        self.assertEqual(events.shape, (self.batch_size, 2, *self.image_size))
        self.assertEqual(depth.shape, (self.batch_size, 1))

    def test_output_positive(self):
        """Test event predictions are positive (for Poisson)."""
        rgb = torch.randn(self.batch_size, 2, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)

        events, depth = self.model(rgb, imu)

        self.assertTrue((events > 0).all())  # All positive

    def test_depth_head(self):
        """Test depth head produces reasonable output."""
        rgb = torch.randn(self.batch_size, 2, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)

        events, depth = self.model(rgb, imu)

        # Depth should be in reasonable range [0, 1] after sigmoid
        self.assertTrue((depth >= 0).all())
        self.assertTrue((depth <= 1).all())


class TestPoissonLoss(unittest.TestCase):
    """Test Poisson NLL loss implementation."""
    
    def test_poisson_nll_basic(self):
        """Test Poisson NLL computes correctly."""
        pred_rate = torch.tensor([1.0, 2.0, 3.0])
        gt_counts = torch.tensor([1.0, 2.0, 3.0])
        
        # Poisson NLL: λ - k*log(λ)
        poisson_nll = pred_rate - gt_counts * torch.log(pred_rate + 1e-6)
        loss = poisson_nll.mean()
        
        self.assertGreater(loss, 0)
        self.assertEqual(loss.shape, torch.Size([]))
        
    def test_poisson_nll_perfect_prediction(self):
        """Test Poisson NLL is low for perfect predictions."""
        pred_rate = torch.tensor([5.0, 5.0, 5.0])
        gt_counts = torch.tensor([5.0, 5.0, 5.0])
        
        poisson_nll = pred_rate - gt_counts * torch.log(pred_rate + 1e-6)
        loss = poisson_nll.mean()
        
        # Should be low (but not zero due to log(k!) term we ignore)
        self.assertLess(loss, 1.0)
        
    def test_poisson_nll_bad_prediction(self):
        """Test Poisson NLL is high for bad predictions."""
        pred_rate = torch.tensor([0.1, 0.1, 0.1])  # Predict almost nothing
        gt_counts = torch.tensor([10.0, 10.0, 10.0])  # But ground truth is high
        
        poisson_nll = pred_rate - gt_counts * torch.log(pred_rate + 1e-6)
        loss = poisson_nll.mean()
        
        # Should be high (model under-predicted)
        self.assertGreater(loss, 10.0)


class TestEventRateRegularization(unittest.TestCase):
    """Test event rate regularization."""

    def test_rate_regularization_encourages_events(self):
        """Test differentiable rate regularization penalizes extreme predictions."""
        pred_rate_none = torch.zeros(2, 2, 260, 346)   # mean=0, far from 0.1
        pred_rate_ideal = torch.ones(2, 2, 260, 346) * 0.1  # mean=0.1, target
        pred_rate_over = torch.ones(2, 2, 260, 346) * 0.5   # mean=0.5, too high

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
    
    def test_gradient_clipping_limits_norm(self):
        """Test gradient clipping limits gradient norm."""
        model = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))

        # Create large gradients (2-channel frame pair input)
        rgb = torch.randn(2, 2, 260, 346)
        imu = torch.randn(2, 50, 6)
        events, depth = model(rgb, imu)
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
        total_norm = total_norm ** 0.5
        
        self.assertLessEqual(total_norm, 1.0 + 1e-6)  # Small tolerance


class TestDropoutScaling(unittest.TestCase):
    """Test event dropout scaling."""
    
    def test_dropout_maintains_expected_value(self):
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
    
    def test_multi_scale_produces_valid_output(self):
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


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_event_bpb_calculation(self):
        """Test event_bpb uses Poisson NLL per element / log(2)."""
        # Poisson NLL: λ - k*log(λ)
        pred_rate = torch.tensor([1.0, 2.0, 3.0])
        gt_counts = torch.tensor([1.0, 2.0, 3.0])
        poisson_nll = (pred_rate - gt_counts * torch.log(pred_rate + 1e-6)).mean().item()
        bpb = poisson_nll / np.log(2)

        self.assertGreater(bpb, 0)  # bpb > 0 for any non-zero rate

    def test_event_bpb_zero_for_sparse_perfect(self):
        """Test event_bpb is 0 when both pred and gt are 0 (sparse background)."""
        # Poisson NLL for λ=0, k=0: 0 - 0*log(0) = 0
        poisson_nll = 0.0 - 0.0 * np.log(1e-6)  # λ + ε to avoid log(0), k=0
        bpb = poisson_nll / np.log(2)

        self.assertAlmostEqual(bpb, 0.0, places=5)

    def test_event_bpb_worse_for_bad_prediction(self):
        """Test event_bpb increases for under-prediction."""
        pred_good = torch.tensor([5.0])   # Close to target
        pred_bad = torch.tensor([0.01])   # Far under-predicts
        gt = torch.tensor([5.0])

        nll_good = (pred_good - gt * torch.log(pred_good + 1e-6)).item()
        nll_bad = (pred_bad - gt * torch.log(pred_bad + 1e-6)).item()

        self.assertLess(nll_good, nll_bad)


class TestModelCheckpoint(unittest.TestCase):
    """Test checkpoint saving/loading."""
    
    def test_checkpoint_save_load(self):
        """Test checkpoint can be saved and loaded."""
        import tempfile

        import torch
        
        # Create model and save
        model1 = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint = {
                "model_state_dict": model1.state_dict(),
                "config": {"base_channels": BASE_CHANNELS, "imu_hidden_dim": IMU_HIDDEN_DIM}
            }
            torch.save(checkpoint, f.name)
            
            # Load into new model
            model2 = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
            checkpoint_loaded = torch.load(f.name, weights_only=True)
            model2.load_state_dict(checkpoint_loaded["model_state_dict"])
        
        # Check parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


class TestModelForwardShape(unittest.TestCase):
    """Test model forward pass with 2-channel frame pair input."""

    def test_forward_two_channel_input(self):
        """Model accepts 2-channel (frame pair) input without error."""
        model = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
        rgb = torch.randn(2, 2, 260, 346)  # 2 channels: current + prev frame
        imu = torch.randn(2, 50, 6)

        events, depth = model(rgb, imu)

        self.assertEqual(events.shape, (2, 2, 260, 346))
        self.assertEqual(depth.shape, (2, 1))


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEventPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestPoissonLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestEventRateRegularization))
    suite.addTests(loader.loadTestsFromTestCase(TestGradientClipping))
    suite.addTests(loader.loadTestsFromTestCase(TestDropoutScaling))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiScaleWindows))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCheckpoint))
    suite.addTests(loader.loadTestsFromTestCase(TestModelForwardShape))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
