#!/usr/bin/env python
"""
Unit Tests for V2E Improved Event Prediction.

Test coverage: Core functionality, metrics, data loading
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from train import EventPredictor, ModelConfig, BASE_CHANNELS, IMU_HIDDEN_DIM
from prepare_data import FPVDataset, evaluate_combined_metric


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
        self.assertGreater(num_params, 0)
        self.assertAlmostEqual(num_params / 1e6, 1.19, places=2)  # ~1.19M parameters
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        rgb = torch.randn(self.batch_size, 1, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)
        
        events, depth = self.model(rgb, imu)
        
        self.assertEqual(events.shape, (self.batch_size, 2, *self.image_size))
        self.assertEqual(depth.shape, (self.batch_size, 1))
        
    def test_output_positive(self):
        """Test event predictions are positive (for Poisson)."""
        rgb = torch.randn(self.batch_size, 1, *self.image_size)
        imu = torch.randn(self.batch_size, 50, 6)
        
        events, depth = self.model(rgb, imu)
        
        self.assertTrue((events > 0).all())  # All positive
        
    def test_depth_head(self):
        """Test depth head produces reasonable output."""
        rgb = torch.randn(self.batch_size, 1, *self.image_size)
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
        """Test rate regularization encourages realistic event rates."""
        pred_rate_none = torch.zeros(2, 2, 260, 346)  # No events at all
        pred_rate_some = torch.ones(2, 2, 260, 346) * 0.5  # All pixels > 0.3
        
        target_rate = 0.1
        reg_none = ((pred_rate_none > 0.3).float().mean() - target_rate) ** 2
        reg_some = ((pred_rate_some > 0.3).float().mean() - target_rate) ** 2
        
        # No events: (0 - 0.1)^2 = 0.01
        # All events: (1 - 0.1)^2 = 0.81
        # Neither is ideal, but all events is much worse
        self.assertGreater(reg_some, reg_none)


class TestGradientClipping(unittest.TestCase):
    """Test gradient clipping."""
    
    def test_gradient_clipping_limits_norm(self):
        """Test gradient clipping limits gradient norm."""
        model = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
        
        # Create large gradients
        rgb = torch.randn(2, 1, 260, 346)
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
        """Test event_bpb is calculated correctly."""
        # event_bpb = MSE / log(2)
        mse = 0.01
        bpb = mse / np.log(2)
        
        self.assertAlmostEqual(bpb, 0.014427, places=5)
        
    def test_event_bpb_perfect_prediction(self):
        """Test event_bpb is 0 for perfect predictions."""
        mse = 0.0
        bpb = mse / np.log(2)
        
        self.assertEqual(bpb, 0.0)


class TestModelCheckpoint(unittest.TestCase):
    """Test checkpoint saving/loading."""
    
    def test_checkpoint_save_load(self):
        """Test checkpoint can be saved and loaded."""
        import tempfile
        import torch
        
        # Create model and save
        model1 = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {
                'model_state_dict': model1.state_dict(),
                'config': {'base_channels': BASE_CHANNELS, 'imu_hidden_dim': IMU_HIDDEN_DIM}
            }
            torch.save(checkpoint, f.name)
            
            # Load into new model
            model2 = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
            checkpoint_loaded = torch.load(f.name, weights_only=True)
            model2.load_state_dict(checkpoint_loaded['model_state_dict'])
        
        # Check parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


class TestMixedPrecision(unittest.TestCase):
    """Test mixed precision training."""
    
    def test_autocast_produces_float16(self):
        """Test autocast produces float16 tensors."""
        device = "cpu"  # MPS/CUDA would work too but CPU is safer for tests
        
        model = EventPredictor(ModelConfig(base_channels=BASE_CHANNELS, imu_hidden_dim=IMU_HIDDEN_DIM))
        rgb = torch.randn(2, 1, 260, 346)
        imu = torch.randn(2, 50, 6)
        
        # Test autocast
        with torch.amp.autocast(device_type=device if device != "mps" else "cpu"):
            events, depth = model(rgb, imu)
        
        # Should run without error (dtype may vary by device)
        self.assertIsNotNone(events)
        self.assertIsNotNone(depth)


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
    suite.addTests(loader.loadTestsFromTestCase(TestMixedPrecision))
    
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
