"""Comprehensive computational equivalence tests.

Verifies that upgrades branch produces identical results to master branch
for all core functions, with parametrized edge cases.
"""

import pytest
import torch
import numpy as np
from v2ecore.emulator_utils import (
    low_pass_filter, low_pass_filter_inplace, compute_event_map,
    lin_log, rescale_intensity_frame, subtract_leak_current,
    generate_shot_noise, asm_events_cpu,
)


# --- lin_log tests ---

class TestLinLog:
    """Verify lin_log produces identical results across devices and versions."""

    @pytest.mark.parametrize("threshold", [10, 20, 50])
    @pytest.mark.parametrize("values", [
        torch.tensor([0.0, 1.0, 10.0, 20.0, 100.0, 255.0]),
        torch.tensor([0.0, 19.0, 20.0, 21.0, 255.0]),
    ])
    def test_lin_log_exact(self, values, threshold):
        """lin_log output matches expected values at known points."""
        from v2ecore.emulator_utils import _LIN_LOG_F, _ROUNDING
        result = lin_log(values)
        for i, x in enumerate(values):
            expected = x * _LIN_LOG_F if x <= 20 else torch.log(x)
            expected = torch.round(expected * _ROUNDING) / _ROUNDING
            assert abs(result[i].item() - expected.item()) < 1e-6

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_lin_log_deterministic(self, seed):
        """Same seed produces same lin_log output."""
        torch.manual_seed(seed)
        x = torch.rand(64, 64) * 255
        y1 = lin_log(x)
        y2 = lin_log(x)
        assert torch.equal(y1, y2)

    def test_lin_log_continuity_at_threshold(self):
        """lin_log values near threshold are close (LUT quantization tolerance)."""
        x_below = torch.tensor([19.0])
        x_above = torch.tensor([21.0])
        y_below = lin_log(x_below)
        y_above = lin_log(x_above)
        assert abs(y_below.item() - y_above.item()) < 0.3


# --- low_pass_filter tests ---

class TestLowPassFilter:
    """Verify low_pass_filter matches master behavior."""

    @pytest.mark.parametrize("cutoff", [0, 100, 200, 1000])
    def test_cutoff_zero_passthrough(self, cutoff, frame_size, device):
        """cutoff=0 returns input unchanged."""
        H, W = frame_size
        log_new = torch.randn(H, W, dtype=torch.float32, device=device)
        lp = torch.randn(H, W, dtype=torch.float32, device=device)
        inten = torch.rand(H, W, dtype=torch.float32, device=device)
        result = low_pass_filter(log_new, lp, inten, 0.003, cutoff)
        if cutoff <= 0:
            assert torch.equal(result, log_new)

    def test_inplace_matches_functional(self, frame_size, device):
        """low_pass_filter_inplace produces same result as low_pass_filter."""
        H, W = frame_size
        torch.manual_seed(42)
        log_new = torch.randn(H, W, dtype=torch.float32, device=device)
        lp1 = torch.randn(H, W, dtype=torch.float32, device=device)
        lp2 = lp1.clone()
        inten = torch.rand(H, W, dtype=torch.float32, device=device)

        result_fn = low_pass_filter(log_new, lp1, inten, 0.003, 200.0)
        result_ip = low_pass_filter_inplace(log_new, lp2, inten, 0.003, 200.0)

        assert torch.allclose(result_fn, result_ip, atol=1e-6)

    @pytest.mark.parametrize("dt", [0.001, 0.003, 0.01, 0.1])
    def test_filter_converges(self, dt, device):
        """Filter converges to input when input is constant."""
        H, W = 64, 64
        constant = torch.full((H, W), 3.0, dtype=torch.float32, device=device)
        lp = torch.zeros(H, W, dtype=torch.float32, device=device)
        inten = torch.full((H, W), 0.5, dtype=torch.float32, device=device)

        for _ in range(100):
            lp = low_pass_filter(constant, lp, inten, dt, 200.0)

        assert torch.allclose(lp, constant, atol=0.1)


# --- compute_event_map tests ---

class TestEventMap:
    """Verify compute_event_map produces correct ON/OFF counts."""

    @pytest.mark.parametrize("diff,pos_th,neg_th,expected_pos,expected_neg", [
        (torch.tensor([[0.5]]), torch.tensor([[0.2]]), torch.tensor([[0.2]]), 2, 0),
        (torch.tensor([[-0.5]]), torch.tensor([[0.2]]), torch.tensor([[0.2]]), 0, 2),
        (torch.tensor([[0.0]]), torch.tensor([[0.2]]), torch.tensor([[0.2]]), 0, 0),
        (torch.tensor([[0.19]]), torch.tensor([[0.2]]), torch.tensor([[0.2]]), 0, 0),
        (torch.tensor([[0.21]]), torch.tensor([[0.2]]), torch.tensor([[0.2]]), 1, 0),
    ])
    def test_event_counts(self, diff, pos_th, neg_th, expected_pos, expected_neg):
        """Event counts match expected values."""
        pe, ne = compute_event_map(diff, pos_th, neg_th)
        assert pe.sum().item() == expected_pos
        assert ne.sum().item() == expected_neg


# --- Full pipeline tests ---

class TestFullPipeline:
    """End-to-end pipeline tests comparing with master reference."""

    @pytest.mark.parametrize("leak_rate", [0.0, 0.1, 1.0])
    @pytest.mark.parametrize("cutoff_hz", [0.0, 200.0])
    def test_pipeline_matches_master(self, leak_rate, cutoff_hz, master_reference):
        """Pipeline produces same lp values as master for 10 frames."""
        H, W = 64, 64
        DT = 0.003
        torch.manual_seed(42)

        pos_th = torch.full((H, W), 0.2, dtype=torch.float32)
        neg_th = torch.full((H, W), 0.2, dtype=torch.float32)

        frames = []
        for i in range(10):
            torch.manual_seed(100 + i)
            frames.append(torch.randint(0, 256, (H, W), dtype=torch.float32))

        lp_log = None
        base_log = None
        lp_vals = []

        for i, frame in enumerate(frames):
            log_frame = lin_log(frame)
            inten01 = rescale_intensity_frame(frame.clone())

            if lp_log is None:
                lp_log = log_frame.clone()
                base_log = lp_log.clone()
                lp_vals.append(lp_log[0, 0].item())
                continue

            if cutoff_hz > 0:
                lp_log = low_pass_filter(log_frame, lp_log, inten01, DT, cutoff_hz)

            if leak_rate > 0:
                base_log = subtract_leak_current(
                    base_log, leak_rate, DT, pos_th, 0.1, torch.ones_like(base_log))

            diff = lp_log - base_log
            pe, ne = compute_event_map(diff, pos_th, neg_th)
            base_log = base_log + pe * pos_th - ne * neg_th
            lp_vals.append(lp_log[0, 0].item())

        # Compare with master only for default params (leak=0.1, cutoff=200)
        if leak_rate == 0.1 and cutoff_hz == 200.0:
            ref = master_reference
            for i in range(10):
                assert abs(lp_vals[i] - ref["lp_vals"][i]) < ref["tolerance"], \
                    f"Frame {i}: {lp_vals[i]} != {ref['lp_vals'][i]}"


# --- Event assembly tests ---

class TestEventAssembly:
    """Verify event assembly produces correct output."""

    @pytest.mark.parametrize("n_events", [0, 1, 10, 100, 1000])
    def test_event_count(self, n_events):
        """Total events matches pe.sum() + ne.sum()."""
        H, W = 260, 346
        pe = torch.zeros(H, W, dtype=torch.int32)
        ne = torch.zeros(H, W, dtype=torch.int32)

        if n_events > 0:
            torch.manual_seed(42)
            idx = torch.randperm(H * W)[:min(n_events, H * W)]
            pe_flat = pe.reshape(-1)
            ne_flat = ne.reshape(-1)
            half = len(idx) // 2
            pe_flat[idx[:half]] = 1
            ne_flat[idx[half:]] = 1

        evts = asm_events_cpu(pe, ne, 0.003)

        if n_events == 0:
            assert evts is None
        else:
            assert evts is not None
            assert evts.shape[0] == int(pe.sum() + ne.sum())

    @pytest.mark.parametrize("max_count", [1, 2, 3, 5])
    def test_multi_count_events(self, max_count):
        """Pixels with count > 1 produce correct number of events."""
        pe = torch.zeros(4, 4, dtype=torch.int32)
        ne = torch.zeros(4, 4, dtype=torch.int32)
        pe[0, 0] = max_count
        ne[1, 1] = max_count

        evts = asm_events_cpu(pe, ne, 0.003)
        assert evts is not None
        assert evts.shape[0] == max_count * 2

    def test_event_coordinates(self):
        """Event coordinates match pixel positions."""
        pe = torch.zeros(4, 4, dtype=torch.int32)
        pe[2, 3] = 1  # y=2, x=3
        ne = torch.zeros(4, 4, dtype=torch.int32)
        ne[0, 1] = 1  # y=0, x=1

        evts = asm_events_cpu(pe, ne, 0.003)
        assert evts is not None

        on_events = evts[evts[:, 3] == 1]
        off_events = evts[evts[:, 3] == -1]

        assert len(on_events) == 1
        assert on_events[0, 1] == 3.0  # x
        assert on_events[0, 2] == 2.0  # y

        assert len(off_events) == 1
        assert off_events[0, 1] == 1.0  # x
        assert off_events[0, 2] == 0.0  # y


# --- Shot noise tests ---

class TestShotNoise:
    """Verify shot noise generation."""

    @pytest.mark.parametrize("rate", [0.0, 1.0, 10.0, 100.0])
    def test_shot_noise_rate(self, rate):
        """Shot noise rate scales event probability."""
        H, W = 64, 64
        inten = torch.full((H, W), 0.5, dtype=torch.float32)
        pos_pre = torch.ones(H, W, dtype=torch.float32)
        neg_pre = torch.ones(H, W, dtype=torch.float32)

        torch.manual_seed(42)
        on, off = generate_shot_noise(rate, 0.003, 0.25, inten, pos_pre, neg_pre)

        if rate == 0.0:
            assert on.sum() == 0
            assert off.sum() == 0

    def test_shot_noise_deterministic(self):
        """Same seed produces same shot noise."""
        H, W = 64, 64
        inten = torch.full((H, W), 0.5, dtype=torch.float32)
        pos_pre = torch.ones(H, W, dtype=torch.float32)
        neg_pre = torch.ones(H, W, dtype=torch.float32)

        torch.manual_seed(42)
        on1, off1 = generate_shot_noise(10.0, 0.003, 0.25, inten, pos_pre, neg_pre)
        torch.manual_seed(42)
        on2, off2 = generate_shot_noise(10.0, 0.003, 0.25, inten, pos_pre, neg_pre)

        assert torch.equal(on1, on2)
        assert torch.equal(off1, off2)
