"""Noise functions for emulator."""

import logging
import math

import torch

logger = logging.getLogger(__name__)


def compute_photoreceptor_noise_voltage(
    shot_noise_rate_hz, f3db, sample_rate_hz, pos_thr, neg_thr, sigma_thr
):
    def compute_vn_from_log_rate_per_hz(thr, x):
        y = -0.0026 * x**3 - 0.036 * x**2 - 0.1949 * x + 0.321
        thr_per_vn = 10**y
        vn = thr / thr_per_vn
        return vn

    if compute_photoreceptor_noise_voltage.last_sample_rate is not None:
        diff = np.abs(
            sample_rate_hz / compute_photoreceptor_noise_voltage.last_sample_rate - 1
        )
        if diff < 0.1:
            return compute_photoreceptor_noise_voltage.last_vn

    rate_per_bw = (shot_noise_rate_hz / f3db) / 2
    if rate_per_bw > 0.5:
        logger.warning(
            f"shot noise rate per hz of bandwidth is larger than 0.1 (rate_hz={shot_noise_rate_hz} Hz, 3dB bandwidth={f3db} Hz)"
        )
    x = math.log10(rate_per_bw)
    if x < -5.0:
        logger.warning(
            f"desired noise rate of {shot_noise_rate_hz}Hz is too low to accurately compute a threshold value"
        )
    elif x > 0.0:
        logger.warning(
            f"desired noise rate of {shot_noise_rate_hz}Hz is too large to accurately compute a threshold value"
        )

    N = 300
    import numpy as np

    pos_samps = pos_thr + sigma_thr * np.random.default_rng().standard_normal(N)
    neg_samps = neg_thr + sigma_thr * np.random.default_rng().standard_normal(N)
    thrs = np.vstack((pos_samps, neg_samps))
    mins = np.min(thrs, axis=0)
    vns = np.zeros_like(mins)
    for i in range(N):
        thr = mins[i]
        vn = compute_vn_from_log_rate_per_hz(thr, x)
        vns[i] = vn
    vn = np.mean(vns)

    compute_photoreceptor_noise_voltage.last_sample_rate = sample_rate_hz
    tau = 1 / (f3db * 2 * math.pi)
    dt = 1 / sample_rate_hz
    t = np.arange(0, 1000 * tau, dt)
    rin = vn * np.random.default_rng().standard_normal(t.shape)
    rms_in = np.std(rin)
    rout = np.zeros_like(rin)
    eps = dt / tau
    eps_limit = 0.1
    if eps > eps_limit:
        logger.warning(
            f"\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce timestep (currently {dt:.3f}s) (using higher frame rate) or decrease cutoff_hz (currently {f3db:.3f} Hz)"
        )
    rout[0] = 0
    for i in range(1, len(rin)):
        rout[i] = rout[i - 1] * (1 - eps) + rin[i] * eps
    rms_out = np.std(rout)
    scale = rms_in / rms_out
    vnscaled = scale * vn

    compute_photoreceptor_noise_voltage.last_vn = vnscaled
    if not compute_photoreceptor_noise_voltage.vrms_computation_printed:
        logger.info(
            f"For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units, scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, sample interval dt={dt * 1000:.3f} ms, cutoff_hz={f3db} Hz, tau={tau * 1000:.3f} ms, Rn/f3dB={rate_per_bw:.3g} Hz, and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units."
        )
        compute_photoreceptor_noise_voltage.vrms_computation_printed = True
    return vnscaled


compute_photoreceptor_noise_voltage.vrms_computation_printed = False
compute_photoreceptor_noise_voltage.last_sample_rate = None
compute_photoreceptor_noise_voltage.last_vn = None


def generate_shot_noise(
    shot_noise_rate_hz,
    delta_time,
    shot_noise_inten_factor,
    inten01,
    pos_thres_pre_prob,
    neg_thres_pre_prob,
):
    if shot_noise_rate_hz * delta_time > 1:
        logger.warning(
            f"shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz * delta_time:.2f} is too large, decrease timestamp resolution or sample rate"
        )

    shot_noise_factor = ((shot_noise_rate_hz / 2) * delta_time) * (
        (shot_noise_inten_factor - 1) * inten01 + 1
    )

    one_minus_shot_ON_prob_this_sample = 1 - shot_noise_factor * pos_thres_pre_prob
    shot_OFF_prob_this_sample = shot_noise_factor * neg_thres_pre_prob

    rand01 = torch.rand(size=inten01.shape, dtype=torch.float32, device=inten01.device)

    shot_on_cord = torch.gt(rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(rand01, shot_OFF_prob_this_sample)

    return shot_on_cord, shot_off_cord
