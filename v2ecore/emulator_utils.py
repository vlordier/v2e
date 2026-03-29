"""Emulator utilities for DVS event computation.

Provides torch-based functions for:
- lin_log mapping (linear to logarithmic intensity)
- IIR lowpass filtering (intensity-dependent photoreceptor model)
- Event map computation (threshold-based ON/OFF event detection)
- Leak current subtraction (pixel leakage modeling)
- Shot noise generation (Poisson temporal noise)

All functions operate on torch Tensors for GPU acceleration.
"""

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def lin_log(x: torch.Tensor, threshold: float = 20) -> torch.Tensor:
    """Piecewise linear-logarithmic intensity mapping.

    Maps linear intensity values to a hybrid linear+log scale:
    - Below threshold: linear mapping (scaled log(threshold)/threshold)
    - Above threshold: natural logarithm

    The transition is continuous at the threshold point. A floating-point
    rounding step prevents precision artifacts that could cause spurious
    OFF events after ON events during motion.

    Args:
        x: Input linear intensity values (any shape). Assumes 8-bit range 0-255.
        threshold: Transition point from linear to log mapping (default 20).

    Returns:
        Logarithmically-mapped values (same shape as x, float32).
    """
    if x.dtype != torch.float32:
        x = x.float()

    f = (1.0 / threshold) * math.log(threshold)

    y = torch.where(x <= threshold, x * f, torch.log(x))

    # Round to avoid precision artifacts that cause spurious events
    rounding = 1e8
    y = torch.round(y * rounding) / rounding

    return y


def rescale_intensity_frame(new_frame):
    """Rescale intensity frames.

    make sure we get no zero time constants
    limit max time constant to ~1/10 of white intensity level
    """
    return (new_frame + 20) / 275.0



def subtract_leak_current(
    base_log_frame: torch.Tensor,
    leak_rate_hz: float,
    delta_time: float,
    pos_thres: torch.Tensor,
    leak_jitter_fraction: float,
    noise_rate_array: torch.Tensor,
) -> torch.Tensor:
    """Subtract leak current from base log frame.

    Models pixel-to-pixel variation in leakage rate via a log-normal
    noise_rate_array multiplied by Gaussian jitter.

    Args:
        base_log_frame: Memorized log intensity values [H, W].
        leak_rate_hz: Nominal leak event rate per pixel (Hz).
        delta_time: Time step since last frame (seconds).
        pos_thres: Per-pixel ON thresholds [H, W].
        leak_jitter_fraction: Fractional jitter std dev for leak rate.
        noise_rate_array: Per-pixel noise rate variation [H, W].

    Returns:
        Updated base_log_frame with leak current subtracted.
    """
    rand = torch.randn(
        noise_rate_array.shape, dtype=torch.float32, device=noise_rate_array.device
    )
    curr_leak_rate = leak_rate_hz * noise_rate_array * (1 - leak_jitter_fraction * rand)
    delta_leak = delta_time * curr_leak_rate * pos_thres
    return base_log_frame - delta_leak


def low_pass_filter(log_new_frame, lp_log_frame, inten01, delta_time, cutoff_hz):
    """Compute intensity-dependent 1st-order IIR low-pass filter.

    The time constant is inversely proportional to local pixel intensity,
    modeling the DVS photoreceptor behavior where brighter regions have
    shorter time constants.

    Args:
        log_new_frame: New frame in lin-log representation [H, W].
        lp_log_frame: Previous low-pass filtered frame state [H, W].
        inten01: Normalized intensity array scaling filter time constant [H, W],
                 or None for uniform filtering.
        delta_time: Time step since last frame (seconds).
        cutoff_hz: 3dB cutoff frequency (Hz). If <=0, returns input unchanged.

    Returns:
        new_lp_log_frame: Updated low-pass filtered frame [H, W].
    """
    if cutoff_hz <= 0:
        return log_new_frame

    tau = 1 / (math.pi * 2 * cutoff_hz)

    if inten01 is not None:
        eps = inten01 * (delta_time / tau)
        max_eps = torch.max(eps)
        if max_eps > 0.3:
            max_warnings = 10
            if low_pass_filter._warning_count < max_warnings:
                logger.warning(
                    f"IIR lowpass filter update has large maximum update eps={max_eps:.2f}"
                    f" from delta_time/tau={delta_time:.3g}/{tau:.3g}"
                )
                low_pass_filter._warning_count += 1
                if low_pass_filter._warning_count == max_warnings:
                    logger.warning(
                        "Suppressing further IIR lowpass warnings;"
                        " check timestamp resolution and DVS photoreceptor cutoff frequency"
                    )
        eps = torch.clamp(eps, max=1)  # keep filter stable
    else:
        eps = delta_time / tau

    new_lp_log_frame = (1 - eps) * lp_log_frame + eps * log_new_frame
    return new_lp_log_frame


low_pass_filter._warning_count = 0


def low_pass_filter_inplace(
    log_new_frame: torch.Tensor,
    lp_log_frame: torch.Tensor,
    inten01: Optional[torch.Tensor],
    delta_time: float,
    cutoff_hz: float,
) -> torch.Tensor:
    """In-place IIR low-pass filter. Modifies lp_log_frame directly.

    Avoids tensor allocation overhead, ~7x faster on MPS than the
    functional version. Safe when lp_log_frame is state that will be
    overwritten anyway (e.g. stored on self.lp_log_frame).

    Args:
        log_new_frame: New frame [H, W].
        lp_log_frame: Filter state to update in-place [H, W].
        inten01: Normalized intensity [H, W], or None for uniform.
        delta_time: Time step (seconds).
        cutoff_hz: Cutoff frequency (Hz). If <=0, copies input to state.

    Returns:
        lp_log_frame (same tensor, modified in-place).
    """
    if cutoff_hz <= 0:
        lp_log_frame.copy_(log_new_frame)
        return lp_log_frame

    tau = 1 / (math.pi * 2 * cutoff_hz)

    if inten01 is not None:
        eps = inten01 * (delta_time / tau)
        eps.clamp_(max=1)
    else:
        eps = delta_time / tau

    lp_log_frame.mul_(1 - eps).add_(eps * log_new_frame)
    return lp_log_frame


def fused_photoreceptor_step(
    log_new_frame: torch.Tensor,
    lp_log_frame: torch.Tensor,
    base_log_frame: torch.Tensor,
    inten01: Optional[torch.Tensor],
    pos_thres: torch.Tensor,
    neg_thres: torch.Tensor,
    delta_time: float,
    cutoff_hz: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused lowpass + diff + event_map in a single GPU kernel chain.

    Eliminates intermediate tensor allocations and reduces GPU kernel
    launch overhead. ~3x faster than calling low_pass_filter + compute_event_map
    separately on MPS.

    Args:
        log_new_frame: New frame in lin-log [H, W].
        lp_log_frame: Lowpass state to update in-place [H, W].
        base_log_frame: Memorized brightness for diff computation [H, W].
        inten01: Normalized intensity [H, W], or None.
        pos_thres: ON thresholds [H, W].
        neg_thres: OFF thresholds [H, W].
        delta_time: Time step (seconds).
        cutoff_hz: Cutoff frequency (Hz).

    Returns:
        (lp_log_frame, pos_evts_frame, neg_evts_frame): Updated state and
        integer event count tensors.
    """
    # Lowpass (in-place on lp_log_frame)
    if cutoff_hz > 0:
        tau = 1 / (math.pi * 2 * cutoff_hz)
        if inten01 is not None:
            eps = inten01 * (delta_time / tau)
            eps.clamp_(max=1)
        else:
            eps = delta_time / tau
        lp_log_frame.mul_(1 - eps).add_(eps * log_new_frame)

    # Diff from memorized value
    diff = lp_log_frame - base_log_frame

    # Event map (reuses diff, no extra alloc)
    pos_evts = torch.div(torch.relu(diff), pos_thres, rounding_mode="floor").to(torch.int32)
    neg_evts = torch.div(torch.relu(-diff), neg_thres, rounding_mode="floor").to(torch.int32)

    return lp_log_frame, pos_evts, neg_evts


def compute_event_map(diff_frame, pos_thres, neg_thres):
    """
        Compute event maps, i.e. 2d arrays of [width,height] containing quantized number of ON and OFF events.

    Args:
        diff_frame:  the input difference frame between stored log intensity and current frame log intensity [width, height]
        pos_thres:  ON threshold values [width, height]
        neg_thres:  OFF threshold values [width, height]

    Returns:
        pos_evts_frame, neg_evts_frame;  2d Tensors of integer ON and OFF event counts
    """
    # extract positive and negative differences
    pos_frame = F.relu(diff_frame)
    neg_frame = F.relu(-diff_frame)

    # compute quantized number of ON and OFF events for each pixel
    pos_evts_frame = torch.div(pos_frame, pos_thres, rounding_mode="floor").type(
        torch.int32
    )
    neg_evts_frame = torch.div(neg_frame, neg_thres, rounding_mode="floor").type(
        torch.int32
    )

    #  max_events = max(pos_evts_frame.max(), neg_evts_frame.max())

    #  # boolean array (max_events, height, width)
    #  # positive events and negative
    #  pos_evts_cord = torch.arange(
    #      1, max_events+1, dtype=torch.int32,
    #      device=diff_frame.device).unsqueeze(-1).unsqueeze(-1).repeat(
    #          1, diff_frame.shape[0], diff_frame.shape[1])
    #  neg_evts_cord = pos_evts_cord.clone().detach()
    #
    #  # generate event cords
    #  pos_evts_cord_post = (pos_evts_cord >= pos_evts_frame.unsqueeze(0))
    #  neg_evts_cord_post = (neg_evts_cord >= neg_evts_frame.unsqueeze(0))

    return pos_evts_frame, neg_evts_frame
    #  return pos_evts_cord_post, neg_evts_cord_post, max_events


def compute_photoreceptor_noise_voltage(
    shot_noise_rate_hz, f3db, sample_rate_hz, pos_thr, neg_thr, sigma_thr
) -> float:
    """
     Computes the necessary photoreceptor noise voltage to result in observed shot noise rate at low light intensity.
     This computation relies on the known f3dB photoreceptor lowpass filter cutoff frequency and the known (nominal) event threshold.
     emulator.py injects Gaussian distributed noise to the photoreceptor that should in principle generate the desired shot noise events.

     See the file media/noise_event_rate_simulation.xlsx for the simulation data and curve fit.

    Parameters
    -----------
     shot_noise_rate_hz: float
        the desired pixel shot noise rate in hz
     f3db: float
        the 1st-order IIR RC lowpass filter cutoff frequency in Hz
     sample_rate_hz: float
        the sample rate (up-sampled frame rate) before IIR lowpassing the noise
     pos_thr:float
        on threshold in ln units
     neg_thr:float
        off threshold in ln units. The on and off thresholds are averaged to obtain a single threshold.
     sigma_thr: float
        the std deviations of the thresholds

    Returns
    -----------
    float
         Noise signal Gaussian RMS value in log_e units, to be added as Gaussian source directly to log photoreceptor output signal
    """

    def compute_vn_from_log_rate_per_hz(thr, x):
        # y = log10(thr/Vn)
        # x = log10(Rn/f3db)
        # see the plot Fig. 3 from Graca, Rui, and Tobi Delbruck. 2021. “Unraveling the Paradox of Intensity-Dependent DVS Pixel Noise.” arXiv [eess.SY]. arXiv. http://arxiv.org/abs/2109.08640.
        # the fit is computed in media/noise_event_rate_simulation.xlsx spreadsheet
        y = -0.0026 * x**3 - 0.036 * x**2 - 0.1949 * x + 0.321
        thr_per_vn = 10**y  # to get thr/vn
        vn = (
            thr / thr_per_vn
        )  # compute necessary vn to give us this noise rate per pixel at this pixel bandwidth
        return vn

    # check if we already estimated the required noise for this sample rate
    if compute_photoreceptor_noise_voltage.last_sample_rate is not None:
        diff = np.abs(
            sample_rate_hz / compute_photoreceptor_noise_voltage.last_sample_rate - 1
        )
        if diff < 0.1:
            return compute_photoreceptor_noise_voltage.last_vn  # return cached value

    rate_per_bw = (
        (shot_noise_rate_hz / f3db) / 2
    )  # simulation data are on ON event rates, divide by 2 here to end up with correct total rate
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

    # now we need to numerically estimate the required Vnrms given the thresholds and the sigma thresholds,
    # since the noise rate varies dramatically with threshold
    N = 300  # num samples
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
    # now we need to find the scaling factor from white noise to get the correct noise vn after RC lowpass.
    # # to get this NEB factor, we generate white samples here, lowpass filter them the same exact way
    # as we do in the emulator (i.e. with same IIR time constant and sample rate)
    # compute the variance, and scale the amplitude to give us vn
    compute_photoreceptor_noise_voltage.last_sample_rate = sample_rate_hz
    tau = 1 / (f3db * 2 * math.pi)
    dt = 1 / sample_rate_hz
    t = np.arange(0, 1000 * tau, dt)
    rin = vn * np.random.default_rng().standard_normal(
        t.shape
    )  # generated Gaussian random sequence with amplitude vn RMS
    rms_in = np.std(rin)  # check the RMS, should be vn
    rout = np.zeros_like(rin)
    # RC lowpass the noise
    eps = dt / tau
    eps_limit = 0.1
    if eps > eps_limit:
        logger.warning(
            f"\neps={eps:.3f} for IIR lowpass is >{eps_limit}, either reduce timestep (currently {dt:.3f}s) (using higher frame rate) or decrease cutuff_hz (currently {f3db:.3f} Hz)"
            f"\n\tExpect the generated shot noise rate to be significantly lower than the desired rate."
            f"\n\tConsider not using --photoreceptor_noise option if you only want simple Poisson shot noise without temporal correlation of lowpass filtering and ON/OFF events."
        )
    rout[0] = 0  # init value is mean 0
    # lp filter the sequence with same tau and dt as v2e
    for i in range(1, len(rin)):
        rout[i] = rout[i - 1] * (1 - eps) + rin[i] * eps
    rms_out = np.std(rout)  # compute the amplitude of this noise
    scale = rms_in / rms_out  #
    vnscaled = (
        scale * vn
    )  # divide the computed vn to get the necessary vn to add before RC lowpass filtering
    new_rms_out = np.std(scale * rin)  # check RMS of scaled noise

    compute_photoreceptor_noise_voltage.last_vn = vnscaled
    # rout*=vnscaled
    # stdout=np.std(rout)
    # import matplotlib.pyplot as plt
    # plt.plot(t,rin,t,rout)
    # plt.xlabel('time (s)')
    # plt.ylabel('filtered noise')
    # plt.show()
    if not compute_photoreceptor_noise_voltage.vrms_computation_printed:
        logger.info(
            f"For desired shot_noise_rate_hz={shot_noise_rate_hz} Hz, computed photoreceptor_noise_rms={vn:.3f} in ln units,"
            f" scaled by factor {scale:.3f} to {vnscaled:.3f} before 1st-order lowpass with sample rate {sample_rate_hz:.3} Hz, "
            f"sample interval dt={dt * 1000:.3f} ms,"
            f", cutoff_hz={f3db} Hz, tau={tau * 1000:.3f} ms,  Rn/f3dB={rate_per_bw:.3g} Hz, "
            f" and nominal on/off threshold={pos_thr}/{neg_thr} +/- {sigma_thr:.3f} ln units."
            # f' The sample lowpass filtered has RMS amplitude {stdout:.3f}.'
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
    """Generate shot noise.
    :param shot_noise_rate_hz: the rate per pixel in hz
    :param delta_time: the delta time for this frame in seconds
    :param shot_noise_inten_factor: factor to model the slight increase
        of shot noise with intensity when shot noise dominates at low intensity
    :param inten01: the pixel light intensities in this frame; shape is used to generate output
    :param pos_thres_pre_prob: per pixel factor to generate more
        noise from pixels with lower ON threshold: self.pos_thres_nominal/self.pos_thres
    :param neg_thres_pre_prob: same for OFF

    :returns: shot_on_coord, shot_off_coord, each are (h,w) arrays of on and off boolean True for noise events per pixel
    """
    # new shot noise generator, generate for the entire batch of iterations over this frame

    if shot_noise_rate_hz * delta_time > 1:
        logger.warning(
            f"shot_noise_rate_hz*delta_time={shot_noise_rate_hz:.2f}*{delta_time:.2g}={shot_noise_rate_hz * delta_time:.2f} is too large, decrease timestamp resolution or sample rate"
        )

    # shot noise factor is the probability of generating an OFF event in this frame (which is tiny typically)
    # we compute it by taking half the total shot noise rate (OFF only),
    # multiplying by the delta time of this frame,
    # and multiplying by the intensity factor
    # division by num_iter is correct if generate_shot_noise is called outside the iteration loop, unless num_iter=1 for calling outside loop
    shot_noise_factor = (
        ((shot_noise_rate_hz / 2) * delta_time)
        * ((shot_noise_inten_factor - 1) * inten01 + 1)
    )  # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1 # TODO check this logic again, the shot noise rate should increase with intensity but factor is negative here

    # probability for each pixel is
    # dt*rate*nom_thres/actual_thres.
    # That way, the smaller the threshold,
    # the larger the rate
    one_minus_shot_ON_prob_this_sample = (
        1 - shot_noise_factor * pos_thres_pre_prob
    )  # ON shot events are generated when uniform sampled random number from range 0-1 is larger than this; the larger shot_noise_factor, the larger the noise rate
    shot_OFF_prob_this_sample = (
        shot_noise_factor * neg_thres_pre_prob
    )  # OFF shot events when 0-1 sample less than this

    # for shot noise generate rands from 0-1 for each pixel
    rand01 = torch.rand(
        size=inten01.shape, dtype=torch.float32, device=inten01.device
    )  # draw_frame samples

    # precompute all the shot noise cords, gets binary array size of chip
    shot_on_cord = torch.gt(rand01, one_minus_shot_ON_prob_this_sample)
    shot_off_cord = torch.lt(rand01, shot_OFF_prob_this_sample)

    return shot_on_cord, shot_off_cord


# ---------------------------------------------------------------------------
# torch.compile fused pipeline (inductor backend)
# Fuses lin_log + lowpass + diff + event_map into a single GPU kernel.
# Falls back gracefully if compile is unavailable.
# ---------------------------------------------------------------------------

_LIN_LOG_THRESHOLD = 20.0
_LIN_LOG_F = (1.0 / _LIN_LOG_THRESHOLD) * math.log(_LIN_LOG_THRESHOLD)
_ROUNDING = 1e8


def _fused_photoreceptor_step_compiled(
    frame: torch.Tensor,
    lp_buf: torch.Tensor,
    base_buf: torch.Tensor,
    pos_thres: torch.Tensor,
    neg_thres: torch.Tensor,
    inten01: torch.Tensor,
    delta_time: float,
    tau: float,
) -> tuple:
    """Compiled fusion of lin_log + lowpass + diff + event_map.

    All element-wise ops are fused into a single GPU kernel by torch inductor,
    giving ~3x speedup over separate calls on MPS.
    """
    log_frame = torch.where(frame <= _LIN_LOG_THRESHOLD, frame * _LIN_LOG_F, torch.log(frame))
    log_frame = torch.round(log_frame * _ROUNDING) / _ROUNDING
    eps = inten01 * (delta_time / tau)
    eps = torch.clamp(eps, max=1)
    lp_buf = (1 - eps) * lp_buf + eps * log_frame
    diff = lp_buf - base_buf
    pe = torch.div(torch.relu(diff), pos_thres, rounding_mode="floor").to(torch.int32)
    ne = torch.div(torch.relu(-diff), neg_thres, rounding_mode="floor").to(torch.int32)
    return lp_buf, pe, ne


def _fused_step_with_leak(
    frame: torch.Tensor,
    lp_buf: torch.Tensor,
    base_buf: torch.Tensor,
    pos_thres: torch.Tensor,
    neg_thres: torch.Tensor,
    inten01: torch.Tensor,
    noise_rate_arr: torch.Tensor,
    rand_vals: torch.Tensor,
    delta_time: float,
    tau: float,
    leak_rate_hz: float,
    leak_jitter: float,
) -> tuple:
    """Fused pipeline: lin_log + lowpass + leak + diff + event_map.

    Leak subtraction is included in the compiled kernel, eliminating
    a separate kernel launch. rand_vals is pre-generated externally
    to keep RNG out of the compiled kernel for maximum fusion.
    """
    log_frame = torch.where(frame <= _LIN_LOG_THRESHOLD, frame * _LIN_LOG_F, torch.log(frame))
    log_frame = torch.round(log_frame * _ROUNDING) / _ROUNDING
    eps = inten01 * (delta_time / tau)
    eps = torch.clamp(eps, max=1)
    lp_buf = (1 - eps) * lp_buf + eps * log_frame

    # Leak: use pre-generated random values
    leak = leak_rate_hz * noise_rate_arr * (1 - leak_jitter * rand_vals)
    base_buf = base_buf - delta_time * leak * pos_thres

    diff = lp_buf - base_buf
    pe = torch.div(torch.relu(diff), pos_thres, rounding_mode="floor").to(torch.int32)
    ne = torch.div(torch.relu(-diff), neg_thres, rounding_mode="floor").to(torch.int32)
    return lp_buf, base_buf, pe, ne


def _fused_batched_step(
    frames_b: torch.Tensor,
    lp_buf: torch.Tensor,
    base_buf: torch.Tensor,
    pos_thres: torch.Tensor,
    neg_thres: torch.Tensor,
    intens_b: torch.Tensor,
    noise_rate_arr: torch.Tensor,
    rand_vals_b: torch.Tensor,
    delta_time: float,
    tau: float,
    leak_rate_hz: float,
    leak_jitter: float,
) -> tuple:
    """Batched fused pipeline for [B, H, W] frame batches.

    Processes B frames sequentially with state accumulation (each frame's
    lp_buf and base_buf feed into the next). All element-wise ops are fused
    into compiled GPU kernels. The batch dimension allows the compiler to
    optimize memory access patterns across frames.

    Returns:
        (lp_bufs, base_bufs, pe_batch, ne_batch): All [B, H, W] tensors.
    """
    B = frames_b.shape[0]
    # lin_log all frames at once (independent)
    log_frames = torch.where(frames_b <= _LIN_LOG_THRESHOLD, frames_b * _LIN_LOG_F, torch.log(frames_b))
    log_frames = torch.round(log_frames * _ROUNDING) / _ROUNDING

    # Pre-compute constants
    eps = intens_b * (delta_time / tau)
    eps = torch.clamp(eps, max=1)
    leak = leak_rate_hz * noise_rate_arr * (1 - leak_jitter * rand_vals_b)
    delta_leak = delta_time * leak * pos_thres

    # Process frames sequentially with state accumulation
    lp_bufs = torch.empty_like(frames_b)
    base_bufs = torch.empty_like(frames_b)
    pe_batch = torch.empty(B, *lp_buf.shape, dtype=torch.int32, device=lp_buf.device)
    ne_batch = torch.empty(B, *lp_buf.shape, dtype=torch.int32, device=lp_buf.device)

    for i in range(B):
        # lowpass
        lp_buf = (1 - eps[i]) * lp_buf + eps[i] * log_frames[i]
        # leak
        base_buf = base_buf - delta_leak[i]
        # event map
        diff = lp_buf - base_buf
        pe_batch[i] = torch.div(torch.relu(diff), pos_thres, rounding_mode="floor").to(torch.int32)
        ne_batch[i] = torch.div(torch.relu(-diff), neg_thres, rounding_mode="floor").to(torch.int32)
        lp_bufs[i] = lp_buf
        base_bufs[i] = base_buf

    return lp_bufs, base_bufs, pe_batch, ne_batch


_compiled_batched = None


def get_compiled_batched():
    """Lazily compile and return the batched fused step function."""
    global _compiled_batched
    if _compiled_batched is None:
        try:
            _compiled_batched = torch.compile(_fused_batched_step, mode="max-autotune")
        except Exception:
            _compiled_batched = _fused_batched_step
    return _compiled_batched
_compiled_step = None
_compiled_step_leak = None


def get_compiled_step():
    """Lazily compile and return the fused step function."""
    global _compiled_step
    if _compiled_step is None:
        try:
            _compiled_step = torch.compile(_fused_photoreceptor_step_compiled, backend="inductor")
        except Exception:
            _compiled_step = _fused_photoreceptor_step_compiled
    return _compiled_step


def get_compiled_step_leak():
    """Lazily compile and return the fused step with leak function."""
    global _compiled_step_leak
    if _compiled_step_leak is None:
        try:
            _compiled_step_leak = torch.compile(_fused_step_with_leak, backend="inductor")
        except Exception:
            _compiled_step_leak = _fused_step_with_leak
    return _compiled_step_leak
