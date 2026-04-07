# v2e-v2 research roadmap

This note turns the current realism-upgrade ideas into a concrete plan for the `SensorsINI/v2e` codebase.

The goal is **not** to replace the current simulator, but to extend its strengths — finite bandwidth, threshold mismatch, leak/shot noise, GPU acceleration, and interpolation support — toward a more faithful **sensor-and-scene emulator**.

---

## Prototype status (April 2026)

A first **Phase 1 prototype** is now implemented directly in the repo:

- soft refractory recovery
- threshold adaptation
- hot / bursty / row-correlated defect noise
- scene-cut reset / suppression controls
- CLI exposure via `v2e.py`
- regression coverage in `test/test_v2ecore.py`
- a reproducible synthetic comparison script in `scripts/realism_sweep.py`

Current sweep snapshot from `scripts/realism_sweep.py`:

| Scenario | Baseline | Experimental variant | Observed effect |
| --- | ---: | ---: | --- |
| `rapid-ramp` | `32000` | `3586` (`soft-refractory`) | `-88.8%` events |
| `rapid-ramp` | `32000` | `24000` (`threshold-adapt`) | `-25.0%` events |
| `static-hold` | `0` | `14` (`hot-pixels`) | defect activity appears on static scenes |
| `scene-cut` | `11376` | `0` (`cut-reset`) | cut burst fully suppressed |

These numbers are **sanity-check results, not calibration targets**. They show that the new knobs are active and materially changing event statistics in the intended directions.

### Real-clip comparative run (April 2026)

End-to-end run on `media/counting.gif` (1 second, DVS346 resolution, no SloMo):

```
python v2e.py -i media/counting.gif -o /tmp/v2e_realism_baseline \
  --no_preview --disable_slomo --dvs346 --skip_video_output --dvs_text events.txt --stop_time 1.0

python v2e.py -i media/counting.gif -o /tmp/v2e_realism_proto \
  --no_preview --disable_slomo --dvs346 --skip_video_output --dvs_text events.txt --stop_time 1.0 \
  --refractory_mode soft --refractory_period 0.001 --refractory_tau_s 0.0005 \
  --threshold_adaptation_gain 0.1 --threshold_adaptation_tau_s 0.05 \
  --hot_pixel_fraction 0.002 --hot_pixel_rate_hz 25 \
  --bursty_pixel_fraction 0.001 --bursty_pixel_rate_hz 10 \
  --row_noise_rate_hz 0.2 \
  --scene_cut_policy reset --scene_cut_threshold 0.2
```

| Metric | Baseline | Experimental | Change |
| --- | ---: | ---: | ---: |
| Total events | 1,034,434 | 833,170 | **−19.5%** |
| ON events | 508,471 | 374,828 | −26.3% |
| OFF events | 525,965 | 458,344 | −12.9% |
| ON/OFF ratio | 0.967 | 0.818 | (more OFF-skewed) |

The asymmetric ON/OFF reduction is expected: the combined soft refractory + threshold adaptation primarily attenuates rapid ON bursts (fast-ramp transitions), while the OFF channel (slower decay paths) is less affected. The ON/OFF ratio shift from ~0.97 to ~0.82 is consistent with real DVS sensors, which exhibit an inherent ON/OFF asymmetry.

### GPU SloMo run (Vast.ai RTX 3090, April 2026)

Same clip with SloMo enabled on a Vast.ai RTX 3090 ($0.147/hr). SloMo upsampled 76 input frames to **2019 interpolated frames** (26.8× average factor), giving **1.49 ms average DVS timestamp resolution**.

```
python v2e.py -i media/counting.gif ... --stop_time 3.0          # (no --disable_slomo)
```

| Metric | Baseline | Experimental | Change |
| --- | ---: | ---: | ---: |
| Total events | 1,988,373 | 1,976,953 | **−0.57%** |
| ON events | 989,030 | 983,110 | −0.60% |
| OFF events | 999,340 | 993,840 | −0.55% |
| ON/OFF ratio | 0.990 | 0.989 | unchanged |

**Key finding — refractory/timestamp coupling:** the effect of the soft refractory controls drops from −19.5% (no SloMo, coarse timestamps) to −0.57% (SloMo 26.8×, 1.49 ms timestamps) because the refractory period (`--refractory_period 0.001 = 1 ms`) is nearly equal to the interpolated timestamp spacing. Once the interpolated frame interval exceeds the refractory period, the filter has no room to suppress inter-frame retriggering.

**Implication for calibration:** to tune the refractory period from real sensor data, you must either:
1. Use `--disable_slomo` (matched to sensor's native frame rate), or
2. Set `--refractory_period` well below the SloMo timestamp resolution (e.g. `≤ 0.5 ms` for 26× upsampling at 25 fps input).

The synthetic sweep results are **identical** on GPU vs CPU (correctness verified on RTX 3090, CUDA 12.6).

### UZH-FPV 5-sequence comparative run (April 2026)

End-to-end validation on five real UZH-FPV sequences with available DAVIS ground-truth events.
Run on a Vast.ai RTX 3090 (Czechia, ~$0.20/hr) with SloMo enabled.
Real event counts are taken from the ground-truth `events.txt` files in the first 10 s window.

| Sequence | Real N | Real R | Baseline N | Cnt Δ% | Base R | ΔR | Experimental N | Cnt Δ% | Exp R | ΔR | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| indoor_forward_3 | 670,600 | 0.891 | 3,625,998 | +440.7% | 0.968 | +0.078 | 3,149,121 | +369.6% | 0.952 | +0.062 | **EXP** |
| indoor_forward_10 | 236,523 | 1.009 | 2,030,616 | +758.5% | 1.044 | +0.034 | 1,941,719 | +720.9% | 1.037 | +0.027 | **EXP** |
| indoor_45_2 | 429,030 | 0.809 | 86,181 | −79.9% | 1.123 | +0.314 | 313,984 | −26.8% | 1.082 | +0.272 | **EXP** |
| outdoor_forward_1 | 3,713,201 | 0.824 | 38,302,480 | +931.5% | 1.012 | +0.187 | 25,726,887 | +592.8% | 0.988 | +0.164 | **EXP** |
| outdoor_forward_3 | 3,081,219 | 0.901 | 39,742,921 | +1189.8% | 1.010 | +0.109 | 26,207,890 | +750.6% | 0.987 | +0.086 | **EXP** |

Columns: **R** = ON/OFF ratio; **ΔR** = |R\_synth − R\_real|; **Winner** = which condition has ON/OFF ratio closer to real sensor.

**Summary statistics:**

| Metric | Baseline | Experimental |
|---|---:|---:|
| Mean event count Δ vs real | +648.1% | +481.4% |
| Mean \|ΔON/OFF\| | 0.1444 | 0.1223 |
| Sequences where ON/OFF closer to real | 0 / 5 | **5 / 5** |

**Key findings:**

1. **Experimental controls win on ON/OFF fidelity in all 5 sequences.** Mean |ΔON/OFF| is 15% better (0.1444 → 0.1223).
2. **Both conditions massively over-count events** at the default threshold (pos/neg = 0.2). Baseline is +648%, experimental is +481% over real. Threshold calibration is the largest remaining gap.
3. **Experimental controls reduce event count toward real in 4/5 sequences.** The exception is `indoor_45_2`, where the low-motion 45° downward view causes hot/bursty pixel noise to *add* events rather than suppress spurious burst events. This anomaly motivates threshold-aware noise scaling.
4. **Outdoor sequences are particularly over-counted** (931–1189% at baseline). High dynamic outdoor lighting raises per-pixel contrast far above the 0.2 threshold, requiring calibrated thresholds in the 0.8–1.5 range.
5. **`indoor_45_2` anomaly**: experimental (313K) > baseline (86K), because the baseline was itself already under-counting (−79.9%), and the noise injection from hot/bursty pixel models is not intensity-gated. Hot-pixel noise should be conditioned on local event rate to avoid inflating already-sparse sequences.

**Committed results script:** `scripts/vast_run.sh` — replicable on any Vast.ai or SLURM GPU node with a CUDA 12.x runtime.

**Next step:** threshold calibration using `v2ecore/calibration.py` (`--calibrate_from` flag). Expected to bring both conditions to within 10–30% of real event counts.

---

## Current baseline in the repo

Today the main pieces are already in place:

- `v2ecore/emulator.py` contains the stateful `EventEmulator`
- `v2ecore/emulator_utils.py` contains noise and helper kernels
- `v2ecore/slomo.py` handles frame interpolation to reduce timestamp quantization
- `v2ecore/v2e_args.py` exposes the key sensor knobs on the CLI

Notable current support already present in the codebase:

- static threshold mismatch (`sigma_thres`)
- finite photoreceptor bandwidth (`cutoff_hz`)
- leak and shot noise
- a **hard** refractory filter via `--refractory_period`

The roadmap below focuses on gaps that are still missing or only modeled in a simplified form.

---

## Recommended implementation order

### Phase 1 — highest payoff, lowest risk

#### 1. Soft refractory recovery

**Why:** `v2e` already has hard suppression through `--refractory_period`, but real pixels typically recover gradually rather than switching from fully blocked to fully ready.

**Proposed upgrade:**

- keep per-pixel `last_event_ts` / `timestamp_mem`
- add a recovery curve, e.g. `recovery = 1 - exp(-dt / tau_ref)`
- either:
  - attenuate event probability during recovery, or
  - temporarily raise the effective threshold after a spike
- optionally make recovery depend on brightness, polarity, or local activity

**Suggested API / flags:**

- `--refractory_mode {hard,soft,adaptive}`
- `--refractory_tau_s`
- `--refractory_floor`
- `--refractory_brightness_dependence`

**Likely code touch points:**

- `v2ecore/emulator.py`
- `v2ecore/emulator_utils.py` (`apply_refractory_recovery(...)` helper)

**Tests to add:**

- max firing rate saturates under fast flicker
- `soft` approaches `hard` as `tau_ref -> 0`
- recovery curve changes event count smoothly instead of causing abrupt clipping

---

#### 2. Hot pixels and correlated defect maps

**Why:** the current leak/noise model is useful, but deployment data often includes persistent hot pixels, bursty defect pixels, polarity-biased defects, and row/column artifacts.

**Proposed upgrade:**

- create a persistent `DefectMap`
- support:
  - always-hot pixels
  - intermittently hot / bursty pixels
  - ON-biased or OFF-biased bad pixels
  - local clusters of defects
  - row/column correlated flicker
- optionally modulate defect rates by illumination or temperature

**Suggested API / flags:**

- `--hot_pixel_fraction`
- `--bursty_pixel_fraction`
- `--defect_cluster_fraction`
- `--row_flicker_rate_hz`
- `--defect_seed`

**Likely code touch points:**

- new file `v2ecore/defects.py`
- integration in `v2ecore/emulator.py`

**Tests to add:**

- hot pixels fire above configurable background rate
- clustered defects remain spatially correlated across frames
- row artifacts produce coherent line-wise spikes without breaking clean-mode behavior

---

#### 3. Dynamic thresholds and adaptation

**Why:** thresholds are currently mismatched but mostly static. A stronger model makes them activity- and context-dependent.

**Proposed upgrade:**

- maintain dynamic per-pixel states:
  - `theta_on(x, y, t)`
  - `theta_off(x, y, t)`
- add short-term adaptation after firing
- allow slow drift with brightness, recent activity, or operating regime
- model ON/OFF asymmetry as a changing state rather than only a fixed mismatch

**Suggested API / flags:**

- `--thres_adapt_tau_s`
- `--thres_adapt_gain`
- `--thres_drift_std`
- `--thres_brightness_coupling`

**Likely code touch points:**

- new file `v2ecore/adaptation.py`
- threshold updates in `v2ecore/emulator.py`

**Tests to add:**

- repeated fast stimulation temporarily reduces sensitivity
- ON/OFF event imbalance changes in a controlled way under bright/dim inputs
- static defaults preserve current behavior when adaptation is disabled

---

#### 4. Cut-aware interpolation and transition handling

**Why:** one of the most visible artifacts in video-to-event conversion is fake flicker or event bursts around scene cuts and abrupt transitions.

**Proposed upgrade:**

- detect cuts using histogram jumps, optical-flow consistency, or simple frame distance heuristics
- add a transition policy:
  - `reset`: reset pixel memory at cuts
  - `suppress`: hold events for a short window
  - `blend`: fade the internal state over a small time horizon
- keep the normal interpolation path for smooth segments

**Suggested API / flags:**

- `--scene_cut_policy {off,reset,suppress,blend}`
- `--scene_cut_threshold`
- `--scene_cut_hold_ms`

**Likely code touch points:**

- `v2ecore/slomo.py`
- new helper file `v2ecore/cut_detection.py`

**Tests to add:**

- concatenated videos do not produce unrealistic event bursts at boundaries
- smooth motion clips are unaffected when no cuts are detected

---

### Phase 2 — deeper sensor realism

#### 5. Richer temporal pixel model

**Why:** the current model is a good practical baseline, but timing realism can likely benefit from a more explicit analog front-end.

**Proposed upgrade:**

- separate stages for:
  - photometric / logarithmic conversion
  - amplifier / comparator latency
  - optional hysteresis or memory
- support second-order or illumination-dependent latency dynamics
- keep per-pixel parameter variation for mismatch realism

**Suggested module split:**

- `v2ecore/pixel_models.py`
  - `BasePixelTemporalModel`
  - `FirstOrderPixelModel`
  - `SecondOrderLatencyModel`

**Validation targets:**

- timestamp error vs. intensity
- inter-spike interval statistics
- event timing under fast edges and low-light conditions

---

#### 6. Calibration-to-real-camera mode

**Why:** the biggest product upgrade is to make `v2e` not only generic, but also camera-specific when calibration data is available.

**Proposed upgrade:**

- fit threshold, latency, leak, noise, and defect profiles from real recordings
- export a reusable sensor profile
- allow profile-conditioned emulation from the CLI

**Suggested files / classes:**

- `v2ecore/calibration.py`
  - `CalibrationProfile`
  - `fit_threshold_profile(...)`
  - `fit_latency_profile(...)`
  - `fit_defect_profile(...)`
- `scripts/calibrate_sensor.py`

**Suggested API / flags:**

- `--sensor_profile path/to/profile.yaml`
- `--export_sensor_profile path/to/profile.yaml`

---

### Phase 3 — bigger research bets

#### 7. Photometric and spectral front-end

**Why:** grayscale conversion is practical, but it does not capture spectral sensitivity, quantum efficiency, or front-end image-formation effects.

**Proposed upgrade:**

- model sensor response from irradiance rather than only luma
- include:
  - spectral sensitivity
  - lens transmission / vignetting
  - dark current and shot noise before log conversion
  - optional NIR sensitivity

**Suggested files / classes:**

- `v2ecore/photometry.py`
  - `SpectralResponseModel`
  - `LensAndExposureModel`

**Note:** this is most valuable when the source data is synthetic or radiometrically meaningful.

---

#### 8. Hybrid physics + neural residual correction

**Why:** keep the current interpretable threshold-crossing core, then learn the residual effects the analytic model still misses.

**Proposed upgrade:**

- learn residual corrections on top of the physics-based precursor field
- candidate targets:
  - low-pass filtered log intensity residuals
  - threshold correction maps
  - timing correction offsets
  - structured defect / noise residuals
- keep event triggering explicit and classical for interpretability

**Suggested files / classes:**

- `v2ecore/neural_residuals.py`
  - `ResidualCorrectionModel`
  - `TimingResidualModel`
- optional continuous-time field models:
  - U-Net for local residuals
  - FNO for `(x, y, t)` continuous precursor correction

---

## Proposed module boundaries

| Module | Responsibility | Candidate classes / functions |
| --- | --- | --- |
| `v2ecore/emulator.py` | Main orchestration and state updates | `EventEmulator` |
| `v2ecore/emulator_utils.py` | Shared kernels / helper math | `apply_refractory_recovery`, noise helpers |
| `v2ecore/defects.py` | Persistent and bursty sensor defects | `DefectMap`, `HotPixelProcess`, `RowArtifactModel` |
| `v2ecore/adaptation.py` | Dynamic thresholds and short-term adaptation | `DynamicThresholdAdapter` |
| `v2ecore/pixel_models.py` | Analog temporal models | `FirstOrderPixelModel`, `SecondOrderLatencyModel` |
| `v2ecore/cut_detection.py` | Scene-cut handling before event generation | `detect_scene_cut`, `TransitionPolicy` |
| `v2ecore/calibration.py` | Sensor fitting and profile export | `CalibrationProfile`, fitting helpers |
| `v2ecore/photometry.py` | Spectral / photometric front-end | `SpectralResponseModel`, `LensAndExposureModel` |
| `v2ecore/neural_residuals.py` | Hybrid learned residuals | `ResidualCorrectionModel`, `TimingResidualModel` |

---

## Suggested realism benchmark loop

To keep the roadmap evidence-based, every new realism feature should be tied to measurable checks.

### Pixel- and stream-level metrics

- event rate histogram
- ON/OFF balance and asymmetry
- inter-event interval distribution
- latency vs. illumination curve
- defect persistence and cluster statistics
- row/column artifact correlation

### Downstream metrics

- optical flow / tracking transfer to real event data
- detection or SLAM robustness under low light, flicker, and fast motion
- regression against a learned realism metric when one is available

### Benchmark script sketch

Add a script such as:

- `scripts/evaluate_realism.py`

that compares synthetic and real event recordings and emits a summary report for parameter tuning.

---

## Immediate next tickets

If work starts now, the best near-term sequence is:

1. **upgrade existing hard refractory logic to soft recovery**
2. **add a defect-map layer for hot / bursty / clustered pixels**
3. **make thresholds adaptive instead of purely static**
4. **add cut-aware handling to the interpolation pipeline**

That sequence is deliberately chosen to improve realism without destabilizing the current core architecture.

---

## Status checklist

- [x] finite bandwidth model
- [x] threshold mismatch
- [x] leak noise
- [x] shot noise
- [x] basic hard refractory period
- [x] soft/adaptive refractory recovery
- [x] hot/bursty/clustered defect model
- [x] dynamic thresholds and adaptation
- [x] cut-aware interpolation policy
- [x] calibration-to-real-camera workflow (`v2ecore/calibration.py`, `--calibrate_from`)
- [x] 5-sequence UZH-FPV baseline vs experimental validation (ON/OFF fidelity: EXP wins 5/5)
- [ ] threshold calibration applied re-run (eliminate 481–648% event overcounting)
- [ ] intensity-gated hot-pixel noise (fix indoor_45_2 anomaly)
- [ ] second-order temporal pixel model
- [ ] spectral / photometric front-end
- [ ] hybrid neural residual correction
