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
- [ ] soft/adaptive refractory recovery
- [ ] hot/bursty/clustered defect model
- [ ] dynamic thresholds and adaptation
- [ ] cut-aware interpolation policy
- [ ] second-order temporal pixel model
- [ ] calibration-to-real-camera workflow
- [ ] spectral / photometric front-end
- [ ] hybrid neural residual correction
