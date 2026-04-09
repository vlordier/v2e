#!/usr/bin/env bash
# Run baseline vs experimental-realism v2e on multiple UZH FPV sequences.
# Compares synthesised events from APS frames against the real DAVIS events.
#
# Dataset:  http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/
# Sequences: indoor_forward_3, indoor_forward_10, indoor_45_2,
#            outdoor_forward_1, outdoor_forward_3
set -euo pipefail
LOG=/root/v2e_run.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) starting v2e FPV multi-sequence run ==="

# ── system deps (idempotent) ───────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 build-essential unzip

# ── clone / update repo ────────────────────────────────────────────────────────
cd /root
if [ ! -d v2e ]; then
  git clone --branch upgrades --depth 1 https://github.com/vlordier/v2e.git
fi
cd v2e
git fetch --depth 1 origin upgrades && git reset --hard origin/upgrades

# ── python env (idempotent) ────────────────────────────────────────────────────
pip install --quiet --upgrade pip
pip install --quiet gdown tqdm Pillow
pip install --quiet -r requirements.txt
pip install --quiet -e .

# ── SuperSloMo checkpoint ──────────────────────────────────────────────────────
CKPT=/root/v2e/input/SuperSloMo39.ckpt
mkdir -p /root/v2e/input
if [ ! -f "$CKPT" ]; then
  echo "Downloading SuperSloMo39.ckpt..."
  gdown --id 1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx -O "$CKPT"
fi

# ── Sequences to evaluate ──────────────────────────────────────────────────────
# Two indoor_forward (high-speed corridor flight)
# One indoor_45      (tilted 45°, different illumination dynamics)
# Two outdoor_forward (trees/sky, broad luminance range)
SEQUENCES=(
  indoor_forward_3
  indoor_forward_10
  indoor_45_2
  outdoor_forward_1
  outdoor_forward_3
)

STOP=10.0        # first 10 s of each sequence
FPV_ROOT=/root/fpv
OUT_ROOT=/root/out
mkdir -p "$FPV_ROOT" "$OUT_ROOT"

BASE_URL="http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3"

# ── Per-sequence loop ──────────────────────────────────────────────────────────
for SEQ in "${SEQUENCES[@]}"; do
  SEQ_DIR="${FPV_ROOT}/${SEQ}"
  DONE_MARKER="${OUT_ROOT}/${SEQ}/.done"
  if [ -f "$DONE_MARKER" ]; then
    echo "--- ${SEQ}: already done, skipping ---"
    continue
  fi

  echo ""
  echo "======================================================================"
  echo "=== $(date)  SEQUENCE: ${SEQ} ==="
  echo "======================================================================"

  # ── Download ──────────────────────────────────────────────────────────────
  mkdir -p "$SEQ_DIR"
  if [ ! -f "${SEQ_DIR}/images.txt" ]; then
    echo "[${SEQ}] Downloading..."
    ZIP="${FPV_ROOT}/${SEQ}.zip"
    wget -q --show-progress \
      "${BASE_URL}/${SEQ}_davis_with_gt.zip" \
      -O "$ZIP"
    echo "[${SEQ}] Extracting..."
    unzip -q "$ZIP" -d "$SEQ_DIR"
    rm "$ZIP"
  fi

  # ── Detect FPS from images.txt ────────────────────────────────────────────
  FPS=$(python3 -c "
lines = [l.split() for l in open('${SEQ_DIR}/images.txt') if not l.startswith('#') and l.strip()]
t0, t1 = float(lines[0][1]), float(lines[1][1])
tN = float(lines[-1][1])
n  = len(lines)
fps = (n - 1) / (tN - t0)
print(f'{fps:.4f}')
")
  echo "[${SEQ}] Detected FPS: ${FPS}"

  # ── Build AVI from APS frames ─────────────────────────────────────────────
  IN_VIDEO="${SEQ_DIR}/aps.avi"
  if [ ! -f "$IN_VIDEO" ]; then
    FRAME_PATTERN="${SEQ_DIR}/img/image_0_%d.png"
    # fall back if frames use a different pattern
    if [ ! -f "${SEQ_DIR}/img/image_0_0.png" ]; then
      FRAME_PATTERN="${SEQ_DIR}/img/frame_%d.png"
    fi
    echo "[${SEQ}] Building AVI at ${FPS} fps..."
    ffmpeg -y -loglevel error \
      -framerate "$FPS" \
      -i "$FRAME_PATTERN" \
      -c:v ffv1 \
      "$IN_VIDEO"
    echo "[${SEQ}] AVI ready: $(du -sh "$IN_VIDEO" | cut -f1)"
  fi

  # ── Baseline v2e ─────────────────────────────────────────────────────────
  echo "[${SEQ}] === BASELINE run ==="
  python /root/v2e/v2e.py \
    -i "$IN_VIDEO" \
    -o "${OUT_ROOT}/${SEQ}/baseline" \
    --overwrite \
    --no_preview \
    --dvs346 \
    --skip_video_output \
    --dvs_text events.txt \
    --stop_time "$STOP" \
    --slomo_model "$CKPT"

  # ── Experimental v2e ──────────────────────────────────────────────────────
  # refractory_period 0.3 ms < SloMo timestamp spacing (~1 ms at ~32 fps x 30x)
  echo "[${SEQ}] === EXPERIMENTAL run ==="
  python /root/v2e/v2e.py \
    -i "$IN_VIDEO" \
    -o "${OUT_ROOT}/${SEQ}/experimental" \
    --overwrite \
    --no_preview \
    --dvs346 \
    --skip_video_output \
    --dvs_text events.txt \
    --stop_time "$STOP" \
    --slomo_model "$CKPT" \
    --refractory_mode soft \
    --refractory_period 0.0003 \
    --refractory_tau_s 0.00015 \
    --threshold_adaptation_gain 0.1 \
    --threshold_adaptation_tau_s 0.05 \
    --hot_pixel_fraction 0.002 \
    --hot_pixel_rate_hz 25 \
    --bursty_pixel_fraction 0.001 \
    --bursty_pixel_rate_hz 10 \
    --row_noise_rate_hz 0.2 \
    --scene_cut_policy reset \
    --scene_cut_threshold 0.3

  touch "$DONE_MARKER"
done

# ── Calibrated runs (--calibrate_from auto-fits pos/neg thresholds per sequence) ─
for SEQ in "${SEQUENCES[@]}"; do
  SEQ_DIR="${FPV_ROOT}/${SEQ}"
  DONE_CAL="${OUT_ROOT}/${SEQ}/.done_calibrated"
  if [ -f "$DONE_CAL" ]; then
    echo "--- ${SEQ}: calibrated already done, skipping ---"
    continue
  fi

  REAL_EVENTS="${SEQ_DIR}/events.txt"
  IN_VIDEO="${SEQ_DIR}/aps.avi"

  echo ""
  echo "======================================================================"
  echo "=== $(date)  CALIBRATED: ${SEQ} ==="
  echo "======================================================================"

  # Calibrated Baseline: threshold fitted to real event count,  no realism extras
  echo "[${SEQ}] === CALIBRATED BASELINE run ==="
  python /root/v2e/v2e.py \
    -i "$IN_VIDEO" \
    -o "${OUT_ROOT}/${SEQ}/cal_baseline" \
    --overwrite \
    --no_preview \
    --dvs346 \
    --skip_video_output \
    --dvs_text events.txt \
    --stop_time "$STOP" \
    --slomo_model "$CKPT" \
    --calibrate_from "$REAL_EVENTS" \
    --calibrate_stop_time "$STOP"

  # Calibrated Experimental: fitted threshold + all realism controls
  echo "[${SEQ}] === CALIBRATED EXPERIMENTAL run ==="
  python /root/v2e/v2e.py \
    -i "$IN_VIDEO" \
    -o "${OUT_ROOT}/${SEQ}/cal_experimental" \
    --overwrite \
    --no_preview \
    --dvs346 \
    --skip_video_output \
    --dvs_text events.txt \
    --stop_time "$STOP" \
    --slomo_model "$CKPT" \
    --calibrate_from "$REAL_EVENTS" \
    --calibrate_stop_time "$STOP" \
    --refractory_mode soft \
    --refractory_period 0.0003 \
    --refractory_tau_s 0.00015 \
    --threshold_adaptation_gain 0.1 \
    --threshold_adaptation_tau_s 0.05 \
    --hot_pixel_fraction 0.002 \
    --hot_pixel_rate_hz 25 \
    --bursty_pixel_fraction 0.001 \
    --bursty_pixel_rate_hz 10 \
    --row_noise_rate_hz 0.2 \
    --scene_cut_policy reset \
    --scene_cut_threshold 0.3

  touch "$DONE_CAL"
done

# ── Aggregate comparison across all sequences ──────────────────────────────────
echo ""
echo "======================================================================"
echo "=== AGGREGATE COMPARISON: synthesised vs real DAVIS events (4 conditions) ==="
echo "======================================================================"
python3 - <<'PYEOF'
import sys
from pathlib import Path

FPV_ROOT = Path("/root/fpv")
OUT_ROOT = Path("/root/out")
STOP     = 10.0

SEQUENCES = [
    "indoor_forward_3",
    "indoor_forward_10",
    "indoor_45_2",
    "outdoor_forward_1",
    "outdoor_forward_3",
]

# Each entry: (output subdir name, short display label)
CONDITIONS = [
    ("baseline",         "Base"),
    ("experimental",     "Exp"),
    ("cal_baseline",     "CalBase"),
    ("cal_experimental", "CalExp"),
]


def load_events(path, t_offset=0.0, t_max=None):
    on = off = 0
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                if len(parts) < 4:
                    continue
                t = float(parts[0]) - t_offset
                if t_max is not None and t > t_max:
                    break
                if float(parts[3]) > 0:
                    on += 1
                else:
                    off += 1
    except FileNotFoundError:
        pass
    return on, off


rows = []
for seq in SEQUENCES:
    real_path = FPV_ROOT / seq / "events.txt"
    if not real_path.exists():
        print(f"  {seq}: real events not found, skipping")
        continue

    t0 = None
    with open(real_path) as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                t0 = float(line.split()[0])
                break

    r_on, r_off = load_events(real_path, t_offset=t0, t_max=STOP)
    r_n = r_on + r_off
    r_ratio = r_on / r_off if r_off else float('inf')

    cond_results = {}
    for cond_dir, cond_label in CONDITIONS:
        p = OUT_ROOT / seq / cond_dir / "events.txt"
        c_on, c_off = load_events(p, t_max=STOP)
        c_n = c_on + c_off
        c_ratio = c_on / c_off if c_off else float('inf')
        cnt_pct = (c_n - r_n) / r_n * 100 if r_n else float('inf')
        ratio_d = c_ratio - r_ratio
        cond_results[cond_label] = (c_n, cnt_pct, c_ratio, ratio_d)

    rows.append((seq, r_n, r_ratio, cond_results))

if not rows:
    print("No results available yet.")
    sys.exit(0)

all_labels  = [lb for _, lb in CONDITIONS]
present_labels = [lb for lb in all_labels
                  if any(lb in r[3] and r[3][lb][0] > 0 for r in rows)]

# ── Per-condition summary tables ─────────────────────────────────────────────
for label in present_labels:
    print(f"\n{'='*72}")
    print(f"  Condition: {label}")
    print(f"{'='*72}")
    hdr = f"  {'Sequence':<22}  {'Real N':>9}  {'Real R':>6}  {'Synth N':>9}  {'Cnt%':>7}  {'Synth R':>7}  {'ΔR':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (seq, r_n, r_ratio, cond_results) in rows:
        if label not in cond_results or cond_results[label][0] == 0:
            print(f"  {seq:<22}  {'(not run)':>9}")
            continue
        c_n, cnt_pct, c_ratio, ratio_d = cond_results[label]
        print(f"  {seq:<22}  {r_n:>9,}  {r_ratio:>6.3f}"
              f"  {c_n:>9,}  {cnt_pct:>+7.1f}%  {c_ratio:>7.3f}  {ratio_d:>+6.3f}")

# ── Head-to-head ON/OFF winner table ──────────────────────────────────────────
print(f"\n{'='*72}")
print("  ON/OFF ratio winners per sequence (lowest |ΔR| wins)")
print(f"{'='*72}")
print(f"  {'Sequence':<22}", end="")
for lb in present_labels:
    print(f"  {lb:>10}", end="")
print("  Winner")
print("  " + "-" * (24 + 12 * len(present_labels)))
winners = {lb: 0 for lb in present_labels}
for (seq, r_n, r_ratio, cond_results) in rows:
    avail = {lb: cond_results[lb][3] for lb in present_labels
             if lb in cond_results and cond_results[lb][0] > 0}
    if not avail:
        continue
    best_lb = min(avail, key=lambda lb: abs(avail[lb]))
    winners[best_lb] += 1
    print(f"  {seq:<22}", end="")
    for lb in present_labels:
        if lb in avail:
            marker = "***" if lb == best_lb else "   "
            print(f"  {avail[lb]:>+7.3f}{marker}", end="")
        else:
            print(f"  {'--':>10}", end="")
    print(f"  {best_lb}")

print()
for lb in present_labels:
    print(f"  {lb}: wins ON/OFF fidelity in {winners[lb]}/{len(rows)} sequences")

# ── Count-error summary ────────────────────────────────────────────────────────
print()
print("  Mean event count delta vs real:")
for lb in present_labels:
    cnts = [r[3][lb][1] for r in rows if lb in r[3] and r[3][lb][0] > 0]
    if cnts:
        mean_cnt = sum(cnts) / len(cnts)
        print(f"    {lb:<16}: {mean_cnt:>+7.1f}%")

print()
print("==> Calibrated conditions (CalBase/CalExp) should have dramatically lower count error.")
PYEOF

echo ""
echo "=== Done at $(date) ==="
touch /root/DONE
