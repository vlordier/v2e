#!/usr/bin/env bash
# Run baseline vs experimental-realism v2e on multiple UZH FPV sequences.
# Compares synthesised events from APS frames against the real DAVIS events.
#
# Dataset:  http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/
# Sequences: indoor_forward_3, indoor_forward_10, indoor_45_1,
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
  indoor_45_1
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

# ── Aggregate comparison across all sequences ──────────────────────────────────
echo ""
echo "======================================================================"
echo "=== AGGREGATE COMPARISON: synthesised vs real DAVIS events ==="
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
    "indoor_45_1",
    "outdoor_forward_1",
    "outdoor_forward_3",
]

def load_events(path, t_offset=0.0, t_max=None):
    """Return (n_on, n_off) for events up to t_max seconds."""
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
    b_path    = OUT_ROOT / seq / "baseline"    / "events.txt"
    e_path    = OUT_ROOT / seq / "experimental"/ "events.txt"

    if not real_path.exists():
        print(f"  {seq}: real events not found, skipping")
        continue
    if not b_path.exists() or not e_path.exists():
        print(f"  {seq}: v2e output not found (may not have run yet)")
        continue

    # find t0 for real events (epoch timestamps)
    t0 = None
    with open(real_path) as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                t0 = float(line.split()[0])
                break

    r_on, r_off = load_events(real_path, t_offset=t0, t_max=STOP)
    b_on, b_off = load_events(b_path,    t_max=STOP)
    e_on, e_off = load_events(e_path,    t_max=STOP)

    r_n = r_on + r_off
    b_n = b_on + b_off
    e_n = e_on + e_off

    r_ratio = r_on / r_off if r_off else float('inf')
    b_ratio = b_on / b_off if b_off else float('inf')
    e_ratio = e_on / e_off if e_off else float('inf')

    b_cnt_pct = (b_n - r_n) / r_n * 100 if r_n else float('inf')
    e_cnt_pct = (e_n - r_n) / r_n * 100 if r_n else float('inf')

    b_ratio_d = b_ratio - r_ratio
    e_ratio_d = e_ratio - r_ratio

    winner = "EXP" if abs(e_ratio_d) < abs(b_ratio_d) else "BASE"

    rows.append((seq, r_n, r_ratio, b_n, b_cnt_pct, b_ratio, b_ratio_d,
                          e_n, e_cnt_pct, e_ratio, e_ratio_d, winner))

if not rows:
    print("No results available yet.")
    sys.exit(0)

# ── Per-sequence table ──────────────────────────────────────────────────────
hdr = f"{'Sequence':<22}  {'Real N':>8}  {'Real R':>6}  {'Base N':>9}  {'Cnt%':>6}  {'B R':>6}  {'ΔR':>6}  {'Exp N':>9}  {'Cnt%':>6}  {'E R':>6}  {'ΔR':>6}  {'Winner'}"
print(hdr)
print("-" * len(hdr))
for (seq, r_n, r_ratio, b_n, b_cnt_pct, b_ratio, b_ratio_d,
          e_n, e_cnt_pct, e_ratio, e_ratio_d, winner) in rows:
    print(f"{seq:<22}  {r_n:>8,}  {r_ratio:>6.3f}"
          f"  {b_n:>9,}  {b_cnt_pct:>+6.1f}%  {b_ratio:>6.3f}  {b_ratio_d:>+6.3f}"
          f"  {e_n:>9,}  {e_cnt_pct:>+6.1f}%  {e_ratio:>6.3f}  {e_ratio_d:>+6.3f}"
          f"  {winner}")

# ── Summary ─────────────────────────────────────────────────────────────────
exp_wins = sum(1 for r in rows if r[-1] == "EXP")
base_wins = len(rows) - exp_wins
print()
print(f"ON/OFF ratio closer to real sensor:  Experimental {exp_wins}/{len(rows)}"
      f"  vs  Baseline {base_wins}/{len(rows)}")

avg_b_delta = sum(abs(r[6]) for r in rows) / len(rows)
avg_e_delta = sum(abs(r[10]) for r in rows) / len(rows)
print(f"Mean |ΔON/OFF|:  Baseline {avg_b_delta:.4f}  Experimental {avg_e_delta:.4f}")
if avg_e_delta < avg_b_delta:
    print("==> Experimental realism controls IMPROVE ON/OFF fidelity on average.")
else:
    print("==> Baseline ON/OFF fidelity is better or equal on average.")

avg_b_cnt = sum(r[4] for r in rows) / len(rows)
avg_e_cnt = sum(r[8] for r in rows) / len(rows)
print(f"Mean count delta vs real:  Baseline {avg_b_cnt:+.1f}%  Experimental {avg_e_cnt:+.1f}%")
PYEOF

echo ""
echo "=== Done at $(date) ==="
touch /root/DONE
