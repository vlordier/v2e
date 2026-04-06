#!/usr/bin/env bash
# Run baseline vs experimental-realism v2e on UZH FPV indoor_forward_10.
# Compares synthesised events from APS frames against the real DAVIS events.
#
# Dataset:  http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/
# Sequence: indoor_forward_10  (33 s, 2412 frames, 346x260, ~30.6M real events)
set -euo pipefail
LOG=/root/v2e_run.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) starting v2e FPV run ==="

# ── system deps ────────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1 libglib2.0-0 build-essential unzip

# ── clone repo ─────────────────────────────────────────────────────────────────
cd /root
if [ ! -d v2e ]; then
  git clone --branch upgrades --depth 1 https://github.com/vlordier/v2e.git
fi
cd v2e

# ── python env ─────────────────────────────────────────────────────────────────
pip install --quiet --upgrade pip
pip install --quiet gdown tqdm Pillow
pip install --quiet -r requirements.txt
pip install --quiet -e .

# ── SuperSloMo checkpoint (151 MB, Google Drive) ───────────────────────────────
CKPT=/root/v2e/input/SuperSloMo39.ckpt
mkdir -p /root/v2e/input
if [ ! -f "$CKPT" ]; then
  echo "Downloading SuperSloMo39.ckpt..."
  gdown --id 1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx -O "$CKPT"
fi

# ── UZH FPV dataset ────────────────────────────────────────────────────────────
SEQ=indoor_forward_10
SEQ_DIR=/root/fpv/${SEQ}
mkdir -p "$SEQ_DIR"

if [ ! -f "${SEQ_DIR}/images.txt" ]; then
  echo "Downloading UZH FPV ${SEQ}..."
  ZIP="${SEQ_DIR}/${SEQ}_davis_with_gt.zip"
  wget -q --show-progress \
    "http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/${SEQ}_davis_with_gt.zip" \
    -O "$ZIP"
  echo "Extracting..."
  unzip -q "$ZIP" -d "$SEQ_DIR"
  rm "$ZIP"
fi

# ── Build AVI from sorted APS frames ───────────────────────────────────────────
# frames: img/image_0_0.png .. image_0_2411.png at 32.69 fps
# ffmpeg %d pattern exactly matches this naming
IN_VIDEO="${SEQ_DIR}/aps.avi"
if [ ! -f "$IN_VIDEO" ]; then
  echo "Building AVI from APS frames..."
  ffmpeg -y -loglevel error \
    -framerate 32.69 \
    -i "${SEQ_DIR}/img/image_0_%d.png" \
    -c:v ffv1 \
    "$IN_VIDEO"
  echo "AVI ready: $(du -sh $IN_VIDEO | cut -f1)"
fi

# ── Run time window: first 10 s (covers ~327 frames, manageable GPU time) ──────
STOP=10.0
OUT=/root/out

# ── Baseline (no realism controls, SloMo on) ───────────────────────────────────
echo "=== BASELINE run ==="
python v2e.py \
  -i "$IN_VIDEO" \
  -o "${OUT}/baseline" \
  --overwrite \
  --no_preview \
  --dvs346 \
  --skip_video_output \
  --dvs_text events.txt \
  --stop_time "$STOP" \
  --slomo_model "$CKPT"

# ── Experimental (all realism controls, SloMo on) ──────────────────────────────
# refractory_period 0.0003 s (0.3 ms) < SloMo timestamp resolution (~1 ms at 32 fps x 30x)
echo "=== EXPERIMENTAL run ==="
python v2e.py \
  -i "$IN_VIDEO" \
  -o "${OUT}/experimental" \
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

# ── Compare synthesised vs real events ─────────────────────────────────────────
echo "=== COMPARISON: synthesised vs real DAVIS events ==="
python3 - <<'PYEOF'
import sys, re, collections
from pathlib import Path

SEQ_DIR = Path("/root/fpv/indoor_forward_10")
OUT     = Path("/root/out")
STOP    = 10.0

def load_events(path, t_max=None):
    """Return (n_total, n_on, n_off) for events in a file up to t_max seconds."""
    n, on, off = 0, 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            if t_max is not None and t > t_max:
                break
            p = float(parts[3])
            n += 1
            if p > 0:
                on += 1
            else:
                off += 1
    return n, on, off

# Real events: timestamps start at epoch time, normalise to 0
real_path = SEQ_DIR / "events.txt"
with open(real_path) as f:
    for line in f:
        if not line.startswith('#'):
            t0_real = float(line.split()[0])
            break

print("Loading real events (first 10 s)...")
real_n, real_on, real_off = 0, 0, 0
with open(real_path) as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.split()
        if len(parts) < 4: continue
        t = float(parts[0]) - t0_real
        if t > STOP: break
        real_n += 1
        if float(parts[3]) > 0: real_on += 1
        else: real_off += 1
real_off = real_n - real_on

print("Loading baseline events...")
b_n, b_on, b_off = load_events(OUT / "baseline" / "events.txt")

print("Loading experimental events...")
e_n, e_on, e_off = load_events(OUT / "experimental" / "events.txt")

print()
print(f"{'Source':<20} {'Total':>10} {'ON':>10} {'OFF':>10} {'ON/OFF':>8}")
print("-" * 62)
for label, n, on, off in [
    ("Real DAVIS",   real_n, real_on, real_off),
    ("Baseline v2e", b_n, b_on, b_off),
    ("Experimental", e_n, e_on, e_off),
]:
    ratio = on/off if off else float('inf')
    print(f"{label:<20} {n:>10,} {on:>10,} {off:>10,} {ratio:>8.3f}")

print()
for label, n, on, off in [
    ("Baseline vs Real",     b_n, real_n),
    ("Experimental vs Real", e_n, real_n),
]:
    delta = (n - off) / off * 100 if off else 0  # reuse off as real_n here
    d = (n - real_n) / real_n * 100
    closer = "" if label.startswith("Exp") else ""
    print(f"{label}: {d:+.1f}% vs real event count")

print()
print("ON/OFF ratio comparison:")
real_ratio = real_on/real_off if real_off else float('inf')
b_ratio    = b_on/b_off if b_off else float('inf')
e_ratio    = e_on/e_off if e_off else float('inf')
print(f"  Real DAVIS:   {real_ratio:.3f}")
print(f"  Baseline v2e: {b_ratio:.3f}  (delta {b_ratio-real_ratio:+.3f})")
print(f"  Experimental: {e_ratio:.3f}  (delta {e_ratio-real_ratio:+.3f})")
if abs(e_ratio - real_ratio) < abs(b_ratio - real_ratio):
    print("  --> Experimental ON/OFF ratio is CLOSER to real sensor.")
else:
    print("  --> Baseline ON/OFF ratio is closer to real sensor.")
PYEOF

echo "=== Done at $(date) ==="
touch /root/DONE
