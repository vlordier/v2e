# IMU-Enhanced Event Camera Research

## Goal
Use IMU as extra input to RGB, and events as extra GT for training vision tasks.

## Dataset
Uses UZH FPV Drone Racing Dataset (https://fpv.ifi.uzh.ch/datasets/):
- DAVIS event camera data (events + images + IMU)
- Text format for easy parsing
- Ground truth available for some sequences

## Architecture
- RGB + IMU → Multi-modal encoder
- IMU conditions RGB processing (FiLM-style modulation)
- Events predicted as 2-channel map (positive/negative)

## Autoresearch Pattern

This research follows the [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) pattern:

- **Fixed time budget**: 10 minutes per experiment
- **Single metric**: `event_bpb` (bits per byte for event prediction)
- **Git-based version control**: Keep improvements, revert failures
- **Autonomous loop**: Run experiments until interrupted

### Files

| File | Purpose |
|------|---------|
| `prepare_data.py` | **READ-ONLY**: Data loading, evaluation, fixed constants |
| `train.py` | **EDITABLE**: Model architecture, optimizer, hyperparameters |
| `program.md` | Experiment protocol and guidelines |
| `results.tsv` | Experiment history (git commit, metric, status) |

### Quick Start

```bash
# 1. Create synthetic data for testing
python download_fpv.py --synthetic

# 2. Run baseline experiment
python train.py

# 3. Start autonomous loop
# See program.md for full protocol
```

### Download Real Data

```bash
# List available sequences
python download_fpv.py --list

# Download a small sequence with ground truth
python download_fpv.py --sequence indoor_forward_3 --output data/fpv/
```

## Experiment Loop

The autonomous experiment loop:

1. Edit `train.py` with an experimental idea
2. `git add research/v2e_imu/train.py && git commit -m "experiment: description"`
3. Run: `python train.py > run.log 2>&1`
4. Read results: `grep "^event_bpb:" run.log`
5. If improved → keep and advance
6. If not → discard and revert
7. Repeat forever

See `program.md` for detailed protocol.

## Results

Results are logged in `results.tsv`:

```
commit	event_bpb	peak_memory_gb	status	description
abc1234	0.150000	4.2	keep	baseline
```

## Architecture Ideas

When exploring, consider:
- Deeper vs wider networks
- Attention mechanisms (self-attention, cross-attention)
- Temporal modeling (GRU vs LSTM vs TCN)
- Feature modulation (FiLM, SPADE, etc.)
- Loss functions (MSE, L1, Huber, perceptual)
- Data augmentation
- Optimizer choices (Adam, AdamW, SGD, Lion, etc.)
- Learning rate schedules
- Normalization (BatchNorm, LayerNorm, GroupNorm)
- Activation functions (ReLU, GELU, SiLU, Mish)

## License

Research code. Dataset from UZH FPV (CC BY-NC-SA 3.0).
