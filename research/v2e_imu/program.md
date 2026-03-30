# Program

## IMU-Enhanced Event Camera Research

Autonomous experiment loop for optimizing event prediction using RGB + IMU data.
Based on the autoresearch-mlx pattern: fixed-time experiments, single metric, git-based version control.

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g., `mar30`). The branch `research/` must not already exist — this is a fresh run.

2. **Create the branch**: `git checkout -b research/` from current master.

3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context
   - `prepare_data.py` — fixed constants, data loading, evaluation. **DO NOT MODIFY**
   - `train.py` — the file you edit. Model architecture, optimizer, training loop

4. **Verify data exists**: Check that `data/fpv/` contains data or synthetic data. If not, tell the human to run `python download_fpv.py --synthetic`.

5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `python train.py` once to establish YOUR baseline on this hardware. Do NOT use baseline numbers from other platforms.

6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time). You launch it as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare_data.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already available.
- Modify the evaluation harness. The `evaluate_event_bpb` function in `prepare_data.py` is the ground truth metric.

**The goal is simple: get the lowest event_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. MLX uses unified memory shared between CPU and GPU. Some increase is acceptable for meaningful event_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 event_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 event_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
event_bpb: 0.123456
event_mse: 0.045678
training_seconds: 600.0
total_seconds: 645.2
peak_vram_mb: 4096.0
samples_per_sec: 150.5
num_steps: 1234
num_params_M: 2.45
base_channels: 32
imu_hidden_dim: 128
```

Note that the script runs for a fixed 10-minute training budget. On Apple Silicon the throughput, step count, and absolute event_bpb will differ from other results — that's expected. Compare only against your own baseline on the same hardware.

```
grep "^event_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	event_bpb	peak_memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. event_bpb achieved (e.g., 0.123456) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g., 4.2 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	event_bpb	peak_memory_gb	status	description
abc1234	0.150000	4.2	keep	baseline
def5678	0.145000	4.2	keep	increase base_channels to 48
ghi9012	0.160000	4.2	discard	larger model, same time budget
```

## The experiment loop

The experiment runs on a dedicated branch (e.g., `research/mar30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. `git add research/v2e_imu/train.py && git commit -m "experiment: "` (never `git add -A` — this may be inside a larger repo)
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^event_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If event_bpb improved (lower), `git add research/v2e_imu/results.tsv && git commit --amend --no-edit` to include the log, advancing the branch
9. If event_bpb is equal or worse, record the discard commit hash, then `git reset --hard ` to discard it cleanly

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~12 minutes total (10 min training + ~1 min eval overhead on Apple Silicon). If a run exceeds 15 minutes, kill it and treat as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g., a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until they manually stop you. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~12 minutes then you can run approx 5/hour, for a total of about 40 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Architecture Ideas to Explore

When you run out of obvious ideas, consider:

1. **Deeper vs wider**: More layers vs more channels per layer
2. **Attention mechanisms**: Self-attention, cross-attention between IMU and RGB
3. **Temporal modeling**: Different sequence lengths, GRU vs LSTM, TCN
4. **Feature modulation**: Different ways to condition RGB on IMU (FiLM, SPADE, etc.)
5. **Loss functions**: L1, Huber, perceptual losses, adversarial losses
6. **Data augmentation**: Random crops, flips, color jitter, IMU noise
7. **Optimizer choices**: Adam, AdamW, SGD with momentum, Lion, etc.
8. **Learning rate schedules**: Cosine, linear, step, warmup strategies
9. **Normalization**: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
10. **Activation functions**: ReLU, GELU, SiLU, Mish

Good luck!
