# Vast.ai + MLflow + S3 + GitHub setup

## 1. Required secrets

Set these locally before launching a Vast.ai instance:

```bash
export GITHUB_TOKEN=github_pat_xxx
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000
export S3_RESULTS_PREFIX=s3://<bucket>/v2e-autoresearch
```

## 2. GitHub authentication model

The autoresearch loop pushes with a **fine-grained GitHub PAT** or GitHub App token passed via `GITHUB_TOKEN`.

It is used through a temporary HTTP auth header:

```bash
git -c http.extraheader="AUTHORIZATION: basic ..." push origin HEAD:research/vastai-apr03
```

This avoids hardcoding the token into the repo or notebook.

## 3. Start a Vast.ai instance

Use the self-bootstrapping launcher:

```bash
bash scripts/vastai_launch_example.sh launch
```

Useful variants:

```bash
bash scripts/vastai_launch_example.sh search
bash scripts/vastai_launch_example.sh dry-run
bash scripts/vastai_launch_example.sh print-onstart
```

The launcher will:

1. auto-load `research/v2e_imu/.env.vastai.local` if present
2. auto-install the `vastai` CLI into the repo `.venv` if it is missing
3. search for a suitable GPU offer
4. create the instance
5. write `research/v2e_imu/.env.vastai.local` on the remote machine
6. run `scripts/vastai_bootstrap.sh`

## 4. Monitor the run

SSH into the instance and tail:

```bash
tail -f ~/v2e/logs/vastai-autoresearch.out
```

Or inspect MLflow and S3 artifacts.

## 5. Notes

- Winning experiments are kept and pushed to the research branch.
- Discards are logged in `results.tsv` and can also be copied to S3.
- On macOS, `prepare_data.py` now falls back to `num_workers=0` to avoid dataloader worker crashes during local testing.
- If you prefer the Vast.ai web UI, use `bash scripts/vastai_launch_example.sh print-onstart` and paste the emitted command into the instance `on-start script` field.
- Lint the automation shell scripts with:

```bash
bash scripts/lint_shell.sh
```
