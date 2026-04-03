from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = Path(__file__).resolve().parent
TRAIN_FILE = RESEARCH_DIR / "train.py"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
RUN_LOG = RESEARCH_DIR / "run.log"
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("TRAIN_TIMEOUT_SECONDS", "1200"))
DEFAULT_REMOTE = os.getenv("GITHUB_REMOTE", "origin")
DEFAULT_BRANCH = os.getenv("RESEARCH_BRANCH", f"research/vastai-{time.strftime('%b%d').lower()}")
DEFAULT_PYTHON = os.getenv("PYTHON_BIN", sys.executable)

BASELINE_CONSTANTS: dict[str, str] = {
    "MODEL_TYPE": '"unet"',
    "BASE_CHANNELS": "32",
    "IMU_HIDDEN_DIM": "128",
    "TOTAL_BATCH_SIZE": "4",
    "DEVICE_BATCH_SIZE": "4",
    "LEARNING_RATE": "1e-3",
    "WEIGHT_DECAY": "0.0",
    "WARMUP_RATIO": "0.1",
    "WARMDOWN_RATIO": "0.3",
    "FINAL_LR_FRAC": "0.01",
}

DEFAULT_PLAN: list[dict[str, Any]] = [
    {
        "name": "AdamW weight_decay=1e-2",
        "description": "Best batch-size configuration with stronger AdamW regularization.",
        "commit_message": "probe: AdamW weight_decay=1e-2",
        "constants": {
            "WEIGHT_DECAY": "1e-2",
        },
    },
    {
        "name": "Scheduler warmup 5% warmdown 50%",
        "description": "Retest the strongest scheduler shape on the best batch-size/LR baseline.",
        "commit_message": "probe: warmup 5% warmdown 50%",
        "constants": {
            "WARMUP_RATIO": "0.05",
            "WARMDOWN_RATIO": "0.5",
            "FINAL_LR_FRAC": "0.01",
        },
    },
]


@dataclass(frozen=True)
class Replacement:
    pattern: str
    replacement: str
    regex: bool = False
    dotall: bool = False


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    commit_message: str
    constants: dict[str, str] = field(default_factory=dict)
    replacements: tuple[Replacement, ...] = ()


def run(
    args: list[str],
    cwd: Path = ROOT,
    check: bool = True,
    env: dict[str, str] | None = None,
    stdout: Any | None = None,
    stderr: Any | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=(stdout is None and stderr is None),
        stdout=stdout,
        stderr=stderr,
        env=env,
        timeout=timeout,
    )


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=ROOT, check=check)


def ensure_git_identity() -> None:
    if not git("config", "user.name", check=False).stdout.strip():
        git("config", "user.name", os.getenv("GIT_AUTHOR_NAME", "vastai-autoresearch"))
    if not git("config", "user.email", check=False).stdout.strip():
        git(
            "config",
            "user.email",
            os.getenv("GIT_AUTHOR_EMAIL", "vastai-autoresearch@example.com"),
        )


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def short_commit() -> str:
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def ensure_results_tsv() -> None:
    if not RESULTS_TSV.exists() or not RESULTS_TSV.read_text(encoding="utf-8").strip():
        RESULTS_TSV.write_text(
            "commit\tval_ap\tpeak_memory_gb\tstatus\tdescription\n",
            encoding="utf-8",
        )


def best_keep_val_ap() -> float:
    ensure_results_tsv()
    best = 0.0
    for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        _, val_ap, _, status, _ = parts[:5]
        if status == "keep":
            try:
                best = max(best, float(val_ap))
            except ValueError:
                continue
    return best


def append_result(
    commit: str, val_ap: float, peak_vram_mb: float, status: str, description: str
) -> None:
    ensure_results_tsv()
    peak_gb = round((peak_vram_mb or 0.0) / 1024.0, 1)
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write(f"{commit}\t{val_ap:.6f}\t{peak_gb:.1f}\t{status}\t{description}\n")


def update_last_result_commit(old_sha: str, new_sha: str) -> None:
    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].startswith(f"{old_sha}\t"):
            lines[idx] = lines[idx].replace(f"{old_sha}\t", f"{new_sha}\t", 1)
            RESULTS_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return


def parse_run_metrics(log_text: str) -> tuple[float | None, float | None]:
    val_ap_match = re.search(r"^val_ap:\s*([0-9]*\.?[0-9]+)", log_text, flags=re.MULTILINE)
    vram_match = re.search(r"^peak_vram_mb:\s*([0-9]*\.?[0-9]+)", log_text, flags=re.MULTILINE)
    val_ap = float(val_ap_match.group(1)) if val_ap_match else None
    peak_vram = float(vram_match.group(1)) if vram_match else None
    return val_ap, peak_vram


def set_constant(text: str, name: str, value_expr: str) -> str:
    pattern = rf"^({re.escape(name)}\s*=\s*).*$"
    updated, count = re.subn(pattern, rf"\g<1>{value_expr}", text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Could not find constant {name} in {TRAIN_FILE}")
    return updated


def restore_baseline(text: str) -> str:
    for name, value in BASELINE_CONSTANTS.items():
        text = set_constant(text, name, value)

    text = re.sub(
        r"(def __init__\(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = )\d+(\) -> None:)",
        r"\g<1>2\g<2>",
        text,
    )
    text = text.replace(
        "            modulated.append(f + bias)", "            modulated.append(f * scale + bias)"
    )
    text = text.replace(
        "            modulated.append(f * scale)", "            modulated.append(f * scale + bias)"
    )
    text = text.replace("    alpha: float = 0.85,", "    alpha: float = 0.75,")
    return text


def apply_experiment(exp: Experiment) -> None:
    text = TRAIN_FILE.read_text(encoding="utf-8")
    updated = restore_baseline(text)

    for name, value in exp.constants.items():
        updated = set_constant(updated, name, value)

    for replacement in exp.replacements:
        if replacement.regex:
            flags = re.MULTILINE | (re.DOTALL if replacement.dotall else 0)
            updated, count = re.subn(
                replacement.pattern, replacement.replacement, updated, flags=flags
            )
            if count == 0:
                raise ValueError(
                    f"Pattern not found for experiment {exp.name}: {replacement.pattern}"
                )
        else:
            if replacement.pattern not in updated:
                raise ValueError(
                    f"Literal not found for experiment {exp.name}: {replacement.pattern}"
                )
            updated = updated.replace(replacement.pattern, replacement.replacement)

    TRAIN_FILE.write_text(updated, encoding="utf-8")


def commit_experiment(exp: Experiment) -> str:
    git("add", str(TRAIN_FILE.relative_to(ROOT)))
    diff = git("diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        raise RuntimeError(f"No staged changes detected for experiment: {exp.name}")
    git("commit", "-m", exp.commit_message)
    return short_commit()


def load_plan(path: Path | None) -> list[Experiment]:
    raw_plan: Any
    if path is not None and path.exists():
        raw_plan = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw_plan = DEFAULT_PLAN

    experiments_data = (
        raw_plan.get("experiments", raw_plan) if isinstance(raw_plan, dict) else raw_plan
    )
    experiments: list[Experiment] = []
    for item in experiments_data:
        replacements = tuple(
            Replacement(
                pattern=rep["pattern"],
                replacement=rep["replacement"],
                regex=bool(rep.get("regex", False)),
                dotall=bool(rep.get("dotall", False)),
            )
            for rep in item.get("replacements", [])
        )
        experiments.append(
            Experiment(
                name=item["name"],
                description=item.get("description", item["name"]),
                commit_message=item.get("commit_message", f"probe: {item['name']}"),
                constants=dict(item.get("constants", {})),
                replacements=replacements,
            )
        )
    return experiments


def run_training(exp: Experiment, timeout_seconds: int, python_bin: str) -> tuple[bool, str]:
    env = os.environ.copy()
    env.setdefault("MLFLOW_RUN_NAME", exp.name)
    with RUN_LOG.open("w", encoding="utf-8") as handle:
        try:
            result = run(
                [python_bin, "train.py"],
                cwd=RESEARCH_DIR,
                check=False,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
            handle.write(f"\nTIMEOUT after {timeout_seconds}s\n")
    return ok, RUN_LOG.read_text(encoding="utf-8", errors="replace") if RUN_LOG.exists() else ""


def maybe_sync_to_s3(experiment_sha: str) -> None:
    s3_prefix = os.getenv("S3_RESULTS_PREFIX", "").strip()
    if not s3_prefix:
        return

    s3_prefix = s3_prefix.rstrip("/")
    for artifact in (RESULTS_TSV, RUN_LOG, RESEARCH_DIR / "event_predictor_checkpoint.pt"):
        if artifact.exists():
            target = f"{s3_prefix}/{experiment_sha}/{artifact.name}"
            run(["aws", "s3", "cp", str(artifact), target], cwd=ROOT, check=False)


def maybe_push_to_github(remote: str, branch: str) -> None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN not set; skipping git push")
        return

    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    run(
        [
            "git",
            "-c",
            f"http.extraheader=AUTHORIZATION: basic {auth}",
            "push",
            remote,
            f"HEAD:{branch}",
        ],
        cwd=ROOT,
        check=True,
    )


def log_and_finalize(
    exp: Experiment,
    experiment_sha: str,
    ok: bool,
    val_ap: float | None,
    peak_vram: float | None,
    push: bool,
    remote: str,
    branch: str,
) -> None:
    best_before = best_keep_val_ap()

    if ok and val_ap is not None and peak_vram is not None and val_ap > best_before:
        append_result(experiment_sha, val_ap, peak_vram, "keep", exp.description)
        git("add", str(RESULTS_TSV.relative_to(ROOT)))
        git("commit", "--amend", "--no-edit")
        final_sha = short_commit()
        if final_sha != experiment_sha:
            update_last_result_commit(experiment_sha, final_sha)
            git("add", str(RESULTS_TSV.relative_to(ROOT)))
            git("commit", "--amend", "--no-edit")
            experiment_sha = short_commit()
        print(f"KEEP {experiment_sha} val_ap={val_ap:.6f} peak_vram_mb={peak_vram:.1f}")
    else:
        git("reset", "--hard", "HEAD~1")
        status = "discard" if ok and val_ap is not None else "crash"
        description = exp.description if status == "discard" else f"{exp.description} failed"
        append_result(experiment_sha, val_ap or 0.0, peak_vram or 0.0, status, description)
        message = (
            f"chore: log discard {experiment_sha} val_ap={val_ap:.4f}"
            if status == "discard" and val_ap is not None
            else f"chore: log crash {experiment_sha}"
        )
        git("add", str(RESULTS_TSV.relative_to(ROOT)))
        git("commit", "-m", message)
        if status == "discard" and val_ap is not None:
            print(f"DISCARD {experiment_sha} val_ap={val_ap:.6f} (best={best_before:.6f})")
        else:
            print(f"CRASH {experiment_sha}; see {RUN_LOG}")

    maybe_sync_to_s3(experiment_sha)
    if push:
        maybe_push_to_github(remote, branch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v2e IMU autoresearch loop outside the notebook."
    )
    parser.add_argument(
        "--plan", type=Path, default=None, help="Path to the experiment plan JSON file."
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--push", action="store_true", help="Push results back to GitHub after each run."
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    ensure_git_identity()
    ensure_results_tsv()

    if current_branch() != args.branch:
        print(f"Using branch {current_branch()} (push target {args.branch})")

    experiments = load_plan(args.plan)
    if not experiments:
        raise SystemExit("No experiments found in the plan.")

    for exp in experiments:
        print(f"=== Running experiment: {exp.name} ===")
        apply_experiment(exp)
        experiment_sha = commit_experiment(exp)
        ok, log_text = run_training(exp, args.timeout_seconds, args.python_bin)
        val_ap, peak_vram = parse_run_metrics(log_text)
        log_and_finalize(
            exp, experiment_sha, ok, val_ap, peak_vram, args.push, args.remote, args.branch
        )


if __name__ == "__main__":
    main()
