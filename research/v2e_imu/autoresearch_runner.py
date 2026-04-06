from __future__ import annotations

import argparse
import base64
import contextlib
import datetime
import email.utils
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = Path(__file__).resolve().parent
TRAIN_FILE = RESEARCH_DIR / "train.py"
RESULTS_TSV = RESEARCH_DIR / "results.tsv"
RUN_LOG = RESEARCH_DIR / "run.log"
DEFAULT_ENV_FILE = RESEARCH_DIR / ".env.vastai.local"
DEFAULT_TIMEOUT_SECONDS = 1200
DEFAULT_PLAN_POLL_SECONDS = 300
DEFAULT_REMOTE = "origin"
DEFAULT_PYTHON = sys.executable

BASELINE_CONSTANTS: dict[str, str] = {
    "MODEL_TYPE": '"unet"',
    "BASE_CHANNELS": "32",
    "IMU_HIDDEN_DIM": "128",
    "TOTAL_BATCH_SIZE": "4",
    "DEVICE_BATCH_SIZE": "4",
    "LEARNING_RATE": "2e-3",
    "WEIGHT_DECAY": "0.0",
    "WARMUP_RATIO": "0.1",
    "WARMDOWN_RATIO": "0.3",
    "FINAL_LR_FRAC": "0.01",
}

DEFAULT_PLAN: list[dict[str, Any]] = [
    {
        "name": "AdamW weight_decay=1e-4 on restored baseline",
        "description": "Test mild AdamW regularization on the restored batch=4 LR=2e-3 baseline.",
        "commit_message": "probe: AdamW weight_decay=1e-4",
        "constants": {
            "WEIGHT_DECAY": "1e-4",
        },
    },
    {
        "name": "Scheduler warmup 5% warmdown 50% on restored baseline",
        "description": "Retest the strongest scheduler shape on the restored batch=4 LR=2e-3 baseline.",
        "commit_message": "probe: warmup 5% warmdown 50% restored baseline",
        "constants": {
            "WARMUP_RATIO": "0.05",
            "WARMDOWN_RATIO": "0.5",
            "FINAL_LR_FRAC": "0.01",
        },
    },
    {
        "name": "LR 2.5e-3 on restored baseline",
        "description": "Test a slightly faster LR on the best batch=4 configuration.",
        "commit_message": "probe: LR 2.5e-3 on batch=4 baseline",
        "constants": {
            "LEARNING_RATE": "2.5e-3",
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
    search_strategy: str = "plan"
    metadata: dict[str, str] = field(default_factory=dict)


DEFAULT_OPTUNA_STORAGE_URL = f"sqlite:///{(RESEARCH_DIR / 'autoresearch_optuna.db').as_posix()}"


@dataclass(frozen=True)
class OptunaConfig:
    enabled: bool = True
    study_name: str = "v2e-imu-autoresearch"
    storage_url: str = DEFAULT_OPTUNA_STORAGE_URL
    max_generated: int = 1
    sampler_seed: int = 42
    batch_choices: tuple[str, ...] = ("4", "8", "16")
    base_channel_choices: tuple[str, ...] = ("32", "48", "64")
    imu_hidden_choices: tuple[str, ...] = ("128", "160", "192")
    lr_low: float = 5e-4
    lr_high: float = 4e-3


class NoOpExperimentError(RuntimeError):
    """Raised when an experiment would not change the current baseline."""


def default_branch_name() -> str:
    return os.getenv("RESEARCH_BRANCH", f"research/vastai-{time.strftime('%b%d').lower()}")


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


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


def _warn_command_failure(action: str, result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "").strip().replace("\n", " | ")
    if detail:
        if len(detail) > 400:
            detail = detail[:400] + "…"
        print(f"WARNING: {action} failed (exit {result.returncode}): {detail}")
    else:
        print(f"WARNING: {action} failed (exit {result.returncode})")


def _measure_aws_clock_offset_seconds() -> int:
    request = urllib.request.Request(
        "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15"
    )
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:
        response = exc
    except Exception as exc:
        print(f"WARNING: could not measure AWS clock skew: {exc}")
        return 0

    date_header = response.headers.get("Date")
    if not date_header:
        return 0

    try:
        server_time = email.utils.parsedate_to_datetime(date_header)
        if server_time.tzinfo is None:
            server_time = server_time.replace(tzinfo=datetime.timezone.utc)
        local_time = datetime.datetime.now(datetime.timezone.utc)
        offset_seconds = int((server_time - local_time).total_seconds())
        if abs(offset_seconds) >= 30:
            print(f"Applying AWS clock skew correction: {offset_seconds}s")
        return offset_seconds
    except Exception as exc:
        print(f"WARNING: could not parse AWS Date header: {exc}")
        return 0


@contextlib.contextmanager
def _patched_botocore_clock(offset_seconds: int) -> Iterator[None]:
    if abs(offset_seconds) < 30:
        yield
        return

    try:
        import botocore.auth
        import botocore.compat
        import botocore.endpoint
        import botocore.signers
    except Exception as exc:
        print(f"WARNING: could not patch botocore clock: {exc}")
        yield
        return

    modules: list[Any] = [
        botocore.compat,
        botocore.auth,
        botocore.endpoint,
        botocore.signers,
    ]
    try:
        import botocore.crt.auth as botocore_crt_auth

        modules.append(botocore_crt_auth)
    except Exception:
        pass

    originals = {
        module: module.get_current_datetime
        for module in modules
        if hasattr(module, "get_current_datetime")
    }

    def skewed_now(remove_tzinfo: bool = True) -> datetime.datetime:
        now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            seconds=offset_seconds
        )
        return now.replace(tzinfo=None) if remove_tzinfo else now

    for module in originals:
        module.get_current_datetime = skewed_now

    try:
        yield
    finally:
        for module, original in originals.items():
            module.get_current_datetime = original


def aws_identity_preflight_ok() -> bool:
    try:
        import boto3
    except Exception as exc:
        print(f"WARNING: boto3 unavailable for AWS preflight: {exc}")
        return False

    try:
        offset_seconds = _measure_aws_clock_offset_seconds()
        with _patched_botocore_clock(offset_seconds):
            identity = boto3.client("sts").get_caller_identity()
        print(f"AWS identity OK: {identity.get('Arn', 'unknown')}")
        return True
    except Exception as exc:
        print(f"WARNING: AWS STS validation failed: {exc}")
        return False


def _parse_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"Expected s3:// URL, got: {url}")
    bucket_and_key = url[5:]
    bucket, sep, key = bucket_and_key.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"Expected full s3://bucket/key URL, got: {url}")
    return bucket, key


def _upload_file_to_s3(artifact: Path, target: str, offset_seconds: int) -> bool:
    try:
        import boto3
    except Exception as exc:
        print(f"WARNING: boto3 unavailable for S3 sync: {exc}")
        return False

    try:
        bucket, key = _parse_s3_url(target)
        with _patched_botocore_clock(offset_seconds):
            boto3.client("s3").upload_file(str(artifact), bucket, key)
        print(f"Synced {artifact.name} to {target}")
        return True
    except Exception as exc:
        print(f"WARNING: S3 sync for {artifact.name} failed: {exc}")
        return False


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


def completed_experiment_descriptions() -> set[str]:
    ensure_results_tsv()
    completed: set[str] = set()
    for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        _, _, _, status, description = parts[:5]
        if status in {"keep", "discard", "crash", "skip"} and description:
            completed.add(description)
            if status == "crash" and description.endswith(" failed"):
                completed.add(description[: -len(" failed")])
    return completed


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
        raise NoOpExperimentError(f"No staged changes detected for experiment: {exp.name}")
    git("commit", "-m", exp.commit_message)
    return short_commit()


def _load_plan_payload(path: Path | None) -> Any:
    if path is not None and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"experiments": DEFAULT_PLAN}


def load_plan(path: Path | None) -> list[Experiment]:
    raw_plan = _load_plan_payload(path)
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
                search_strategy=item.get("search_strategy", "plan"),
                metadata={str(k): str(v) for k, v in item.get("metadata", {}).items()},
            )
        )
    return experiments


def load_optuna_config(path: Path | None) -> OptunaConfig:
    raw_plan = _load_plan_payload(path)
    raw_config = raw_plan.get("optuna", {}) if isinstance(raw_plan, dict) else {}

    env_enabled = os.getenv("AUTORESEARCH_OPTUNA_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    def tuple_of_strings(values: Any, default: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            return default
        return tuple(str(v) for v in values)

    raw_enabled = raw_config.get("enabled", env_enabled)
    enabled = (
        raw_enabled
        if isinstance(raw_enabled, bool)
        else str(raw_enabled).strip().lower() not in {"0", "false", "no"}
    )

    return OptunaConfig(
        enabled=enabled,
        study_name=str(raw_config.get("study_name", "v2e-imu-autoresearch")),
        storage_url=str(raw_config.get("storage_url", DEFAULT_OPTUNA_STORAGE_URL)),
        max_generated=max(1, int(raw_config.get("max_generated", 1))),
        sampler_seed=int(raw_config.get("sampler_seed", 42)),
        batch_choices=tuple_of_strings(raw_config.get("batch_choices"), ("4", "8", "16")),
        base_channel_choices=tuple_of_strings(
            raw_config.get("base_channel_choices"), ("32", "48", "64")
        ),
        imu_hidden_choices=tuple_of_strings(
            raw_config.get("imu_hidden_choices"), ("128", "160", "192")
        ),
        lr_low=float(raw_config.get("lr_low", 5e-4)),
        lr_high=float(raw_config.get("lr_high", 4e-3)),
    )


def _format_float_literal(value: float) -> str:
    return f"{value:.6g}"


def build_training_env(exp: Experiment, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ.copy())
    env.setdefault("MLFLOW_RUN_NAME", exp.name)
    env["AUTORESEARCH_EXPERIMENT_NAME"] = exp.name
    env["AUTORESEARCH_EXPERIMENT_DESCRIPTION"] = exp.description
    env["AUTORESEARCH_SEARCH_KIND"] = exp.search_strategy

    trial_number = exp.metadata.get("trial_number", "")
    if trial_number:
        env["AUTORESEARCH_TRIAL_NUMBER"] = trial_number
    else:
        env.pop("AUTORESEARCH_TRIAL_NUMBER", None)
    return env


def build_optuna_experiment(
    config: OptunaConfig, completed_descriptions: set[str]
) -> Experiment | None:
    if not config.enabled:
        return None

    try:
        import optuna
    except Exception as exc:
        print(f"WARNING: Optuna not available; skipping generated experiments: {exc}")
        return None

    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=config.study_name,
        storage=config.storage_url,
        load_if_exists=True,
        sampler=sampler,
    )

    for _ in range(max(1, config.max_generated * 4)):
        trial = study.ask()
        total_batch = int(trial.suggest_categorical("TOTAL_BATCH_SIZE", list(config.batch_choices)))
        device_batch_choices = [int(v) for v in config.batch_choices if int(v) <= total_batch]
        device_batch = int(
            trial.suggest_categorical(
                "DEVICE_BATCH_SIZE",
                device_batch_choices or [total_batch],
            )
        )
        base_channels = int(
            trial.suggest_categorical("BASE_CHANNELS", list(config.base_channel_choices))
        )
        imu_hidden = int(
            trial.suggest_categorical("IMU_HIDDEN_DIM", list(config.imu_hidden_choices))
        )
        learning_rate = trial.suggest_float(
            "LEARNING_RATE", config.lr_low, config.lr_high, log=True
        )
        weight_decay = trial.suggest_categorical("WEIGHT_DECAY", [0.0, 1e-5, 1e-4, 3e-4])
        warmup_ratio = trial.suggest_float("WARMUP_RATIO", 0.03, 0.15)
        warmdown_ratio = trial.suggest_float("WARMDOWN_RATIO", 0.2, 0.6)
        final_lr_frac = trial.suggest_float("FINAL_LR_FRAC", 0.003, 0.05, log=True)

        constants = {
            "TOTAL_BATCH_SIZE": str(total_batch),
            "DEVICE_BATCH_SIZE": str(device_batch),
            "BASE_CHANNELS": str(base_channels),
            "IMU_HIDDEN_DIM": str(imu_hidden),
            "LEARNING_RATE": _format_float_literal(learning_rate),
            "WEIGHT_DECAY": _format_float_literal(weight_decay),
            "WARMUP_RATIO": _format_float_literal(warmup_ratio),
            "WARMDOWN_RATIO": _format_float_literal(warmdown_ratio),
            "FINAL_LR_FRAC": _format_float_literal(final_lr_frac),
        }
        name = (
            f"Optuna trial {trial.number}: batch={total_batch}, dev={device_batch}, "
            f"ch={base_channels}, lr={constants['LEARNING_RATE']}"
        )
        description = (
            f"Optuna trial {trial.number} exploring batch={total_batch}, device_batch={device_batch}, "
            f"channels={base_channels}, imu_hidden={imu_hidden}, lr={constants['LEARNING_RATE']}, "
            f"wd={constants['WEIGHT_DECAY']}."
        )
        if description in completed_descriptions:
            study.tell(trial, 0.0)
            continue

        trial.set_user_attr("name", name)
        trial.set_user_attr("description", description)
        return Experiment(
            name=name,
            description=description,
            commit_message=f"probe: optuna trial {trial.number}",
            constants=constants,
            search_strategy="optuna",
            metadata={
                "trial_number": str(trial.number),
                "study_name": config.study_name,
                "storage_url": config.storage_url,
            },
        )

    return None


def maybe_record_optuna_result(exp: Experiment, ok: bool, val_ap: float | None) -> None:
    if exp.search_strategy != "optuna":
        return

    try:
        import optuna
    except Exception:
        return

    trial_number_raw = exp.metadata.get("trial_number")
    storage_url = exp.metadata.get("storage_url")
    study_name = exp.metadata.get("study_name")
    if not trial_number_raw or not storage_url or not study_name:
        return

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    objective = float(val_ap or 0.0)
    state = (
        optuna.trial.TrialState.COMPLETE
        if ok and val_ap is not None
        else optuna.trial.TrialState.FAIL
    )
    try:
        study.tell(int(trial_number_raw), objective, state=state)
    except ValueError:
        pass


def run_training(exp: Experiment, timeout_seconds: int, python_bin: str) -> tuple[bool, str]:
    env = build_training_env(exp)
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
    offset_seconds = _measure_aws_clock_offset_seconds()
    for artifact in (RESULTS_TSV, RUN_LOG, RESEARCH_DIR / "event_predictor_checkpoint.pt"):
        if artifact.exists():
            target = f"{s3_prefix}/{experiment_sha}/{artifact.name}"
            _upload_file_to_s3(artifact, target, offset_seconds)


def maybe_push_to_github(remote: str, branch: str) -> None:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN not set; skipping git push")
        return

    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    git_auth = ["git", "-c", f"http.extraheader=AUTHORIZATION: basic {auth}"]
    remote_ref = f"{remote}/{branch}"

    fetch_result = run(
        [*git_auth, "fetch", remote, f"{branch}:refs/remotes/{remote}/{branch}"],
        cwd=ROOT,
        check=False,
    )
    remote_exists = (
        git("show-ref", "--verify", "--quiet", f"refs/remotes/{remote_ref}", check=False).returncode
        == 0
    )
    if fetch_result.returncode != 0 and remote_exists:
        _warn_command_failure(f"git fetch {remote}/{branch}", fetch_result)

    if remote_exists:
        ancestor_check = git("merge-base", "--is-ancestor", remote_ref, "HEAD", check=False)
        if ancestor_check.returncode != 0:
            rebase_result = run([*git_auth, "rebase", remote_ref], cwd=ROOT, check=False)
            if rebase_result.returncode != 0:
                _warn_command_failure(f"git rebase {remote_ref}", rebase_result)
                git("rebase", "--abort", check=False)
                return

    push_result = run([*git_auth, "push", remote, f"HEAD:{branch}"], cwd=ROOT, check=False)
    if push_result.returncode == 0:
        print(f"Pushed latest result to {remote}/{branch}")
    else:
        _warn_command_failure(f"git push to {remote}/{branch}", push_result)


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
        description = exp.description
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

    maybe_record_optuna_result(exp, ok, val_ap)
    maybe_sync_to_s3(experiment_sha)
    if push:
        maybe_push_to_github(remote, branch)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    pre_args, remaining = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        description="Run the v2e IMU autoresearch loop outside the notebook."
    )
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument(
        "--plan", type=Path, default=None, help="Path to the experiment plan JSON file."
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TRAIN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
    )
    parser.add_argument("--python-bin", default=os.getenv("PYTHON_BIN", DEFAULT_PYTHON))
    parser.add_argument("--remote", default=os.getenv("GITHUB_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--branch", default=default_branch_name())
    parser.add_argument(
        "--plan-poll-seconds",
        type=int,
        default=int(os.getenv("PLAN_POLL_SECONDS", str(DEFAULT_PLAN_POLL_SECONDS))),
        help="Seconds to sleep before re-checking the plan for new work. Set to 0 to exit once the current plan is exhausted.",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push results back to GitHub after each run."
    )
    args = parser.parse_args(remaining)

    os.chdir(ROOT)
    ensure_git_identity()
    ensure_results_tsv()

    if current_branch() != args.branch:
        print(f"Using branch {current_branch()} (push target {args.branch})")

    while True:
        experiments = load_plan(args.plan)
        optuna_config = load_optuna_config(args.plan)

        completed = completed_experiment_descriptions()
        pending = [exp for exp in experiments if exp.description not in completed]

        if not pending:
            generated = build_optuna_experiment(optuna_config, completed)
            if generated is not None:
                pending = [generated]
                print(f"Generated new Optuna trial: {generated.name}")

        if not pending:
            if args.plan_poll_seconds <= 0:
                print("No pending experiments in the plan. Exiting.")
                break
            print(
                f"No pending experiments in the plan; sleeping {args.plan_poll_seconds}s before retrying."
            )
            time.sleep(args.plan_poll_seconds)
            continue

        for exp in pending:
            print(f"=== Running experiment [{exp.search_strategy}]: {exp.name} ===")
            apply_experiment(exp)
            try:
                experiment_sha = commit_experiment(exp)
            except NoOpExperimentError as exc:
                append_result(short_commit(), 0.0, 0.0, "skip", exp.description)
                print(f"SKIP {exp.name}: {exc}")
                continue
            ok, log_text = run_training(exp, args.timeout_seconds, args.python_bin)
            val_ap, peak_vram = parse_run_metrics(log_text)
            log_and_finalize(
                exp,
                experiment_sha,
                ok,
                val_ap,
                peak_vram,
                args.push,
                args.remote,
                args.branch,
            )

        if args.plan_poll_seconds <= 0:
            break


if __name__ == "__main__":
    main()
