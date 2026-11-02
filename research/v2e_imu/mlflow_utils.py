from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]


def _load_mlflow() -> Any | None:
    try:
        import mlflow
    except Exception:
        return None
    return mlflow


_MLFLOW = _load_mlflow()


def mlflow_is_enabled() -> bool:
    return _MLFLOW is not None and bool(os.getenv("MLFLOW_TRACKING_URI", "").strip())


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def default_tags() -> dict[str, str]:
    tags = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }

    git_sha = _git_value("rev-parse", "--short", "HEAD")
    git_branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    if git_sha:
        tags["git.commit"] = git_sha
    if git_branch:
        tags["git.branch"] = git_branch

    for env_name, tag_name in {
        "VAST_CONTAINERLABEL": "vast.container_label",
        "VAST_MACHINE_ID": "vast.machine_id",
        "VAST_INSTANCE_ID": "vast.instance_id",
    }.items():
        value = os.getenv(env_name, "").strip()
        if value:
            tags[tag_name] = value

    return tags


class MlflowRunManager:
    """Small optional MLflow wrapper for local/Vast.ai experiments."""

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        self._mlflow = _MLFLOW
        self.enabled = mlflow_is_enabled()
        self.run_name = run_name or os.getenv("MLFLOW_RUN_NAME") or None
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "v2e-imu-autoresearch"
        )
        self._active = False

    def start(
        self,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        if not self.enabled or self._mlflow is None:
            return

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            return

        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(self.experiment_name)
        self._mlflow.start_run(run_name=self.run_name)
        self._active = True

        merged_tags = default_tags()
        if tags:
            merged_tags.update(tags)
        if merged_tags:
            self._mlflow.set_tags(merged_tags)

        if params:
            clean_params = {k: _param_value(v) for k, v in params.items()}
            self._mlflow.log_params(clean_params)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.enabled and self._active and self._mlflow is not None and tags:
            self._mlflow.set_tags(tags)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or not self._active or self._mlflow is None or not metrics:
            return

        clean_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not clean_metrics:
            return

        if step is None:
            self._mlflow.log_metrics(clean_metrics)
        else:
            self._mlflow.log_metrics(clean_metrics, step=step)

    def log_artifacts(self, paths: Iterable[Path | str]) -> None:
        if not self.enabled or not self._active or self._mlflow is None:
            return

        for path_like in paths:
            path = Path(path_like)
            if path.exists():
                self._mlflow.log_artifact(str(path))

    def log_text(self, text: str, artifact_file: str) -> None:
        if self.enabled and self._active and self._mlflow is not None:
            self._mlflow.log_text(text, artifact_file)

    def end(self, status: str = "FINISHED") -> None:
        if self.enabled and self._active and self._mlflow is not None:
            self._mlflow.end_run(status=status)
            self._active = False
