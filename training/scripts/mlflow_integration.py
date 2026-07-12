"""T-021 — MLflow tracking integration.

Wraps training runs with MLflow logging so every run records params,
metrics, artefacts, and the ONNX export hash. Uses a SQLite backend by
default (file:./mlruns.db), with an optional remote tracking URI via
the MLFLOW_TRACKING_URI environment variable.

Usage (called automatically by train.py when MLflow is installed):
    from mlflow_integration import MLflowTracker
    tracker = MLflowTracker(experiment="eegconformer-bciiv2a")
    tracker.start_run({"model": "EEGConformer", "dataset": "BCI-IV-2a"})
    tracker.log_param("lr", 1e-3)
    tracker.log_metric("val_loss", 0.45)
    tracker.log_artifact("eegconformer.pt")
    tracker.end_run()
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class MLflowTracker:
    """Thin wrapper around MLflow that degrades gracefully when not installed."""

    def __init__(self, experiment: str, tracking_uri: str | None = None):
        self.experiment = experiment
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///training/cache/mlruns.db"
        )
        self._mlflow = None
        self._run = None

        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._mlflow is not None

    def start_run(self, params: dict[str, Any] | None = None) -> None:
        if not self.available:
            return
        self._run = self._mlflow.start_run()
        if params:
            for k, v in params.items():
                self._mlflow.log_param(k, v)

    def log_param(self, key: str, value: Any) -> None:
        if self._mlflow and self._run:
            self._mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._mlflow and self._run:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow and self._run:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log an artefact and its SHA-256 hash as a tag."""
        if not self._mlflow or not self._run:
            return
        p = Path(path)
        if not p.exists():
            return
        self._mlflow.log_artifact(str(p))
        # Log the hash so the registry (T-022) can verify integrity.
        sha = _file_sha256(p)
        self._mlflow.set_tag(f"sha256_{p.name}", sha)

    def end_run(self) -> None:
        if self._mlflow and self._run:
            self._mlflow.end_run()
            self._run = None
