"""Shared helpers for the EEGConformer training package."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = TRAINING_ROOT / "configs" / "eegconformer-bciiv2a.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass
class Paths:
    cache: Path
    processed: Path
    artefacts: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Paths":
        name = cfg["name"]
        root = TRAINING_ROOT
        p = cls(
            cache=root / "cache" / "moabb",
            processed=root / "cache" / "processed" / name,
            artefacts=root / "artefacts" / name,
        )
        for d in (p.cache, p.processed, p.artefacts):
            d.mkdir(parents=True, exist_ok=True)
        return p