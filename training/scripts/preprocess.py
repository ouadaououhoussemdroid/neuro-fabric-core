"""Preprocess BCI-IV-2a to the Neuro-Fabric contract.

Output: training/cache/processed/<name>/{train,holdout}.npz with
    X: float32 [N, 22, 1000]
    y: int64   [N]
    subjects: int64 [N]

Pipeline:
    1. MOABB MotorImagery paradigm (bandpass, epoching)
    2. Channel/sample sanity check vs config
    3. Per-trial z-score per channel (matches the runtime preprocessing
       contract in src/lib/eeg/preprocessing/normalize.ts)
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from _common import Paths, load_config, set_seed


def _zscore(x: np.ndarray) -> np.ndarray:
    # x: [N, C, T] -> per-trial per-channel
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    paths = Paths.from_config(cfg)
    os.environ.setdefault("MNE_DATA", str(paths.cache))
    os.environ.setdefault("MOABB_DATA", str(paths.cache))

    import moabb
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    moabb.set_log_level("WARNING")
    sig = cfg["signal"]
    paradigm = MotorImagery(
        fmin=sig["bandpass_hz"][0],
        fmax=sig["bandpass_hz"][1],
        tmin=sig["tmin_s"],
        tmax=sig["tmax_s"],
        resample=sig["sample_rate_hz"],
        n_classes=cfg["model"]["n_classes"],
    )
    dataset = BNCI2014_001()
    subjects = cfg["dataset"]["subjects"]
    holdout = set(cfg["dataset"]["holdout_subjects"])

    print("[preprocess] extracting epochs via MOABB MotorImagery paradigm")
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = np.asarray(X, dtype=np.float32)
    # Contract enforcement
    target_T = sig["window_samples"]
    if X.shape[-1] != target_T:
        # Defensive crop / pad if MOABB returns ±1 sample due to rounding.
        if X.shape[-1] > target_T:
            X = X[..., :target_T]
        else:
            pad = target_T - X.shape[-1]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="edge")
    assert X.shape[1] == sig["channels"], f"channel mismatch: {X.shape[1]} != {sig['channels']}"
    assert X.shape[2] == target_T, f"sample mismatch: {X.shape[2]} != {target_T}"

    # Label encoding
    classes = sorted(set(labels))
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.asarray([label_map[c] for c in labels], dtype=np.int64)
    subs = metadata["subject"].to_numpy().astype(np.int64)

    X = _zscore(X)

    train_mask = ~np.isin(subs, list(holdout))
    np.savez_compressed(
        paths.processed / "train.npz",
        X=X[train_mask], y=y[train_mask], subjects=subs[train_mask],
        classes=np.array(classes),
    )
    np.savez_compressed(
        paths.processed / "holdout.npz",
        X=X[~train_mask], y=y[~train_mask], subjects=subs[~train_mask],
        classes=np.array(classes),
    )
    print(f"[preprocess] train={train_mask.sum()}  holdout={(~train_mask).sum()}  classes={classes}")
    print(f"[preprocess] wrote → {paths.processed}")


if __name__ == "__main__":
    main()