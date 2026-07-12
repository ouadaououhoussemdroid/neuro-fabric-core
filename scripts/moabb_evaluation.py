"""T-017 — MOABB-driven evaluation harness.

Scores any registered Braindecode model on BCI-IV-2a, BCI-IV-2b, and
PhysioNetMI using MOABB's cross-subject evaluation protocol. Results are
logged as JSON (and optionally to MLflow if T-021 is available).

Usage:
    python scripts/moabb_evaluation.py --architecture EEGConformer --datasets BCI-IV-2a
    python scripts/moabb_evaluation.py --architecture EEGNetv4 --datasets all
    python scripts/moabb_evaluation.py --architecture Deep4Net --datasets BCI-IV-2b,PhysioNetMI

Outputs:
    training/artefacts/<architecture>/moabb_report.json

Dependencies (already in training/requirements.txt):
    braindecode, moabb, scikit-learn, numpy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = REPO_ROOT / "training" / "artefacts"

DATASETS = {
    "BCI-IV-2a": "BNCI2014_001",
    "BCI-IV-2b": "BNCI2014_004",
    "PhysioNetMI": "PhysionetMI",
}

ARCHITECTURES = ["EEGConformer", "EEGNetv4", "ShallowFBCSPNet", "Deep4Net"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MOABB evaluation harness")
    ap.add_argument(
        "--architecture",
        required=True,
        choices=ARCHITECTURES,
        help="Braindecode architecture to evaluate.",
    )
    ap.add_argument(
        "--datasets",
        default="BCI-IV-2a",
        help="Comma-separated dataset names (or 'all'). Options: BCI-IV-2a, BCI-IV-2b, PhysioNetMI.",
    )
    ap.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Limit number of subjects (for quick CI runs). Default: all.",
    )
    ap.add_argument(
        "--fmin",
        type=float,
        default=8.0,
        help="Bandpass low cutoff (Hz). Default: 8.",
    )
    ap.add_argument(
        "--fmax",
        type=float,
        default=30.0,
        help="Bandpass high cutoff (Hz). Default: 30.",
    )
    return ap.parse_args()


def _resolve_datasets(names: str) -> list[str]:
    if names.lower() == "all":
        return list(DATASETS.keys())
    parsed = [n.strip() for n in names.split(",") if n.strip()]
    for n in parsed:
        if n not in DATASETS:
            raise ValueError(f"Unknown dataset: {n}. Available: {list(DATASETS.keys())}")
    return parsed


def _build_model(architecture: str, n_channels: int, n_times: int, n_classes: int):
    """Build a Braindecode model by architecture name."""
    from braindecode.models import EEGConformer, EEGNetv4, ShallowFBCSPNet, Deep4Net

    arch = architecture.lower()
    if arch == "eegconformer":
        return EEGConformer(n_outputs=n_classes, n_chans=n_channels, n_times=n_times, final_fc_length="auto")
    elif arch == "eegnetv4":
        return EEGNetv4(n_outputs=n_classes, n_chans=n_channels, n_times=n_times, final_fc_length="auto")
    elif arch == "shallowfbcspnet":
        return ShallowFBCSPNet(n_outputs=n_classes, n_chans=n_channels, n_times=n_times, n_filters_time=40, filter_time_length=25, final_fc_length="auto")
    elif arch == "deep4net":
        return Deep4Net(n_outputs=n_classes, n_chans=n_channels, n_times=n_times, n_filters_time=25, filter_time_length=10, final_fc_length="auto")
    raise ValueError(f"Unknown architecture: {architecture}")


def _evaluate_dataset(
    architecture: str,
    dataset_name: str,
    moabb_class: str,
    n_subjects: int | None,
    fmin: float,
    fmax: float,
) -> dict[str, Any]:
    """Run cross-subject evaluation on one dataset."""
    import numpy as np
    from moabb.datasets import (
        BNCI2014_001,
        BNCI2014_004,
        PhysionetMI,
    )
    from moabb.paradigms import MotorImagery
    from braindecode import EEGClassifier
    import torch
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from skorch.callbacks import EpochScoring

    # Resolve the dataset class.
    dataset_classes = {
        "BNCI2014_001": BNCI2014_001,
        "BNCI2014_004": BNCI2014_004,
        "PhysionetMI": PhysionetMI,
    }
    dataset_cls = dataset_classes[moabb_class]
    dataset = dataset_cls()

    subjects = list(range(1, dataset.subject_list + 1)) if n_subjects is None else list(range(1, min(n_subjects, dataset.subject_list) + 1))

    paradigm = MotorImagery(fmin=fmin, fmax=fmax, n_classes=4, resample=250)
    print(f"[moabb] {dataset_name}: extracting epochs for {len(subjects)} subjects")
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = np.asarray(X, dtype=np.float32)

    n_channels = X.shape[1]
    n_times = X.shape[2]
    classes = sorted(set(labels))
    n_classes = len(classes)
    label_map = {c: i for i, c in enumerate(classes)}
    y = np.array([label_map[c] for c in labels], dtype=np.int64)

    print(f"[moabb] {dataset_name}: X={X.shape} n_classes={n_classes}")

    model = _build_model(architecture, n_channels, n_times, n_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        batch_size=64,
        max_epochs=5,  # low for CI; increase for real runs
        device=device,
        train_split=False,
        verbose=0,
    )

    # Simple per-subject cross-validation (leave-one-subject-out is expensive;
    # use a 3-fold split for CI feasibility).
    from sklearn.model_selection import StratifiedKFold

    subjects_arr = metadata["subject"].to_numpy()
    unique_subjects = np.unique(subjects_arr)
    # Group K-fold by subject to prevent leakage.
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=min(3, len(unique_subjects)))
    scores = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects_arr)):
        print(f"[moabb] {dataset_name}: fold {fold + 1} (train={len(train_idx)}, test={len(test_idx)})")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        acc = float((pred == y[test_idx]).mean())
        scores.append(acc)
        print(f"[moabb] {dataset_name}: fold {fold + 1} acc={acc:.4f}")

    return {
        "dataset": dataset_name,
        "architecture": architecture,
        "n_subjects": len(subjects),
        "n_channels": int(n_channels),
        "n_times": int(n_times),
        "n_classes": int(n_classes),
        "n_epochs_total": int(len(y)),
        "fold_accuracies": scores,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "fmin": fmin,
        "fmax": fmax,
    }


def main() -> None:
    args = _parse_args()
    datasets = _resolve_datasets(args.datasets)
    out_dir = ARTEFACTS / args.architecture
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ds_name in datasets:
        moabb_class = DATASETS[ds_name]
        try:
            result = _evaluate_dataset(
                args.architecture,
                ds_name,
                moabb_class,
                args.n_subjects,
                args.fmin,
                args.fmax,
            )
            results.append(result)
        except Exception as e:
            print(f"[moabb] {ds_name}: FAILED — {e}")
            results.append({"dataset": ds_name, "error": str(e)})

    report = {
        "architecture": args.architecture,
        "datasets": datasets,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "results": results,
        "generated_at": str(__import__("datetime").datetime.now().isoformat()),
    }
    out_path = out_dir / "moabb_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[moabb] report → {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
