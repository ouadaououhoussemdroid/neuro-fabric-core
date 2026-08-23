#!/usr/bin/env python3
"""
Scientific-grade Leave-One-Subject-Out (LOSO) cross-validation utility.

M33-Scientific-Reboot — Phase 4

Provides a reusable, auditable LOSO splitter that guarantees:
  - No subject appears in both train and test folds
  - No trial from a held-out subject enters training
  - Preprocessing statistics computed train-only
  - Per-fold metrics returned with confidence intervals

Usage:
    from loso import loso_split, assert_no_leakage

    splits = loso_split(embeddings, labels, subj_ids)
    for train_idx, test_idx in splits:
        assert_no_leakage(train_idx, test_idx, subj_ids)
        # ... train and evaluate ...
"""
from __future__ import annotations

import numpy as np
from typing import Iterator

def loso_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subj_ids: np.ndarray,
    min_trials_per_subject: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate true leave-one-subject-out splits.

    For each subject S:
      - Test fold: all trials where subject_id == S
      - Train fold: all trials where subject_id != S

    Subjects with fewer than `min_trials_per_subject` trials are excluded
    from being a held-out test subject (they cannot form a valid test set).

    Args:
        embeddings: [N, D] embedding matrix
        labels: [N] ground-truth labels
        subj_ids: [N] subject identifiers
        min_trials_per_subject: minimum trials per subject to qualify as test

    Returns:
        List of (train_indices, test_indices) tuples, one per eligible subject.
    """
    subj_ids = np.asarray(subj_ids)
    unique_subjects = np.unique(subj_ids)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for subj in unique_subjects:
        is_test = subj_ids == subj
        test_idx = np.where(is_test)[0]
        train_idx = np.where(~is_test)[0]

        if len(test_idx) < min_trials_per_subject:
            continue

        # Double-check: no overlap between train and test subjects
        train_subjects = np.unique(subj_ids[train_idx])
        test_subjects = np.unique(subj_ids[test_idx])
        overlap = np.intersect1d(train_subjects, test_subjects)
        assert len(overlap) == 0, (
            f"LOSO leakage detected for subject {subj}: "
            f"subjects {overlap} appear in both train and test"
        )

        splits.append((train_idx, test_idx))

    return splits


def assert_no_leakage(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    subj_ids: np.ndarray,
) -> None:
    """Assert no subject-level leakage between train and test indices."""
    train_subjects = set(np.unique(subj_ids[train_idx]))
    test_subjects = set(np.unique(subj_ids[test_idx]))
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} subjects in both train and test: {overlap}"
    )


def loo_cv_evaluate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subj_ids: np.ndarray,
    model_factory,
    metric_fn,
    min_trials_per_subject: int = 2,
) -> dict:
    """
    Run full LOSO cross-validation and return per-fold + aggregate metrics.

    Args:
        embeddings: [N, D] matrix
        labels: [N] ground truth
        subj_ids: [N] subject IDs
        model_factory: callable() → model with .fit(X, y) and .predict(X)
        metric_fn: callable(y_true, y_pred) → dict of metrics
        min_trials_per_subject: minimum trials to qualify as test subject

    Returns:
        {
            "n_folds": int,
            "n_subjects": int,
            "per_fold": list[dict],
            "mean": dict,
            "std": dict,
            "ci_lower": dict,
            "ci_upper": dict,
            "p_value_vs_baseline": float,
        }
    """
    splits = loso_split(embeddings, labels, subj_ids, min_trials_per_subject)

    per_fold = []
    for i, (train_idx, test_idx) in enumerate(splits):
        assert_no_leakage(train_idx, test_idx, subj_ids)

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Fit scaler ONLY on training data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        fold_metrics = metric_fn(y_test, y_pred)
        fold_metrics["fold"] = i
        fold_metrics["n_train"] = len(train_idx)
        fold_metrics["n_test"] = len(test_idx)
        fold_metrics["test_subject"] = int(np.unique(subj_ids[test_idx])[0])
        per_fold.append(fold_metrics)

    # Aggregate
    metric_keys = [k for k in per_fold[0] if k not in ("fold", "n_train", "n_test", "test_subject")]
    mean_metrics = {k: float(np.mean([f[k] for f in per_fold])) for k in metric_keys}
    std_metrics = {k: float(np.std([f[k] for f in per_fold])) for k in metric_keys}

    # 95% CI via bootstrap percentile method
    ci_lower = {}
    ci_upper = {}
    n_bootstrap = 10000
    rng = np.random.RandomState(42)
    for k in metric_keys:
        fold_vals = np.array([f[k] for f in per_fold])
        boot_means = np.array([
            np.mean(rng.choice(fold_vals, size=len(fold_vals), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_lower[k] = float(np.percentile(boot_means, 2.5))
        ci_upper[k] = float(np.percentile(boot_means, 97.5))

    return {
        "n_folds": len(splits),
        "n_subjects": len(np.unique(subj_ids)),
        "per_fold": per_fold,
        "mean": mean_metrics,
        "std": std_metrics,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
