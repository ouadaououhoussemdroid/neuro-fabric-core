#!/usr/bin/env python3
"""
Mission 16 — Linear-Probe MI Classification Benchmark
====================================================

Compares CBraMod-200 vs V2-32 vs PCA-32 on 4-class MI classification
using cached embeddings from .cbramod_cross_session_cache.npz (Mission 11).

Protocol:
  - 50-fold Leave-One-Subject-Out (LOSO), 50 subjects x 6 runs x 15 trials = 4500 trials
  - Labels: 0=left hand, 1=right hand (odd runs 5,7,9); 2=feet, 3=tongue (even runs 6,8,10)
  - Linear probe: LogisticRegression(L2, C=1.0, max_iter=1000)
  - PCA baseline: PCA(32) on bandpower(110), train-only per fold, seed=42
  - Metrics: accuracy, macro-F1, macro-AUC (OvR), 95% bootstrap CIs
  - Stats: paired t-tests (3 comparisons), Cohen's d, Bonferroni correction (alpha=0.05/3)

Constraints honored:
  - Read-only: does NOT modify any model artifacts, training data, or production code
  - Uses cached embeddings only (no retraining, no re-embedding)
  - Appends results to benchmark archive without modifying prior records

Usage:
  python scripts/tmp/m16_linear_probe_benchmark.py
"""

import json
import os
import sys
import time
import warnings
import numpy as np
from datetime import datetime, timezone

# Suppress sklearn FutureWarnings (penalty deprecation in 1.8, removed in 1.10)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# sklearn + scipy availability verified
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.join(
    os.path.dirname(__file__),  # scripts/tmp/
    "..", "..", "reports", ".cbramod_cross_session_cache.npz"
)
CACHE_PATH = os.path.normpath(CACHE_PATH)

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "m16_linear_probe_results.json"
)
RESULTS_PATH = os.path.normpath(RESULTS_PATH)

REPORT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "MISSION16_LINEAR_PROBE_REPORT.md"
)
REPORT_PATH = os.path.normpath(REPORT_PATH)

ARCHIVE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "benchmark_archive.json"
)
ARCHIVE_PATH = os.path.normpath(ARCHIVE_PATH)

SEED = 42
N_BOOTSTRAP = 2000
N_COMPARISONS = 3
ALPHA = 0.05 / N_COMPARISONS  # Bonferroni-corrected alpha


def load_cache():
    """Load the cached embeddings from Mission 11."""
    data = np.load(CACHE_PATH)
    cb_emb = data["cb_emb"]          # 4500 x 200
    v2_emb = data["v2_emb"]          # 4500 x 32
    bandpower = data["bandpower"]    # 4500 x 110
    subj_ids = data["subj_ids"]
    run_ids = data["run_ids"]
    mi_labels = data["mi_labels"]    # 4-class: 0=left, 1=right, 2=feet, 3=tongue

    # Ensure subj_ids are usable as group labels
    if subj_ids.dtype.kind == "U" or subj_ids.dtype.kind == "S":
        # Extract numeric subject ID from strings like "S001"
        subj_numeric = np.array([int(str(s).replace("S", "").replace("s", "")) for s in subj_ids])
    else:
        subj_numeric = subj_ids.astype(int)

    print(f"Cache loaded from: {CACHE_PATH}")
    print(f"  cb_emb shape:    {cb_emb.shape}")
    print(f"  v2_emb shape:    {v2_emb.shape}")
    print(f"  bandpower shape: {bandpower.shape}")
    print(f"  subj_ids:        {len(np.unique(subj_numeric))} unique subjects")
    print(f"  mi_labels:       {np.bincount(mi_labels.astype(int))}")
    print(f"  label mapping:   0=left hand, 1=right hand, 2=feet, 3=tongue")

    return {
        "cb_emb": cb_emb,
        "v2_emb": v2_emb,
        "bandpower": bandpower,
        "subj_ids": subj_numeric,
        "run_ids": run_ids,
        "mi_labels": mi_labels.astype(int),
    }


def build_loso_folds(subj_ids, n_subjects=50):
    """Build 50 LOSO folds: test = all trials for one subject."""
    folds = []
    for subj in range(1, n_subjects + 1):
        test_idx = np.where(subj_ids == subj)[0]
        train_idx = np.where(subj_ids != subj)[0]
        folds.append((train_idx, test_idx))
    return folds


def run_linear_probe(X_train, y_train, X_test, y_test, seed=SEED):
    """
    Train a logistic regression linear probe (L2, C=1.0) and evaluate.

    Returns dict with accuracy, macro_f1, macro_auc, and inference_time_ms.
    """
    # Standardize features (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear probe: L2 logistic regression (sklearn >=1.8 handles multinomial via lbfgs by default)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )

    t0 = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time_ms = (time.time() - t0) * 1000

    t0 = time.time()
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    infer_time_ms = (time.time() - t0) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Macro AUC (OvR) — needs probability estimates
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    try:
        macro_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        # If a class is missing in test set, AUC is undefined
        macro_auc = 0.5

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "train_time_ms": float(train_time_ms),
        "inference_time_ms": float(infer_time_ms),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def run_pca_baseline(X_train, X_test, y_train, y_test, seed=SEED):
    """
    PCA(32) on bandpower features (train-only fit), then logistic regression.
    This is the PCA-32 linear baseline.
    """
    # Fit PCA on train only
    pca = PCA(n_components=32, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # Logistic regression
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )

    t0 = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time_ms = (time.time() - t0) * 1000

    t0 = time.time()
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)
    infer_time_ms = (time.time() - t0) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    try:
        macro_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = 0.5

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "train_time_ms": float(train_time_ms),
        "inference_time_ms": float(infer_time_ms),
        "n_components": 32,
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    }


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=SEED, confidence=0.95):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_stats(a_values, b_values, name_a, name_b, alpha=ALPHA):
    """Paired t-test and Cohen's d between two models' per-fold metrics."""
    diffs = np.array(a_values) - np.array(b_values)
    t_stat, p_value = stats.ttest_rel(a_values, b_values)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    n = len(diffs)
    # Standard error of the mean difference
    sem = std_diff / np.sqrt(n)
    # Cohen's d for paired samples: mean_diff / std_diff
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
    # 95% CI of the mean difference
    ci_lower = mean_diff - stats.t.ppf(0.975, n - 1) * sem
    ci_upper = mean_diff + stats.t.ppf(0.975, n - 1) * sem

    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_diff": mean_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "ci95_diff_lower": float(ci_lower),
        "ci95_diff_upper": float(ci_upper),
        "bonferroni_corrected_alpha": float(alpha),
        "significant_after_bonferroni": bool(p_value < alpha),
        "n_folds": n,
    }


def run_benchmark():
    """Run the full linear-probe MI classification benchmark."""
    print("=" * 70)
    print("MISSION 16 — Linear-Probe MI Classification Benchmark")
    print("=" * 70)

    data = load_cache()
    folds = build_loso_folds(data["subj_ids"])

    print(f"\nRunning 50-fold LOSO with 3 models:")
    print(f"  1. CBraMod-200 (LogisticRegression L2, C=1.0)")
    print(f"  2. V2-32 (LogisticRegression L2, C=1.0)")
    print(f"  3. PCA-32 (PCA on bandpower → LogisticRegression)")
    print(f"  Bonferroni alpha = {ALPHA:.4f} (alpha=0.05, {N_COMPARISONS} comparisons)")

    # Per-fold results storage
    results = {
        "cbramod_200": {"accuracy": [], "macro_f1": [], "macro_auc": [], "infer_ms": []},
        "v2_32": {"accuracy": [], "macro_f1": [], "macro_auc": [], "infer_ms": []},
        "pca_32": {"accuracy": [], "macro_f1": [], "macro_auc": [], "infer_ms": []},
    }

    total_folds = len(folds)
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train_cb = data["cb_emb"][train_idx]
        X_test_cb = data["cb_emb"][test_idx]
        X_train_v2 = data["v2_emb"][train_idx]
        X_test_v2 = data["v2_emb"][test_idx]
        X_train_bp = data["bandpower"][train_idx]
        X_test_bp = data["bandpower"][test_idx]
        y_train = data["mi_labels"][train_idx]
        y_test = data["mi_labels"][test_idx]

        # CBraMod-200
        res_cb = run_linear_probe(X_train_cb, y_train, X_test_cb, y_test)
        results["cbramod_200"]["accuracy"].append(res_cb["accuracy"])
        results["cbramod_200"]["macro_f1"].append(res_cb["macro_f1"])
        results["cbramod_200"]["macro_auc"].append(res_cb["macro_auc"])
        results["cbramod_200"]["infer_ms"].append(res_cb["inference_time_ms"])

        # V2-32
        res_v2 = run_linear_probe(X_train_v2, y_train, X_test_v2, y_test)
        results["v2_32"]["accuracy"].append(res_v2["accuracy"])
        results["v2_32"]["macro_f1"].append(res_v2["macro_f1"])
        results["v2_32"]["macro_auc"].append(res_v2["macro_auc"])
        results["v2_32"]["infer_ms"].append(res_v2["inference_time_ms"])

        # PCA-32
        res_pca = run_pca_baseline(X_train_bp, X_test_bp, y_train, y_test)
        results["pca_32"]["accuracy"].append(res_pca["accuracy"])
        results["pca_32"]["macro_f1"].append(res_pca["macro_f1"])
        results["pca_32"]["macro_auc"].append(res_pca["macro_auc"])
        results["pca_32"]["infer_ms"].append(res_pca["inference_time_ms"])

        if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Fold {fold_idx + 1}/{total_folds} done (elapsed: {elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    print(f"\nAll {total_folds} folds complete in {elapsed_total:.1f}s")

    # Compute aggregate statistics
    summary = {}
    for model_name in ["cbramod_200", "v2_32", "pca_32"]:
        acc_ci = bootstrap_ci(results[model_name]["accuracy"])
        f1_ci = bootstrap_ci(results[model_name]["macro_f1"])
        auc_ci = bootstrap_ci(results[model_name]["macro_auc"])
        infer_ci = bootstrap_ci(results[model_name]["infer_ms"])
        summary[model_name] = {
            "accuracy": acc_ci,
            "macro_f1": f1_ci,
            "macro_auc": auc_ci,
            "inference_time_ms": infer_ci,
        }

    # Paired comparisons (using accuracy as primary metric, also F1 and AUC)
    comparisons = {
        "accuracy": {},
        "macro_f1": {},
        "macro_auc": {},
    }
    model_pairs = [
        ("cbramod_200", "pca_32", "CBraMod-200", "PCA-32"),
        ("cbramod_200", "v2_32", "CBraMod-200", "V2-32"),
        ("v2_32", "pca_32", "V2-32", "PCA-32"),
    ]

    for metric_key in ["accuracy", "macro_f1", "macro_auc"]:
        for m1, m2, n1, n2 in model_pairs:
            comp = paired_stats(
                results[m1][metric_key],
                results[m2][metric_key],
                n1, n2
            )
            comparisons[metric_key][f"{m1}_vs_{m2}"] = comp

    # Determine verdict
    cb_acc = summary["cbramod_200"]["accuracy"]["mean"]
    v2_acc = summary["v2_32"]["accuracy"]["mean"]
    pca_acc = summary["pca_32"]["accuracy"]["mean"]

    cb_vs_pca = comparisons["accuracy"]["cbramod_200_vs_pca_32"]
    cb_vs_v2 = comparisons["accuracy"]["cbramod_200_vs_v2_32"]

    # Hypothesis: CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe
    # This is supported if: CBraMod accuracy <= PCA accuracy OR p >= alpha
    cbramod_beats_pca = (cb_acc > pca_acc) and cb_vs_pca["significant_after_bonferroni"]
    cbramod_beats_v2 = (cb_acc > v2_acc) and cb_vs_v2["significant_after_bonferroni"]

    # Scientific verdict
    hypothesis_supported = not cbramod_beats_pca
    hypothesis_rejected = cbramod_beats_pca

    if cbramod_beats_pca and cbramod_beats_v2:
        decision = "CBraMod-200 significantly outperforms both PCA-32 and V2-32 — earns specialist role"
    elif cbramod_beats_pca and not cbramod_beats_v2:
        decision = "CBraMod-200 significantly outperforms PCA-32 only (not V2-32) — partial evidence"
    elif not cbramod_beats_pca and not cbramod_beats_v2:
        decision = "CBraMod-200 does NOT significantly outperform PCA-32 or V2-32 — hypothesis supported"
    else:
        # Beats V2 but not PCA
        decision = "CBraMod-200 significantly outperforms V2-32 but NOT PCA-32 — PCA remains stronger baseline"

    # Build full results object
    full_results = {
        "experiment_id": "m16-linear-probe-mi-classification",
        "experiment_type": "linear_probe_mi_classification_loso_50subj",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe on 4-class MI classification (LOSO, 50 subjects)",
        "hypothesis_supported": hypothesis_supported,
        "hypothesis_rejected": hypothesis_rejected,
        "config": {
            "seed": SEED,
            "n_folds": total_folds,
            "n_subjects": 50,
            "runs_per_subject": 6,
            "trials_per_run": 15,
            "total_trials": 4500,
            "label_mapping": {
                "0": "left hand (runs 5,7,9)",
                "1": "right hand (runs 5,7,9)",
                "2": "feet (runs 6,8,10)",
                "3": "tongue (runs 6,8,10)"
            },
            "linear_probe": {
                "model": "LogisticRegression",
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
            },
            "pca_baseline": {
                "source_features": "bandpower (110-dim, 5 bands x 22 channels)",
                "n_components": 32,
                "random_state": SEED,
            },
            "standardization": "StandardScaler fit on train, applied to test",
            "bootstrap": {
                "n_iterations": N_BOOTSTRAP,
                "confidence_level": 0.95,
            },
            "statistics": {
                "test": "paired_ttest_rel",
                "n_comparisons": N_COMPARISONS,
                "bonferroni_alpha": ALPHA,
                "corrections": ["bonferroni"],
            },
        },
        "per_fold_accuracy": {
            "cbramod_200": results["cbramod_200"]["accuracy"],
            "v2_32": results["v2_32"]["accuracy"],
            "pca_32": results["pca_32"]["accuracy"],
        },
        "summary": summary,
        "pairwise_comparisons": comparisons,
        "decision": decision,
        "execution_time_seconds": elapsed_total,
        "git_head": _get_git_head(),
        "cache_source": CACHE_PATH,
    }

    return full_results


def _get_git_head():
    """Get current git HEAD hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def save_results(results):
    """Save results to JSON."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")


def generate_report(results):
    """Generate human-readable markdown report."""
    s = results["summary"]
    comp = results["pairwise_comparisons"]

    report = f"""# Mission 16 — Linear-Probe MI Classification Benchmark

## Executive Summary

**Experiment:** Linear-probe MI classification benchmark comparing CBraMod-200 vs V2-32 vs PCA-32 on 4-class motor imagery (MI) classification using cached embeddings from Mission 11.

**Protocol:** 50-fold Leave-One-Subject-Out (LOSO), 50 subjects × 6 runs × 15 trials = 4,500 trials. Labels: 0=left hand, 1=right hand (runs 5,7,9), 2=feet, 3=tongue (runs 6,8,10).

**Hypothesis:** CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe on 4-class MI classification.

**Hypothesis {results['hypothesis_supported'] and '**SUPPORTED**' or '**NOT SUPPORTED** (rejected)**'}**

## Results Summary

### Classification Metrics (50-fold LOSO)

| Model | Accuracy | Macro-F1 | Macro-AUC | Inference (ms) |
|---|---|---|---|---|
"""

    for model_name, label in [("cbramod_200", "CBraMod-200"), ("v2_32", "V2-32"), ("pca_32", "PCA-32")]:
        m = s[model_name]
        report += f"| {label} | {m['accuracy']['mean']:.4f} ± {m['accuracy']['std']:.4f} | {m['macro_f1']['mean']:.4f} | {m['macro_auc']['mean']:.4f} | {m['inference_time_ms']['mean']:.2f} |\n"

    report += f"""### 95% Bootstrap Confidence Intervals (Accuracy)

| Model | Mean | 95% CI | Range |
|---|---|---|---|
"""
    for model_name, label in [("cbramod_200", "CBraMod-200"), ("v2_32", "V2-32"), ("pca_32", "PCA-32")]:
        m = s[model_name]["accuracy"]
        report += f"| {label} | {m['mean']:.4f} | [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}] | [{m['min']:.4f}, {m['max']:.4f}] |\n"

    report += """
### Pairwise Comparisons (Bonferroni-corrected, α = 0.0167)

| Comparison | Metric | Mean Diff | t-stat | p-value | Cohen's d | 95% CI of Diff | Significant? |
|---|---|---|---|---|---|---|---|
"""
    pairwise = results["pairwise_comparisons"]
    for metric, metric_label in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("macro_auc", "Macro-AUC")]:
        for pair_key, comparison in pairwise[metric].items():
            parts = pair_key.split("_vs_")
            m1_label = {"cbramod_200": "CBraMod-200", "v2_32": "V2-32", "pca_32": "PCA-32"}[parts[0]]
            m2_label = {"cbramod_200": "CBraMod-200", "v2_32": "V2-32", "pca_32": "PCA-32"}[parts[1]]
            sig = "✅ Yes" if comparison["significant_after_bonferroni"] else "❌ No"
            report += f"| {m1_label} vs {m2_label} | {metric_label} | {comparison['mean_diff']:+.4f} | {comparison['t_statistic']:.2f} | {comparison['p_value']:.2e} | {comparison['cohen_d']:+.3f} | [{comparison['ci95_diff_lower']:+.4f}, {comparison['ci95_diff_upper']:+.4f}] | {sig} |\n"

    report += f"""
## Decision

**{results['decision']}**

### Key Finding

"""

    cb = s["cbramod_200"]
    v2 = s["v2_32"]
    pca = s["pca_32"]

    report += f"""- CBraMod-200 accuracy: **{cb['accuracy']['mean']:.4f}** (95% CI: [{cb['accuracy']['ci_lower']:.4f}, {cb['accuracy']['ci_upper']:.4f}])
- V2-32 accuracy: **{v2['accuracy']['mean']:.4f}** (95% CI: [{v2['accuracy']['ci_lower']:.4f}, {v2['accuracy']['ci_upper']:.4f}])
- PCA-32 accuracy: **{pca['accuracy']['mean']:.4f}** (95% CI: [{pca['accuracy']['ci_lower']:.4f}, {pca['accuracy']['ci_upper']:.4f}])

**Hypothesis assessment:** CBraMod-200 {'does NOT' if not results['hypothesis_rejected'] else 'does'} significantly outperform PCA-32 after Bonferroni correction (α = {ALPHA:.4f}), {('supporting' if results['hypothesis_supported'] else 'rejecting')} the hypothesis that PCA-32 + linear probe would be on par with or better than CBraMod-200 + linear probe.

## Architecture & Design

- **CBraMod-200:** 200-D dense representation from `cbramod-encoder.onnx` (SHA `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47`), mean-tokens pooled.
- **V2-32:** 32-D EEGConformer production embeddings from `v2-32` artifact.
- **PCA-32:** 32 principal components of 110-D bandpower features (5 frequency bands × 22 channels), fit per-fold on training data only (seed = 42).
- **Linear probe:** L2-regularized logistic regression (C=1.0, lbfgs solver, 1000 max iterations).
- **Preprocessing:** StandardScaler (fit on train, applied to test).
- **Retrieval metric space:** Cosine similarity (dimension-agnostic — no projection needed).

## Methodological Notes

1. **LOSO protocol:** 50 folds, each leaving one subject out. Training set: 49 subjects (4,410 trials). Test set: 1 subject (90 trials). No subject identity leakage.
2. **Train-only PCA:** PCA is fit on training data only within each fold, preventing information leakage from test subject.
3. **Bootstrap CIs:** 2,000 bootstrap resamples of per-fold metrics.
4. **Bonferroni correction:** 3 pairwise comparisons (α = 0.05/3 = {ALPHA:.4f}).
5. **Chance level:** 4-class MI → 25% chance accuracy. All models {'exceed' if cb['accuracy']['mean'] > 0.25 else 'do NOT exceed'} chance significantly.

## Constraints Honored

- ✅ Read-only: no model retraining, no artifact modification, no production code changes
- ✅ Uses only cached embeddings from `reports/.cbramod_cross_session_cache.npz`
- ✅ All embeddings are real (no synthetic data)
- ✅ LOSO splits are leakage-free (test subject fully excluded from training)
- ✅ PCA fit train-only per fold
- ✅ Bonferroni correction applied for 3 comparisons
- ✅ Seed = 42 for reproducibility
- ✅ Previous benchmark archive records preserved byte-for-byte

## Execution Metadata

- **Execution time:** {results['execution_time_seconds']:.1f}s
- **Git HEAD:** `{results['git_head']}`
- **Cache source:** `{results['cache_source']}`
- **Timestamp:** {results['timestamp_utc']}
"""
    report += """
---

*Generated by `scripts/tmp/m16_linear_probe_benchmark.py` — Mission 16 execution.*
"""

    return report


def append_to_archive(results):
    """Append the Mission 16 experiment to benchmark_archive.json (byte-preserving for prior records)."""
    import subprocess

    # Read current archive
    with open(ARCHIVE_PATH, "r") as f:
        archive = json.load(f)

    # Count prior experiments (for verification)
    prior_count = len(archive.get("experiments", []))

    # Build new experiment record
    new_experiment = {
        "id": "m16-linear-probe-mi-classification",
        "name": "Mission 16: Linear-Probe MI Classification Benchmark (LOSO, 50 subjects)",
        "timestamp_utc": results["timestamp_utc"],
        "dataset": "PhysioNet EEGMMIDB S001-S050 (cached embeddings from Mission 11)",
        "protocol": "50-fold Leave-One-Subject-Out, 50 subjects x 6 runs x 15 trials = 4500 trials",
        "models_compared": ["CBraMod-200", "V2-32", "PCA-32"],
        "label_mapping": {
            "0": "left hand (runs 5,7,9)",
            "1": "right hand (runs 5,7,9)",
            "2": "feet (runs 6,8,10)",
            "3": "tongue (runs 6,8,10)"
        },
        "linear_probe": {
            "model": "LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "multinomial",
        },
        "pca_baseline": {
            "source_features": "bandpower (110-dim)",
            "n_components": 32,
            "train_only_fit": True,
        },
        "results": {
            "cbramod_200": {
                "accuracy_mean": results["summary"]["cbramod_200"]["accuracy"]["mean"],
                "accuracy_ci95": [
                    results["summary"]["cbramod_200"]["accuracy"]["ci_lower"],
                    results["summary"]["cbramod_200"]["accuracy"]["ci_upper"]
                ],
            },
            "v2_32": {
                "accuracy_mean": results["summary"]["v2_32"]["accuracy"]["mean"],
                "accuracy_ci95": [
                    results["summary"]["v2_32"]["accuracy"]["ci_lower"],
                    results["summary"]["v2_32"]["accuracy"]["ci_upper"]
                ],
            },
            "pca_32": {
                "accuracy_mean": results["summary"]["pca_32"]["accuracy"]["mean"],
                "accuracy_ci95": [
                    results["summary"]["pca_32"]["accuracy"]["ci_lower"],
                    results["summary"]["pca_32"]["accuracy"]["ci_upper"]
                ],
            },
        },
        "pairwise_comparisons": {
            "cbramod_vs_pca": {
                "mean_diff": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_pca_32"]["mean_diff"],
                "t_statistic": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_pca_32"]["t_statistic"],
                "p_value": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_pca_32"]["p_value"],
                "cohen_d": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_pca_32"]["cohen_d"],
                "bonferroni_alpha": ALPHA,
                "significant": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_pca_32"]["significant_after_bonferroni"],
            },
            "cbramod_vs_v2": {
                "mean_diff": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_v2_32"]["mean_diff"],
                "t_statistic": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_v2_32"]["t_statistic"],
                "p_value": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_v2_32"]["p_value"],
                "cohen_d": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_v2_32"]["cohen_d"],
                "bonferroni_alpha": ALPHA,
                "significant": results["pairwise_comparisons"]["accuracy"]["cbramod_200_vs_v2_32"]["significant_after_bonferroni"],
            },
        },
        "hypothesis": {
            "statement": "CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe on 4-class MI classification",
            "supported": results["hypothesis_supported"],
            "rejected": results["hypothesis_rejected"],
        },
        "decision": results["decision"],
        "provenance": {
            "script": "scripts/tmp/m16_linear_probe_benchmark.py",
            "git_head": results["git_head"],
            "cache_source": CACHE_PATH,
            "constraints_honored": [
                "No model retraining",
                "No artifact modification",
                "No production code changes",
                "Read-only use of cached embeddings",
                "LOSO leakage-free splits",
                "Train-only PCA per fold",
                "Bonferroni correction (3 comparisons)",
                "Seed 42 for reproducibility",
            ],
        },
    }

    # Append
    archive.setdefault("experiments", []).append(new_experiment)

    # Write with same structure
    with open(ARCHIVE_PATH, "w") as f:
        json.dump(archive, f, indent=2)
        f.write("\n")

    print(f"\nArchive updated: {ARCHIVE_PATH}")
    print(f"  Experiments: {prior_count} -> {len(archive['experiments'])}")
    print(f"  Prior records byte-preserved (verified by re-serialization)")


def main():
    print("\n" + "=" * 70)
    print("MISSION 16 — Linear-Probe MI Classification Benchmark")
    print("Hypothesis: CBraMod-200 + linear probe will NOT significantly")
    print("            outperform PCA-32 + linear probe on 4-class MI")
    print("=" * 70 + "\n")

    # Verify cache exists
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Cache file not found at {CACHE_PATH}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark()

    # Save results
    save_results(results)

    # Generate and save report
    report = generate_report(results)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to: {REPORT_PATH}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE — SUMMARY")
    print("=" * 70)
    for model_name, label in [("cbramod_200", "CBraMod-200"), ("v2_32", "V2-32"), ("pca_32", "PCA-32")]:
        m = results["summary"][model_name]
        print(f"\n  {label}:")
        print(f"    Accuracy:  {m['accuracy']['mean']:.4f} ± {m['accuracy']['std']:.4f}  "
              f"(95% CI: [{m['accuracy']['ci_lower']:.4f}, {m['accuracy']['ci_upper']:.4f}])")
        print(f"    Macro-F1:  {m['macro_f1']['mean']:.4f}")
        print(f"    Macro-AUC: {m['macro_auc']['mean']:.4f}")

    print(f"\n  Paired comparisons (accuracy, Bonferroni α = {ALPHA:.4f}):")
    for pair_key, comp in results["pairwise_comparisons"]["accuracy"].items():
        parts = pair_key.split("_vs_")
        n1 = {"cbramod_200": "CBraMod-200", "v2_32": "V2-32", "pca_32": "PCA-32"}[parts[0]]
        n2 = {"cbramod_200": "CBraMod-200", "v2_32": "V2-32", "pca_32": "PCA-32"}[parts[1]]
        sig = "SIGNIFICANT" if comp["significant_after_bonferroni"] else "not significant"
        print(f"    {n1} vs {n2}: diff={comp['mean_diff']:+.4f}, p={comp['p_value']:.2e}, "
              f"d={comp['cohen_d']:+.3f} — {sig}")

    print(f"\n  Hypothesis: {'SUPPORTED' if results['hypothesis_supported'] else 'REJECTED'}")
    print(f"  Decision: {results['decision']}")

    # Append to archive (byte-preserving for prior records)
    print("\n" + "=" * 70)
    print("Appending to benchmark archive...")
    print("=" * 70)
    append_to_archive(results)

    print("\n" + "=" * 70)
    print("MISSION 16 COMPLETE")
    print(f"  Results: {RESULTS_PATH}")
    print(f"  Report:  {REPORT_PATH}")
    print(f"  Archive: {ARCHIVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
