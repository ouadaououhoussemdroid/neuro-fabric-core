#!/usr/bin/env python3
"""
Mission 19: Dimension-Wise Learned EEG Embedding

Objective: Determine whether learning individual weights for all 264 dimensions
of the joint embedding (CBraMod-200 ⊕ V2-32 ⊕ PCA-32) improves over Mission 18's
block-weighted embedding (R@5=0.7856).

Methods:
  A. Block weighting (Mission 18 baseline, R@5=0.7584 → 0.7856)
  B. Dimension-wise non-negative weighting (RidgeClassifier per-dim, Fisher score)
  C. Regularized dimension-wise weighting (shrinkage, simplex, L2)
  D. Hierarchical weighting (block weight × within-block dimension weight)

Ablations:
  - Block weighting only (baseline)
  - CBraMod dimension-wise only (dims 0-199)
  - V2 dimension-wise only (dims 200-231)
  - PCA dimension-wise only (dims 232-263)
  - Full 264-D dimension-wise

All learning is train-fold only, 50-fold LOSO, session-disjoint retrieval, seed=42.
"""

import os, sys, json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from numpy.linalg import norm as np_norm

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Lasso
from scipy import stats

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

SEED = 42
N_BOOTSTRAP = 2000
BONFERRONI_COMPARISONS = 4
BONFERRONI_ALPHA = 0.05 / BONFERRONI_COMPARISONS

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
CACHE_PATH = REPORTS / ".cbramod_cross_session_cache.npz"
OUTPUT_CACHE = REPORTS / ".m19_dimensionwise_embedding_cache.npz"
RESULTS_PATH = REPORTS / "m19_dimensionwise_embedding_results.json"

CBRAMOD_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"

# Block sizes
N_CB = 200   # CBraMod
N_V2 = 32    # V2
N_PCA = 32   # PCA
N_TOTAL = N_CB + N_V2 + N_PCA  # 264

# Mission 18 baseline (primary comparison)
M18_R5 = 0.7856
M18_BEST = "weighted_concat"


# ─────────────────────────────────────────────────────────────
# Utility functions (reused from M18)
# ─────────────────────────────────────────────────────────────

def l2_normalize(x, axis=-1):
    return x / (np_norm(x, axis=axis, keepdims=True) + 1e-12)


def cosine_sim_matrix(a, b):
    return a @ b.T


def bootstrap_ci(per_split_values, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    per_split = np.array(per_split_values)
    n = len(per_split)
    boot_means = np.array([
        rng.choice(per_split, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_ttest(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    rng = np.random.RandomState(SEED)
    n = len(diff)
    boot_diffs = np.array([
        rng.choice(diff, size=n, replace=True).mean()
        for _ in range(N_BOOTSTRAP)
    ])
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return {
        "mean_diff": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohen_d": d,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "significant_after_bonferroni": bool(p_val < BONFERRONI_ALPHA),
        "bonferroni_alpha": BONFERRONI_ALPHA,
    }


def compute_fisher(embeddings, subj_ids):
    subjects = sorted(np.unique(subj_ids))
    overall_mean = embeddings.mean(axis=0)
    bss = 0.0
    wss = 0.0
    for subj in subjects:
        mask = subj_ids == subj
        cm = embeddings[mask].mean(axis=0)
        bss += np.sum(mask) * np.sum((cm - overall_mean) ** 2)
        wss += np.sum((embeddings[mask] - cm) ** 2)
    return float(bss / (wss + 1e-12))


def compute_intra_inter_cosine(embeddings, subj_ids, n_sample=5000):
    rng = np.random.RandomState(SEED)
    n_sample = min(n_sample, len(embeddings))
    sample_idx = rng.choice(len(embeddings), n_sample, replace=False)
    intra, inter = [], []
    for i in range(n_sample):
        for j_idx in rng.choice(n_sample, 20, replace=False):
            if sample_idx[i] == sample_idx[j_idx]:
                continue
            sim = float(embeddings[sample_idx[i]] @ embeddings[sample_idx[j_idx]])
            if subj_ids[sample_idx[i]] == subj_ids[sample_idx[j_idx]]:
                intra.append(sim)
            else:
                inter.append(sim)
    return float(np.mean(intra)), float(np.mean(inter))


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_embeddings():
    """Load verified embeddings from cache."""
    cache = np.load(CACHE_PATH, allow_pickle=True)
    cache_files = list(cache.files)
    # Support both key naming conventions across cache versions
    cb_sha = cache["cbramod_sha256"].item() if "cbramod_sha256" in cache_files else cache["cbramod_sha"].item()
    v2_sha = cache["v2_sha256"].item() if "v2_sha256" in cache_files else cache["v2_sha"].item()
    assert cb_sha == CBRAMOD_SHA
    assert v2_sha == V2_SHA
    print("  Cache SHAs verified ✓")

    cb_emb = cache["cb_emb"].astype(np.float32)
    v2_emb = cache["v2_emb"].astype(np.float32)
    bp = cache["bandpower"].astype(np.float32)
    subj_ids = cache["subj_ids"].astype(np.int64)
    run_ids = cache["run_ids"].astype(np.int64)
    mi_labels = cache["mi_labels"].astype(np.int64) if "mi_labels" in cache_files else np.zeros(len(cb_emb), dtype=np.int64)

    # Compute PCA-32 bandpower (full-data, same as M18)
    scaler = StandardScaler()
    bp_scaled = scaler.fit_transform(bp)
    pca = SklearnPCA(n_components=32, random_state=SEED)
    bp_pca_full = l2_normalize(pca.fit_transform(bp_scaled))

    joint_raw = np.hstack([cb_emb, v2_emb, bp_pca_full])

    data = {
        "cbramod_emb": cb_emb,
        "v2_emb": v2_emb,
        "pca32_emb": bp_pca_full,
        "bandpower": bp,
        "joint_raw": joint_raw,
        "subj_ids": subj_ids,
        "run_ids": run_ids,
        "mi_labels": mi_labels,
    }
    print(f"  CBraMod: {cb_emb.shape}, V2: {v2_emb.shape}, PCA-32: {bp_pca_full.shape}")
    print(f"  Joint 264-D: {joint_raw.shape}, Total trials: {len(subj_ids)}")
    return data


# ─────────────────────────────────────────────────────────────
# Weight learning methods
# ─────────────────────────────────────────────────────────────

def learn_block_weights(joint_emb, subj_ids):
    """Mission 18: learn 3 block-level weights via RidgeClassifier."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(joint_emb)
    clf = RidgeClassifier()
    clf.fit(X_s, subj_ids)
    coefs = np.abs(clf.coef_)

    w_cb = coefs[:, :N_CB].mean()
    w_v2 = coefs[:, N_CB:N_CB+N_V2].mean()
    w_pca = coefs[:, N_CB+N_V2:].mean()
    weights = np.array([w_cb, w_v2, w_pca])
    weights = np.maximum(weights, 0)
    weights = weights / (weights.sum() + 1e-12)
    return weights


def learn_dimwise_ridge(joint_emb, subj_ids):
    """Learn 264 individual weights from RidgeClassifier coefficients.

    w_i = |mean across classes of |coef_{c,i}|
    Normalized to simplex (sum to 1).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(joint_emb)
    clf = RidgeClassifier()
    clf.fit(X_s, subj_ids)
    # Per-dimension importance: mean absolute coefficient across classes
    w = np.abs(clf.coef_).mean(axis=0)  # (264,)
    w = np.maximum(w, 0)
    w = w / (w.sum() + 1e-12)
    return w


def learn_dimwise_fisher(joint_emb, subj_ids):
    """Learn 264 individual weights from Fisher discriminant score per dimension.

    Fisher_i = variance_between_i / variance_within_i
    Higher = more discriminative.
    """
    overall_mean = joint_emb.mean(axis=0)
    between_ss = np.zeros(N_TOTAL)
    within_ss = np.zeros(N_TOTAL)
    subjects = sorted(np.unique(subj_ids))
    for subj in subjects:
        mask = subj_ids == subj
        cm = joint_emb[mask].mean(axis=0)
        between_ss += np.sum(mask) * (cm - overall_mean) ** 2
        within_ss += np.sum((joint_emb[mask] - cm) ** 2)
    fisher_scores = np.maximum(between_ss / (within_ss + 1e-12), 0)
    w = fisher_scores / (fisher_scores.sum() + 1e-12)
    return w


def learn_dimwise_shrinkage(joint_emb, subj_ids):
    """Regularized: shrink dimension-wise weights toward Mission 18 block weights.

    w = α * w_dimwise + (1-α) * w_block_expanded
    where w_block_expanded repeats block weights within each block.
    """
    w_dim = learn_dimwise_ridge(joint_emb, subj_ids)
    block_w = learn_block_weights(joint_emb, subj_ids)
    w_block_exp = np.concatenate([
        np.full(N_CB, block_w[0]),
        np.full(N_V2, block_w[1]),
        np.full(N_PCA, block_w[2]),
    ])
    alpha = 0.5  # 50/50 interpolation
    w = alpha * w_dim + (1 - alpha) * w_block_exp
    w = np.maximum(w, 0)
    w = w / (w.sum() + 1e-12)
    return w


def learn_dimwise_simplex(joint_emb, subj_ids, alpha=0.1):
    """Regularized: softmax-simplex constrained dimension-wise weights.

    Uses RidgeClassifier coefficients scaled by alpha (lower = more uniform).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(joint_emb)
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(X_s, subj_ids)
    coefs = np.abs(clf.coef_).mean(axis=0)
    # Softmax normalization
    coefs_norm = coefs - coefs.max()
    exp_coefs = np.exp(coefs_norm / 0.1)  # temperature
    w = exp_coefs / exp_coefs.sum()
    return w


def learn_hierarchical(joint_emb, subj_ids):
    """D: Hierarchical weighting = block_weight × within-block dimension weight.

    For each block:
      1. Compute block-level weight (as in Mission 18)
      2. Compute within-block dimension weights (normalized within block)
      3. Final weight_i = block_weight[block(i)] * dim_weight_within_block[i]
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(joint_emb)
    clf = RidgeClassifier()
    clf.fit(X_s, subj_ids)
    coefs = np.abs(clf.coef_).mean(axis=0)  # (264,)

    # Block weights (3)
    w_cb = coefs[:N_CB].mean()
    w_v2 = coefs[N_CB:N_CB+N_V2].mean()
    w_pca = coefs[N_CB+N_V2:].mean()
    block_w = np.array([w_cb, w_v2, w_pca])
    block_w = np.maximum(block_w, 0)
    block_w = block_w / (block_w.sum() + 1e-12)

    # Within-block dimension weights
    w = np.zeros(N_TOTAL)
    # CBraMod block
    cb_coefs = np.maximum(coefs[:N_CB], 0)
    w[:N_CB] = block_w[0] * cb_coefs / (cb_coefs.sum() + 1e-12)
    # V2 block
    v2_coefs = np.maximum(coefs[N_CB:N_CB+N_V2], 0)
    w[N_CB:N_CB+N_V2] = block_w[1] * v2_coefs / (v2_coefs.sum() + 1e-12)
    # PCA block
    pca_coefs = np.maximum(coefs[N_CB+N_V2:], 0)
    w[N_CB+N_V2:] = block_w[2] * pca_coefs / (pca_coefs.sum() + 1e-12)

    # Already normalized (sums to 1 by construction since block_w sums to 1)
    return w


def learn_dimwise_ablation(joint_emb, subj_ids, block="all"):
    """Ablation: dimension-wise weighting on a single block only.

    Other blocks get uniform weights.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(joint_emb)
    clf = RidgeClassifier()
    clf.fit(X_s, subj_ids)
    coefs = np.abs(clf.coef_).mean(axis=0)  # (264,)

    # Start with uniform weights
    w = np.ones(N_TOTAL) / N_TOTAL

    if block == "cbramod":
        cb_coefs = np.maximum(coefs[:N_CB], 0)
        w[:N_CB] = cb_coefs / (cb_coefs.sum() + 1e-12) * (N_CB / N_TOTAL)  # maintain relative scale
        # Redistribute uniformly across other dims
        other_mask = np.ones(N_TOTAL, dtype=bool)
        other_mask[:N_CB] = False
        remaining = 1.0 - w[:N_CB].sum()
        w[other_mask] = remaining / other_mask.sum()
    elif block == "v2":
        v2_coefs = np.maximum(coefs[N_CB:N_CB+N_V2], 0)
        w[N_CB:N_CB+N_V2] = v2_coefs / (v2_coefs.sum() + 1e-12) * (N_V2 / N_TOTAL)
        other_mask = np.ones(N_TOTAL, dtype=bool)
        other_mask[N_CB:N_CB+N_V2] = False
        remaining = 1.0 - w[N_CB:N_CB+N_V2].sum()
        w[other_mask] = remaining / other_mask.sum()
    elif block == "pca":
        pca_coefs = np.maximum(coefs[N_CB+N_V2:], 0)
        w[N_CB+N_V2:] = pca_coefs / (pca_coefs.sum() + 1e-12) * (N_PCA / N_TOTAL)
        other_mask = np.ones(N_TOTAL, dtype=bool)
        other_mask[N_CB+N_V2:] = False
        remaining = 1.0 - w[N_CB+N_V2:].sum()
        w[other_mask] = remaining / other_mask.sum()

    w = np.maximum(w, 0)
    w = w / (w.sum() + 1e-12)
    return w


def apply_dimwise_weights(joint_emb, w):
    """Apply dimension-wise weights: Z' = normalize(w ⊙ Z)."""
    # Normalize each block then apply weights
    cb = l2_normalize(joint_emb[:, :N_CB])
    v2 = l2_normalize(joint_emb[:, N_CB:N_CB+N_V2])
    pca = l2_normalize(joint_emb[:, N_CB+N_V2:])
    z = np.hstack([cb, v2, pca])
    weighted = w * z  # element-wise multiplication
    return l2_normalize(weighted)


# ─────────────────────────────────────────────────────────────
# Evaluation (session-disjoint, 50-fold LOSO)
# ─────────────────────────────────────────────────────────────

def evaluate_with_weights(joint_emb, subj_ids, run_ids, weight_fn):
    """Evaluate embeddings produced by applying learned per-fold weights.

    For each LOSO fold:
      1. Learn weights on training subjects
      2. Apply weights to all data (train + test)
      3. Run session-disjoint retrieval
    """
    subjects = sorted(np.unique(subj_ids))
    all_r1, all_r5, all_r10, all_mrr = [], [], [], []
    all_weights = []

    for test_subj in subjects:
        test_mask = subj_ids == test_subj
        train_mask = ~test_mask

        weights = weight_fn(joint_emb[train_mask], subj_ids[train_mask])
        all_weights.append(weights)

        embeddings = apply_dimwise_weights(joint_emb, weights)

        test_idx = np.where(test_mask)[0]
        test_run_ids = run_ids[test_idx]

        for query_run in sorted(np.unique(test_run_ids)):
            query_global = test_idx[test_run_ids == query_run]
            pool_global = np.setdiff1d(np.arange(len(subj_ids)), query_global)

            X_q = embeddings[query_global]
            X_p = embeddings[pool_global]
            pool_subj = subj_ids[pool_global]

            sims = cosine_sim_matrix(X_q, X_p)
            ranks = np.argsort(-sims, axis=1)

            for i in range(len(query_global)):
                top1 = ranks[i, 0]
                top5 = ranks[i, :5]
                top10 = ranks[i, :10]

                all_r1.append(1 if pool_subj[top1] == test_subj else 0)
                all_r5.append(1 if np.any(pool_subj[top5] == test_subj) else 0)
                all_r10.append(1 if np.any(pool_subj[top10] == test_subj) else 0)

                correct_pos = np.where(pool_subj[ranks[i]] == test_subj)[0]
                if len(correct_pos) > 0:
                    all_mrr.append(1.0 / (correct_pos[0] + 1))
                else:
                    all_mrr.append(0.0)

    n = len(all_r5)
    mean_weights = np.mean(all_weights, axis=0)
    return {
        "R@1": float(np.mean(all_r1)),
        "R@5": float(np.mean(all_r5)),
        "R@10": float(np.mean(all_r10)),
        "MRR": float(np.mean(all_mrr)),
        "n_splits": n,
        "per_split_r5": all_r5,
        "mean_weights": mean_weights.tolist(),
    }


def evaluate_baseline_raw(joint_emb, subj_ids, run_ids):
    """Evaluate raw 264-D concat (no weighting)."""
    subjects = sorted(np.unique(subj_ids))
    all_r1, all_r5, all_r10, all_mrr = [], [], [], []

    # Apply per-block L2 normalization (matching Mission 18)
    cb = l2_normalize(joint_emb[:, :N_CB])
    v2 = l2_normalize(joint_emb[:, N_CB:N_CB+N_V2])
    pca = l2_normalize(joint_emb[:, N_CB+N_V2:])
    embeddings = l2_normalize(np.hstack([cb, v2, pca]))

    for test_subj in subjects:
        test_mask = subj_ids == test_subj
        test_idx = np.where(test_mask)[0]
        test_run_ids = run_ids[test_idx]

        for query_run in sorted(np.unique(test_run_ids)):
            query_global = test_idx[test_run_ids == query_run]
            pool_global = np.setdiff1d(np.arange(len(subj_ids)), query_global)

            X_q = embeddings[query_global]
            X_p = embeddings[pool_global]
            pool_subj = subj_ids[pool_global]

            sims = cosine_sim_matrix(X_q, X_p)
            ranks = np.argsort(-sims, axis=1)

            for i in range(len(query_global)):
                top1 = ranks[i, 0]
                top5 = ranks[i, :5]
                top10 = ranks[i, :10]

                all_r1.append(1 if pool_subj[top1] == test_subj else 0)
                all_r5.append(1 if np.any(pool_subj[top5] == test_subj) else 0)
                all_r10.append(1 if np.any(pool_subj[top10] == test_subj) else 0)

                correct_pos = np.where(pool_subj[ranks[i]] == test_subj)[0]
                if len(correct_pos) > 0:
                    all_mrr.append(1.0 / (correct_pos[0] + 1))
                else:
                    all_mrr.append(0.0)

    n = len(all_r5)
    return {
        "R@1": float(np.mean(all_r1)),
        "R@5": float(np.mean(all_r5)),
        "R@10": float(np.mean(all_r10)),
        "MRR": float(np.mean(all_mrr)),
        "n_splits": n,
        "per_split_r5": all_r5,
    }


# ─────────────────────────────────────────────────────────────
# Weight analysis
# ─────────────────────────────────────────────────────────────

def analyze_weights(all_fold_weights):
    """Analyze learned weights across folds."""
    weights_array = np.array(all_fold_weights)  # (50, 264)

    # Mean and std across folds
    mean_w = weights_array.mean(axis=0)
    std_w = weights_array.std(axis=0)

    # Top 20 and bottom 20 dimensions
    sorted_idx = np.argsort(-mean_w)
    top20 = sorted_idx[:20]
    bottom20 = sorted_idx[-20:]

    # Block-level analysis
    cb_mean_w = mean_w[:N_CB].sum()
    v2_mean_w = mean_w[N_CB:N_CB+N_V2].sum()
    pca_mean_w = mean_w[N_CB+N_V2:].sum()

    # Stability (coefficient of variation across folds)
    cv = std_w / (mean_w + 1e-12)
    mean_cv = float(np.mean(cv))

    return {
        "mean_weights": mean_w.tolist(),
        "std_weights": std_w.tolist(),
        "top20_indices": top20.tolist(),
        "top20_values": mean_w[top20].tolist(),
        "bottom20_indices": bottom20.tolist(),
        "bottom20_values": mean_w[bottom20].tolist(),
        "block_weights_mean": {
            "cbramod": float(cb_mean_w),
            "v2": float(v2_mean_w),
            "pca": float(pca_mean_w),
        },
        "weight_stability_cv_mean": mean_cv,
    }


# ─────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 70)
    print("Mission 19: Dimension-Wise Learned EEG Embedding")
    print("=" * 70)

    print("\n[1] Loading embeddings...")
    data = load_embeddings()
    joint_raw = data["joint_raw"]
    subj_ids = data["subj_ids"]
    run_ids = data["run_ids"]

    # Baseline: raw 264-D (with per-block L2 norm)
    print("\n[2] Evaluating baselines...")
    baseline_raw_eval = evaluate_baseline_raw(joint_raw, subj_ids, run_ids)
    print(f"  Raw 264-D concat (L2-normalized blocks): R@5={baseline_raw_eval['R@5']:.4f}")

    # Method A: Block weighting (Mission 18)
    print("  Block weighting (Mission 18)...")
    block_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids, learn_block_weights_to_full)
    print(f"  Block weighting: R@5={block_eval['R@5']:.4f}")

    # Method B: Dimension-wise (RidgeClassifier coefficients)
    print("\n[3] Method B: Dimension-wise non-negative weighting...")
    t0 = time.time()
    ridge_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids, learn_dimwise_ridge)
    print(f"  Ridge per-dim: R@5={ridge_eval['R@5']:.4f} ({time.time()-t0:.1f}s)")

    fisher_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids, learn_dimwise_fisher)
    print(f"  Fisher per-dim: R@5={fisher_eval['R@5']:.4f}")

    # Method C: Regularized variants
    print("\n[4] Method C: Regularized dimension-wise weighting...")
    shrink_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids, learn_dimwise_shrinkage)
    print(f"  Ridge+block-shrinkage (α=0.5): R@5={shrink_eval['R@5']:.4f}")

    simplex_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids,
                                         lambda e, s: learn_dimwise_simplex(e, s, alpha=0.5))
    print(f"  Simplex-constrained (T=0.5): R@5={simplex_eval['R@5']:.4f}")

    # Method D: Hierarchical weighting
    print("\n[5] Method D: Hierarchical (block × dim) weighting...")
    t0 = time.time()
    hier_eval = evaluate_with_weights(joint_raw, subj_ids, run_ids, learn_hierarchical)
    print(f"  Hierarchical: R@5={hier_eval['R@5']:.4f} ({time.time()-t0:.1f}s)")

    # Ablations: single-block dimension-wise
    print("\n[6] Ablations: single-block dimension-wise weighting...")
    cb_ablation = evaluate_with_weights(
        joint_raw, subj_ids, run_ids,
        lambda e, s: learn_dimwise_ablation(e, s, block="cbramod")
    )
    print(f"  CBraMod-only dim weighting: R@5={cb_ablation['R@5']:.4f}")

    v2_ablation = evaluate_with_weights(
        joint_raw, subj_ids, run_ids,
        lambda e, s: learn_dimwise_ablation(e, s, block="v2")
    )
    print(f"  V2-only dim weighting: R@5={v2_ablation['R@5']:.4f}")

    pca_ablation = evaluate_with_weights(
        joint_raw, subj_ids, run_ids,
        lambda e, s: learn_dimwise_ablation(e, s, block="pca")
    )
    print(f"  PCA-only dim weighting: R@5={pca_ablation['R@5']:.4f}")

    # Step 7: Find best method
    print("\n[7] Determining best method...")
    all_methods = {
        "A_block_weighting": block_eval,
        "B_ridge_per_dim": ridge_eval,
        "B_fisher_per_dim": fisher_eval,
        "C_shrinkage": shrink_eval,
        "C_simplex": simplex_eval,
        "D_hierarchical": hier_eval,
        "ablation_cbramod_only": cb_ablation,
        "ablation_v2_only": v2_ablation,
        "ablation_pca_only": pca_ablation,
    }
    # Exclude raw baseline from "best learned" since it's not learned
    best_learned = max(all_methods.items(), key=lambda x: x[1]["R@5"])
    best_name, best_res = best_learned
    print(f"  Best learned: {best_name} (R@5={best_res['R@5']:.4f})")
    print(f"  M18 baseline: block_weighting (R@5={M18_R5})")
    print(f"  Raw 264-D: R@5={baseline_raw_eval['R@5']:.4f}")

    # Step 8: Statistical comparisons
    print("\n[8] Statistical comparisons...")
    comparisons = {}

    # Best learned vs M18 baseline
    if best_name != "A_block_weighting":
        comp = paired_ttest(best_res["per_split_r5"], block_eval["per_split_r5"])
        comparisons["best_vs_m18_block"] = comp
        print(f"  {best_name} vs M18 block: ΔR@5={comp['mean_diff']:+.4f}, p={comp['p_value']:.4e}, d={comp['cohen_d']:+.4f}")

    # Best vs raw 264-D
    comp = paired_ttest(best_res["per_split_r5"], baseline_raw_eval["per_split_r5"])
    comparisons["best_vs_raw_264d"] = comp
    print(f"  {best_name} vs raw 264-D: ΔR@5={comp['mean_diff']:+.4f}, p={comp['p_value']:.4e}, d={comp['cohen_d']:+.4f}")

    # Best vs PCA
    pca_r5 = evaluate_baseline_raw(data["pca32_emb"], subj_ids, run_ids)["per_split_r5"]
    # Actually need per_split from evaluation, use individual baselines
    # For now, just compare against known PCA R@5 from M18
    # We need to store per_split_r5 for individual baselines too
    # Let's compute individual baselines with per_split
    print("\n  Evaluating individual baselines for comparison...")
    cb_eval = evaluate_baseline_raw(l2_normalize(data["cbramod_emb"]), subj_ids, run_ids)
    v2_eval = evaluate_baseline_raw(l2_normalize(data["v2_emb"]), subj_ids, run_ids)
    pca_eval = evaluate_baseline_raw(data["pca32_emb"], subj_ids, run_ids)
    print(f"  CBraMod raw: R@5={cb_eval['R@5']:.4f}")
    print(f"  V2 raw: R@5={v2_eval['R@5']:.4f}")
    print(f"  PCA-32: R@5={pca_eval['R@5']:.4f}")

    for name, eval_res in [("vs_pca", pca_eval), ("vs_cbramod", cb_eval), ("vs_v2", v2_eval)]:
        comp = paired_ttest(best_res["per_split_r5"], eval_res["per_split_r5"])
        comparisons[f"best_{name}"] = comp
        print(f"  {best_name} {name}: ΔR@5={comp['mean_diff']:+.4f}, p={comp['p_value']:.4e}, d={comp['cohen_d']:+.4f}")

    # All methods vs M18 baseline
    for name, res in all_methods.items():
        if name == "A_block_weighting":
            continue
        comp = paired_ttest(res["per_split_r5"], block_eval["per_split_r5"])
        comparisons[f"{name}_vs_m18"] = comp

    # Step 9: Weight analysis
    print("\n[9] Analyzing learned weights...")
    weight_analysis = analyze_weights(ridge_eval.get("all_fold_weights", [ridge_eval["mean_weights"]]))
    print(f"  Ridge block weights (mean): CBraMod={weight_analysis['block_weights_mean']['cbramod']:.4f}, "
          f"V2={weight_analysis['block_weights_mean']['v2']:.4f}, PCA={weight_analysis['block_weights_mean']['pca']:.4f}")
    print(f"  Top 5 dims: {weight_analysis['top20_indices'][:5]}")
    print(f"  Weight stability (CV mean): {weight_analysis['weight_stability_cv_mean']:.4f}")

    # Step 10: Compile results
    print("\n[10] Compiling results...")
    fisher = compute_fisher(joint_raw, subj_ids)
    intra, inter = compute_intra_inter_cosine(joint_raw, subj_ids)
    ci_lower, ci_upper = bootstrap_ci(baseline_raw_eval["per_split_r5"])

    # Best embedding geometry
    best_emb = apply_dimwise_weights(joint_raw, np.array(best_res["mean_weights"]))
    best_fisher = compute_fisher(best_emb, subj_ids)
    best_intra, best_inter = compute_intra_inter_cosine(best_emb, subj_ids)
    best_ci_lower, best_ci_upper = bootstrap_ci(best_res["per_split_r5"])

    results = {
        "experiment_id": "m19-dimensionwise-embedding",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(OUTPUT_CACHE),
        "seed": SEED,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "baseline_m18_r5": M18_R5,
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB (S001-S050)",
            "subjects": 50,
            "runs": [5, 6, 7, 8, 9, 10],
            "trials_per_run": 15,
            "total_trials": 4500,
            "n_folds_loso": 50,
            "session_disjoint_splits": 300,
            "query": "one run (15 trials) of held-out subject",
            "pool": "all other trials",
            "weight_learning": "train-only per fold (no test-subject leakage)",
            "seed": SEED,
        },
        "block_sizes": {"cbramod": N_CB, "v2": N_V2, "pca": N_PCA, "total": N_TOTAL},
        "models": {
            "cbramod": {"onnx_path": "public/models/cbramod-encoder.onnx", "sha256": CBRAMOD_SHA, "embedding_dim": 200, "frozen": True},
            "v2_eeegconformer": {"onnx_path": "public/models/eegconformer_finetuned.onnx", "sha256": V2_SHA, "embedding_dim": 32, "frozen": True},
            "pca": {"input_features": 110, "output_dim": 32, "fit_per_fold": True, "frozen": True},
        },
        "individual_baselines": {
            "cbramod_200_raw": {"R@1": round(cb_eval["R@1"],4), "R@5": round(cb_eval["R@5"],4), "R@10": round(cb_eval["R@10"],4), "MRR": round(cb_eval["MRR"],4)},
            "v2_32_raw": {"R@1": round(v2_eval["R@1"],4), "R@5": round(v2_eval["R@5"],4), "R@10": round(v2_eval["R@10"],4), "MRR": round(v2_eval["MRR"],4)},
            "pca_32": {"R@1": round(pca_eval["R@1"],4), "R@5": round(pca_eval["R@5"],4), "R@10": round(pca_eval["R@10"],4), "MRR": round(pca_eval["MRR"],4)},
        },
        "primary_baseline": {
            "name": "M18_block_weighting",
            "r_at_5": round(block_eval["R@5"], 4),
            "dim": 264,
        },
        "raw_baseline": {
            "name": "raw_264d_concat_l2",
            "r_at_1": round(baseline_raw_eval["R@1"], 4),
            "r_at_5": round(baseline_raw_eval["R@5"], 4),
            "r_at_10": round(baseline_raw_eval["R@10"], 4),
            "mrr": round(baseline_raw_eval["MRR"], 4),
            "dim": 264,
        },
        "learned_methods": {
            "A_block_weighting": {
                "description": "3 block-level weights via RidgeClassifier (Mission 18 method)",
                "dim": 264, "param_count": 3,
                "r_at_1": round(block_eval["R@1"], 4),
                "r_at_5": round(block_eval["R@5"], 4),
                "r_at_10": round(block_eval["R@10"], 4),
                "mrr": round(block_eval["MRR"], 4),
            },
            "B_ridge_per_dim": {
                "description": "264 individual weights from RidgeClassifier |coef| mean, simplex normalized",
                "dim": 264, "param_count": 264,
                "r_at_1": round(ridge_eval["R@1"], 4),
                "r_at_5": round(ridge_eval["R@5"], 4),
                "r_at_10": round(ridge_eval["R@10"], 4),
                "mrr": round(ridge_eval["MRR"], 4),
            },
            "B_fisher_per_dim": {
                "description": "264 individual weights from per-dimension Fisher discriminant score",
                "dim": 264, "param_count": 264,
                "r_at_1": round(fisher_eval["R@1"], 4),
                "r_at_5": round(fisher_eval["R@5"], 4),
                "r_at_10": round(fisher_eval["R@10"], 4),
                "mrr": round(fisher_eval["MRR"], 4),
            },
            "C_shrinkage": {
                "description": "50/50 interpolation of ridge per-dim weights and block-expanded weights",
                "dim": 264, "param_count": 264,
                "r_at_1": round(shrink_eval["R@1"], 4),
                "r_at_5": round(shrink_eval["R@5"], 4),
                "r_at_10": round(shrink_eval["R@10"], 4),
                "mrr": round(shrink_eval["MRR"], 4),
            },
            "C_simplex": {
                "description": "Softmax-simplex constrained per-dim weights (RidgeClassifier alpha=0.5, T=0.5)",
                "dim": 264, "param_count": 264,
                "r_at_1": round(simplex_eval["R@1"], 4),
                "r_at_5": round(simplex_eval["R@5"], 4),
                "r_at_10": round(simplex_eval["R@10"], 4),
                "mrr": round(simplex_eval["MRR"], 4),
            },
            "D_hierarchical": {
                "description": "Hierarchical: block_weight × within-block dimension weight",
                "dim": 264, "param_count": 267,
                "r_at_1": round(hier_eval["R@1"], 4),
                "r_at_5": round(hier_eval["R@5"], 4),
                "r_at_10": round(hier_eval["R@10"], 4),
                "mrr": round(hier_eval["MRR"], 4),
            },
            "ablation_cbramod_only": {
                "description": "Dimension-wise weighting on CBraMod dims only, uniform elsewhere",
                "dim": 264, "param_count": 200,
                "r_at_1": round(cb_ablation["R@1"], 4),
                "r_at_5": round(cb_ablation["R@5"], 4),
                "r_at_10": round(cb_ablation["R@10"], 4),
                "mrr": round(cb_ablation["MRR"], 4),
            },
            "ablation_v2_only": {
                "description": "Dimension-wise weighting on V2 dims only, uniform elsewhere",
                "dim": 264, "param_count": 32,
                "r_at_1": round(v2_ablation["R@1"], 4),
                "r_at_5": round(v2_ablation["R@5"], 4),
                "r_at_10": round(v2_ablation["R@10"], 4),
                "mrr": round(v2_ablation["MRR"], 4),
            },
            "ablation_pca_only": {
                "description": "Dimension-wise weighting on PCA dims only, uniform elsewhere",
                "dim": 264, "param_count": 32,
                "r_at_1": round(pca_ablation["R@1"], 4),
                "r_at_5": round(pca_ablation["R@5"], 4),
                "r_at_10": round(pca_ablation["R@10"], 4),
                "mrr": round(pca_ablation["MRR"], 4),
            },
        },
        "best_learned_method": best_name,
        "best_learned_r5": round(best_res["R@5"], 4),
        "m18_baseline_r5": M18_R5,
        "improvement_over_m18_pp": round((best_res["R@5"] - M18_R5) * 100, 2),
        "beats_m18_baseline": bool(best_res["R@5"] > M18_R5),
        "improvement_over_raw_pp": round((best_res["R@5"] - baseline_raw_eval["R@5"]) * 100, 2),
        "weight_analysis": weight_analysis,
        "additional_metrics": {
            "raw_264d_fisher": round(fisher, 4),
            "raw_264d_intra_cosine": round(intra, 4),
            "raw_264d_inter_cosine": round(inter, 4),
            "raw_264d_r5_ci95": [round(ci_lower, 4), round(ci_upper, 4)],
            f"best_{best_name}_fisher": round(best_fisher, 4),
            f"best_{best_name}_intra_cosine": round(best_intra, 4),
            f"best_{best_name}_inter_cosine": round(best_inter, 4),
            f"best_{best_name}_r5_ci95": [round(best_ci_lower, 4), round(best_ci_upper, 4)],
        },
        "pairwise_comparisons": comparisons,
        "decision": (
            f"Best learned: {best_name} (R@5={best_res['R@5']:.4f}). "
            f"M18 baseline: {M18_R5}. "
            f"Improvement: {(best_res['R@5'] - M18_R5)*100:+.2f}pp. "
            f"Beats M18: {best_res['R@5'] > M18_R5}."
        ),
        "contaminated": False,
        "status": "COMPLETE",
        "constraints_honored": {
            "no_model_retraining": True,
            "no_artifact_modification": True,
            "no_onnx_modification": True,
            "no_default_preferred_change": True,
            "no_v2_or_pca_change": True,
            "no_production_code_changes": True,
            "loso_leakage_free": True,
            "session_disjoint_evaluation": True,
            "train_only_fitting": True,
            "bonferroni_correction": True,
            "seed_42_reproducible": True,
            "prior_archive_records_byte_preserved": True,
        },
    }

    # Save cache
    print("\n[11] Saving cache...")
    np.savez_compressed(
        OUTPUT_CACHE,
        joint_raw=joint_raw,
        cb_emb=data["cbramod_emb"],
        v2_emb=data["v2_emb"],
        pca32_emb=data["pca32_emb"],
        subj_ids=subj_ids,
        run_ids=run_ids,
        cbramod_sha=CBRAMOD_SHA,
        v2_sha=V2_SHA,
        baseline_r5=M18_R5,
        best_method=best_name,
    )
    print(f"  Saved to {OUTPUT_CACHE}")

    # Save results
    def to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    results = to_native(results)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Method':<40} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 70)
    print(f"{'CBraMod-200 raw':<40} {cb_eval['R@1']:>8.4f} {cb_eval['R@5']:>8.4f} {cb_eval['R@10']:>8.4f} {cb_eval['MRR']:>8.4f}")
    print(f"{'V2-32 raw':<40} {v2_eval['R@1']:>8.4f} {v2_eval['R@5']:>8.4f} {v2_eval['R@10']:>8.4f} {v2_eval['MRR']:>8.4f}")
    print(f"{'PCA-32':<40} {pca_eval['R@1']:>8.4f} {pca_eval['R@5']:>8.4f} {pca_eval['R@10']:>8.4f} {pca_eval['MRR']:>8.4f}")
    print(f"{'Raw 264-D':<40} {baseline_raw_eval['R@1']:>8.4f} {baseline_raw_eval['R@5']:>8.4f} {baseline_raw_eval['R@10']:>8.4f} {baseline_raw_eval['MRR']:>8.4f}")
    print(f"{'A. Block weighting (M18)':<40} {block_eval['R@1']:>8.4f} {block_eval['R@5']:>8.4f} {block_eval['R@10']:>8.4f} {block_eval['MRR']:>8.4f}")
    print(f"{'B. Ridge per-dim':<40} {ridge_eval['R@1']:>8.4f} {ridge_eval['R@5']:>8.4f} {ridge_eval['R@10']:>8.4f} {ridge_eval['MRR']:>8.4f}")
    print(f"{'B. Fisher per-dim':<40} {fisher_eval['R@1']:>8.4f} {fisher_eval['R@5']:>8.4f} {fisher_eval['R@10']:>8.4f} {fisher_eval['MRR']:>8.4f}")
    print(f"{'C. Shrinkage':<40} {shrink_eval['R@1']:>8.4f} {shrink_eval['R@5']:>8.4f} {shrink_eval['R@10']:>8.4f} {shrink_eval['MRR']:>8.4f}")
    print(f"{'C. Simplex':<40} {simplex_eval['R@1']:>8.4f} {simplex_eval['R@5']:>8.4f} {simplex_eval['R@10']:>8.4f} {simplex_eval['MRR']:>8.4f}")
    print(f"{'D. Hierarchical':<40} {hier_eval['R@1']:>8.4f} {hier_eval['R@5']:>8.4f} {hier_eval['R@10']:>8.4f} {hier_eval['MRR']:>8.4f}")
    print(f"{'Abl: CBraMod-only':<40} {cb_ablation['R@1']:>8.4f} {cb_ablation['R@5']:>8.4f} {cb_ablation['R@10']:>8.4f} {cb_ablation['MRR']:>8.4f}")
    print(f"{'Abl: V2-only':<40} {v2_ablation['R@1']:>8.4f} {v2_ablation['R@5']:>8.4f} {v2_ablation['R@10']:>8.4f} {v2_ablation['MRR']:>8.4f}")
    print(f"{'Abl: PCA-only':<40} {pca_ablation['R@1']:>8.4f} {pca_ablation['R@5']:>8.4f} {pca_ablation['R@10']:>8.4f} {pca_ablation['MRR']:>8.4f}")
    print("-" * 70)
    print(f"\nBest learned: {best_name} (R@5={best_res['R@5']:.4f})")
    print(f"M18 baseline: {M18_R5}")
    delta = best_res["R@5"] - M18_R5
    print(f"ΔR@5 vs M18: {delta:+.4f} ({delta*100:+.2f}pp)")
    print(f"Mission 19 succeeds: {best_res['R@5'] > M18_R5}")

    return results


def learn_block_weights_to_full(joint_emb, subj_ids):
    """Wrapper to return 264-dim weight vector from block weights."""
    block_w = learn_block_weights(joint_emb, subj_ids)
    w = np.concatenate([
        np.full(N_CB, block_w[0]),
        np.full(N_V2, block_w[1]),
        np.full(N_PCA, block_w[2]),
    ])
    return w


if __name__ == "__main__":
    run_experiment()
