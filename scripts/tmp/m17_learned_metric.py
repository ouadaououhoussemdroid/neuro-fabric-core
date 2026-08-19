#!/usr/bin/env python3
"""
Mission 17 — Learned Similarity Projection Experiment

Tests whether a linear projection W: R^200 -> R^200 (or R^49 for LDA),
trained with subject-identity supervision, can improve CBraMod-200
retrieval beyond raw cosine NN, centroid matching, and PCA baselines.

CBraMod remains COMPLETELY FROZEN. W is a post-processing layer trained
on cached embeddings only.

Protocol:
  - 50-fold LOSO: train W on 49 subjects, evaluate on 1 held-out subject
  - Session-disjoint retrieval: for each held-out subject,
    evaluate all 6 runs as queries (300 total session-disjoint splits)
  - Pool per query: all trials NOT in the query (subject, run) pair
  - R@1, R@5, R@10, MRR computed

Methods compared:
  - Baseline A: Raw CBraMod-200 cosine NN
  - Baseline B: Centroid-based CBraMod-200
  - Baseline C: PCA-32 bandpower
  - Baseline D: V2-32 cosine NN
  - Method 1: LDA projection (closed-form, Fisher discriminant)
  - Method 2: SupCon-trained linear projection (PyTorch)

Usage: python scripts/tmp/m17_learned_metric.py
"""

import json
import os
import sys
import time
import warnings
import numpy as np
import re
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ============================================================
# Configuration
# ============================================================
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", ".cbramod_cross_session_cache.npz"
)
CACHE_PATH = os.path.normpath(CACHE_PATH)

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "m17_learned_metric_results.json"
)
RESULTS_PATH = os.path.normpath(RESULTS_PATH)

SEED = 42
N_SUBJECTS = 50
N_RUNS_PER_SUBJECT = 6
TRIALS_PER_RUN = 15
N_FOLDS = 50
N_BOOTSTRAP = 2000

# Try importing PyTorch for SupCon
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    print(f"PyTorch available: {torch.__version__}")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available — skipping SupCon method")

# sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def load_cache():
    """Load cached embeddings from Mission 11."""
    data = np.load(CACHE_PATH)
    cb_emb = data["cb_emb"]
    v2_emb = data["v2_emb"]
    bandpower = data["bandpower"]
    subj_ids = data["subj_ids"]
    run_ids = data["run_ids"]
    mi_labels = data["mi_labels"]

    # Convert subj_ids to numeric
    if subj_ids.dtype.kind in ("U", "S"):
        subj_numeric = np.array([int(str(s).replace("S", "").replace("s", "")) for s in subj_ids])
    else:
        subj_numeric = subj_ids.astype(int)

    # Extract run numbers
    run_nums = np.zeros(len(run_ids), dtype=int)
    for i, r in enumerate(run_ids):
        s = str(r)
        nums = re.findall(r"\d+", s)
        if nums:
            run_nums[i] = int(nums[-1])

    return {
        "cb_emb": cb_emb,
        "v2_emb": v2_emb,
        "bandpower": bandpower,
        "subj_ids": subj_numeric,
        "run_nums": run_nums,
        "mi_labels": mi_labels.astype(int),
    }


def l2_normalize(X):
    """L2 normalize rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def compute_recall_at_k(queries, pool, query_subj_ids, pool_subj_ids, k_values=[1, 5, 10]):
    """Compute R@K (subject-recall) for a batch of queries against a pool."""
    sims = queries @ pool.T  # (n_queries, n_pool)

    results = {}
    for k in k_values:
        if k > pool.shape[0]:
            k = pool.shape[0]
        topk = np.argsort(-sims, axis=1)[:, :k]
        retrieved_subjs = pool_subj_ids[topk]
        hits = np.any(retrieved_subjs == query_subj_ids[:, None], axis=1)
        results[f"R@{k}"] = float(np.mean(hits))
    return results


def compute_mrr(queries, pool, query_subj_ids, pool_subj_ids):
    """Compute Mean Reciprocal Rank for subject retrieval."""
    sims = queries @ pool.T
    mrrs = []
    for i in range(len(queries)):
        order = np.argsort(-sims[i])
        retrieved_subjs = pool_subj_ids[order]
        ranks = np.where(retrieved_subjs == query_subj_ids[i])[0]
        if len(ranks) > 0:
            mrrs.append(1.0 / (ranks[0] + 1))
        else:
            mrrs.append(0.0)
    return float(np.mean(mrrs))


def evaluate_session_disjoint(X, subj_ids, run_nums, k_values=[1, 5, 10]):
    """
    Session-disjoint retrieval evaluation.

    For each (subject, run) pair:
      - Query = all 15 trials in that run
      - Pool = all trials NOT in that (subject, run) pair (same subject diff runs + all other subjects)
      - Compute R@K and MRR

    Total splits: 50 subjects × 6 runs = 300
    """
    Xn = l2_normalize(X)
    all_results = {f"R@{k}": [] for k in k_values}
    mrr_values = []

    subjects = sorted(np.unique(subj_ids))
    for subj in subjects:
        subj_idx = np.where(subj_ids == subj)[0]

        for run in sorted(np.unique(run_nums)):
            run_idx = subj_idx[run_nums[subj_idx] == run]

            if len(run_idx) == 0:
                continue

            # Query: this run's trials
            query_idx = run_idx
            # Pool: everything else (same subject diff runs + all other subjects)
            pool_idx = np.setdiff1d(np.arange(len(subj_ids)), query_idx)

            queries = Xn[query_idx]
            pool = Xn[pool_idx]

            query_subjs = subj_ids[query_idx]
            pool_subjs = subj_ids[pool_idx]

            r = compute_recall_at_k(queries, pool, query_subjs, pool_subjs, k_values)
            mrr = compute_mrr(queries, pool, query_subjs, pool_subjs)

            for k in k_values:
                all_results[f"R@{k}"].append(r[f"R@{k}"])
            mrr_values.append(mrr)

    summary = {}
    for k in k_values:
        vals = np.array(all_results[f"R@{k}"])
        summary[f"R@{k}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "ci95_lower": float(np.percentile(np.random.RandomState(SEED).choice(vals, size=(N_BOOTSTRAP, len(vals))).mean(axis=1), 2.5)),
            "ci95_upper": float(np.percentile(np.random.RandomState(SEED).choice(vals, size=(N_BOOTSTRAP, len(vals))).mean(axis=1), 97.5)),
            "n_splits": len(vals),
        }
    summary["MRR"] = {
        "mean": float(np.mean(mrr_values)),
        "std": float(np.std(mrr_values, ddof=1)),
        "n_splits": len(mrr_values),
    }
    return summary, all_results


def train_lda_projection(X_train, subj_ids_train, target_dim=None):
    """
    Train LDA projection for subject identity.

    Returns the LDA transformation matrix and the fitted model.
    """
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=None)
    lda.fit(X_train, subj_ids_train)

    # The scalings_ matrix projects from original space to LDA space
    # scalings_ shape: (n_features, n_classes - 1)
    W = lda.scalings_  # (200, 49) for 50 subjects
    if target_dim is not None and W.shape[1] > target_dim:
        W = W[:, :target_dim]

    return W, lda


def train_supcon_projection(X_train, subj_ids_train, input_dim=200, output_dim=200,
                            n_epochs=1000, lr=0.01, batch_size=256,
                            temperature=0.1, reg_weight=0.01, seed=SEED):
    """
    Train a linear projection W: R^input_dim -> R^output_dim
    using supervised contrastive loss.

    Uses SGD/Adam optimizer. W is a full-rank square matrix (possibly with regularization).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(subj_ids_train, dtype=torch.long)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    # Parameterize W
    W = torch.nn.Parameter(torch.eye(input_dim, output_dim) * 0.1)
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=reg_weight)

    n = len(X_scaled)
    rng = np.random.RandomState(seed)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Sample a batch
        batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X_batch = X_t[batch_idx]
        y_batch = y_t[batch_idx]

        # Project and normalize
        Z = torch.mm(X_batch, W)
        Z = F.normalize(Z, p=2, dim=1)

        # Compute similarity matrix
        sim = torch.mm(Z, Z.T) / temperature

        # Supervised contrastive loss
        y_onehot = (y_batch[:, None] == y_batch[None, :]).float()
        # Exclude self
        sim.fill_diagonal_(-1e9)
        y_onehot.fill_diagonal_(0)

        # Softmax
        exp_sim = torch.exp(sim)
        # Mask: only positive pairs
        pos_mask = y_onehot.bool()
        neg_mask = ~pos_mask & ~torch.eye(len(y_batch), dtype=torch.bool, device=y_batch.device)

        # Compute loss: -log(sum(exp(sim_pos)) / sum(exp(sim_all_non_self)))
        denom = exp_sim[neg_mask].sum() + exp_sim[pos_mask].sum()
        numer = exp_sim[pos_mask].sum()

        loss = -torch.log(numer / (denom + 1e-8))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"    SupCon epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}")

    return W.detach().numpy().T, scaler  # scaler for applying at test time


def apply_projection(X, W):
    """Apply projection W and L2 normalize."""
    Z = X @ W
    return l2_normalize(Z)


def centroid_based_retrieval(X, subj_ids, run_nums, k_values=[1, 5, 10]):
    """
    Centroid-based retrieval: compute per-subject centroids from training runs
    (runs != 5), then match run-5 queries to centroids.
    """
    Xn = l2_normalize(X)
    subjects = sorted(np.unique(subj_ids))

    # Compute centroids per subject (using runs 6-10 as training)
    centroids = {}
    for subj in subjects:
        subj_idx = np.where(subj_ids == subj)[0]
        train_idx = subj_idx[run_nums[subj_idx] != 5]
        if len(train_idx) > 0:
            centroid = Xn[train_idx].mean(axis=0)
            centroids[subj] = l2_normalize(centroid[None, :])[0]

    all_centroids = np.array([centroids[s] for s in subjects])

    # Evaluate: use run 5 as queries
    results = {f"R@{k}": [] for k in k_values}
    mrr_values = []

    for subj in subjects:
        run5_idx = np.where((subj_ids == subj) & (run_nums == 5))[0]
        if len(run5_idx) == 0:
            continue

        queries = Xn[run5_idx]
        subj_pos = subjects.index(subj)

        sims = queries @ all_centroids.T
        for k in k_values:
            topk = np.argsort(-sims, axis=1)[:, :k]
            hits = np.any(topk == subj_pos, axis=1)
            results[f"R@{k}"].append(float(np.mean(hits)))

        # MRR
        for i in range(len(queries)):
            order = np.argsort(-sims[i])
            rank = np.where(order == subj_pos)[0][0] + 1
            mrr_values.append(1.0 / rank)

    summary = {}
    for k in k_values:
        vals = np.array(results[f"R@{k}"])
        summary[f"R@{k}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "n_queries": len(vals),
        }
    summary["MRR"] = {
        "mean": float(np.mean(mrr_values)),
        "n_queries": len(mrr_values),
    }
    return summary


def v2_pca_results():
    """Return V2 and PCA results from existing benchmark archive for reference."""
    # From Mission 13/14 benchmark results
    return {
        "V2-32": {
            "R@1": 0.0687,
            "R@5": 0.2158,
            "R@10": 0.3364,
            "source": "Mission 13/14 (full 4500 dataset, 300 session-disjoint splits)"
        },
        "PCA-32": {
            "R@1": 0.4400,
            "R@5": 0.6920,
            "R@10": 0.7853,
            "source": "Mission 13/14 (full 4500 dataset, 300 session-disjoint splits)"
        }
    }


def paired_ttest(a, b):
    """Paired t-test between two arrays of per-split scores."""
    from scipy import stats
    diff = np.array(a) - np.array(b)
    t_stat, p_value = stats.ttest_rel(a, b)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    n = len(diff)
    sem = std_diff / np.sqrt(n)
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
    ci_lower = mean_diff - stats.t.ppf(0.975, n - 1) * sem
    ci_upper = mean_diff + stats.t.ppf(0.975, n - 1) * sem
    return {
        "mean_diff": mean_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "ci95_lower": float(ci_lower),
        "ci95_upper": float(ci_upper),
        "n_splits": n,
    }


def run_experiment():
    """Run the full learned metric experiment."""
    print("=" * 70)
    print("Mission 17 — Learned Similarity Projection Experiment")
    print("=" * 70)

    data = load_cache()
    print(f"\nCache loaded: cb_emb={data['cb_emb'].shape}, v2_emb={data['v2_emb'].shape}, "
          f"bandpower={data['bandpower'].shape}")
    print(f"  Subjects: {len(np.unique(data['subj_ids']))}, Runs: {sorted(np.unique(data['run_nums']))}")
    print(f"  MI labels: {np.bincount(data['mi_labels'])}")

    cb_emb = data["cb_emb"]
    subj_ids = data["subj_ids"]
    run_nums = data["run_nums"]
    v2_emb = data["v2_emb"]
    bandpower = data["bandpower"]

    # ============================================================
    # BASELINE A: Raw CBraMod-200 cosine NN (session-disjoint)
    # ============================================================
    print("\n" + "=" * 50)
    print("Baseline A: Raw CBraMod-200 cosine NN")
    print("=" * 50)
    t0 = time.time()
    raw_results, raw_per_split = evaluate_session_disjoint(cb_emb, subj_ids, run_nums)
    print(f"  R@1={raw_results['R@1']['mean']:.4f}  R@5={raw_results['R@5']['mean']:.4f}  R@10={raw_results['R@10']['mean']:.4f}")
    print(f"  MRR={raw_results['MRR']['mean']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # BASELINE B: Centroid-based CBraMod-200
    # ============================================================
    print("\n" + "=" * 50)
    print("Baseline B: Centroid-based CBraMod-200")
    print("=" * 50)
    t0 = time.time()
    centroid_results = centroid_based_retrieval(cb_emb, subj_ids, run_nums)
    print(f"  R@1={centroid_results['R@1']['mean']:.4f}  R@5={centroid_results['R@5']['mean']:.4f}  R@10={centroid_results['R@10']['mean']:.4f}")
    print(f"  MRR={centroid_results['MRR']['mean']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # BASELINE C: PCA-32 bandpower (from Mission 13/14)
    # ============================================================
    print("\n" + "=" * 50)
    print("Baseline C: PCA-32 bandpower (reference from Mission 13)")
    print("=" * 50)
    ref_results = v2_pca_results()
    pca_results = ref_results["PCA-32"]
    v2_results = ref_results["V2-32"]
    print(f"  PCA-32: R@1={pca_results['R@1']:.4f}  R@5={pca_results['R@5']:.4f}  R@10={pca_results['R@10']:.4f}")
    print(f"  V2-32:  R@1={v2_results['R@1']:.4f}  R@5={v2_results['R@5']:.4f}  R@10={v2_results['R@10']:.4f}")

    # ============================================================
    # BASELINE D: V2-32 cosine NN (session-disjoint, same protocol)
    # ============================================================
    print("\n" + "=" * 50)
    print("Baseline D: V2-32 cosine NN (this experiment, same protocol)")
    print("=" * 50)
    t0 = time.time()
    v2_raw_results, v2_per_split = evaluate_session_disjoint(v2_emb, subj_ids, run_nums)
    print(f"  R@1={v2_raw_results['R@1']['mean']:.4f}  R@5={v2_raw_results['R@5']['mean']:.4f}  R@10={v2_raw_results['R@10']['mean']:.4f}")
    print(f"  MRR={v2_raw_results['MRR']['mean']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Method 1: LDA projection
    # ============================================================
    print("\n" + "=" * 50)
    print("Method 1: LDA projection (Fisher discriminant)")
    print("=" * 50)

    Xn = l2_normalize(cb_emb)
    lda_results = {f"R@{k}": [] for k in [1, 5, 10]}
    lda_mrr = []
    lda_projections = []  # Store projection info
    fold_times = []

    subjects = sorted(np.unique(subj_ids))
    for fold_idx, test_subj in enumerate(subjects):
        if fold_idx % 10 == 0:
            print(f"  Fold {fold_idx+1}/{N_FOLDS}...", end=" ", flush=True)

        # Train/test split (LOSO)
        test_mask = subj_ids == test_subj
        train_mask = ~test_mask

        X_train = cb_emb[train_mask]
        y_train = subj_ids[train_mask]
        X_test = cb_emb[test_mask]

        t0 = time.time()
        # Train LDA
        W_lda, lda_model = train_lda_projection(X_train, y_train)
        fold_times.append(time.time() - t0)

        # Apply projection to test
        Z_test = apply_projection(X_test, W_lda)

        # Build pool: all other trials (same subject diff runs + all other subjects)
        # For session-disjoint: for each run, query = this run, pool = rest
        test_subj_idx = np.where(test_mask)[0]

        for run in sorted(np.unique(run_nums)):
            run_mask = run_nums[test_subj_idx] == run
            run_idx = test_subj_idx[run_mask]

            if len(run_idx) == 0:
                continue

            # Query: this run's trials
            queries = Z_test[run_mask]

            # Pool: all OTHER trials
            pool_idx = np.setdiff1d(np.arange(len(subj_ids)), run_idx)
            pool = apply_projection(cb_emb[pool_idx], W_lda)

            query_subjs = subj_ids[run_idx]
            pool_subjs = subj_ids[pool_idx]

            r = compute_recall_at_k(queries, pool, query_subjs, pool_subjs, [1, 5, 10])
            mrr = compute_mrr(queries, pool, query_subjs, pool_subjs)

            for k in [1, 5, 10]:
                lda_results[f"R@{k}"].append(r[f"R@{k}"])
            lda_mrr.append(mrr)

        if fold_idx % 10 == 0:
            print("done")

    lda_summary = {}
    for k in [1, 5, 10]:
        vals = np.array(lda_results[f"R@{k}"])
        bootstrap_rng = np.random.RandomState(SEED)
        boot_means = bootstrap_rng.choice(vals, size=(N_BOOTSTRAP, len(vals))).mean(axis=1)
        lda_summary[f"R@{k}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "ci95_lower": float(np.percentile(boot_means, 2.5)),
            "ci95_upper": float(np.percentile(boot_means, 97.5)),
            "n_splits": len(vals),
        }
    lda_summary["MRR"] = {
        "mean": float(np.mean(lda_mrr)),
        "std": float(np.std(lda_mrr, ddof=1)),
        "n_splits": len(lda_mrr),
    }
    lda_summary["mean_training_time_ms"] = float(np.mean(fold_times) * 1000)
    lda_summary["projection_shape"] = list(W_lda.shape)

    print(f"\n  LDA R@1={lda_summary['R@1']['mean']:.4f}  R@5={lda_summary['R@5']['mean']:.4f}  R@10={lda_summary['R@10']['mean']:.4f}")
    print(f"  LDA MRR={lda_summary['MRR']['mean']:.4f}")
    print(f"  Mean training time: {lda_summary['mean_training_time_ms']:.1f}ms per fold")

    # ============================================================
    # Method 2: SupCon projection (PyTorch)
    # ============================================================
    supcon_summary = None
    if HAS_TORCH:
        print("\n" + "=" * 50)
        print("Method 2: SupCon-trained linear projection (PyTorch)")
        print("=" * 50)

        supcon_results = {f"R@{k}": [] for k in [1, 5, 10]}
        supcon_mrr = []

        for fold_idx, test_subj in enumerate(subjects):
            if fold_idx % 10 == 0:
                print(f"  Fold {fold_idx+1}/{N_FOLDS}...", end=" ", flush=True)

            test_mask = subj_ids == test_subj
            train_mask = ~test_mask

            X_train = cb_emb[train_mask]
            y_train = subj_ids[train_mask]
            X_test = cb_emb[test_mask]

            # Train SupCon projection
            W_supcon, scaler = train_supcon_projection(
                X_train, y_train, input_dim=200, output_dim=200,
                n_epochs=300, lr=0.01, batch_size=256, temperature=0.1,
                reg_weight=0.001, seed=SEED + fold_idx
            )

            # Apply projection
            Z_test = apply_projection(scaler.transform(X_test), W_supcon)

            # Session-disjoint evaluation
            test_subj_idx = np.where(test_mask)[0]
            for run in sorted(np.unique(run_nums)):
                run_mask = run_nums[test_subj_idx] == run
                run_idx = test_subj_idx[run_mask]
                if len(run_idx) == 0:
                    continue
                queries = Z_test[run_mask]
                pool_idx = np.setdiff1d(np.arange(len(subj_ids)), run_idx)
                pool = apply_projection(scaler.transform(cb_emb[pool_idx]), W_supcon)

                query_subjs = subj_ids[run_idx]
                pool_subjs = subj_ids[pool_idx]

                r = compute_recall_at_k(queries, pool, query_subjs, pool_subjs, [1, 5, 10])
                mrr = compute_mrr(queries, pool, query_subjs, pool_subjs)

                for k in [1, 5, 10]:
                    supcon_results[f"R@{k}"].append(r[f"R@{k}"])
                supcon_mrr.append(mrr)

            if fold_idx % 10 == 0:
                print("done")

        supcon_summary = {}
        for k in [1, 5, 10]:
            vals = np.array(supcon_results[f"R@{k}"])
            supcon_summary[f"R@{k}"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "n_splits": len(vals),
            }
        supcon_summary["MRR"] = {
            "mean": float(np.mean(supcon_mrr)),
            "n_splits": len(supcon_mrr),
        }
        supcon_summary["projection_shape"] = [200, 200]

        print(f"\n  SupCon R@1={supcon_summary['R@1']['mean']:.4f}  R@5={supcon_summary['R@5']['mean']:.4f}  R@10={supcon_summary['R@10']['mean']:.4f}")
        print(f"  SupCon MRR={supcon_summary['MRR']['mean']:.4f}")
    else:
        print("\n  Skipping SupCon (PyTorch not available)")

    # ============================================================
    # Statistical Comparison
    # ============================================================
    print("\n" + "=" * 50)
    print("Statistical Comparison (paired t-test on per-split R@5)")
    print("=" * 50)

    comparisons = {}
    # LDA vs raw cosine
    comp_lda_vs_raw = paired_ttest(lda_results["R@5"], raw_per_split["R@5"])
    # LDA vs centroid
    # Note: centroid uses different query set (run 5 only), so need to compare differently
    # For fair comparison, also compute raw cosine on run-5-only for centroid comparison
    comp_lda_vs_centroid = {
        "note": "Centroid uses run-5-only queries; LDA uses all 6 runs (session-disjoint). Comparing different protocols.",
        "lda_r5": lda_summary["R@5"]["mean"],
        "centroid_r5": centroid_results["R@5"]["mean"],
        "delta": lda_summary["R@5"]["mean"] - centroid_results["R@5"]["mean"]
    }
    # LDA vs PCA (reference values)
    comp_lda_vs_pca = {
        "note": "PCA results from Mission 13 reference (300 splits, different protocol)",
        "lda_r5": lda_summary["R@5"]["mean"],
        "pca_r5": pca_results["R@5"],
        "delta": lda_summary["R@5"]["mean"] - pca_results["R@5"]
    }

    # LDA vs V2 raw (same protocol, 300 splits)
    comp_lda_vs_v2 = paired_ttest(lda_results["R@5"], v2_per_split["R@5"])

    print(f"\n  LDA vs Raw CBraMod R@5:")
    print(f"    Δ = {comp_lda_vs_raw['mean_diff']:+.4f}, t={comp_lda_vs_raw['t_statistic']:.2f}, "
          f"p={comp_lda_vs_raw['p_value']:.2e}, d={comp_lda_vs_raw['cohen_d']:+.3f}")
    print(f"    CI95: [{comp_lda_vs_raw['ci95_lower']:+.4f}, {comp_lda_vs_raw['ci95_upper']:+.4f}]")

    print(f"\n  LDA vs Centroid (different protocols):")
    print(f"    LDA R@5={lda_summary['R@5']['mean']:.4f}, Centroid R@5={centroid_results['R@5']['mean']:.4f}")
    print(f"    Δ = {comp_lda_vs_centroid['delta']:+.4f}")

    print(f"\n  LDA vs PCA reference:")
    print(f"    LDA R@5={lda_summary['R@5']['mean']:.4f}, PCA R@5={pca_results['R@5']:.4f}")
    print(f"    Δ = {comp_lda_vs_pca['delta']:+.4f}")

    if HAS_TORCH and supcon_summary:
        comp_supcon_vs_raw = paired_ttest(supcon_results["R@5"], raw_per_split["R@5"])
        comp_supcon_vs_lda = paired_ttest(supcon_results["R@5"], lda_results["R@5"])
        print(f"\n  SupCon vs Raw CBraMod R@5:")
        print(f"    Δ = {comp_supcon_vs_raw['mean_diff']:+.4f}, t={comp_supcon_vs_raw['t_statistic']:.2f}, "
              f"p={comp_supcon_vs_raw['p_value']:.2e}, d={comp_supcon_vs_raw['cohen_d']:+.3f}")
        print(f"\n  SupCon vs LDA R@5:")
        print(f"    Δ = {comp_supcon_vs_lda['mean_diff']:+.4f}, t={comp_supcon_vs_lda['t_statistic']:.2f}, "
              f"p={comp_supcon_vs_lda['p_value']:.2e}, d={comp_supcon_vs_lda['cohen_d']:+.3f}")

    # ============================================================
    # Compile results
    # ============================================================
    results = {
        "experiment_id": "m17-learned-metric-projection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_source": CACHE_PATH,
        "seed": SEED,
        "protocol": {
            "n_folds_loso": N_FOLDS,
            "session_disjoint_splits_per_fold": N_RUNS_PER_SUBJECT,
            "total_splits": N_FOLDS * N_RUNS_PER_SUBJECT,
            "query": "one run of held-out subject (15 trials)",
            "pool": "all other trials (same subject diff runs + all other subjects' all runs)",
            "train_w": "49 subjects (4410 trials)",
            "no_test_subject_info_during_training": True,
        },
        "baselines": {
            "raw_cbramod_200_cosine": raw_results,
            "centroid_cbramod_200": centroid_results,
            "pca_32_bandpower_reference": pca_results,
            "v2_32_cosine_reference": v2_results,
            "v2_32_same_protocol": v2_raw_results,
        },
        "methods": {
            "lda_projection": lda_summary,
            **({"supcon_projection": supcon_summary} if HAS_TORCH and supcon_summary else {}),
        },
        "pairwise_comparisons": {
            "lda_vs_raw_cosine": comp_lda_vs_raw,
            "lda_vs_centroid": comp_lda_vs_centroid,
            "lda_vs_pca_reference": comp_lda_vs_pca,
            "lda_vs_v2_same_protocol": comp_lda_vs_v2,
            **({"supcon_vs_raw_cosine": comp_supcon_vs_raw} if HAS_TORCH and supcon_summary else {}),
            **({"supcon_vs_lda": comp_supcon_vs_lda} if HAS_TORCH and supcon_summary else {}),
        },
        "geometry_notes": {
            "cbramod_anisotropy": 0.9621,
            "cbramod_participation_ratio": 4.16,
            "top_pc1_variance_ratio": 0.4683,
            "whitening_hurts": True,
            "centroid_beats_nn": True,
            "centroid_r5_gain_over_nn": 0.115,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("-" * 64)
    print(f"{'CBraMod raw cosine (Baseline A)':<30} {raw_results['R@1']['mean']:>8.4f} {raw_results['R@5']['mean']:>8.4f} {raw_results['R@10']['mean']:>8.4f} {raw_results['MRR']['mean']:>8.4f}")
    print(f"{'CBraMod centroid (Baseline B)':<30} {centroid_results['R@1']['mean']:>8.4f} {centroid_results['R@5']['mean']:>8.4f} {centroid_results['R@10']['mean']:>8.4f} {centroid_results['MRR']['mean']:>8.4f}")
    print(f"{'PCA-32 bandpower (Baseline C)':<30} {pca_results['R@1']:>8.4f} {pca_results['R@5']:>8.4f} {pca_results['R@10']:>8.4f} {'n/a':>8}")
    print(f"{'V2-32 cosine (Baseline D)':<30} {v2_raw_results['R@1']['mean']:>8.4f} {v2_raw_results['R@5']['mean']:>8.4f} {v2_raw_results['R@10']['mean']:>8.4f} {v2_raw_results['MRR']['mean']:>8.4f}")
    print(f"{'CBraMod LDA (Method 1)':<30} {lda_summary['R@1']['mean']:>8.4f} {lda_summary['R@5']['mean']:>8.4f} {lda_summary['R@10']['mean']:>8.4f} {lda_summary['MRR']['mean']:>8.4f}")
    if HAS_TORCH and supcon_summary:
        print(f"{'CBraMod SupCon (Method 2)':<30} {supcon_summary['R@1']['mean']:>8.4f} {supcon_summary['R@5']['mean']:>8.4f} {supcon_summary['R@10']['mean']:>8.4f} {supcon_summary['MRR']['mean']:>8.4f}")
    print("-" * 64)

    # Decision framework
    lda_r5 = lda_summary['R@5']['mean']
    raw_r5 = raw_results['R@5']['mean']
    centroid_r5 = centroid_results['R@5']['mean']
    pca_r5 = pca_results['R@5']

    print(f"\n  LDA R@5 vs raw cosine R@5: {lda_r5:.4f} vs {raw_r5:.4f} → Δ={lda_r5-raw_r5:+.4f}")
    print(f"  LDA R@5 vs centroid R@5: {lda_r5:.4f} vs {centroid_r5:.4f} → Δ={lda_r5-centroid_r5:+.4f}")
    print(f"  LDA R@5 vs PCA R@5: {lda_r5:.4f} vs {pca_r5:.4f} → Δ={lda_r5-pca_r5:+.4f}")

    return results


if __name__ == "__main__":
    run_experiment()
