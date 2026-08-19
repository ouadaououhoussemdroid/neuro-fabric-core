#!/usr/bin/env python3
"""
Mission 17 — Embedding Geometry & Retrieval Decomposition Analysis

Analyzes the CBraMod-200 embedding space from cached Mission-11 embeddings to
inform the scientific similarity audit. Uses the same cache as Mission 16.

Computes:
  - Anisotropy (mean pairwise cosine similarity)
  - Variance distribution across dimensions
  - Embedding concentration (cosine similarity distribution)
  - Intra-subject vs inter-subject cosine distributions
  - Intra-session vs cross-session cosine distributions
  - Same-task vs different-task cosine distributions
  - Retrieval decomposition: subject identity, session, task, run effects
  - NN gap analysis (same vs different subject)
  - Dimension utilization (effective rank)
  - Principal angle / PCA of embeddings
"""

import numpy as np
import json
import os
from datetime import datetime, timezone

CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", ".cbramod_cross_session_cache.npz"
)
CACHE_PATH = os.path.normpath(CACHE_PATH)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports", "m17_geometry_analysis.json"
)
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)


def load_cache():
    data = np.load(CACHE_PATH)
    cb_emb = data["cb_emb"]          # 4500 x 200
    v2_emb = data["v2_emb"]          # 4500 x 32
    bandpower = data["bandpower"]    # 4500 x 110
    subj_ids = data["subj_ids"]
    run_ids = data["run_ids"]
    mi_labels = data["mi_labels"]

    # Convert subj_ids to numeric
    if subj_ids.dtype.kind in ("U", "S"):
        subj_numeric = np.array([int(str(s).replace("S", "").replace("s", "")) for s in subj_ids])
    else:
        subj_numeric = subj_ids.astype(int)

    # Run IDs — determine run number
    # run_ids likely encode run number (5, 7, 9 for hand; 6, 8, 10 for foot/tongue)
    if run_ids.dtype.kind in ("U", "S"):
        run_str = np.array([str(r) for r in run_ids])
    else:
        run_str = run_ids.astype(str)

    # Extract run number (assume format like "run5" or "5" or "S001_run5")
    run_nums = np.zeros(len(run_ids), dtype=int)
    for i, r in enumerate(run_str):
        s = str(r)
        # Try to extract a number from the run string
        import re
        nums = re.findall(r'\d+', s)
        if nums:
            run_nums[i] = int(nums[-1])  # Take the last number found

    # Task mapping
    # Runs 5,7,9 → hand task (classes 0=left, 1=right)
    # Runs 6,8,10 → foot/tongue task (classes 2=feet, 3=tongue)
    task_ids = np.where(np.isin(run_nums, [5, 7, 9]), 0, 1)  # 0=hand, 1=foot/tongue

    # Session/run group (same run number within a subject = same session)
    # Session = subject + run combination
    session_ids = subj_numeric * 100 + run_nums

    return {
        "cb_emb": cb_emb,
        "v2_emb": v2_emb,
        "bandpower": bandpower,
        "subj_ids": subj_numeric,
        "run_nums": run_nums,
        "run_strs": run_str,
        "mi_labels": mi_labels.astype(int),
        "task_ids": task_ids,
        "session_ids": session_ids,
    }


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def compute_cosine_matrix(X_sample):
    """Compute pairwise cosine similarities for a sample of embeddings."""
    Xn = l2_normalize(X_sample)
    return Xn @ Xn.T


def sample_pairs(n, subj_ids, run_nums, task_ids, mi_labels, seed=42):
    """Sample pairs and classify them."""
    rng = np.random.RandomState(seed)
    # Sample up to 5000 pairs to keep memory manageable
    n_pairs = min(n, len(subj_ids) * (len(subj_ids) - 1) // 2)

    same_subject = []
    diff_subject = []
    same_session = []     # same subject, same run
    diff_session = []     # same subject, different run
    same_task = []
    diff_task = []
    same_label = []       # same MI class
    diff_label = []

    # Get all unique indices
    indices = np.arange(len(subj_ids))

    # Sample pairs
    sampled = 0
    attempts = 0
    max_attempts = n_pairs * 100
    while sampled < n_pairs and attempts < max_attempts:
        i, j = rng.choice(indices, size=2, replace=False)
        attempts += 1

        # Same/different subject
        if subj_ids[i] == subj_ids[j]:
            same_subject.append((i, j))
        else:
            diff_subject.append((i, j))

        # Same/different session (subject + run)
        if subj_ids[i] == subj_ids[j] and run_nums[i] == run_nums[j]:
            same_session.append((i, j))
        if subj_ids[i] == subj_ids[j] and run_nums[i] != run_nums[j]:
            diff_session.append((i, j))

        # Same/different task
        if task_ids[i] == task_ids[j]:
            same_task.append((i, j))
        else:
            diff_task.append((i, j))

        # Same/different MI label
        if mi_labels[i] == mi_labels[j]:
            same_label.append((i, j))
        else:
            diff_label.append((i, j))

        sampled += 1

    return {
        "same_subject": same_subject,
        "diff_subject": diff_subject,
        "same_session": same_session,
        "diff_session": diff_session,
        "same_task": same_task,
        "diff_task": diff_task,
        "same_label": same_label,
        "diff_label": diff_label,
    }


def cosine_for_pairs(X, pairs):
    """Compute cosine similarity for a list of index pairs."""
    Xn = l2_normalize(X)
    sims = []
    for i, j in pairs:
        sims.append(float(Xn[i] @ Xn[j]))
    return np.array(sims)


def compute_statistics(values):
    """Compute summary statistics for an array."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q05": float(np.percentile(values, 5)),
        "q95": float(np.percentile(values, 95)),
        "count": len(values),
    }


def analyze_geometry(X, name, subj_ids, run_nums, task_ids, mi_labels):
    """Full geometry analysis for a given embedding space."""
    Xn = l2_normalize(X)
    n, d = X.shape

    # 1. Dimensional utilization: how many dimensions carry significant variance
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    variance_explained = (S ** 2) / (S ** 2).sum()
    cumulative_var = np.cumsum(variance_explained)

    # Effective rank: number of dimensions needed for 90% / 95% / 99% variance
    n_dims_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
    n_dims_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
    n_dims_99 = int(np.argmax(cumulative_var >= 0.99) + 1)

    # Participation ratio (effective dimensionality)
    participation_ratio = float((S ** 2).sum() ** 2 / (S ** 4).sum())

    # 2. Anisotropy: mean pairwise cosine similarity (sample for memory)
    rng = np.random.RandomState(42)
    sample_size = min(500, n)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    cos_sample = Xn[sample_idx] @ Xn[sample_idx].T
    cos_vals = cos_sample[np.triu_indices(sample_size, k=1)]
    anisotropy_mean = float(np.mean(cos_vals))
    anisotropy_std = float(np.std(cos_vals, ddof=1))

    # 3. Embedding concentration: mean norm before L2 normalization
    norms = np.linalg.norm(X, axis=1)
    norm_stats = compute_statistics(norms)

    # 4. Cosine similarity distribution for sampled pairs
    pairs = sample_pairs(5000, subj_ids, run_nums, task_ids, mi_labels)
    pair_results = {}
    for pair_type, pair_list in pairs.items():
        if len(pair_list) > 0:
            sims = cosine_for_pairs(X, pair_list)
            pair_results[pair_type] = compute_statistics(sims)
        else:
            pair_results[pair_type] = {"count": 0}

    # 5. NN gap analysis (same vs different subject)
    # Compute for a subset for speed
    nn_sample = min(1000, n)
    nn_idx = rng.choice(n, size=nn_sample, replace=False)
    nn_cos_sims = np.zeros(nn_sample)
    nn_same_subject = np.zeros(nn_sample, dtype=bool)
    nn_same_session = np.zeros(nn_sample, dtype=bool)

    for idx_pos, q_idx in enumerate(nn_idx):
        sims = Xn[q_idx] @ Xn.T
        sims[q_idx] = -np.inf  # Exclude self
        nn_idx_pos = np.argmax(sims)
        nn_cos_sims[idx_pos] = sims[nn_idx_pos]
        nn_same_subject[idx_pos] = (subj_ids[q_idx] == subj_ids[nn_idx_pos])
        nn_same_session[idx_pos] = (subj_ids[q_idx] == subj_ids[nn_idx_pos] and
                                     run_nums[q_idx] == run_nums[nn_idx_pos])

    same_subj_sim = nn_cos_sims[nn_same_subject]
    diff_subj_sim = nn_cos_sims[~nn_same_subject]

    nn_analysis = {
        "mean_nn_cos_same_subject": float(np.mean(same_subj_sim)) if len(same_subj_sim) > 0 else None,
        "mean_nn_cos_diff_subject": float(np.mean(diff_subj_sim)) if len(diff_subj_sim) > 0 else None,
        "nn_gap": float(np.mean(same_subj_sim) - np.mean(diff_subj_sim)) if len(same_subj_sim) > 0 and len(diff_subj_sim) > 0 else None,
        "n_same_subject_nn": int(nn_same_subject.sum()),
        "n_diff_subject_nn": int((~nn_same_subject).sum()),
        "n_same_session_nn": int(nn_same_session.sum()),
        "fraction_same_subject_nn": float(nn_same_subject.sum() / nn_sample),
        "fraction_same_session_nn": float(nn_same_session.sum() / nn_sample),
    }

    # 6. Variance of norms across dimensions (to check for dead dimensions)
    dim_means = X.mean(axis=0)
    dim_stds = X.std(axis=0, ddof=1)
    dim_vars = X.var(axis=0, ddof=1)
    dead_dims = int(np.sum(dim_stds < 1e-6))
    near_dead_dims = int(np.sum(dim_stds < 0.01 * X.std()))

    result = {
        "model": name,
        "n_samples": int(n),
        "n_dimensions": int(d),
        "anisotropy": {
            "mean_pairwise_cosine": anisotropy_mean,
            "std_pairwise_cosine": anisotropy_std,
            "sample_size": sample_size,
            "interpretation": "High positive mean indicates embedding anisotropy (all vectors point in similar direction)"
        },
        "variance_distribution": {
            "variance_explained_ratio_top1": float(variance_explained[0]),
            "variance_explained_ratio_top5": float(cumulative_var[4]) if d >= 5 else float(cumulative_var[-1]),
            "variance_explained_ratio_top10": float(cumulative_var[9]) if d >= 10 else float(cumulative_var[-1]),
            "variance_explained_ratio_top50": float(cumulative_var[49]) if d >= 50 else float(cumulative_var[-1]),
            "n_dims_for_90pct_variance": n_dims_90,
            "n_dims_for_95pct_variance": n_dims_95,
            "n_dims_for_99pct_variance": n_dims_99,
            "participation_ratio": participation_ratio,
            "interpretation": f"Effective dimensionality ~{participation_ratio:.1f} (of {d} total)"
        },
        "embedding_concentration": norm_stats,
        "pairwise_cosine_distribution": pair_results,
        "nn_analysis": nn_analysis,
        "dimension_analysis": {
            "dead_dimensions": dead_dims,
            "near_dead_dimensions": near_dead_dims,
            "dim_mean_range": [float(dim_means.min()), float(dim_means.max())],
        },
    }

    return result


def analyze_retrieval_decomposition(X, subj_ids, run_nums, task_ids, mi_labels, seed=42):
    """
    Decompose what makes CBraMod's retrieval work.

    For each query, find the nearest neighbor and classify:
    - same subject, same run (true intra-session match)
    - same subject, same task, different run
    - same subject, different task, different run
    - different subject, same task
    - different subject, different task
    """
    Xn = l2_normalize(X)
    n = len(subj_ids)

    rng = np.random.RandomState(seed)
    sample_size = min(500, n)
    query_idx = rng.choice(n, size=sample_size, replace=False)

    # For each query, compute cosine to all, exclude self, find NN
    decomposition = {
        "total_queries": sample_size,
        "nn_is_same_subject": 0,
        "nn_is_same_subject_same_run": 0,
        "nn_is_same_subject_same_task_diff_run": 0,
        "nn_is_same_subject_diff_task": 0,
        "nn_is_diff_subject_same_task": 0,
        "nn_is_diff_subject_diff_task": 0,
        "nn_is_same_mi_label": 0,
        "nn_is_diff_mi_label": 0,
    }

    nn_cosine_sims = []

    for q in query_idx:
        sims = Xn[q] @ Xn.T
        sims[q] = -np.inf  # Exclude self
        nn_pos = np.argmax(sims)
        nn_cos = sims[nn_pos]
        nn_cosine_sims.append(float(nn_cos))

        if subj_ids[q] == subj_ids[nn_pos]:
            decomposition["nn_is_same_subject"] += 1
            if run_nums[q] == run_nums[nn_pos]:
                decomposition["nn_is_same_subject_same_run"] += 1
            elif task_ids[q] == task_ids[nn_pos]:
                decomposition["nn_is_same_subject_same_task_diff_run"] += 1
            else:
                decomposition["nn_is_same_subject_diff_task"] += 1
        else:
            if task_ids[q] == task_ids[nn_pos]:
                decomposition["nn_is_diff_subject_same_task"] += 1
            else:
                decomposition["nn_is_diff_subject_diff_task"] += 1

        if mi_labels[q] == mi_labels[nn_pos]:
            decomposition["nn_is_same_mi_label"] += 1
        else:
            decomposition["nn_is_diff_mi_label"] += 1

    # Restructure for clarity (don't modify dict during iteration)
    result = {
        "total_queries": sample_size,
        "nn_composition": {
            "same_subject_same_run": {
                "count": decomposition["nn_is_same_subject_same_run"],
                "fraction": decomposition["nn_is_same_subject_same_run"] / sample_size
            },
            "same_subject_same_task_diff_run": {
                "count": decomposition["nn_is_same_subject_same_task_diff_run"],
                "fraction": decomposition["nn_is_same_subject_same_task_diff_run"] / sample_size
            },
            "same_subject_diff_task": {
                "count": decomposition["nn_is_same_subject_diff_task"],
                "fraction": decomposition["nn_is_same_subject_diff_task"] / sample_size
            },
            "diff_subject_same_task": {
                "count": decomposition["nn_is_diff_subject_same_task"],
                "fraction": decomposition["nn_is_diff_subject_same_task"] / sample_size
            },
            "diff_subject_diff_task": {
                "count": decomposition["nn_is_diff_subject_diff_task"],
                "fraction": decomposition["nn_is_diff_subject_diff_task"] / sample_size
            },
        },
        "nn_label_composition": {
            "same_mi_label": {
                "count": decomposition["nn_is_same_mi_label"],
                "fraction": decomposition["nn_is_same_mi_label"] / sample_size
            },
            "diff_mi_label": {
                "count": decomposition["nn_is_diff_mi_label"],
                "fraction": decomposition["nn_is_diff_mi_label"] / sample_size
            },
        },
        "nn_cosine_stats": compute_statistics(np.array(nn_cosine_sims)),
    }

    return result


def run_analysis():
    """Run the full geometry + retrieval decomposition analysis."""
    print("=" * 70)
    print("Mission 17 — Embedding Geometry & Retrieval Decomposition Analysis")
    print("=" * 70)

    data = load_cache()

    print(f"\nCache loaded:")
    print(f"  CBraMod-200: {data['cb_emb'].shape}")
    print(f"  V2-32:       {data['v2_emb'].shape}")
    print(f"  Bandpower:   {data['bandpower'].shape}")
    print(f"  Subjects:    {len(np.unique(data['subj_ids']))}")
    print(f"  Runs:        {sorted(np.unique(data['run_nums']))}")
    print(f"  Tasks:       hand(0)={int((data['task_ids']==0).sum())}, foot/tongue(1)={int((data['task_ids']==1).sum())}")
    print(f"  MI labels:   {np.bincount(data['mi_labels'])}")

    # Analyze CBraMod-200 geometry
    print("\n--- CBraMod-200 Geometry Analysis ---")
    cb_geom = analyze_geometry(
        data["cb_emb"], "CBraMod-200",
        data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    print(f"  Anisotropy (mean pairwise cosine): {cb_geom['anisotropy']['mean_pairwise_cosine']:.4f}")
    print(f"  Participation ratio: {cb_geom['variance_distribution']['participation_ratio']:.2f}")
    print(f"  Dims for 95% variance: {cb_geom['variance_distribution']['n_dims_for_95pct_variance']}")
    print(f"  NN gap (same vs diff subject): {cb_geom['nn_analysis']['nn_gap']:.6f}")
    print(f"  NN frac same-subject: {cb_geom['nn_analysis']['fraction_same_subject_nn']:.4f}")

    # Analyze PCA geometry
    print("\n--- PCA-32 Geometry Analysis ---")
    pca_geom = analyze_geometry(
        data["bandpower"][:, :32], "PCA-32-bandpower",
        data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    print(f"  Anisotropy (mean pairwise cosine): {pca_geom['anisotropy']['mean_pairwise_cosine']:.4f}")
    print(f"  Participation ratio: {pca_geom['variance_distribution']['participation_ratio']:.2f}")
    print(f"  NN gap (same vs diff subject): {pca_geom['nn_analysis']['nn_gap']:.6f}")

    # Analyze V2 geometry
    print("\n--- V2-32 Geometry Analysis ---")
    v2_geom = analyze_geometry(
        data["v2_emb"], "V2-32",
        data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    print(f"  Anisotropy (mean pairwise cosine): {v2_geom['anisotropy']['mean_pairwise_cosine']:.4f}")
    print(f"  Participation ratio: {v2_geom['variance_distribution']['participation_ratio']:.2f}")
    print(f"  NN gap (same vs diff subject): {v2_geom['nn_analysis']['nn_gap']:.6f}")

    # Retrieval decomposition
    print("\n--- Retrieval Decomposition (CBraMod-200) ---")
    cb_decomp = analyze_retrieval_decomposition(
        data["cb_emb"], data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    for k, v in cb_decomp["nn_composition"].items():
        print(f"  NN {k}: {v['fraction']:.4f} (count={v['count']})")
    for k, v in cb_decomp["nn_label_composition"].items():
        print(f"  NN {k}: {v['fraction']:.4f} (count={v['count']})")

    print("\n--- Retrieval Decomposition (V2-32) ---")
    v2_decomp = analyze_retrieval_decomposition(
        data["v2_emb"], data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    for k, v in v2_decomp["nn_composition"].items():
        print(f"  NN {k}: {v['fraction']:.4f} (count={v['count']})")

    print("\n--- Retrieval Decomposition (PCA-32 bandpower) ---")
    pca_decomp = analyze_retrieval_decomposition(
        data["bandpower"][:, :32], data["subj_ids"], data["run_nums"], data["task_ids"], data["mi_labels"]
    )
    for k, v in pca_decomp["nn_composition"].items():
        print(f"  NN {k}: {v['fraction']:.4f} (count={v['count']})")

    # Summary comparison
    print("\n--- Geometry Comparison Summary ---")
    print(f"{'Model':<15} {'Anisotropy':>12} {'Part. Ratio':>12} {'Dims@95%':>10} {'NN Gap':>12}")
    for name, geom in [("CBraMod-200", cb_geom), ("V2-32", v2_geom), ("PCA-32", pca_geom)]:
        print(f"{name:<15} {geom['anisotropy']['mean_pairwise_cosine']:>12.4f} {geom['variance_distribution']['participation_ratio']:>12.2f} {geom['variance_distribution']['n_dims_for_95pct_variance']:>10} {geom['nn_analysis']['nn_gap']:>12.6f}")

    # Cosine similarity distribution (intra vs inter subject)
    print("\n--- Cosine Similarity Distribution (CBraMod-200) ---")
    ps = cb_geom["pairwise_cosine_distribution"]
    for pair_type in ["same_subject", "diff_subject", "same_session", "diff_session", "same_task", "diff_task", "same_label", "diff_label"]:
        if pair_type in ps and "count" in ps[pair_type]:
            s = ps[pair_type]
            if s["count"] > 0:
                print(f"  {pair_type:<25}: mean={s['mean']:.4f} std={s['std']:.4f}  [min={s['min']:.4f}, max={s['max']:.4f}]  n={s['count']}")

    # Compile results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_source": CACHE_PATH,
        "data_summary": {
            "cb_emb_shape": list(data["cb_emb"].shape),
            "v2_emb_shape": list(data["v2_emb"].shape),
            "bandpower_shape": list(data["bandpower"].shape),
            "n_subjects": len(np.unique(data["subj_ids"])),
            "runs": sorted(np.unique(data["run_nums"]).tolist()),
            "mi_labels": np.bincount(data["mi_labels"]).tolist(),
            "task_mapping": {
                "0": "left hand (runs 5,7,9)",
                "1": "right hand (runs 5,7,9)",
                "2": "feet (runs 6,8,10)",
                "3": "tongue (runs 6,8,10)"
            }
        },
        "geometry_analysis": {
            "cbramod_200": cb_geom,
            "v2_32": v2_geom,
            "pca_32": pca_geom,
        },
        "retrieval_decomposition": {
            "cbramod_200": cb_decomp,
            "v2_32": v2_decomp,
            "pca_32": pca_decomp,
        }
    }

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run_analysis()
