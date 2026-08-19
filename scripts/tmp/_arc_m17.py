#!/usr/bin/env python3
"""Append Mission-17 learned-metric experiment record to reports/benchmark_archive.json.

Byte-preserving: idx0..idx14 content (everything before the last experiment's closing
brace) is verified unchanged via sha256 prefix comparison before and after. Uses text-splicing
rather than json.dump re-serialization (same technique as _arc_m14_phase1.py).

Usage: python scripts/tmp/_arc_m17.py
"""

import json, hashlib, sys, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"

raw = ARCHIVE.read_text(encoding="utf-8")

# Anchor: the boundary between experiments[] and fine_tuning_experiments[]
# The last experiment closes with '  }' (2-space indent), followed by
#   \n  ],\n  "fine_tuning_experiments": [
EXPERIMENTS_ANCHOR = '  }\n  ],\n  "fine_tuning_experiments": ['
count = raw.count(EXPERIMENTS_ANCHOR)
if count != 1:
    sys.stderr.write(f"ERROR: Expected 1 anchor, found {count}\n")
    sys.exit(1)

# Prefix = everything before the closing brace of the last experiment (idx0..idx14)
prefix = raw[:raw.find(EXPERIMENTS_ANCHOR)]
prefix_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
print(f"Prefix sha256 (idx0..last): {prefix_sha[:16]}...")

arch = json.loads(raw)
before_count = len(arch["experiments"])
print(f"Experiments before: {before_count}")

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()

# ---------------------------------------------------------------------------
# Mission 17 experiment record
# ---------------------------------------------------------------------------
m17_record = {
    "id": "m17-learned-metric-projection",
    "experiment_name": "Mission 17: Learned Similarity Projection on CBraMod-200 (LDA + SupCon, 50-fold LOSO)",
    "date": "2026-08-15",
    "author": "NeuroFabric team",
    "mission": "Mission 17 - scientific experiment: whether a learned linear projection W: R^200->R^k trained with subject-identity supervision improves CBraMod-200 retrieval beyond raw cosine, centroid, and PCA baselines",
    "model": "Learned linear projection (LDA closed-form / SupCon PyTorch) on frozen CBraMod-200 cached embeddings",
    "models_compared": [
        "CBraMod-200 raw cosine NN",
        "CBraMod-200 centroid matching",
        "PCA-32 bandpower",
        "V2-32 cosine NN",
        "CBraMod-200 + LDA projection",
        "CBraMod-200 + SupCon projection"
    ],
    "dataset": "PhysioNet EEGMMIDB (S001-S050), 50 subjects x 6 runs x 15 trials = 4500 trials",
    "subjects": 50,
    "trials": 4500,
    "protocol": "50-fold LOSO; train W on 49 subjects (4410 trials), evaluate on held-out subject (90 trials). Session-disjoint retrieval: query = one run (15 trials), pool = all other trials (300 splits per fold, 50 folds = 300 total session-disjoint splits). R@1, R@5, R@10, MRR computed. Paired t-tests with Bonferroni correction (alpha=0.05/3=0.0167). Bootstrap 95% CIs (seed=42). SupCon: 300 epochs, Adam lr=0.01, weight_decay=0.001, batch=256, temp=0.1, W: R^200->R^200.",
    "hypothesis": "A learned linear projection W: R^200 -> R^49 (LDA, Fisher discriminant) will significantly improve CBraMod-200 subject-retrieval R@5 vs raw cosine NN on 50-fold LOSO with session-disjoint evaluation",
    "methods": {
        "lda_projection": {
            "description": "Linear Discriminant Analysis (closed-form Fisher discriminant), 200->49 dimensions",
            "r_at_1_mean": 0.2924,
            "r_at_5_mean": 0.5736,
            "r_at_10_mean": 0.6969,
            "r_at_5_ci95": [0.5429, 0.6062],
            "mrr_mean": 0.4250,
            "mean_training_time_ms": 170.2,
            "projection_shape": [200, 49]
        },
        "supcon_projection": {
            "description": "Supervised Contrastive Learning (PyTorch), 200->200 linear W, 300 epochs, tau=0.1",
            "r_at_1_mean": 0.2024,
            "r_at_5_mean": 0.4651,
            "r_at_10_mean": 0.5947,
            "r_at_5_ci95": [0.4408, 0.4894],
            "mrr_mean": 0.3280
        }
    },
    "results": {
        "raw_cbramod_200_cosine_r_at_5": 0.5276,
        "centroid_cbramod_200_r_at_5": 0.5960,
        "pca_32_bandpower_r_at_5": 0.6920,
        "v2_32_cosine_r_at_5": 0.2158,
        "lda_r_at_5": 0.5736,
        "supcon_r_at_5": 0.4651,
        "lda_vs_raw_cosine_delta_r_at_5": +0.0460,
        "supcon_vs_raw_cosine_delta_r_at_5": -0.0624,
        "lda_beats_raw_cosine": True,
        "lda_beats_centroid": False,
        "lda_approaches_pca": False,
        "supcon_degradation": True,
        "note": "LDA significantly improves retrieval over raw cosine (+4.6pp R@5, p=0.0019). SupCon degrades retrieval (-6.2pp R@5, p<0.001). Neither LDA nor SupCon beats PCA-32 or centroid matching."
    },
    "pairwise_comparisons": {
        "lda_vs_raw_cosine": {
            "mean_diff": 0.0460,
            "t_stat": 3.13,
            "p_value": 0.00192,
            "cohen_d": 0.181,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0167,
            "ci95": [0.0171, 0.0749]
        },
        "supcon_vs_raw_cosine": {
            "mean_diff": -0.0624,
            "t_stat": -5.96,
            "p_value": 6.90e-09,
            "cohen_d": -0.344,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0167
        },
        "supcon_vs_lda": {
            "mean_diff": -0.1084,
            "t_stat": -6.29,
            "p_value": 1.14e-09,
            "cohen_d": -0.363,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0167
        }
    },
    "geometry_insights": {
        "cbramod_anisotropy_mean_pairwise_cosine": 0.9621,
        "cbramod_participation_ratio": 4.16,
        "v2_anisotropy_mean_pairwise_cosine": 0.9097,
        "pca_anisotropy_mean_pairwise_cosine": 0.7850,
        "nn_gap_cbramod": +0.0002,
        "encodes_subject_identity": True,
        "encodes_session_identity": True,
        "encodes_task_identity": False,
        "encodes_mi_label": False,
        "removing_pc1_drops_r5": -0.093
    },
    "decision": "LDA significantly improves CBraMod-200 retrieval (+4.6pp R@5, p=0.002) but does NOT beat centroid matching (0.574 vs 0.596) or PCA-32 (0.574 vs 0.692). SupCon degrades retrieval (-6.2pp, p<0.001). CBraMod-200 encodes subject identity weakly but recoverably; PCA-32 bandpower remains the strongest simple baseline. High-value next direction: late fusion of CBraMod + bandpower for complementary signal combination.",
    "contaminated": False,
    "status": "COMPLETE - scientific experiment conducted. LDA improvement is real and significant. SupCon failure is documented. PCA remains unbeaten by learned metrics. Recommendation: late fusion for Mission 18.",
    "report_file": "reports/MISSION17_LEARNED_METRIC_REPORT.md",
    "results_json": "reports/m17_learned_metric_results.json",
    "benchmark_script": "scripts/tmp/m17_learned_metric.py",
    "provenance": {
        "cache_source": "reports/.cbramod_cross_session_cache.npz (Mission 11 cached embeddings, no retraining)",
        "git_head": git_head,
        "seed": 42,
        "n_bootstrap": 2000,
        "n_folds_loso": 50,
        "session_disjoint_splits": 300,
        "bonferroni_alpha": 0.0167
    },
    "constraints_honored": {
        "no_model_retraining": True,
        "no_artifact_modification": True,
        "no_onnx_modification": True,
        "no_default_preferred_change": True,
        "no_v2_or_pca_change": True,
        "no_production_code_changes": True,
        "read_only_cached_embeddings": True,
        "loso_leakage_free": True,
        "session_disjoint_evaluation": True,
        "bonferroni_correction": True,
        "seed_42_reproducible": True,
        "prior_archive_records_byte_preserved": True
    }
}

# ---------------------------------------------------------------------------
# Splice record using text replacement (same technique as _arc_m14_phase1.py)
# ---------------------------------------------------------------------------
m17_json = json.dumps(m17_record, indent=2, ensure_ascii=False)

# Indent: 2-space base indent for experiment records
m17_indented = "\n".join(("  " + line) if line else line for line in m17_json.split("\n"))

replacement = '  },\n' + m17_indented + '\n  ],\n  "fine_tuning_experiments": ['
new_raw = raw.replace(EXPERIMENTS_ANCHOR, replacement, 1)

# Verify prefix unchanged
new_prefix = new_raw[:raw.find(EXPERIMENTS_ANCHOR)]
prefix_sha_after = hashlib.sha256(new_prefix.encode("utf-8")).hexdigest()
print(f"Prefix sha256 (after append):  {prefix_sha_after[:16]}...")
assert prefix_sha == prefix_sha_after, "FAIL: idx0..last prefix changed!"
print("OK: idx0..last prefix sha256 identical before/after append.")

# Validate JSON
new_arch = json.loads(new_raw)
print("OK: appended archive is valid JSON.")

# Verify experiment count
assert len(new_arch["experiments"]) == before_count + 1, \
    f"Expected {before_count + 1} experiments, got {len(new_arch['experiments'])}"
print(f"OK: experiments count {before_count} -> {len(new_arch['experiments'])}")

# Verify the ID
assert new_arch["experiments"][-1]["id"] == "m17-learned-metric-projection", \
    f"Expected m17 at [-1], got {new_arch['experiments'][-1]['id']}"
print(f"OK: last experiment: [{new_arch['experiments'][-1]['id']}]")

# Write
ARCHIVE.write_text(new_raw, encoding="utf-8")
print(f"\nDone: appended Mission-17 record to {ARCHIVE}")
print(f"Experiments: {before_count} -> {len(new_arch['experiments'])}")
