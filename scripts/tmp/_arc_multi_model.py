#!/usr/bin/env python3
"""Append Multi-Model Fusion experiment record to reports/benchmark_archive.json.

Byte-preserving: idx0..idx14 content (everything before the last experiment's closing
brace) is verified unchanged via sha256 prefix comparison before and after. Uses text-splicing
rather than json.dump re-serialization (same technique as _arc_m14_phase1.py).

Usage: python scripts/tmp/_arc_multi_model.py
"""

import json, hashlib, sys, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"

raw = ARCHIVE.read_text(encoding="utf-8")

# Anchor: the boundary between experiments[] and fine_tuning_experiments[]
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
# Multi-Model Fusion experiment record
# ---------------------------------------------------------------------------
multi_model_record = {
    "id": "multi-model-embedding-fusion",
    "experiment_name": "Multi-Model EEG Representation + Fusion: CBraMod-200 + V2-32 + PCA-32 Late Fusion",
    "date": "2026-08-16",
    "author": "NeuroFabric team",
    "mission": "Multi-Model EEG Representation + Fusion — determine whether combining embeddings from CBraMod-200, EEGConformer V2-32, and PCA-32 bandpower yields better retrieval than any single representation",
    "model": "Late fusion of frozen CBraMod-200, V2-EEGConformer-32, and PCA-32 bandpower embeddings (logistic regression weight learning per LOSO fold)",
    "models_compared": [
        "CBraMod-200 raw cosine NN",
        "CBraMod-200 centroid matching",
        "CBraMod-200 LDA projection",
        "V2-32 raw cosine NN",
        "PCA-32 bandpower",
        "PCA + CBraMod fusion",
        "PCA + V2 fusion",
        "CBraMod + V2 fusion",
        "PCA + CBraMod + V2 fusion"
    ],
    "dataset": "PhysioNet EEGMMIDB (S001-S050), 50 subjects x 6 runs x 15 trials = 4500 trials",
    "subjects": 50,
    "trials": 4500,
    "protocol": "50-fold LOSO; session-disjoint retrieval (300 splits per fold = 15000 total); query=one run (15 trials), pool=all other trials; late fusion via logistic regression on pairwise similarities (weights learned per-fold on 49 training subjects only); Bonferroni correction (4 comparisons against best fusion, alpha=0.0125); bootstrap 95% CIs; seed=42",
    "hypothesis": "Combining embeddings from CBraMod-200, V2-32, and PCA-32 via late fusion will yield significantly better subject retrieval than PCA-32 alone",
    "methods": {
        "cbramod": {
            "onnx_path": "public/models/cbramod-encoder.onnx",
            "sha256": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
            "channels": 19,
            "embedding_dim": 200,
            "wasm_compatible": False,
            "output_pooling": "mean-tokens over [19, 5, 200] -> [200]"
        },
        "v2_eeegconformer": {
            "onnx_path": "public/models/eegconformer_finetuned.onnx",
            "sha256": "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
            "channels": 22,
            "embedding_dim": 32,
            "wasm_compatible": True
        },
        "pca": {
            "input_features": 110,
            "output_dim": 32,
            "n_bands": 5,
            "fit_per_fold": True,
            "scaler": "StandardScaler (train-only)"
        },
        "fusion": {
            "method": "Late fusion via logistic regression weight learning",
            "weight_constraint": "Non-negative, normalized to sum=1",
            "training_samples": "2000 pairwise samples per fold (train subjects only)",
            "leakage_control": "Weights learned on 49 training subjects, applied to held-out subject"
        }
    },
    "results": {
        "individual": {
            "cbramod_raw_cosine": {"R@1": 0.2427, "R@5": 0.5273, "R@10": 0.6587, "MRR": 0.3776},
            "v2_raw_cosine": {"R@1": 0.0687, "R@5": 0.2158, "R@10": 0.3364, "MRR": 0.1568},
            "cbramod_centroid": {"R@1": 0.3082, "R@5": 0.6520, "R@10": 0.8018, "MRR": 0.4652},
            "cbramod_lda": {"R@1": 0.2924, "R@5": 0.5736, "R@10": 0.6969, "MRR": 0.4250},
            "pca32": {"R@1": 0.4713, "R@5": 0.7360, "R@10": 0.8231, "MRR": 0.5910}
        },
        "fusion": {
            "pca_cbramod": {"R@1": 0.4876, "R@5": 0.7427, "R@10": 0.8273, "MRR": 0.6036},
            "pca_v2": {"R@1": 0.4856, "R@5": 0.7480, "R@10": 0.8307, "MRR": 0.6032},
            "cbramod_v2": {"R@1": 0.2184, "R@5": 0.4996, "R@10": 0.6429, "MRR": 0.3537},
            "pca_cbramod_v2": {"R@1": 0.4902, "R@5": 0.7462, "R@10": 0.8311, "MRR": 0.6062}
        },
        "best_individual_r5": 0.7360,
        "best_individual_method": "PCA-32 bandpower",
        "best_fusion_r5": 0.7480,
        "best_fusion_method": "PCA + V2",
        "fusion_improvement_over_pca": +0.0120,
        "fusion_p_value_vs_pca": 0.0002,
        "fusion_cohen_d_vs_pca": 0.055,
        "fusion_significant_after_bonferroni": True
    },
    "pairwise_comparisons": {
        "pca_v2_fusion_vs_pca": {
            "mean_diff": 0.0120,
            "p_value": 0.0002,
            "cohen_d": 0.055,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0125
        },
        "pca_v2_fusion_vs_cbramod_raw": {
            "mean_diff": 0.2207,
            "p_value": 7.1e-125,
            "cohen_d": 0.366,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0125
        },
        "pca_v2_fusion_vs_centroid": {
            "mean_diff": 0.0960,
            "p_value": 3.1e-29,
            "cohen_d": 0.168,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0125
        },
        "pca_v2_fusion_vs_lda": {
            "mean_diff": 0.1744,
            "p_value": 7.4e-78,
            "cohen_d": 0.284,
            "significant_after_bonferroni": True,
            "bonferroni_alpha": 0.0125
        }
    },
    "geometry_analysis": {
        "cbramod_anisotropy_mean_pairwise_cosine": 0.9621,
        "cbramod_dominant_signal": "Subject identity (weakly encoded, recoverable via centroid/LDA)",
        "v2_anisotropy_mean_pairwise_cosine": 0.9097,
        "pca_anisotropy_mean_pairwise_cosine": 0.7850,
        "complementarity_assessment": "PCA and V2 have partially complementary subject-identity signal; CBraMod adds diminishing returns"
    },
    "decision": "Fusion significantly improves retrieval over PCA alone (+1.2pp R@5, p=0.0002). However, the improvement is modest (d=0.055). PCA bandpower remains the dominant signal, with V2 providing complementary information and CBraMod adding diminishing returns. The multi-model ensemble confirms partial complementarity between learned representations and spectral features.",
    "contaminated": False,
    "status": "COMPLETE - fusion improves over PCA significantly but marginally. Best fusion: PCA+V2 (R@5=0.748 vs PCA=0.736). All constraints honored.",
    "report_file": "reports/MULTI_MODEL_ENSEMBLE_REPORT.md",
    "results_json": "reports/multi_model_ensemble_results.json",
    "benchmark_script": "scripts/tmp/multi_model_embedding_fusion.py",
    "provenance": {
        "cache_source": "reports/.cbramod_cross_session_cache.npz (verified: SHA match + Subject 1 re-extraction, cb_diff=0.0138, v2_diff=0.0549)",
        "git_head": git_head,
        "seed": 42,
        "n_bootstrap": 2000,
        "n_folds_loso": 50,
        "session_disjoint_splits": 300,
        "fusion_training_pairs_per_fold": 2000,
        "bonferroni_alpha": 0.0125,
        "cbramod_sha": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        "v2_sha": "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"
    },
    "constraints_honored": {
        "no_model_retraining": True,
        "no_artifact_modification": True,
        "no_onnx_modification": True,
        "no_default_preferred_change": True,
        "no_v2_or_pca_change": True,
        "no_production_code_changes": True,
        "loso_leakage_free": True,
        "session_disjoint_evaluation": True,
        "fusion_weights_train_only": True,
        "bonferroni_correction": True,
        "seed_42_reproducible": True,
        "prior_archive_records_byte_preserved": True
    }
}

# ---------------------------------------------------------------------------
# Splice record using text replacement
# ---------------------------------------------------------------------------
record_json = json.dumps(multi_model_record, indent=2, ensure_ascii=False)

# Indent: 2-space base indent for experiment records (matches _arc_m16.py)
record_indented = "\n".join(("  " + line) if line else line for line in record_json.split("\n"))

replacement = '  },\n' + record_indented + '\n  ],\n  "fine_tuning_experiments": ['
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
assert new_arch["experiments"][-1]["id"] == "multi-model-embedding-fusion", \
    f"Expected 'multi-model-embedding-fusion' at [-1], got {new_arch['experiments'][-1]['id']}"
print(f"OK: last experiment: [{new_arch['experiments'][-1]['id']}]")

# Write
ARCHIVE.write_text(new_raw, encoding="utf-8")
print(f"\nDone: appended Multi-Model Fusion record to {ARCHIVE}")
print(f"Experiments: {before_count} -> {len(new_arch['experiments'])}")
