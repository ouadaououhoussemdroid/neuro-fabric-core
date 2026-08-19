#!/usr/bin/env python3
"""Append M19 experiment record to benchmark_archive.json.

Uses byte-preserving text-splicing: loads the archive as raw text,
finds the closing ']' of the 'experiments' array, and splices in the
new record BEFORE that close bracket. All prior bytes are untouched.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = REPO / "reports" / "benchmark_archive.json"
M19_RESULTS_PATH = REPO / "reports" / "m19_dimensionwise_embedding_results.json"

# ── Load M19 results to source all numbers ──────────────────────
with open(M19_RESULTS_PATH) as f:
    m19 = json.load(f)

# ── Construct archive experiment record (matches existing schema) ──
m19_record = {
    "id": "m19-dimensionwise-embedding",
    "experiment_name": "Mission 19: Dimension-Wise Learned EEG Embedding (50-subject validation)",
    "date": "2026-08-17",
    "author": "NeuroFabric team",
    "mission": "Mission 19 — determine whether learning individual weights for all 264 dimensions of the joint embedding (CBraMod-200 ⊕ V2-32 ⊕ PCA-32) improves over Mission 18's block-weighted embedding (R@5=0.7856)",
    "model": "6 learned weight methods (block, ridge per-dim, Fisher per-dim, shrinkage, simplex, hierarchical) + 3 single-block ablations — all applied as element-wise scaling to the 264-D joint embedding",
    "models_compared": [
        "CBraMod-200 raw cosine NN",
        "V2-32 raw cosine NN",
        "PCA-32 bandpower",
        "Raw 264-D concat (baseline)",
        "M18 block weighting (3 params, R@5=0.7856)",
        "Ridge per-dim (264 params)",
        "Fisher per-dim (264 params)",
        "Shrinkage per-dim (264 params, α=0.5)",
        "Simplex-constrained per-dim (264 params)",
        "Hierarchical (264 params)",
        "CBraMod-only dim weighting (ablation)",
        "V2-only dim weighting (ablation)",
        "PCA-only dim weighting (ablation)"
    ],
    "dataset": "PhysioNet EEGMMIDB (S001-S050)",
    "subjects": 50,
    "trials": 4500,
    "protocol": m19["protocol"],
    "hypothesis": "Learning individual weights for all 264 dimensions of the joint embedding (CBraMod-200 ⊕ V2-32 ⊕ PCA-32) will significantly improve subject retrieval over Mission 18's 3-parameter block-weighted embedding (R@5=0.7856)",
    "methods": {
        "A_block_weighting": {
            "description": "3 block-level weights for [CBraMod|V2|PCA] via RidgeClassifier coefficient aggregation (Mission 18 method)",
            "dim": 264,
            "param_count": 3,
            "r_at_1": m19["learned_methods"]["A_block_weighting"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["A_block_weighting"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["A_block_weighting"]["r_at_10"],
            "mrr": m19["learned_methods"]["A_block_weighting"]["mrr"]
        },
        "B_ridge_per_dim": {
            "description": "264 individual weights from RidgeClassifier |coef| mean, simplex-normalized",
            "dim": 264,
            "param_count": 264,
            "r_at_1": m19["learned_methods"]["B_ridge_per_dim"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["B_ridge_per_dim"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["B_ridge_per_dim"]["r_at_10"],
            "mrr": m19["learned_methods"]["B_ridge_per_dim"]["mrr"]
        },
        "B_fisher_per_dim": {
            "description": "264 individual weights from per-dimension Fisher discriminant score",
            "dim": 264,
            "param_count": 264,
            "r_at_1": m19["learned_methods"]["B_fisher_per_dim"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["B_fisher_per_dim"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["B_fisher_per_dim"]["r_at_10"],
            "mrr": m19["learned_methods"]["B_fisher_per_dim"]["mrr"]
        },
        "C_shrinkage": {
            "description": "50/50 interpolation of ridge per-dim weights and block-expanded weights (shrinkage toward block weights)",
            "dim": 264,
            "param_count": 264,
            "r_at_1": m19["learned_methods"]["C_shrinkage"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["C_shrinkage"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["C_shrinkage"]["r_at_10"],
            "mrr": m19["learned_methods"]["C_shrinkage"]["mrr"]
        },
        "C_simplex": {
            "description": "Softmax-simplex constrained per-dim weights (RidgeClassifier alpha=0.5, T=0.5)",
            "dim": 264,
            "param_count": 264,
            "r_at_1": m19["learned_methods"]["C_simplex"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["C_simplex"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["C_simplex"]["r_at_10"],
            "mrr": m19["learned_methods"]["C_simplex"]["mrr"]
        },
        "D_hierarchical": {
            "description": "Hierarchical: block_weight × within-block dimension weight",
            "dim": 264,
            "param_count": 264,
            "r_at_1": m19["learned_methods"]["D_hierarchical"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["D_hierarchical"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["D_hierarchical"]["r_at_10"],
            "mrr": m19["learned_methods"]["D_hierarchical"]["mrr"]
        },
        "ablation_cbramod_only": {
            "description": "Dimension-wise weighting on CBraMod dims only, uniform elsewhere",
            "dim": 264,
            "param_count": 200,
            "r_at_1": m19["learned_methods"]["ablation_cbramod_only"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["ablation_cbramod_only"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["ablation_cbramod_only"]["r_at_10"],
            "mrr": m19["learned_methods"]["ablation_cbramod_only"]["mrr"]
        },
        "ablation_v2_only": {
            "description": "Dimension-wise weighting on V2 dims only, uniform elsewhere",
            "dim": 264,
            "param_count": 32,
            "r_at_1": m19["learned_methods"]["ablation_v2_only"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["ablation_v2_only"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["ablation_v2_only"]["r_at_10"],
            "mrr": m19["learned_methods"]["ablation_v2_only"]["mrr"]
        },
        "ablation_pca_only": {
            "description": "Dimension-wise weighting on PCA dims only, uniform elsewhere",
            "dim": 264,
            "param_count": 32,
            "r_at_1": m19["learned_methods"]["ablation_pca_only"]["r_at_1"],
            "r_at_5": m19["learned_methods"]["ablation_pca_only"]["r_at_5"],
            "r_at_10": m19["learned_methods"]["ablation_pca_only"]["r_at_10"],
            "mrr": m19["learned_methods"]["ablation_pca_only"]["mrr"]
        }
    },
    "individual_baselines": m19["individual_baselines"],
    "results": {
        "primary_baseline": m19["primary_baseline"],
        "raw_baseline": m19["raw_baseline"],
        "best_learned_method": m19["best_learned_method"],
        "best_learned_r5": m19["best_learned_r5"],
        "best_learned_r_at_1": m19["learned_methods"]["C_shrinkage"]["r_at_1"],
        "best_learned_r_at_10": m19["learned_methods"]["C_shrinkage"]["r_at_10"],
        "best_learned_mrr": m19["learned_methods"]["C_shrinkage"]["mrr"],
        "m18_baseline_r5": m19["m18_baseline_r5"],
        "improvement_over_m18_pp": m19["improvement_over_m18_pp"],
        "beats_m18_baseline": m19["beats_m18_baseline"],
        "improvement_over_raw_pp": m19["improvement_over_raw_pp"],
        "additional_metrics": m19["additional_metrics"]
    },
    "pairwise_comparisons": {
        k: v for k, v in m19["pairwise_comparisons"].items()
    },
    "weight_analysis": {
        "block_weights_mean": m19["weight_analysis"]["block_weights_mean"],
        "weight_stability_cv_mean": m19["weight_analysis"]["weight_stability_cv_mean"],
        "top20_indices": m19["weight_analysis"]["top20_indices"],
        "top20_values": m19["weight_analysis"]["top20_values"],
        "bottom20_indices": m19["weight_analysis"]["bottom20_indices"],
        "bottom20_values": m19["weight_analysis"]["bottom20_values"]
    },
    "decision": "Mission 19 is a marginal improvement (Result B). C-shrinkage (R@5=0.7860) numerically beats M18 block weighting (R@5=0.7856) by +0.04pp but the improvement is NOT statistically significant (p=0.157, d=0.021, Bonferroni alpha=0.0125). The 264-parameter Ridge weighting is significantly worse (p=1.6e-11). Block-level weighting (3 parameters) is the optimal parameter-count tradeoff. The Mission 18 block-weighted 264-D concatenation remains the final research embedding.",
    "result_classification": "B — Marginal improvement (not significant)",
    "contaminated": False,
    "status": "COMPLETE - C-shrinkage marginally beats M18 block weighting (+0.04pp R@5) but NOT statistically significant (p=0.157). 264-parameter per-dim weighting overfits. Block weighting (3 params) is optimal. All constraints honored.",
    "report_file": "reports/MISSION19_DIMENSIONWISE_EMBEDDING_REPORT.md",
    "results_json": "reports/m19_dimensionwise_embedding_results.json",
    "benchmark_script": "scripts/tmp/m19_dimensionwise_embedding.py",
    "provenance": {
        "cache_source": "reports/.cbramod_cross_session_cache.npz (verified: SHA match)",
        "learned_cache_path": "reports/.m19_dimensionwise_embedding_cache.npz",
        "git_head": "b9164a664fce039df24c23656427a30c3a966926",
        "seed": 42,
        "n_bootstrap": 2000,
        "n_folds_loso": 50,
        "session_disjoint_splits": 300,
        "bonferroni_alpha": 0.0125,
        "bonferroni_comparisons": 4,
        "cbramod_sha": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        "v2_sha": "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
        "cbramod_emb_dim": 200,
        "v2_emb_dim": 32,
        "pca_emb_dim": 32,
        "joint_dim": 264
    },
    "constraints_honored": m19["constraints_honored"]
}

# ── Serialize the record with indent=2 ─────────────────────────
record_str = json.dumps(m19_record, indent=2, ensure_ascii=False)
# Re-indent to match archive style (2-space indent at top level for experiments array items)
lines = record_str.split('\n')
indented_lines = []
for line in lines:
    if line.startswith(' '):
        indented_lines.append('  ' + line)  # indent one level under experiments array
    else:
        indented_lines.append('  ' + line)
indented_record = '\n'.join(indented_lines)

# ── Byte-preserving text splice ───────────────────────────────
archive_text = ARCHIVE_PATH.read_text(encoding='utf-8')

# Find the experiments array close bracket.
# The m18 entry ends with "  }\n  ],\n" (within experiments array)
# We need to find the FIRST "  ]," that comes after the m18 experiment's closing "  }"
# Strategy: find "m18-learned-joint-embedding" and then locate the closing of its object + array

marker = '"id": "m18-learned-joint-embedding"'
idx = archive_text.find(marker)
assert idx >= 0, "Could not find m18-learned-joint-embedding marker"

# From idx, find the closing of the experiments array: search for '  ],'
# after the m18 entry's closing brace
search_from = idx
# Find the next occurrence of a line that is exactly '  ],' after the m18 object
# We look for the pattern: "  }\n  ],\n" (closing brace of m18, then close bracket of experiments)
close_pattern = '  }\n  ],'
close_idx = archive_text.find(close_pattern, search_from)
assert close_idx >= 0, "Could not find experiments array close pattern"

# The insertion point is right before the '  ]' — i.e., after '  }\n'
# We insert the new record (comma + formatted JSON) between '  }\n' and '  ],\n'
insertion_point = close_idx + len('  }\n')  # right after '  }\n'

# Add comma after existing entry, then the new record
new_text = archive_text[:insertion_point]
new_text += ', ' + indented_record.strip()  # comma-separated new entry
new_text += archive_text[insertion_point:]

# ── Write back ────────────────────────────────────────────────
# First, validate the result is valid JSON
try:
    json.loads(new_text)
except json.JSONDecodeError as e:
    print(f"ERROR: Spliced archive is invalid JSON: {e}")
    raise

ARCHIVE_PATH.write_text(new_text, encoding='utf-8')

# ── Verify ───────────────────────────────────────────────────
with open(ARCHIVE_PATH) as f:
    final = json.load(f)

exp_ids = [e['id'] for e in final['experiments']]
print(f"Total experiments in archive: {len(final['experiments'])}")
print(f"Experiment IDs: {exp_ids}")
print(f"M19 appended: {'m19-dimensionwise-embedding' in exp_ids}")

# Verify prior records unchanged
for e in final['experiments']:
    if e['id'] == 'm18-learned-joint-embedding':
        assert e['results']['best_learned_r5'] == 0.7856, "M18 record was modified!"
    if e['id'] == 'joint-embedding-fusion':
        assert e['results']['best_joint_r_at_5'] == 0.7584, "Joint fusion record was modified!"
print("All prior records preserved ✓")
