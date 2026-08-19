#!/usr/bin/env python3
"""Append M20 experiment record to benchmark_archive.json.

Uses byte-preserving text-splicing: finds the closing ']' of the
'experiments' array (before "fine_tuning_experiments") and splices
in the new record. All prior bytes are untouched.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = REPO / "reports" / "benchmark_archive.json"
M20_RESULTS_PATH = REPO / "reports" / "m20_embedding_robustness_results.json"

# ── Load M20 results to source all numbers ──
with open(M20_RESULTS_PATH) as f:
    m20 = json.load(f)

# ── Construct archive experiment record ──
m20_record = {
    "id": "m20-embedding-robustness",
    "experiment_name": "Mission 20: Robust Validation of Best Learned 264-D EEG Embedding",
    "date": "2026-08-17",
    "author": "NeuroFabric team",
    "mission": "Mission 20 — independent replication and robustness validation of M18 block-weighted 264-D embedding and M19 C-shrinkage embedding on the 50-subject session-disjoint LOSO protocol",
    "model": "Three candidates evaluated: (A) raw 264-D concat, (B) M18 block-weighted 264-D (3 params), (C) M19 C-shrinkage 264-D (264 params) — all from frozen CBraMod-200 + V2-32 + PCA-32",
    "models_compared": [
        "Raw 264-D concat (baseline)",
        "M18 block-weighted 264-D (3 params, RidgeClassifier train-only per fold)",
        "M19 C-shrinkage 264-D (264 params, 50/50 ridge + block-expanded)"
    ],
    "dataset": "PhysioNet EEGMMIDB (S001-S050)",
    "subjects": 50,
    "trials": 4500,
    "protocol": m20["protocol"],
    "preprocessing": {
        "channels_cbramod": 19,
        "channels_v2_prod": 22,
        "sample_rate_hz": 250,
        "window_samples": 1000,
        "bandpass_hz": [4.0, 38.0],
        "pca_features": 110,
        "pca_components": 32,
        "pca_fit": "train-only, per LOSO fold (no leakage)",
        "embedding_dim": {
            "cbramod": 200,
            "v2": 32,
            "pca32": 32,
            "joint": 264
        },
        "normalization": "per-block L2 normalize + global L2 normalize",
        "seed": 42
    },
    "hypothesis": "M18 block-weighted 264-D embedding will be independently reproduced and confirmed as robustly superior to raw 264-D concatenation; M19 C-shrinkage will NOT significantly improve over M18 (independently verifying the M19 negative result).",
    "results": {
        "A_raw_264d_concat": m20["candidates"]["A_raw_264d_concat"],
        "B_m18_block_weighted": m20["candidates"]["B_m18_block_weighted"],
        "C_m19_c_shrinkage": m20["candidates"]["C_m19_c_shrinkage"],
        "primary_comparison_m18_vs_raw": m20["primary_comparison"],
        "secondary_comparison_m19_vs_m18": m20["secondary_comparison"],
        "fisher_analysis": m20["fisher_analysis"],
    },
    "weight_analysis": {
        "description": "Per-fold M18 block weight statistics across 50 LOSO folds",
        "weight_stability": m20["candidates"]["B_m18_block_weighted"]["weight_stability"],
        "weight_performance_correlation": m20["robustness_checks"]["weight_performance_correlation"],
        "weight_zeroed_check": m20["robustness_checks"]["weight_zeroed_check"],
        "weight_matches_m18_expected": m20["robustness_checks"]["weight_matches_m18_expected"],
    },
    "robustness_checks": {
        "fold_dominance_m18_vs_raw": m20["robustness_checks"]["fold_dominance_m18_vs_raw"],
        "bootstrap_ci_m18_r5": m20["candidates"]["B_m18_block_weighted"]["bootstrap_ci_r5"],
        "bootstrap_ci_raw_r5": m20["candidates"]["A_raw_264d_concat"]["bootstrap_ci_r5"],
        "bootstrap_ci_m19_r5": m20["candidates"]["C_m19_c_shrinkage"]["bootstrap_ci_r5"],
        "reproducibility_seed42": m20["robustness_checks"]["reproducibility_seed42"],
    },
    "pairwise_comparisons": {
        "primary_m18_vs_raw": {
            "mean_diff": m20["primary_comparison"]["delta_r_at_5"],
            "t_statistic": m20["primary_comparison"]["t_statistic"],
            "p_value": m20["primary_comparison"]["p_value"],
            "cohen_d": m20["primary_comparison"]["cohen_d"],
            "ci95": list(m20["primary_comparison"]["ci95_diff"]),
            "significant_after_bonferroni": m20["primary_comparison"]["significant_after_bonferroni"],
            "bonferroni_alpha": m20["primary_comparison"]["bonferroni_alpha"],
        },
        "secondary_m19_vs_m18": {
            "mean_diff": m20["secondary_comparison"]["delta_r_at_5"],
            "t_statistic": m20["secondary_comparison"]["t_statistic"],
            "p_value": m20["secondary_comparison"]["p_value"],
            "cohen_d": m20["secondary_comparison"]["cohen_d"],
            "ci95": list(m20["secondary_comparison"]["ci95_diff"]),
            "significant_after_bonferroni": m20["secondary_comparison"]["significant_after_bonferroni"],
            "bonferroni_alpha": m20["secondary_comparison"]["bonferroni_alpha"],
            "m18_r5_matches_expected": m20["secondary_comparison"]["m18_r5_matches_expected"],
            "m19_r5_matches_expected": m20["secondary_comparison"]["m19_r5_matches_expected"],
            "p_value_matches_expected": m20["secondary_comparison"]["p_value_matches_expected"],
            "cohen_d_matches_expected": m20["secondary_comparison"]["cohen_d_matches_expected"],
        },
    },
    "cross_check_with_historical": {
        "m18_r5_historical": m20["m18_expected_r5"],
        "m18_r5_reproduced": m20["candidates"]["B_m18_block_weighted"]["r_at_5"],
        "m18_r5_matches": m20["secondary_comparison"]["m18_r5_matches_expected"],
        "m19_r5_historical": m20["m19_expected_r5"],
        "m19_r5_reproduced": m20["candidates"]["C_m19_c_shrinkage"]["r_at_5"],
        "m19_r5_matches": m20["secondary_comparison"]["m19_r5_matches_expected"],
        "m19_p_historical": m20["m19_expected_p_vs_m18"],
        "m19_p_reproduced": m20["secondary_comparison"]["p_value"],
        "m19_p_matches": m20["secondary_comparison"]["p_value_matches_expected"],
        "m19_d_historical": m20["m19_expected_d_vs_m18"],
        "m19_d_reproduced": m20["secondary_comparison"]["cohen_d"],
        "m19_d_matches": m20["secondary_comparison"]["cohen_d_matches_expected"],
    },
    "leakage_audit": m20["leakage_audit"],
    "decision": m20["decision"],
    "decision_reason": m20["decision_reason"],
    "replace_recommendation": m20["replace_recommendation"],
    "result_classification": "PASS — M18 robustly validated as canonical embedding; M19 C-shrinkage adds no significant improvement",
    "contaminated": False,
    "status": m20["status"],
    "report_file": "reports/MISSION20_EMBEDDING_ROBUSTNESS_REPORT.md",
    "results_json": "reports/m20_embedding_robustness_results.json",
    "benchmark_script": "scripts/tmp/m20_embedding_robustness.py",
    "provenance": {
        "cache_source": "reports/.cbramod_cross_session_cache.npz (verified: SHA match)",
        "output_cache": "reports/.m20_embedding_robustness_cache.npz",
        "git_head": "b9164a664fce039df24c23656427a30c3a966926",
        "seed": 42,
        "n_bootstrap": 2000,
        "n_folds_loso": 50,
        "session_disjoint_splits": 4500,
        "bonferroni_alpha": 0.025,
        "bonferroni_comparisons": 2,
        "cbramod_sha": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        "v2_sha": "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
        "runtime_seconds_total": 24,
        "runtime_seconds_raw": 3.1,
        "runtime_seconds_m18": 7.5,
        "runtime_seconds_m19": 13.1,
    },
    "constraints_honored": m20["constraints_honored"],
}

# ── Serialize and indent one level under experiments array ──
record_str = json.dumps(m20_record, indent=2, ensure_ascii=False)
lines = record_str.split('\n')
indented_lines = ['  ' + line for line in lines]
indented_record = '\n'.join(indented_lines)

# ── Byte-preserving text splice ──
archive_text = ARCHIVE_PATH.read_text(encoding='utf-8')

# Find the experiments array close bracket (before "fine_tuning_experiments")
# Search from the cbramod-remap marker
marker = '"id": "cbramod-remap-50subj"'
idx = archive_text.find(marker)
assert idx >= 0, "Could not find cbramod-remap marker"

# Find the '],' that closes the experiments array, right before "fine_tuning_experiments"
close_pattern = '  ],\n  "fine_tuning_experiments"'
close_idx = archive_text.find(close_pattern, idx)
assert close_idx >= 0, "Could not find experiments array close pattern"

# Find the last '  }' before the '],' (closing the last experiment entry)
brace_idx = archive_text.rfind('  }', idx, close_idx)
assert brace_idx >= idx, "Could not find last experiment closing brace"

# Insertion point: right after '  }'
insertion_point = brace_idx + len('  }')

new_text = archive_text[:insertion_point]
new_text += ', ' + indented_record.strip()
new_text += archive_text[insertion_point:]

# ── Validate ──
try:
    json.loads(new_text)
except json.JSONDecodeError as e:
    print(f"ERROR: Spliced archive is invalid JSON: {e}")
    raise

ARCHIVE_PATH.write_text(new_text, encoding='utf-8')

# ── Verify ──
with open(ARCHIVE_PATH) as f:
    final = json.load(f)

exp_ids = [e['id'] for e in final['experiments']]
print(f"Total experiments in archive: {len(final['experiments'])}")
print(f"M20 present: {'m20-embedding-robustness' in exp_ids}")
print(f"Archive experiment count: {len(final['experiments'])} (was 20, now {len(final['experiments'])})")

# Verify prior records unchanged
for e in final['experiments']:
    if e['id'] == 'm18-learned-joint-embedding':
        assert e['results']['best_learned_r5'] == 0.7856, "M18 record was modified!"
    if e['id'] == 'm19-dimensionwise-embedding':
        assert e['results']['best_learned_r5'] == 0.786, "M19 record was modified!"
    if e['id'] == 'cbramod-remap-50subj':
        assert abs(e['decision']['cbramod_accuracy'] - 0.3043) < 0.001, "CBRaMod remap record was modified!"

print("All prior records preserved ✓")

# Verify M20 record
m20_arch = [e for e in final['experiments'] if e['id'] == 'm20-embedding-robustness'][0]
print(f"\nM20 archive entry:")
print(f"  Decision: {m20_arch['decision']}")
print(f"  M18 R@5: {m20_arch['results']['B_m18_block_weighted']['r_at_5']}")
print(f"  M19 R@5: {m20_arch['results']['C_m19_c_shrinkage']['r_at_5']}")
print(f"  M18 vs Raw p: {m20_arch['results']['primary_comparison_m18_vs_raw']['p_value']}")
print(f"  M19 vs M18 p: {m20_arch['results']['secondary_comparison_m19_vs_m18']['p_value']}")
