#!/usr/bin/env python3
"""Append CBraMod remap experiment (id=cbramod-remap-50subj) to benchmark_archive.json.

Uses byte-preserving text-splicing: inserts the new record before the closing
']' of the 'experiments' array, leaving all prior bytes intact.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = REPO / "reports" / "benchmark_archive.json"
RESULTS_PATH = REPO / "reports" / "cbramod_remap_50subj_results.json"

# ── Load source results ────────────────────────────────────────
with open(RESULTS_PATH) as f:
    src = json.load(f)

# ── Construct archive experiment record ────────────────────────
remap_record = {
    "id": "cbramod-remap-50subj",
    "experiment_name": "CBraMod 19→22 Channel Remap Study + 50-Subject LOSO Validation",
    "date": "2026-08-14",
    "author": "NeuroFabric team",
    "mission": "Next Model Mission — CBraMod remap study: determine whether CBraMod (native 19-ch, fixed-shape ONNX) earns a server-side specialist role vs V2 (production GA, 22-ch) and PCA bandpower, on the locked T-032 50-subject LOSO protocol. Decision rule: CBraMod acc ≥ PCA AND ≥ V2 with Bonferroni-corrected p < 0.05 (3 pairwise comparisons).",
    "model": "CBraMod (server-side ONNX, 19 channels, 200-D mean-tokens) vs EEGConformer v2 (22 channels, 32-D) vs PCA bandpower (110→32-D, train-only)",
    "models_compared": [
        "CBraMod-200 (onnx, native 19-ch, mean-tokens pooling)",
        "EEGConformer v2 (onnx, 22-ch, 32-D)",
        "PCA bandpower (110 features → PCA(32), train-only per fold)"
    ],
    "dataset": src["data"]["dataset"],
    "subjects": 50,
    "trials": src["data"]["n_trials"],
    "protocol": "LOSO 50-fold, nearest-centroid (cosine), Recall@K with train-only pool + self-retrieval exclusion, paired t-tests on 50 per-fold accuracies, Bonferroni-corrected over 3 model pairs (alpha=0.05/3=0.01667), seed=42",
    "remap_design": src["remap_design"],
    "preprocessing": src["preprocessing"],
    "artifacts": src["artifacts"],
    "results": src["results"],
    "class_separability": src["class_separability"],
    "latency_ms": src["latency_ms"],
    "statistical_comparisons": src["statistical_comparisons"],
    "bonferroni": src["bonferroni"],
    "decision": src["decision"],
    "contaminated": False,
    "status": "COMPLETE - negative result. CBraMod (0.304 acc) does NOT significantly beat V2 (0.325) or PCA (0.306) after Bonferroni correction (p=0.353, p=1.0). CBraMod is NOT promoted or routed. All constraints honored. CBraMod remains server-side-only (wasmCompatible:false).",
    "report_file": "reports/CBRAMOD_REMAP_50SUBJ_REPORT.md",
    "results_json": "reports/cbramod_remap_50subj_results.json",
    "benchmark_script": "scripts/tmp/cbramod_remap_50subj.py",
    "provenance": {
        "data_dir": src["data"]["data_dir"],
        "git_head": src["git_head"],
        "seed": 42,
        "n_folds_loso": 50,
        "n_trials": src["data"]["n_trials"],
        "bonferroni_corrected_alpha": src["bonferroni"]["corrected_alpha"],
        "n_pairwise_comparisons": src["bonferroni"]["n_comparisons"],
        "cbramod_sha": src["artifacts"]["cbramod"]["sha256"],
        "eegconformer_v2_sha": src["artifacts"]["eegconformer_v2"]["sha256"],
        "cbramod_wasm_compatible": False,
        "v2_wasm_compatible": True
    },
    "constraints_honored": {
        "v2_production_unchanged": True,
        "v2_rollout_unchanged": True,
        "no_default_preferred_change": True,
        "env_unchanged": True,
        "no_model_retraining": True,
        "no_artifact_modification": True,
        "no_onnx_modification": True,
        "no_pca_change": True,
        "no_eegpt_labram_femba_change": True,
        "no_production_code_changes": True,
        "cbramod_not_deployed": True,
        "cbramod_wasm_blocker_preserved": True,
        "loso_leakage_free": True,
        "train_only_pca_fit": True,
        "no_channel_interpolation_or_zero_fill": True,
        "bonferroni_correction": True,
        "seed_42_reproducible": True,
        "prior_archive_records_byte_preserved": True,
        "no_faked_staging_soak": True
    }
}

# ── Serialize with indent=2 and indent one level under experiments array ──
record_str = json.dumps(remap_record, indent=2, ensure_ascii=False)
lines = record_str.split('\n')
indented_lines = ['  ' + line for line in lines]
indented_record = '\n'.join(indented_lines)

# ── Byte-preserving text splice ────────────────────────────────
archive_text = ARCHIVE_PATH.read_text(encoding='utf-8')

# Find the experiments array close bracket (after m19-dimensionwise-embedding)
marker = '"id": "m19-dimensionwise-embedding"'
idx = archive_text.find(marker)
assert idx >= 0, "Could not find m19 marker"

# Find the closing '],' of the experiments array (followed by fine_tuning_experiments key)
close_pattern = '  ],\n  "fine_tuning_experiments"'
close_idx = archive_text.find(close_pattern, idx)
assert close_idx >= 0, "Could not find experiments array close pattern after M19"

# Insertion point: right before '  ],' that closes experiments array
# The M19 entry's closing brace ends somewhere before this '],'
# We search backwards from close_idx to find the last '  }' before it
brace_idx = archive_text.rfind('  }', idx, close_idx)
assert brace_idx >= idx, "Could not find M19 closing brace"

insertion_point = brace_idx + len('  }')

new_text = archive_text[:insertion_point]
new_text += ', ' + indented_record.strip()
new_text += archive_text[insertion_point:]

# ── Validate ──────────────────────────────────────────────────
try:
    json.loads(new_text)
except json.JSONDecodeError as e:
    print(f"ERROR: Spliced archive is invalid JSON: {e}")
    raise

ARCHIVE_PATH.write_text(new_text, encoding='utf-8')

# ── Verify ────────────────────────────────────────────────────
with open(ARCHIVE_PATH) as f:
    final = json.load(f)

exp_ids = [e['id'] for e in final['experiments']]
print(f"Total experiments in archive: {len(final['experiments'])}")
print(f"M19 present: {'m19-dimensionwise-embedding' in exp_ids}")
print(f"CBRaMod remap present: {'cbramod-remap-50subj' in exp_ids}")

# Verify all prior records preserved
for e in final['experiments']:
    if e['id'] == 'm18-learned-joint-embedding':
        assert e['results']['best_learned_r5'] == 0.7856, "M18 modified!"
    if e['id'] == 'm19-dimensionwise-embedding':
        assert e['results']['best_learned_r5'] == 0.786, "M19 modified!"

print("All prior records preserved ✓")

# Verify remap record
remap = [e for e in final['experiments'] if e['id'] == 'cbramod-remap-50subj'][0]
print(f"\nRemap record:")
print(f"  Decision: {remap['decision']['result']}")
print(f"  CBraMod acc: {remap['decision']['cbramod_accuracy']}")
print(f"  V2 acc: {remap['decision']['v2_accuracy']}")
print(f"  PCA acc: {remap['decision']['pca_accuracy']}")
