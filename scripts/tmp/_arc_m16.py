#!/usr/bin/env python3
"""Append Mission-15 and Mission-16 experiment records to reports/benchmark_archive.json.

Byte-preserving: all prior content (idx0..idxN) is verified unchanged via
sha256 prefix comparison before and after. Uses text-splicing (same technique
as _arc_m14_phase1.py) rather than json.dump re-serialization.

Usage: python scripts/tmp/_arc_m16.py
"""

import json, hashlib, sys, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"

raw = ARCHIVE.read_text(encoding="utf-8")

# Anchor: the boundary between experiments[] and fine_tuning_experiments[]
# After _arc_m14_phase1.py ran, the last experiment's closing brace uses 2-space indent:
#   "git_head": "..."
#  }
#  ],
#  "fine_tuning_experiments": [
EXPERIMENTS_ANCHOR = '  }\n  ],\n  "fine_tuning_experiments": ['
assert raw.count(EXPERIMENTS_ANCHOR) == 1, f"Expected 1 anchor, found {raw.count(EXPERIMENTS_ANCHOR)}"

# Prefix = everything before the closing brace of the last experiment (idx0..last)
prefix = raw[:raw.find(EXPERIMENTS_ANCHOR)]
prefix_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
print(f"Prefix sha256 (idx0..last): {prefix_sha[:16]}...")

arch = json.loads(raw)
before_count = len(arch["experiments"])
print(f"Experiments before: {before_count}")

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()

# ---------------------------------------------------------------------------
# Mission 15 experiment record (reconstructed from MISSION15_COMPLETION_REPORT.md)
# ---------------------------------------------------------------------------
m15_record = {
    "id": "mission15-operational-validation",
    "experiment_name": "Mission 15: Production-Like Operational Validation & Conditional Opt-In Readiness",
    "date": "2026-08-15",
    "author": "NeuroFabric team",
    "mission": "Mission 15 - close the 4 INCONCLUSIVE operational gates (A1-A4) from Mission 14",
    "model": "CBraMod Tier-2 server-native path (onnx-cbramod-foundation-200d)",
    "model_version": "cbramod-encoder.onnx @ c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    "dataset": "PhysioNet EEGMMIDB (S001-S050), 50 subjects, 4-class MI (left hand, right hand, feet, tongue)",
    "subjects": 50,
    "trials": 4500,
    "protocol": "50-fold LOSO (4500 trials). 4 INCONCLUSIVE operational gates (A1-A4) closed against local Supabase stack with real JWT auth, real RPC, real pgvector, real ONNX inference (onnxruntime-node CPU EP). No mocks for auth/rate/vector-store/inference (except error-path injection in Phases 1/4).",
    "gates": {
        "A1_browser_wasm_runtime": "PASS - wasmCompatible:false verified in manifest; route is .server.ts (excluded from browser bundle); onnxruntime-node dynamically imported only in Node SSR context",
        "A2_signed_artifact_sha": "PASS - Phase 4: corrupted artifact -> 424 (SHA mismatch); size mismatch -> 424; restored artifact -> 200 with real ONNX; SHA-256 verified byte-for-byte",
        "A3_rate_limit_concurrency": "PASS - Phase 2: 20->200, 21st->429 with retry_after_ms; per-user isolation; 50 concurrent -> 20x200 + 30x429 (atomic UPSERT race-free); Phase 3: ramp 1/5/10/20/50 all PASS",
        "A4_api_contract_real_jwt": "PASS - Phase 1: 8/8 real GoTrue JWT auth + RLS; Phase 5: 16/16 status code contract through real route handler; embedEEG never called on non-200 paths"
    },
    "results": {
        "all_gates": "PASS (4/4)",
        "verdict": "READY_FOR_OPT_IN",
        "new_tests": 41,
        "new_tests_pass": True,
        "v2_artifact_sha_unchanged": True,
        "default_preferred_unchanged": True,
        "onnx_restored_byte_for_byte": True
    },
    "constraints_honored": {
        "default_preferred_unchanged": True,
        "v2_replaced": False,
        "pca_behavior_modified": False,
        "retrained_cbramod": False,
        "altered_model_weights": False,
        "weakened_ci": False,
        "deleted_tests": False,
        "mocked_production_validation": False,
        "reported_inconclusive_as_pass": False,
        "archive_byte_preserved_except_append": True,
        "ga_promotion_not_started": True
    },
    "contaminated": False,
    "status": "READY_FOR_OPT_IN - all 4 operational gates closed; CBraMod remains opt-in server-side only (not DEFAULT_PREFERRED)",
    "report_file": "reports/MISSION15_COMPLETION_REPORT.md",
    "git_head": git_head
}

# ---------------------------------------------------------------------------
# Mission 16 experiment record
# ---------------------------------------------------------------------------
m16_record = {
    "id": "m16-linear-probe-mi-classification",
    "experiment_name": "Mission 16: Linear-Probe MI Classification Benchmark (LOSO, 50 subjects)",
    "date": "2026-08-15",
    "author": "NeuroFabric team",
    "mission": "Mission 16 - scientific benchmark: CBraMod-200 vs V2-32 vs PCA-32 on 4-class MI classification",
    "model": "Linear probe (LogisticRegression L2, C=1.0) on cached embeddings",
    "models_compared": ["CBraMod-200", "V2-32", "PCA-32"],
    "dataset": "PhysioNet EEGMMIDB (S001-S050), 50 subjects x 6 runs x 15 trials = 4500 trials, 4-class MI",
    "subjects": 50,
    "trials": 4500,
    "protocol": "50-fold LOSO; LogisticRegression(L2, C=1.0, lbfgs) linear probe; PCA(32) on bandpower(110) train-only per fold; 2000 bootstrap CIs; paired t-tests with Bonferroni correction (3 comparisons, alpha=0.0167); seed=42",
    "label_mapping": {
        "0": "left hand (runs 5,7,9)",
        "1": "right hand (runs 5,7,9)",
        "2": "feet (runs 6,8,10)",
        "3": "tongue (runs 6,8,10)"
    },
    "hypothesis": "CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe on 4-class MI classification (LOSO, 50 subjects)",
    "results": {
        "cbramod_200_accuracy": 0.3020,
        "cbramod_200_accuracy_ci95": [0.2878, 0.3162],
        "v2_32_accuracy": 0.3167,
        "v2_32_accuracy_ci95": [0.3009, 0.3333],
        "pca_32_accuracy": 0.3244,
        "pca_32_accuracy_ci95": [0.3047, 0.3442],
        "chance_accuracy": 0.25,
        "all_models_above_chance": True
    },
    "pairwise_comparisons": {
        "cbramod_vs_pca": {"mean_diff": -0.0224, "t_stat": -2.08, "p_value": 0.04236, "cohen_d": -0.295, "significant_after_bonferroni": False, "bonferroni_alpha": 0.0167},
        "cbramod_vs_v2": {"mean_diff": -0.0147, "t_stat": -1.43, "p_value": 0.15919, "cohen_d": -0.202, "significant_after_bonferroni": False, "bonferroni_alpha": 0.0167},
        "v2_vs_pca": {"mean_diff": -0.0078, "t_stat": -0.59, "p_value": 0.55720, "cohen_d": -0.084, "significant_after_bonferroni": False, "bonferroni_alpha": 0.0167}
    },
    "hypothesis_supported": True,
    "hypothesis_rejected": False,
    "decision": "CBraMod-200 does NOT significantly outperform PCA-32 or V2-32 - hypothesis supported. PCA-32 + linear probe (32.44%) marginally outperforms CBraMod-200 (30.20%) and V2-32 (31.67%), but pairwise differences are not significant after Bonferroni correction (alpha=0.0167).",
    "contaminated": False,
    "status": "COMPLETE - hypothesis SUPPORTED. CBraMod-200 linear probe does NOT significantly outperform PCA-32 + linear probe. All models above chance (25%).",
    "report_file": "reports/MISSION16_LINEAR_PROBE_REPORT.md",
    "results_json": "reports/m16_linear_probe_results.json",
    "benchmark_script": "scripts/tmp/m16_linear_probe_benchmark.py",
    "provenance": {
        "cache_source": "reports/.cbramod_cross_session_cache.npz (Mission 11 cached embeddings, no retraining)",
        "git_head": git_head,
        "seed": 42,
        "n_bootstrap": 2000,
        "n_comparisons": 3,
        "bonferroni_alpha": 0.0167
    },
    "constraints_honored": {
        "no_model_retraining": True,
        "no_artifact_modification": True,
        "no_production_code_changes": True,
        "read_only_cached_embeddings": True,
        "loso_leakage_free": True,
        "train_only_pca": True,
        "bonferroni_correction": True,
        "seed_42_reproducible": True,
        "prior_archive_records_byte_preserved": True
    }
}

# ---------------------------------------------------------------------------
# Splice records using text replacement (same technique as _arc_m14_phase1.py)
# ---------------------------------------------------------------------------
# The current anchor is:  '  }\n  ],\n  "fine_tuning_experiments": ['
# The last experiment closes with '  }' (2-space indent from _arc_m14_phase1.py).
# We replace the anchor with: '  },\n' + indented_m15 + ',\n' + indented_m16 + '\n  ],\n  "fine_tuning_experiments": ['

m15_json = json.dumps(m15_record, indent=2, ensure_ascii=False)
m16_json = json.dumps(m16_record, indent=2, ensure_ascii=False)

# Indent matches _arc_m14_phase1.py: 2-space base indent for experiment records
m15_indented = "\n".join(("  " + line) if line else line for line in m15_json.split("\n"))
m16_indented = "\n".join(("  " + line) if line else line for line in m16_json.split("\n"))

replacement = '  },\n' + m15_indented + ',\n' + m16_indented + '\n  ],\n  "fine_tuning_experiments": ['
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
assert len(new_arch["experiments"]) == before_count + 2, \
    f"Expected {before_count + 2} experiments, got {len(new_arch['experiments'])}"
print(f"OK: experiments count {before_count} -> {len(new_arch['experiments'])}")

# Verify the IDs
assert new_arch["experiments"][-2]["id"] == "mission15-operational-validation", \
    f"Expected mission15 at [-2], got {new_arch['experiments'][-2]['id']}"
assert new_arch["experiments"][-1]["id"] == "m16-linear-probe-mi-classification", \
    f"Expected m16 at [-1], got {new_arch['experiments'][-1]['id']}"
print(f"OK: last two experiments: [{new_arch['experiments'][-2]['id']}], [{new_arch['experiments'][-1]['id']}]")

# ---------------------------------------------------------------------------
# Add 3 preserved artifacts for Mission 15
# ---------------------------------------------------------------------------
m15_artifacts = [
    {"type": "test", "path": "src/routes/api/eeg/embed/__tests__/-foundation-artifact-integrity-live.test.ts", "description": "Mission 15 Phase 4: artifact SHA serving-path tests (corrupt->424, size mismatch->424, restore->200, byte-for-byte)"},
    {"type": "test", "path": "src/routes/api/eeg/embed/__tests__/-foundation-api-contract-live.test.ts", "description": "Mission 15 Phase 5: full API contract under real JWT auth (16 status codes, no V2/PCA fallback)"},
    {"type": "json", "path": "reports/m15_jwt_test_tokens.json", "description": "Mission 15: real JWT test tokens (valid, expired, tampered) for GoTrue auth validation"}
]

# Find preserved_artifacts array
pa_key_idx = new_raw.find('"preserved_artifacts"')
assert pa_key_idx != -1, "ERROR: preserved_artifacts key not found"
pa_arr_start = new_raw.find('[', pa_key_idx)
# Find matching closing ]
depth = 0
pa_end = pa_arr_start
for i, c in enumerate(new_raw[pa_arr_start:], pa_arr_start):
    if c == '[':
        depth += 1
    elif c == ']':
        depth -= 1
        if depth == 0:
            pa_end = i
            break

# Get existing content inside the array
existing_inside = new_raw[pa_arr_start + 1:pa_end]
# Build new entries
new_art_text = existing_inside.rstrip()
for art in m15_artifacts:
    art_json = json.dumps(art, ensure_ascii=False)
    new_art_text += ",\n    " + art_json

new_pa = "[" + new_art_text + "\n  ]"
new_raw = new_raw[:pa_arr_start] + new_pa + new_raw[pa_end + 1:]

# Final validation
json.loads(new_raw)
final_arch = json.loads(new_raw)
print(f"\nOK: preserved_artifacts extended with 3 Mission 15 entries")
print(f"OK: final archive has {len(final_arch['experiments'])} experiments and {len(final_arch['preserved_artifacts'])} preserved artifacts")
print(f"OK: final archive is valid JSON")

# Write
ARCHIVE.write_text(new_raw, encoding="utf-8")
print(f"\nDone: appended Mission-15 and Mission-16 records to {ARCHIVE}")
print(f"Experiments: {before_count} -> {len(final_arch['experiments'])}")
print(f"Preserved artifacts: {len(arch['preserved_artifacts'])} -> {len(final_arch['preserved_artifacts'])}")
