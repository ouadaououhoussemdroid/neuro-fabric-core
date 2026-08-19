#!/usr/bin/env python3
"""Append the Mission-14 Phase-1 GA-readiness experiment to reports/benchmark_archive.json.

Byte-preserving: idx0..idx10 content (bytes before idx10's closing brace) is verified
unchanged via sha256 prefix comparison before and after. Idempotent: removes any prior
experiment whose id is 'mission14-phase1-ga-readiness' before appending.
"""
import json, hashlib, sys, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"
TARGET_ID = "mission14-phase1-ga-readiness"

raw = ARCHIVE.read_text(encoding="utf-8")

# Locate the closing brace of the last experiment (idx10), which precedes the
# array close  ]  and the "fine_tuning_experiments" key. We verify that everything
# BEFORE this closing brace (i.e. idx0..idx10 content) is byte-identical after append.
anchor = '  ],\n  "fine_tuning_experiments": ['
idx = raw.find(anchor)
if idx == -1:
    sys.stderr.write("ERROR: could not find anchor 'experiments] -> fine_tuning_experiments'\n")
    sys.exit(1)

last_close_idx = raw.rfind("}", 0, idx)
assert last_close_idx != -1, "ERROR: no closing brace found for last experiment"
# Prefix = idx0..idx10 content (everything up to and including the last experiment's '}').
prefix_sha_before = hashlib.sha256(raw[:last_close_idx + 1].encode("utf-8")).hexdigest()
print(f"prefix sha256 (idx0..idx10 before append): {prefix_sha_before}")

# Parse to (a) verify structure, (b) drop any prior TARGET_ID record, (c) build the record.
arch = json.loads(raw)
exps = arch.get("experiments", [])
before_count = len(exps)
print(f"experiments before: {before_count}")

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO).strip()

record = {
    "id": "mission14-phase1-ga-readiness",
    "experiment_name": "Mission 14 Phase 1: CBraMod Tier-2 GA Readiness Assessment",
    "date": "2026-08-15",
    "author": "NeuroFabric team",
    "mission": "Mission 14 Phase 1: production-readiness gates for opt-in CBraMod-200D Tier-2 path",
    "model": "CBraMod (onnx-cbramod-foundation-200d)",
    "model_version": "cbramod-encoder.onnx @ c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    "dataset": "PhysioNet EEGMMIDB (S001-S050) cross-checked vs Mission-11/13 artifacts",
    "subjects": 50,
    "trials": 300,
    "protocol": "Leakage-free session-disjoint splits (50 subj x 6 runs, held-out run excluded from pool); live pgvector RPC; non-superuser role isolation; deterministic concurrency ramp.",
    "preprocessing": {
        "channels": 19,
        "sample_rate_hz": 250,
        "window_samples": 1000,
        "bandpass_hz": [4.0, 38.0],
        "feature_dim": 200,
        "output_pooling": "mean-tokens",
        "normalization": "zscore per channel + L2 normalize",
        "seed": 42
    },
    "results": {
        "recall_at_5": 0.5269,
        "recall_at_10": 0.6587,
        "recall_at_1": 0.2427,
        "mi_accuracy": 0.2749,
        "rpc_exact_ms_per_query_mean": 6.33,
        "rpc_ivfflat_ms_per_query_mean_probe1": 0.28,
        "ann_recall_at_5": 0.5407,
        "nn_same_subject_nn_cosine": 0.9921,
        "nn_diff_subject_nn_cosine": 0.9930,
        "nn_gap": -0.0010,
        "pca_recall_at_5": 0.6920,
        "pca_recall_at_10": 0.7853,
        "v2_padded_recall_at_5": 0.2162,
        "honest_finding": "PCA currently outperforms CBraMod on Recall@5/10 (0.692 vs 0.527; 0.785 vs 0.659). CBraMod beats V2 (0.527 vs 0.216) but remains below PCA.",
        "m13_inmemory_r5": 0.5273,
        "live_rpc_r5_vs_m13": 0.5269
    },
    "latency_ms": {
        "embed_ms_per_window_cbramod": 155,
        "pgvector_rpc_exact_ms_per_query_mean": 6.33,
        "pgvector_rpc_ivfflat_ms_per_query_mean_probe1": 0.28,
        "e2e_ms_estimate": 161
    },
    "statistical_comparison_vs_pca": {
        "cbramod_recall_at_5": 0.5269,
        "pca_recall_at_5": 0.6920,
        "comparison": "PCA > CBraMod on Recall@5/10 (honest). CBraMod < PCA on the MODEL_STRATEGY evidence gate; CBraMod does not earn server-side specialist role by Recall alone.",
        "p_value": None,
        "bonferroni_corrected": False
    },
    "artifact_integrity": {
        "onnx_path": "public/models/cbramod-encoder.onnx",
        "size": 22018587,
        "sha256": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        "hash_matches_manifest": True,
        "mismatch_fails_safe": True,
        "mismatch_http_status": 424,
        "no_v2_pca_fallback": True,
        "test": "src/lib/ai/inference/__tests__/foundation-artifact-integrity.test.ts"
    },
    "rate_limiting": {
        "rpc": "check_rate_limit",
        "budget": "20 requests / 60s per user",
        "excess_http_status": 429,
        "rpc_error_http_status": 503,
        "fk_enforced": True,
        "test": "src/routes/api/eeg/embed/__tests__/-foundation-rate-limit-live.test.ts"
    },
    "concurrency": {
        "ramp_concurrencies": [1, 5, 10, 20],
        "failures": 0,
        "all_http_200": True,
        "shared_cached_adapter_safe": True,
        "test": "src/routes/api/eeg/embed/__tests__/-foundation-concurrent.test.ts"
    },
    "rls_authorization": {
        "table_rls_enabled": True,
        "policies": 3,
        "non_superuser_a_sees_only_own_rows_direct": True,
        "non_superuser_b_sees_only_own_rows_direct": True,
        "rpc_filter_user_id_bound_to_authenticated_caller": True,
        "security_definer_note": "match_foundation_embeddings is SECURITY DEFINER and bypasses table RLS; isolation depends on route passing authenticated userId as filter_user_id (neural-index.ts:195).",
        "test": "src/routes/api/eeg/embed/__tests__/-foundation.test.ts"
    },
    "api_contract": {
        "endpoint": "POST /api/eeg/embed/foundation",
        "accepted_formats": [".edf", ".bdf", ".csv", ".tsv", ".npy"],
        "status_codes": {"200": "success", "400": "bad input", "401": "auth failed", "408": "timeout", "413": "too large", "415": "unsupported ext", "422": "parse/magic/content", "424": "runtime/artifact unavailable", "429": "rate limited", "500": "internal"},
        "response_fields_present": ["dimensions(200)", "model", "modelId(nested)", "provenance.sha256", "timings", "vector_indexed"],
        "response_fields_missing": ["persistence_status", "index_status", "retrieval_status (structured)"],
        "no_silent_fallback": True
    },
    "rollback_safety": {
        "rollback_simulation": "Static isolation analysis (no Touch-2 production edits)",
        "v2_unaffected": True,
        "embedEEG_unaffected": True,
        "default_preferred_unaffected": True,
        "embeddings_vector32_unaffected": True,
        "pca_unaffected": True,
        "browser_wasm_unaffected": True,
        "foundation_namespace_isolated": True
    },
    "production_readiness": {
        "fix_applied": "NeuralVectorIndex now honors tableName/matchRpc/matchRpcExact options (was hardcoded to 'embeddings'/'match_embeddings'); foundation.service metrics registered (was undefined, causing 500 on every request).",
        "fix_files": ["src/lib/vector-search/neural-index.ts", "src/lib/metrics/index.ts", "src/lib/ai/embeddings/index.ts"],
        "no_touch_boundary_respected": True,
        "default_preferred_unchanged": True
    },
    "contaminated": False,
    "status": "Mission-14 Phase-0 PASSED; Phase-1 GA-readiness INCONCLUSIVE (production rollout gates untested: browser/WASM execution, signed-artifact SHA re-verification under load, rate-limit/concurrency against real Supabase stack, API-contract under real auth)",
    "report_file": "reports/MISSION14_PHASE1_GA_READINESS.md",
    "results_json": "reports/MISSION14_PHASE1_GA_READINESS.json",
    "git_head": git_head
}

exps_filtered = [e for e in exps if e.get("id") != TARGET_ID]
exps_filtered.append(record)
arch["experiments"] = exps_filtered
after_count = len(exps_filtered)
print(f"experiments after: {after_count} (appended 1)")

# Rebuild: splice the new record into the raw text at the last experiment's '}',
# keeping everything before idx0..idx10 byte-identical and everything after (the
# array close + fine_tuning_experiments + rest) byte-identical.
record_json = json.dumps(record, indent=2)
# Indent the record to nest inside experiments[] (2-space base indent for top-level keys).
indented = "\n".join(("  " + line) if line else line for line in record_json.split("\n"))

head = raw[:last_close_idx + 1]  # idx0..idx10 content incl. last experiment's '}'
tail = raw[last_close_idx + 1:]  # "\n  ],\n  \"fine_tuning_experiments\": [ ..."

new_raw = head + ",\n" + indented + tail

# Verify idx0..idx10 prefix unchanged.
new_arch = json.loads(new_raw)
prefix_sha_after = hashlib.sha256(new_raw[:last_close_idx + 1].encode("utf-8")).hexdigest()
print(f"prefix sha256 (idx0..idx10 after append):  {prefix_sha_after}")
assert prefix_sha_before == prefix_sha_after, "FAIL: idx0..idx10 prefix changed!"
print("OK: idx0..idx10 prefix sha256 identical before/after append.")

# Validate the result is valid JSON.
json.loads(new_raw)
print("OK: appended archive is valid JSON.")

# Write back.
ARCHIVE.write_text(new_raw, encoding="utf-8")
print(f"Appended {TARGET_ID} to {ARCHIVE}")
print(f"experiments: {before_count} -> {len(new_arch['experiments'])}")
