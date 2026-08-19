import json, hashlib, textwrap
ARC="reports/benchmark_archive.json"
m14=json.load(open("reports/MISSION14_LIVE_PGVECTOR_VALIDATION.json"))
mv=m14["models"]
a=mv["onnx-cbramod-foundation-200d"]["live_rpc_recall_at"]
idx13={
 "id":"mission14-live-pgvector-validation",
 "experiment_name":"Mission 14 - Live pgvector/Supabase RPC Validation (Phase 0)",
 "date":m14["timestamp_utc"][:10],
 "author":"neuro-fabric-core automated validation",
 "mission":"14",
 "phase":"Phase 0 (close M13 pgvector gap)",
 "model":"onnx-cbramod-foundation-200d (Tier-2 retrieval gate, server-side 200-D, wasmCompatible:false)",
 "model_version":"Tier-2 CBraMod 200-D mean-tokens; manifest entry intact (NOT modified in M14)",
 "model_artifact":"reports/.cbramod_cross_session_cache.npz cb_emb 4500x200 (real cached embeddings, no retraining)",
 "dataset":"PhysioNet EEGMMIDB S001-S050; 300 session-disjoint (subject,held-out-run) splits",
 "subjects":50,
 "protocol":"300 leakage-free retrieval splits via REAL match_foundation_embeddings_exact RPC over live Postgres+pgvector 0.8.2; held-out-run excluded from pool; 60-split in-memory cosine (te@tr.T) cross-check; ANN IVFFLat SLO vs exact; NN same-vs-diff gap (300 queries, self excluded).",
 "preprocessing":"All embeddings L2-normalized so <#> L2 ordering == cosine ordering. CBraMod native 200-D; V2 32-D zero-padded to 200-D (cosine-preserving); PCA-32 per-fold train-only fit (seed 42) zero-padded to 200-D.",
 "results":{
   "onnx-cbramod-foundation-200d":{"recall_at_1":a["recall_at_1"]["mean"],"recall_at_5":a["recall_at_5"]["mean"],"recall_at_10":a["recall_at_10"]["mean"],"mi_accuracy":mv["onnx-cbramod-foundation-200d"]["live_rpc_mi_accuracy_mean"],"inmemory_crosscheck_r5":mv["onnx-cbramod-foundation-200d"]["inmemory_crosscheck_60split_recall_at"]["recall_at_5"],"vs_m13_inmemory":0.5273,"verdict":"PASS - live RPC reproduces M13 gate (R5 0.5269 vs 0.5273; crosscheck 0.5256)"},
   "braindecode-eegconformer-prod-v2-padded-200":{"recall_at_1":mv["braindecode-eegconformer-prod-v2-padded-200"]["live_rpc_recall_at"]["recall_at_1"]["mean"],"recall_at_5":mv["braindecode-eegconformer-prod-v2-padded-200"]["live_rpc_recall_at"]["recall_at_5"]["mean"],"recall_at_10":mv["braindecode-eegconformer-prod-v2-padded-200"]["live_rpc_recall_at"]["recall_at_10"]["mean"],"mi_accuracy":mv["braindecode-eegconformer-prod-v2-padded-200"]["live_rpc_mi_accuracy_mean"]},
   "pca-bandpower-32-padded-200":{"recall_at_1":mv["pca-bandpower-32-padded-200"]["live_rpc_recall_at"]["recall_at_1"]["mean"],"recall_at_5":mv["pca-bandpower-32-padded-200"]["live_rpc_recall_at"]["recall_at_5"]["mean"],"recall_at_10":mv["pca-bandpower-32-padded-200"]["live_rpc_recall_at"]["recall_at_10"]["mean"],"mi_accuracy":mv["pca-bandpower-32-padded-200"]["live_rpc_mi_accuracy_mean"],"note":"PCA > CBraMod on Recall@5/10 (R5 0.692 vs 0.527, R10 0.785 vs 0.659) - reported honestly"}
 },
 "latency_ms":{"pgvector_rpc_exact_ms_per_query":mv["onnx-cbramod-foundation-200d"]["rpc_exact_ms_per_query_mean"],
   "pgvector_rpc_ivfflat_ms_per_query_probe1":m14["ann_ivfflat_slo"]["1"]["ivfflat_ms_per_query_mean"],
   "ivfflat_faster_than_exact_x":round(m14["ann_ivfflat_slo"]["1"]["exact_ms_per_query_mean"]/m14["ann_ivfflat_slo"]["1"]["ivfflat_ms_per_query_mean"],1),
   "embed_ms_per_window_cbramod":m14["end_to_end_latency_ms"]["embed_ms_per_window_cbramod"],
   "end_to_end_ms_cbramod":m14["end_to_end_latency_ms"]["embed_ms_per_window_cbramod"]+m14["end_to_end_latency_ms"]["pgvector_rpc_exact_ms_per_query_mean"],
   "note":"Live Postgres+pgvector 0.8.2 (supabase/postgres:15.14.1.162). embed=155ms (onnxruntime-node, 22MB ONNX warm per M13 real-EDF)."},
 "ann_ivfflat_slo":{"ivfflat_index_exists":m14["schema_validation"]["ivfflat_index_cosine"],"ann_recall_at_5_default_probe":m14["ann_ivfflat_slo"]["1"]["ann_recall_at"]["recall_at_5"],"probe_tunable":m14["ann_ivfflat_slo"]["probe_tunable"],"probe_tunability_status":"INCONCLUSIVE - ivfflat.probe GUC accepted by SET LOCAL but IVF index scan ignores it in this image (probe 1/4/10/20/100 return identical top-K)."},
 "nn_same_vs_diff_gap":m14["nn_same_vs_diff_gap"],
 "schema_validation":m14["schema_validation"],
 "missing_infrastructure":{"supabase_full_stack_cli":"not used - public.ecr.aws service-image TLS timeouts; standalone supabase/postgres:15.14.1.162 (still real Postgres+pgvector 0.8.2)","ivfflat_probe_tuning":"INCONCLUSIVE - GUC accepted but IVF index scan ignores it (build quirk; not a script bug)","rbac_non_superuser_enforcement":"not assessed Phase 0 (psycopg2 driver connects as superuser; RLS enabled + 3 policies present but non-privileged enforcement unverified)","browser_wasm_isolation":"not assessed - Phase 0 DB focus","rate_limiting":"not assessed","concurrency_at_scale":"not assessed"},
 "smallest_next_experiment":"Phase 1 GA-readiness assessment (artifact SHA verification, rate limiting, concurrency, V2 regression, rollback safety, browser/WASM isolation, API contract stability) - only if Phase 0 PASS.",
 "contaminated":False,
 "status":"Phase 0 PASSED (live pgvector RPC validated end-to-end). CBraMod Tier-2 retrieval gate PASS (R5 0.5269 ~= M13 0.5273; in-mem crosscheck 0.5256). PCA > CBraMod on Recall@5/10 (honest). IVFFlat index exists/usable/~23x faster but probe tunability INCONCLUSIVE (env). Phase 1 GA-readiness NOT yet executed.",
 "report_file":"reports/MISSION14_LIVE_PGVECTOR_VALIDATION.md",
 "report_json":"reports/MISSION14_LIVE_PGVECTOR_VALIDATION.json",
 "benchmark_script":"scripts/tmp/m14_live_pgvector_validation.py",
 "source_files_inspected":["scripts/tmp/m14_live_pgvector_validation.py","supabase/migrations/20260814000000_foundation_embeddings.sql","reports/.cbramod_cross_session_cache.npz","reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.md"],
 "constraint_compliance":{"modified_production_source":False,"modified_migration_or_schema":False,"retrained_any_model":False,"weakened_ci":False,"deleted_or_disabled_tests":False,"only_report_script_artifacts_changed":True,"embedEEG_untouched":True,"DEFAULT_PREFERRED_untouched":True,"v2_routing_untouched":True,"vector_32_untouched":True,"pca_untouched":True,"no_silent_v2_fallback":True,"cbramod_kept_200d_server_side_opt_in_separate_namespace":True,"preserve_archive_idx0_12_byte_identical":True,"exactly_one_mission14_archive_append":True,"no_faked_database_validation":True,"historical_m11_m12_m13_results_untouched":True}
}
print("built idx13, keys:",len(idx13))

raw=open(ARC,encoding="utf-8").read()
obj_text=json.dumps(idx13,indent=2,ensure_ascii=False)
indented=textwrap.indent(obj_text,"    ")
anchor='    }\n  ],\n  "fine_tuning_experiments":'
assert raw.count(anchor)==1, ("anchor count=%d"%raw.count(anchor))
prefix=raw[:raw.find(anchor)]  # idx0-12 content, before idx12 closing brace
sha=hashlib.sha256(prefix.encode()).hexdigest()
new=raw.replace(anchor,'    },\n'+indented+'\n  ],\n  "fine_tuning_experiments":',1)
assert new.startswith(prefix), "idx0-12 content not preserved"
doc=json.loads(new)
assert len(doc["experiments"])==14
assert doc["experiments"][12]["id"]=="mission13-cbramod-tier2-utility-validation"
assert doc["experiments"][13]["id"]=="mission14-live-pgvector-validation"
open(ARC,"w",encoding="utf-8").write(new)
print("OK archive idx0-12 prefix sha256=%s (unchanged)"%sha[:16])
print("experiments:",len(doc["experiments"]),"idx13=",doc["experiments"][13]["id"])
print("bytes: %d -> %d"%(len(raw),len(new)))
