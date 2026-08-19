/* Idempotent archive appender for the Mission-13 utility-validation (INCONCLUSIVE) result.
 * Adds exactly ONE record to reports/benchmark_archive.json as the next experiments[]
 * element, after asserting every pre-existing experiment + every sibling top-level
 * key is byte-identical before/after. Run once with:
 *   node scripts/tmp/append_mission13_archive.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const archivePath = join(process.cwd(), "reports", "benchmark_archive.json");
const before = readFileSync(archivePath, "utf8");
const archive = JSON.parse(before);

const id = "mission13-cbramod-tier2-utility-validation";
const priorLen = archive.experiments.length;

if (archive.experiments.some((e) => e.id === id)) {
  archive.experiments = archive.experiments.filter((e) => e.id !== id);
}

const entry = {
  id,
  experiment_name:
    "Mission 13 — CBraMod Tier-2 utility / platform-integrated retrieval validation",
  date: "2026-08-14",
  author: "NeuroFabric team",
  mission: "Mission 13 — CBraMod Tier-2 utility (platform-integrated retrieval) validation",
  model: "onnx-cbramod-foundation-200d (wasmCompatible:false, server-only)",
  model_version: "1.0.0",
  model_artifact: {
    url: "/models/cbramod-encoder.onnx",
    sha256: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    size_bytes: 22018587,
    input_shape: [1, 19, 1000],
    output_shape: [1, 19, 5, 200],
    embedding_dim: 200,
    wasmCompatible: false,
    wasmBlockers: ["DFT", "ReduceL2"],
    runtime: "onnxruntime-node CPU EP",
    shasum_verified_by: "foundation-e2e.test.ts (real 22MB forward) + Mission 11 validation",
  },
  dataset:
    "Investigation target: PhysioNet EEGMMIDB S001-S050 runs {5,6,7,8,9,10} (same session-disjoint " +
    "protocol as Mission 11). NOT downloaded/executed in Mission 13 — no .edf data present on disk " +
    "(0 files outside node_modules); scripts download at runtime into ephemeral /tmp.",
  subjects: 50,
  protocol:
    "Read-only inspection (Mission 13): determine whether a leakage-free, session-disjoint " +
    "retrieval benchmark can run THROUGH the actual platform path " +
    "(/api/eeg/embed/foundation -> foundation_embeddings -> match_foundation_embeddings RPC), " +
    "comparing CBraMod 200-D vs V2 32-D vs PCA 32-D. No benchmark executed (see verdict).",
  preprocessing: {
    channel_selection: "selectCbraModChannels (19-ch 10-20 montage, no zero-fill)",
    resample: "160 -> 250 Hz",
    bandpass: [4, 38],
    segment: { window_seconds: 4, overlap: 0.5 },
    zscore: "per channel",
    output_pooling: "mean-tokens -> 200-D L2",
  },
  results: {
    verdict: "INCONCLUSIVE",
    reason:
      "All building blocks exist, but the integrated retrieval layer is missing: (1) no call site " +
      "invokes match_foundation_embeddings (route is write-only); (2) no real pgvector DB in any " +
      "test/CI (Supabase CLI + Docker absent; all tests mock Supabase); (3) 0 .edf files on disk " +
      "(runtime PhysioNet download only); (4) V2-32 path broken in-env (onnxruntime-web WASM backend " +
      "fails -> embedEEG degrades to PCA), so a fair CBraMod-200 vs V2-32 vs PCA-32 comparison " +
      "through the real services cannot be produced here. CBraMod 200-D REPRESENTATION is already " +
      "proven (Mission 11, SUCCESS: Recall@5 +0.312 p=1.66e-59); Mission 13's PLATFORM question is " +
      "deferred pending the smallest next experiment (additive search route + local DB).",
    representation_gate_cited_from_mission: 11,
    representation_gate_decision: "SUCCESS",
    cbramod_200_recall_at_5: 0.5273,
    v2_32_recall_at_5: 0.2158,
    delta_recall_at_5: 0.3116,
    p_bonferroni_vs_v2: 1.663e-59,
    platform_retrieval_executed: false,
    metrics_measured_through_real_rpc: false,
    leakage_prevention_reviewed: true,
    leakage_note:
      "Mission 11's in-process protocol (held-out-run queries vs cross-run pool) is leakage-free; " +
      "no query session was reused in any retrieval-pool construction in Mission 13.",
    no_positive_result_manufactured: true,
  },
  latency_ms: 57.71,
  latency_note:
    "Tier-2 E2E real ONNX forward (onnxruntime-node CPU EP), warm; observed 38.74-57.71 ms across runs. " +
    "This is embedding-generation latency only — RPC/ivfflat retrieval latency is NOT measured " +
    "(no real DB to exercise match_foundation_embeddings).",
  missing_infrastructure: {
    no_tier2_retrieval_call_site: true,
    no_real_pgvector_database: true,
    supabase_cli_installed: false,
    docker_compose_present: false,
    eeg_edf_files_on_disk: 0,
    v2_32_path_operational_in_env: false,
  },
  smallest_next_experiment: {
    step_1: "Add opt-in GET /api/eeg/embed/foundation/search (Tier-2 NeuralVectorIndex); additive, no V2/routing change.",
    step_2: "Local pgvector (supabase start OR Postgres+pgvector + migration 20260814000000_foundation_embeddings.sql) so match_foundation_embeddings is executed.",
    step_3: "Session-disjoint ingestion (S001-S050 runs {5-10}) -> Recall@1/5/10 + cosine descriptors + per-stage latency + bootstrap CIs + paired-t/Bonferroni, CBraMod-200 vs V2-32 vs PCA-32.",
    scope_boundary: "Mission-13 utility validation only — NOT Mission 14 (GA promotion).",
    modification_awaiting_approval: true,
  },
  contaminated: false,
  status: "INCONCLUSIVE",
  report_file: "reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.md",
  report_json: "reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.json",
  benchmark_script: "src/lib/ai/inference/foundation.server.ts (Tier-2 service; retrieval surface proposed in smallest_next_experiment)",
  source_files_inspected: [
    "reports/MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md",
    "reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.md",
    "src/lib/ai/inference/foundation.server.ts",
    "src/routes/api/eeg/embed/foundation.ts",
    "src/lib/vector-search/neural-index.ts",
    "src/lib/vector-search/recall-slo.ts",
    "src/lib/ai/benchmark/validation-metrics.ts",
    "src/lib/evaluation/benchmark.ts",
    "src/lib/evaluation/loso.ts",
    "src/lib/evaluation/model-comparison.ts",
    "supabase/migrations/20260814000000_foundation_embeddings.sql",
    "scripts/tmp/cbramod_cross_session_validation.py",
    "scripts/tmp/cbramod_remap_50subj.py",
    "scripts/t032-embedding-quality.py",
    "src/lib/eeg/loaders/physionet.ts",
    "src/lib/eeg/parsers/edf.ts",
    "src/lib/eeg/channels.ts",
    "src/lib/eeg/preprocessing/resample.ts",
  ],
  constraint_compliance: {
    no_replace_V2: true,
    no_modify_DEFAULT_PREFERRED: true,
    no_modify_embedEEG: true,
    no_modify_embeddings_vector32: true,
    no_remove_or_alter_PCA: true,
    no_make_CBraMod_default: true,
    no_silent_V2_PCA_fallback_in_foundation_path: true,
    no_change_browser_WASM_path: true,
    no_retrain_CBraMod: true,
    no_redesign_before_validation: true,
    mission12_infra_intact: true,
    preserve_all_previous_archive_records_byte_for_byte: true,
    not_starting_mission14: true,
  },
};

archive.experiments.push(entry);

// Byte-untouched assertion: the first `priorLen` experiments + every sibling
// top-level key must be identical to the file on disk before the append.
const rebuilt = JSON.parse(before);
const siblingKeys = [
  "archive_created", "description", "fine_tuning_experiments", "bugs_and_corrections",
  "model_artifacts", "rollout_system", "preserved_artifacts",
];
if (JSON.stringify(rebuilt.experiments) !== JSON.stringify(archive.experiments.slice(0, priorLen))) {
  throw new Error("pre-existing experiments[] changed — aborting; prior records not preserved");
}
for (const k of siblingKeys) {
  if (JSON.stringify(rebuilt[k]) !== JSON.stringify(archive[k])) {
    throw new Error(`sibling key ${k} changed — aborting`);
  }
}

const out = JSON.stringify(archive, null, 2) + "\n";
writeFileSync(archivePath, out, "utf8");
console.log("appended exactly one Mission-13 experiment; total experiments:", archive.experiments.length, "(was", priorLen + ")");
console.log("new id present:", archive.experiments.some((e) => e.id === id));
console.log("prior experiments byte-identical:", JSON.stringify(rebuilt.experiments) === JSON.stringify(archive.experiments.slice(0, priorLen)));
console.log("sibling top-level keys untouched:", siblingKeys.every((k) => JSON.stringify(rebuilt[k]) === JSON.stringify(archive[k])));
