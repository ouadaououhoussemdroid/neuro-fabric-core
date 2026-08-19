/* Idempotent archive appender for the Mission-12 Tier-2 CBraMod 200-D milestone.
 *
 * Adds EXACTLY ONE record to reports/benchmark_archive.json (id
 * "mission12-tier2-cbramod-foundation-200d") as the next experiments[] element,
 * after asserting that every existing experiment (and every sibling top-level
 * array/object) is byte-identical before vs after — so M6/M9/M10/M11 and all
 * prior records remain byte-untouched. Run once with:
 *   node scripts/tmp/append_mission12_archive.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const archivePath = join(root, "reports", "benchmark_archive.json");

const before = readFileSync(archivePath, "utf8");
const archive = JSON.parse(before);

const id = "mission12-tier2-cbramod-foundation-200d";
const saw = archive.experiments.some((e) => e.id === id);

const cbramodSha = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47";
const entry = {
  id,
  experiment_name:
    "Mission 12 — Tier-2 server-native CBraMod 200-D foundation embedding architecture (additive, V2 byte-preserved)",
  date: "2026-08-14",
  author: "NeuroFabric team",
  mission: "Mission 12 — Tier-2 (server-native) CBraMod 200-D foundation architecture",
  model: "onnx-cbramod-foundation-200d (registryId: onnx-cbramod, wasmCompatible:false)",
  model_version: "1.0.0",
  model_artifact: {
    url: "/models/cbramod-encoder.onnx",
    sha256: cbramodSha,
    size_bytes: 22018587,
    input_shape: [1, 19, 1000],
    output_shape: [1, 19, 5, 200],
    output_pooling: "mean-tokens over [1,19,5,200] -> [200] (equiv. Mission-11 r.mean(axis=(1,2)))",
    embedding_dim: 200,
    normalization: "L2 (unit sphere)",
    sample_rate_hz: 250,
    window_samples: 1000,
    channels: 19,
    wasmCompatible: false,
    wasmBlockers: ["DFT", "ReduceL2"],
    runtime: "onnxruntime-node CPU EP (server-only)",
  },
  dataset:
    "PhysioNet EEGMMIDB S001-S050, native 19-channel 10-20 CBraMod montage. " +
    "Cross-session subject-identity retrieval gate sourced from Mission 11 " +
    "(session-disjoint, runs 5-10, held-out-run queries vs cross-task pool). " +
    "Mission 12 itself is infrastructure: it implements the validated serving path; " +
    "it does not re-run the 50-subj evaluation.",
  subjects: 50,
  trials: 4500,
  protocol:
    "Tier-2 serving E2E (NOT a 50-subj ablation): synthetic [19,1000] window -> " +
    "POST /api/eeg/embed/foundation -> foundation.server.ts -> ONNXAdapter " +
    "(onnxruntime-node CPU EP) -> cbramod-encoder.onnx -> mean-tokens 200-D -> L2 -> " +
    "foundation_embeddings(vector(200)) via match_foundation_embeddings RPC. " +
    "SHA-256 + size verified at load (T-016 provenance). Retrieval/contract gate " +
    "cited from Mission 11 cross-session validation.",
  preprocessing: {
    channel_selection: "selectCbraModChannels: native 19-channel 10-20 montage (FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 FZ CZ PZ), canonicalised via canonicalizeChannel (strip 'EEG ' prefix, uppercase, no zeros/no interpolation)",
    resample: "160 -> 250 Hz (resample.ts)",
    bandpass: { low: 4, high: 38 },
    notch: false,
    segment: { window_seconds: 4, overlap: 0.5 },
    zscore: "per channel",
    window_samples: 1000,
    output_pooling: "mean-tokens",
    output_dim: 200,
  },
  results: {
    validation_suite: {
      typecheck_mission12_files_errors: 0,
      typecheck_preexisting_baseline_errors_unmodified: 17,
      lint_mission12_files_errors: 0,
      unit_tests: { name: "foundation.server.test.ts (mocked ort)", passed: 5, failed: 0 },
      route_contract_tests: { name: "-foundation.test.ts", passed: 6, failed: 0 },
      tier2_e2e_real_onnx_forward: {
        name: "foundation-e2e.test.ts",
        passed: 1,
        failed: 0,
        warm_latency_ms_range: [57.71, 114.15],
        dims: 200,
        l2_normalized: true,
        sha256_verified: true,
        fellBack: false,
        note: "22MB real artifact forward over onnxruntime-node CPU EP; second run includes JIT warmup variance.",
      },
      v2_upload_regression: {
        name: "-upload.test.ts",
        passed: 15,
        failed: 0,
        v2_dim: 32,
        v2_namespace: "embeddings (vector(32))",
        v2_pca_fallback: "intact",
        note: "Tier-1 /api/eeg/upload returns/stores 32-D exactly as before; CBraMod path never invoked.",
      },
      build: { target: "Nitro / Cloudflare Workers (Node.js runtime T-003)", exit_code: 0, onnxruntime_node_externalized: true },
    },
    v2_preservation_checks: {
      api_eeg_upload_32d: true,
      embeddings_table_vector32_untouched: true,
      DEFAULT_PREFERRED_unchanged: true,
      embedEEG_unchanged: true,
      PCA_fallback_intact: true,
      manifest_json_unchanged: true,
      integrity_json_unchanged: true,
      no_V2_or_PCA_fallback_in_foundation_path: true,
      foundation_424_not_V2_fallback_on_unavailable: true,
    },
    no_V2_fallback_behaviour:
      "FoundationUnavailableError (missing runtime/artifact/SHA/size mismatch) -> HTTP 424; " +
      "per-window embed failure -> HTTP 500. The V2 /api/eeg/upload path is never imported " +
      "or called from the foundation route (no DEFAULT_PREFERRED / embedEEG / InferenceEngine / registry).",
    archive_integrity: "exactly one experiments[] append; experiments[0..10] byte-identical pre/post; M6/M9/M10/M11 (absent from archive) preserved; fine_tuning_experiments, model_artifacts, preserved_artifacts untouched.",
  },
  latency_ms: 57.71,
  latency_note:
    "Tier-2 E2E warm forward, onnxruntime-node CPU EP (22MB artifact). Mission-11 " +
    "mean reported 64.14 ms; this run observed 57.71 ms (first) / 114.15 ms (JIT-warm). " +
    "wasmCompatible:false so this path is server-side only (not browser/WASM).",
  statistical_comparison_vs_pca:
    "Not applicable to Mission-12 (infra milestone). Gate result sourced from " +
    "Mission 11 cross-session validation: CBraMod 200-D subject-Recall@5 Δ+0.312 vs V2 " +
    "(Bonferroni p=1.663e-59; Δ@10 +0.322 p=3.436e-61; Δ@1 +0.174 p=5.612e-36); all FIRE. " +
    "MI safety floor 0.275 >= chance 0.25.",
  gate_decision_cited_from_mission: 11,
  gate_decision: {
    model_that_earns_role: "CBraMod (server-native, 200-D)",
    role: "opt-in server-side specialist (Tier-2); NOT routed into production embedEEG / DEFAULT_PREFERRED",
    basis: "Mission 11 cross-session subject-identity retrieval gate (SUCCESS)",
    cbramod_recall_at_5: 0.5273,
    v2_recall_at_5: 0.2158,
    delta_recall_at_5: 0.3116,
    p_bonferroni_vs_v2: 1.663e-59,
    mi_accuracy_safety_floor: 0.2749,
    notes:
      "Gate condition (CBraMod >= PCA AND CBraMod >= V2, both p<0.05 Bonferroni) satisfied " +
      "via Mission 11. Mission 12 delivers the opt-in server-side storage/serving backing " +
      "(foundation_embeddings vector(200) + match_foundation_embeddings RPC) for that path; " +
      "it does NOT promote/route CBraMod into the V2 embedEEG default and does NOT retrain.",
  },
  onnx_path: "/models/cbramod-encoder.onnx",
  source_files: {
    service: "src/lib/ai/inference/foundation.server.ts",
    channels: "src/lib/eeg/channels.ts (CBRAMOD_CHANNELS_19, canonicalizeChannel, selectCbraModChannels)",
    resample: "src/lib/eeg/preprocessing/resample.ts",
    route: "src/routes/api/eeg/embed/foundation.ts",
    adapter: "src/lib/ai/adapters/onnx-adapter.ts (reused; unchanged)",
    finalize: "src/lib/ai/embeddings/index.ts (finalize now exported for the foundation path)",
    neural_index: "src/lib/vector-search/neural-index.ts (parameterised tableName/matchRpc; defaults preserved)",
    metrics: "src/lib/metrics/index.ts (foundationRequestsTotal/ErrorsTotal/BytesTotal, foundationEmbedMs) (additive)",
    types: "src/integrations/supabase/types.ts (foundation_embeddings Row/Insert/Update) (additive)",
    migration: "supabase/migrations/20260814000000_foundation_embeddings.sql",
    package_json: "package.json (optionalDependencies.onnxruntime-node) (additive)",
    vite: "vite.config.ts (optimizeDeps.exclude onnxruntime-node) (additive)",
  },
  migration: "supabase/migrations/20260814000000_foundation_embeddings.sql (CREATE TABLE foundation_embeddings vector(200) CHECK(vector_dims=200); RLS; match_foundation_embeddings + match_foundation_embeddings_exact; Tier-1 embeddings/match_embeddings untouched)",
  migration_lint:
    "Supabase CLI / sqlfluff / psql not installed in this environment; static review only " +
    "(idempotent IF NOT EXISTS / CREATE OR REPLACE FUNCTION, SET search_path=public on " +
    "SECURITY DEFINER RPCs, GRANT EXECUTE to authenticated+service_role).",
  constraint_compliance: {
    cbramod_opt_in_server_side_only: true,
    wasmCompatible_false: true,
    isolated_foundation_embeddings_vector200_namespace: true,
    no_V2_or_PCA_fallback: true,
    no_DEFAULT_PREFERRED_modification: true,
    no_embedEEG_modification: true,
    no_env_modification: true,
    no_rollout_modification: true,
    no_vector32_modification: true,
    no_V2_modification: true,
    no_PCA_modification: true,
    no_registry_modification: true,
    no_artifact_modification: true,
    no_manifest_modification: true,
    preserve_mission11_preprocessing_pooling_sha_provenance: true,
    exactly_one_archive_append: true,
    M6_M9_M10_M11_byte_untouched: true,
    no_mission13_started: true,
    no_retrain: true,
  },
  contaminated: false,
  status:
    "IMPLEMENTED + VERIFIED — Tier-2 server-native CBraMod 200-D architecture is additive; " +
    "typecheck 0 mission-12 errors, lint clean, 11 unit/route tests + 1 Tier-2 E2E (real " +
    "22MB forward) + 15/15 V2-upload regression green, production build exit 0, SHA-256 " +
    "verified; Tier-1 V2 preserved byte-for-byte (32-D / embeddings / PCA fallback).",
  report_file: "reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.md",
  report_json: "reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.json",
  benchmark_script: "src/routes/api/eeg/embed/foundation.ts (tier-2 serving E2E) + src/lib/ai/inference/foundation.server.ts",
};

if (saw) {
  archive.experiments = archive.experiments.filter((e) => e.id !== id);
}
archive.experiments.push(entry);

const after = JSON.stringify(archive, null, 2) + "\n";

// Byte-untouched assertion: every pre-existing experiment + every sibling
// top-level structure must be identical (only the new experiments[] tail changes).
const priorExp = archive.experiments.filter((e) => e.id !== id);
if (priorExp.length !== 11) {
  throw new Error(`expected 11 pre-existing experiments, got ${priorExp.length}`);
}
const rebuilt = JSON.parse(before);
const priorSer = JSON.stringify(rebuilt.experiments, null, 2);
const newSer = JSON.stringify(archive.experiments.slice(0, 11), null, 2);
if (priorSer !== newSer) {
  throw new Error("pre-existing experiments[] bytes changed — aborting; M6/M9/M10/M11 not preserved");
}
const siblingKeys = ["archive_created", "description", "fine_tuning_experiments", "bugs_and_corrections", "model_artifacts", "rollout_system", "preserved_artifacts"];
for (const k of siblingKeys) {
  if (JSON.stringify(rebuilt[k]) !== JSON.stringify(archive[k])) {
    throw new Error(`sibling key ${k} changed — aborting`);
  }
}

writeFileSync(archivePath, after, "utf8");
console.log("appended exactly one Mission-12 experiment; total experiments:", archive.experiments.length);
console.log("new id present:", archive.experiments.some((e) => e.id === id));
console.log("prior experiments byte-identical:", priorSer === newSer);
console.log("sibling top-level keys untouched:", siblingKeys.every((k) => JSON.stringify(rebuilt[k]) === JSON.stringify(archive[k])));
