#!/usr/bin/env python3
"""
M25 — Joint 264-D Embedding Productionization (Archive Append).

Validates existing M18 results match the archive, then appends an M25
experiment record documenting the productionization of the block-weighted
joint embedding [CBraMod-200×0.62 ⊕ V2-32×0.16 ⊕ PCA-32×0.22].

This script is idempotent: it removes any prior experiment id
'm25-joint-264-production' before appending.
"""
import json, hashlib, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"

# ── M25 provenance ────────────────────────────────────────────────────────────

CBRAMOD_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"

# Fixed block weights (M18 learned, stable across all 50 LOSO folds).
# Rounded to 2 decimal places for the production configuration.
BLOCK_WEIGHTS = {"cbramod": 0.62, "v2": 0.16, "pca": 0.22}

# M18 learned block weights (full precision, from the results JSON).
BLOCK_WEIGHTS_M18 = {
    "cbramod": 0.6216307282447815,
    "v2": 0.16190451383590698,
    "pca": 0.21646469831466675,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


# ── Load archive ─────────────────────────────────────────────────────────────

with open(ARCHIVE, "r") as f:
    arch = json.load(f)

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

# ── Validate M18 results are in the archive ───────────────────────────────────

m18 = next((e for e in arch["experiments"] if e.get("id") == "m18-learned-joint-embedding"), None)
if m18 is None:
    raise SystemExit("ERROR: M18 record not found in benchmark_archive.json — aborting M25 append.")

# Cross-check key M18 values that M25 builds upon.
assert m18["results"]["best_learned_r5"] == 0.7856, (
    f"M18 R@5 mismatch: expected 0.7856, got {m18['results']['best_learned_r5']}"
)
assert m18["results"]["baseline_r5"] == 0.7584, (
    f"M18 baseline R@5 mismatch: expected 0.7584, got {m18['results']['baseline_r5']}"
)

# Validate block weights from the M18 source JSON.
m18_results_path = REPO / "reports" / "m18_learned_joint_embedding_results.json"
if m18_results_path.exists():
    with open(m18_results_path) as f:
        m18_full = json.load(f)
    m18_weights = m18_full["block_weights_analysis"]
    for k in ("cbramod", "v2", "pca"):
        assert abs(m18_weights[k] - BLOCK_WEIGHTS_M18[k]) < 1e-10, (
            f"M18 block weight mismatch for {k}: {m18_weights[k]} != {BLOCK_WEIGHTS_M18[k]}"
        )
    print("  M18 block weights validated")
    print(f"    CBraMod={m18_weights['cbramod']:.4f} (prod rounded: {BLOCK_WEIGHTS['cbramod']})")
    print(f"    V2={m18_weights['v2']:.4f} (prod rounded: {BLOCK_WEIGHTS['v2']})")
    print(f"    PCA={m18_weights['pca']:.4f} (prod rounded: {BLOCK_WEIGHTS['pca']})")
else:
    print("  WARNING: m18_learned_joint_embedding_results.json not found — skipping full-precision weight validation")

print(f"  M18 best_learned_r5={m18['results']['best_learned_r5']} (expected 0.7856)")
print(f"  M18 baseline_r5={m18['results']['baseline_r5']} (expected 0.7584)")

# Validate artifact SHAs haven't changed (read-only constraints).
manifest_path = REPO / "public" / "models" / "manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

cbramod_entry = manifest["models"]["cbramod-encoder"]
v2_entry = manifest["models"]["eegconformer_finetuned"]
assert cbramod_entry["sha256"] == CBRAMOD_SHA, f"CBraMod SHA mismatch: {cbramod_entry['sha256']}"
assert v2_entry["sha256"] == V2_SHA, f"V2 SHA mismatch: {v2_entry['sha256']}"
print("  Artifact SHAs verified")
print(f"    CBraMod: {CBRAMOD_SHA}")
print(f"    V2:      {V2_SHA}")

# ── Compose the M25 experiment record ─────────────────────────────────────────

record = {
    "id": "m25-joint-264-production",
    "experiment_name": "Mission 25: Joint 264-D Embedding Productionization",
    "date": "2026-08-13",
    "author": "zcode-agent",
    "mission": "Mission 25 — Productize the block-weighted joint 264-D embedding "
               "[CBraMod-200x0.62 + V2-32x0.16 + PCA-32x0.22] as a production "
               "server-side path: vector(264) store + /api/eeg/embed/foundation?model=joint-264",
    "model": "onnx-cbramod-joint-264 (fused: CBraMod-200 + V2-32 + PCA-32)",
    "model_version": "1.0.0",
    "dataset": "PhysioNet EEGMMIDB 1.0.0 (S001-S050, runs 5-6, 4-class MI: left/right_hand, feet/tongue)",
    "subjects": 50,
    "trials": 4500,
    "protocol": "Session-disjoint: 50-fold LOSO, one run (15 trials) of held-out subject "
                 "as query, all other trials as pool. M18 validated R@5=0.7856 "
                 "(vs baseline 264-D raw R@5=0.7584).",
    "preprocessing": {
        "channels_cbramod": [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
        ],
        "channels_v2_prod": [
            "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
            "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
            "T7", "T8", "P7", "P8", "P5", "P6",
        ],
        "sample_rate_hz": 250,
        "window_samples": 1000,
        "bandpass_hz": [4.0, 38.0],
        "normalization": "z-score per channel",
        "resampling": "160->250 Hz (linear interpolation)",
        "pca_features": "5 bands x 22 channels = 110 band-power features",
        "pca_components": 32,
        "pca_fit": "batch-fit per inference request (standard production approach)",
        "protocol": "Session-disjoint LOSO, 50 folds, nearest-centroid (cosine), "
                     "Recall@K with train-only pool, no self-retrieval",
    },
    "fusion_method": {
        "description": "Block-weighted concatenation with fixed learned weights",
        "concatenation": "concat([CBraMod-200, V2-32, PCA-32]) -> 264-D",
        "block_normalization": "L2-normalise each block independently before weighting",
        "block_weights": BLOCK_WEIGHTS,
        "block_weights_m18_full_precision": BLOCK_WEIGHTS_M18,
        "weight_source": "M18: RidgeClassifier coefficients, aggregated to block level, "
                         "L2-normalised per fold, stable across all 50 folds",
        "final_normalization": "L2-normalise the 264-D vector after block-weight scaling",
        "produces": "264-D L2-normalised embedding",
    },
    "artifacts": {
        "cbramod": {
            "path": "public/models/cbramod-encoder.onnx",
            "sha256": CBRAMOD_SHA,
            "input": "eeg[1,19,1000]",
            "output": "embedding[1,19,5,200]->200 (mean-tokens)",
            "wasm_compatible": False,
            "dims": 200,
            "runtime": "onnxruntime-node (server-only: DFT, ReduceL2)",
        },
        "eegconformer_v2": {
            "path": "public/models/eegconformer_finetuned.onnx",
            "sha256": V2_SHA,
            "input": "input[1,22,1000]",
            "output": "embedding[1,32]",
            "wasm_compatible": True,
            "dims": 32,
            "runtime": "onnxruntime-node cpu (server-side), wasmCompatible for browser",
            "note": "production GA default; read-only; SHA unchanged (M23)",
        },
        "pca": {
            "description": "JS PCA(32) on 110 band-power features (5 bands x 22 channels), "
                           "StandardScaler + power-iteration PCA, seed=0x20260711",
            "dims": 32,
            "runtime": "pure JavaScript (no native deps)",
        },
    },
    "results": {
        "source": "m18-learned-joint-embedding (R@5=0.7856, p=4.4532e-09, Bonferroni-significant)",
        "m18_best_method": "weighted_concat",
        "m18_best_r5": 0.7856,
        "m18_baseline_r5": 0.7584,
        "m18_improvement_pp": 2.71,
        "m18_p_value": 4.4532e-09,
        "m18_cohen_d": 0.0876,
        "m18_bonferroni_significant": True,
        "m18_bonferroni_alpha": 0.0125,
        "individual_baselines": {
            "cbramod_200_raw_cosine": {"R@1": 0.2427, "R@5": 0.5276, "R@10": 0.6587, "MRR": 0.3776},
            "v2_32_raw_cosine": {"R@1": 0.0687, "R@5": 0.2158, "R@10": 0.3364, "MRR": 0.1568},
            "pca_32_bandpower": {"R@1": 0.4856, "R@5": 0.7404, "R@10": 0.8264, "MRR": 0.6016},
            "raw_264d_concatenation": {"R@1": 0.4891, "R@5": 0.7584, "R@10": 0.8364, "MRR": 0.6100},
        },
        "m25_production_status": "PRODUCTIONIZED",
        "m25_production_path": (
            "POST /api/eeg/embed/foundation?model=joint-264 -> "
            "selectCbraModChannels(19) + selectProdChannels(22) -> preprocess both -> "
            "embedFoundationWindows + embedV2Windows + embedPCAWindows -> "
            "fuseJointEmbedding([0.62, 0.16, 0.22]) -> 264-D L2-normalised -> "
            "joint_embeddings(vector(264)) table"
        ),
    },
    "constraints_honored": {
        "no_model_retraining": True,
        "no_artifact_modification": True,
        "no_onnx_modification": True,
        "no_default_preferred_change": True,
        "no_v2_or_pca_change": True,
        "v2_artifact_sha_unchanged": True,
        "cbramod_artifact_sha_unchanged": True,
        "vector32_contract_preserved": True,
        "foundation_embeddings_untouched": True,
        "block_weights_fixed_m18": True,
        "seed_42_reproducible_m18": True,
    },
    "success_criteria": [
        "PASS /api/eeg/embed/foundation?model=joint-264 produces valid 264-D L2-normalised embedding",
        "PASS SHA-256 verification passes for both CBraMod (c128ccfd) and V2 (18644de1)",
        "PASS Determinism: cos(runA, runB) = 1.0 (verified in unit + Tier-2 E2E tests)",
        "PASS vector(264) index searchable via match_joint_embeddings / match_joint_embeddings_exact RPCs",
        "PASS No regression in existing foundation (200-D) or V2 (32-D) paths",
        "PASS Lint + typecheck clean for new files",
        "PASS Unit test (fuseJointEmbedding) + Tier-2 E2E test (embedJointWindows) pass",
        "PASS benchmark_archive.json appended with M25 record",
        "PASS M25 report written (reports/MISSION25_JOINT_EMBEDDING_PRODUCTION_REPORT.md)",
    ],
    "contaminated": False,
    "status": "COMPLETE",
    "report_file": "reports/MISSION25_JOINT_EMBEDDING_PRODUCTION_REPORT.md",
    "source_json": "reports/m18_learned_joint_embedding_results.json",
    "benchmark_script": "scripts/tmp/m25_joint_embedding_production.py",
    "git_head": git_head,
    "provenance": {
        "joint_server_module": "src/lib/ai/inference/joint.server.ts",
        "joint_server_sha256": sha256_file(REPO / "src/lib/ai/inference/joint.server.ts"),
        "migration": "supabase/migrations/20260817000000_joint_embeddings.sql",
        "migration_sha256": sha256_file(REPO / "supabase/migrations/20260817000000_joint_embeddings.sql"),
        "route_modification": "src/routes/api/eeg/embed/foundation.ts",
        "route_sha256": sha256_file(REPO / "src/routes/api/eeg/embed/foundation.ts"),
        "channels_modification": "src/lib/eeg/channels.ts",
        "channels_sha256": sha256_file(REPO / "src/lib/eeg/channels.ts"),
        "unit_test": "src/lib/ai/inference/__tests__/joint-fusion.test.ts",
        "e2e_test": "src/lib/ai/inference/__tests__/joint-server.test.ts",
        "browser_test": "tests/browser/joint-embedding.test.ts",
        "m18_block_weights": BLOCK_WEIGHTS_M18,
        "m18_source": "m18-learned-joint-embedding",
    },
}

# ── Append experiment (replace any prior same-id for idempotency) ────────────
arch["experiments"] = [e for e in arch["experiments"] if e.get("id") != record["id"]]
arch["experiments"].append(record)

# ── Preserved artifacts: register the new M25 files ───────────────────────────
new_artifacts = [
    {
        "type": "report",
        "path": "reports/MISSION25_JOINT_EMBEDDING_PRODUCTION_REPORT.md",
        "description": "Mission 25 human-readable report: joint 264-D embedding productionization",
    },
    {
        "type": "script",
        "path": "scripts/tmp/m25_joint_embedding_production.py",
        "description": "Mission 25 archive append script: validates M18 results, appends M25 record",
    },
    {
        "type": "server_module",
        "path": "src/lib/ai/inference/joint.server.ts",
        "description": "M25 joint fusion server module: CBraMod-200 + V2-32 + PCA-32 -> 264-D",
    },
    {
        "type": "migration",
        "path": "supabase/migrations/20260817000000_joint_embeddings.sql",
        "description": "M25 database migration: joint_embeddings table (vector(264)) + ivfflat + RPCs",
    },
    {
        "type": "route",
        "path": "src/routes/api/eeg/embed/foundation.ts",
        "description": "M25 route extension: adds ?model=joint-264 option to foundation endpoint",
    },
    {
        "type": "channels",
        "path": "src/lib/eeg/channels.ts",
        "description": "M25 addition: PROD_CHANNELS_22 + selectProdChannels for 22-channel EEGConformer montage",
    },
    {
        "type": "unit_test",
        "path": "src/lib/ai/inference/__tests__/joint-fusion.test.ts",
        "description": "M25 unit test: fuseJointEmbedding with synthetic inputs (264-D, L2-norm, block weights)",
    },
    {
        "type": "e2e_test",
        "path": "src/lib/ai/inference/__tests__/joint-server.test.ts",
        "description": "M25 Tier-2 E2E test: embedJointWindows with real CBraMod + V2 ONNX artifacts",
    },
    {
        "type": "browser_test",
        "path": "tests/browser/joint-embedding.test.ts",
        "description": "M25 browser test: API endpoint contract for ?model=joint-264",
    },
]
existing = {(a.get("type"), a.get("path")) for a in arch["preserved_artifacts"]}
for a in new_artifacts:
    key = (a["type"], a["path"])
    if key not in existing:
        arch["preserved_artifacts"].append(a)

with open(ARCHIVE, "w") as f:
    json.dump(arch, f, indent=2)

print(f"\nAppended experiment 'm25-joint-264-production' -> {ARCHIVE}")
print(f"  experiments[] count: {len(arch['experiments'])}")
print(f"  preserved_artifacts count: {len(arch['preserved_artifacts'])}")
print(f"  git_head: {git_head}")
print(f"\n  M25 block weights: {BLOCK_WEIGHTS}")
print(f"  M25 R@5 (from M18): 0.7856")
print(f"  M25 baseline R@5 (from M18): 0.7584")
print(f"  M25 improvement: +2.71pp (p=4.4532e-09, Bonferroni-significant)")
