# Mission 28: Joint-2312 4-Block Embedding Productionization

## Executive Summary

**Mission:** Productionize the Joint-2312 4-block joint embedding (CBRaMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048 → 2312-D) as a server-side production path: `vector(2312)` store + `/api/eeg/embed/foundation?model=joint-2312` route.

**Verdict: STRONG_SUCCESS** — M27 verified that adding EEGPT-2048 as a 4th fusion block to the M25 Joint-264 pipeline significantly improves retrieval:

```
FINAL SUMMARY
Model                              R@5     Δ vs Joint-264
----------------------------------------------------------------------
Joint-264 (M25, 3 blocks)          0.7858    —
Joint-2312 (M28, 4 blocks)         0.8527   +6.69pp  ← BEST ✓
----------------------------------------------------------------------
Statistical: p=4.80×10⁻²⁸, Cohen's d=0.704 (medium-to-large effect)
Weight stability: CV < 0.5% across 50 LOSO folds
```

The learned block weights from M27 are fixed for production: CBaMod=0.3062, V2=0.1434, PCA=0.1519, EEGPT=0.3985.

## Architecture

```
POST /api/eeg/embed/foundation?model=joint-2312
  │
  ├── parse EDF/CSV/NPY → EEGSignal (64ch raw)
  ├── selectCbraModChannels(19) → resampleSignal(250) → preprocess(bandpass:[4,38]) → 19×1000×N windows
  ├── selectProdChannels(22)    → resampleSignal(250) → preprocess(bandpass:[4,38]) → 22×1000×N windows
  ├── selectEEGPTChannels(62)   → resampleSignal(250) → preprocess(bandpass:[1,40]) → 62×1000×N windows
  │
  ├── embedFoundationWindows(windows19)    → CBaMod-200  [200-D, L2]  (onnxruntime-node)
  ├── embedV2Windows(windows22)            → V2-32       [32-D, L2]   (onnxruntime-node)
  ├── embedPCA(windows22)                  → PCA-32      [32-D, L2]   (JS)
  ├── embedEEGPTWindows(windows62)          → EEGPT-2048  [2048-D, L2] (onnxruntime-node)
  │
  ├── fuse: concat([L2(CBraMod-200), L2(V2-32), L2(PCA-32), L2(EEGPT-2048)]) → 2312-D
  │   block weights: [0.3062, 0.1434, 0.1519, 0.3985] (element-wise within each block)
  │   final: L2-normalize
  │
  └── write to joint_embeddings_2312(vector(2312)) via NeuralVectorIndex
```

## What Changed

### Files Created (3)

1. **`supabase/migrations/20260817000001_joint_embeddings_2312.sql`** — SQL migration creating `joint_embeddings_2312` table (`vector(2312)`), CHECK constraint, ivfflat index, RLS policies, `match_joint_embeddings_2312()` (ANN) and `match_joint_embeddings_2312_exact()` (exact) RPCs — mirroring the M25 `20260817000000_joint_embeddings.sql` pattern.

2. **`src/lib/ai/inference/__tests__/joint-fusion-2312.test.ts`** — 17 unit tests for `fuseJoint2312Embedding`: dimension contract (2312-D), L2 normalization, zero-input handling, block weight energy ratios, dimension mismatch errors, determinism, non-degenerate outputs.

3. **`scripts/tmp/m28_joint_2312_production.py`** — Archive append script that validates M27 results match the archive, then appends the M28 experiment record.

### Files Modified (4)

1. **`src/lib/eeg/channels.ts`** — Added `EEGPT_CHANNELS_62` (exact 62-channel standard 10-20 montage from M26), `EEGPT_CHANNEL_COUNT = 62`, `EEGPT_INTERPOLATED` mapping (PO5→[PO7, PO3], PO6→[PO4, PO8]), and `selectEEGPTChannels(signal)` function. The function selects 62 channels, interpolates PO5/PO6 from spatial neighbors when absent (matching M26's `preprocess_eegpt_trial`), and throws on missing channels (fail loud, never zero-pad).

2. **`src/lib/ai/inference/joint.server.ts`** — Extended with:
   - Constants: `EEGPT_ARTIFACT_ID="eegpt-encoder-int8"`, `EEGPT_EMBEDDING_DIM=2048`, `EEGPT_CHANNELS=62`, `EEGPT_SHA256="a92daf44..."`, `JOINT_2312_MODEL_ID="onnx-cbramod-joint-2312"`, `JOINT_2312_EMBEDDING_DIM=2312`, `JOINT_2312_BLOCK_WEIGHTS={cbramod:0.3062, v2:0.1434, pca:0.1519, eegpt:0.3985}`, `JOINT_2312_COMPONENT_DIMS`
   - `EEGPTUnavailableError` class (extends `FoundationUnavailableError`)
   - `ensureEEGPTAdapter()` — SHA-verified ONNXAdapter loader mirroring `ensureV2Adapter()` pattern
   - `embedEEGPTWindows(windows62: EEGWindow[]): Promise<number[][]>` — embeds 62-channel windows via EEGPT, returns 2048-D L2-normalized
   - `fuseJoint2312Embedding(cbVector, v2Vector, pcaVector, eegptVector): number[]` — 4-block fusion: L2-normalize each block → scale by M27 weights → concat → 2312-D → L2-normalize
   - `embedJoint2312Windows(windows19, windows22, windows62): Promise<EmbedResult[]>` — runs all 4 embedders, fuses per-window
   - `Joint2312Provenance` interface and `joint2312Provenance()` function
   - Updated `resetJointAdapter()` to also reset EEGPT adapter

3. **`src/routes/api/eeg/embed/foundation.ts`** — Added `model=joint-2312` branch that: (1) selects 19-ch/22-ch/62-ch channels, (2) preprocesses with appropriate bandpass, (3) calls `embedJoint2312Windows`, (4) writes to `joint_embeddings_2312` table via `NeuralVectorIndex`, (5) returns provenance via `joint2312Provenance()`.

4. **`src/lib/ai/inference/__tests__/joint-server.test.ts`** — Added 6 M28 E2E test cases for `embedJoint2312Windows` (2312-D output + L2 normalization, non-degenerate, determinism, multiple windows, mismatch error, empty windows error). Moved `makeCBRaModWindow`/`makeV2Window`/`makeEEGPTWindow` to module level (fixing TS2304 scope error). Added `testTimeout: 60000` for EEGPT-2048 inference latency.

## Verification

### TypeScript Compilation
- `npx tsc --noEmit` — **CLEAN** for all M28 files (`channels.ts`, `joint.server.ts`, `foundation.ts`, both test files)
- Pre-existing errors in `model-comparison.ts`, `test-harness.ts`, and `*`-live test files are unrelated (Supabase type-gen gaps from prior missions)

### Unit Tests (`joint-fusion-2312.test.ts`)
```
17 tests passed in 124ms
✓ constants: JOINT_2312_EMBEDDING_DIM=2312, block weights sum to 1.0, weights match M27 values
✓ fuseJoint2312Embedding: 2312-D output, L2 norm, zero-input handling, dim mismatch errors
✓ block weight energy ratios verified (CBRaMod/EEGPT)
✓ determinism (byte-identical + cos=1.0)
✓ non-degenerate random inputs produce valid L2-normalized output
```

### E2E Tests (`joint-server.test.ts`)
```
12 tests passed (6 M25 + 6 M28), 0 skipped, 24.73s
M28 E2E (6 tests):
✓ embeds aligned 19/22/62-channel windows into 2312-D L2-normalised vector  5889ms
✓ produces non-degenerate embeddings (not all-zero)                          4503ms
✓ is deterministic — cos(runA, runB) ≈ 1.0                                   8158ms
✓ embeds multiple windows independently (one result per window)              6108ms
✓ throws on 3-way window count mismatch
✓ throws on empty windows
```

### Regression Check
- All 122 EEG library tests pass (no regression in channels, preprocessing, loaders, parsers)
- M25 Joint-264 E2E tests pass (6/6)
- M25 `joint-fusion.test.ts` passes (15/15)

### Artifact SHA Verification
| Model | SHA-256 | Status |
|-------|---------|--------|
| CBaMod-200 | `c128ccfd…` | ✅ Verified |
| V2-32 (EEGConformer) | `18644de1…` | ✅ Verified |
| EEGPT-2048 (INT8) | `a92daf44…` | ✅ Verified |

## Block Weights (M27 Learned, Productionized)

| Block | Weight | Dimension | Full Precision (M27) |
|-------|--------|-----------|---------------------|
| CBaMod-200 | 0.3062 | 200-D | 0.3061501818 |
| V2-32 | 0.1434 | 32-D | 0.1434483265 |
| PCA-32 | 0.1519 | 32-D | 0.1518943262 |
| **EEGPT-2048** | **0.3985** | **2048-D** | **0.3985071656** |
| **Total** | **1.0000** | **2312-D** | |

Weight stability: CV < 0.5% across all 50 LOSO folds. EEGPT dominates the fusion (39.85%), reflecting its strong individual retrieval performance (R@5=0.8118 from M27).

## Constraints Honored

- ✅ No model retraining
- ✅ No artifact modification (all SHAs unchanged)
- ✅ No ONNX modification
- ✅ No `DEFAULT_PREFERRED` / rollout / `.env` changes (V2 remains GA default)
- ✅ V2 artifact SHA `18644de1…` unchanged
- ✅ CBaMod artifact SHA `c128ccfd…` unchanged
- ✅ `embeddings` table (vector(32)) untouched
- ✅ `foundation_embeddings` table untouched
- ✅ `joint_embeddings` table (vector(264), M25) untouched
- ✅ Block weights from M27 (not relearned)
- ✅ `.server.ts` suffix ensures `onnxruntime-node` never bundled for browser

## Success Criteria Checklist

- ✅ `/api/eeg/embed/foundation?model=joint-2312` produces valid 2312-D L2-normalised embedding
- ✅ SHA-256 verification passes for CBaMod (c128ccfd), V2 (18644de1), and EEGPT (a92daf44)
- ✅ Determinism: cos(runA, runB) = 1.0 (verified in unit + Tier-2 E2E tests)
- ✅ `vector(2312)` index searchable via `match_joint_embeddings_2312` / `match_joint_embeddings_2312_exact` RPCs
- ✅ No regression in existing foundation (200-D), V2 (32-D), or Joint-264 (264-D) paths
- ✅ Lint + typecheck clean for new M28 files
- ✅ Unit test (17 tests) + Tier-2 E2E test (6 tests) pass
- ✅ `benchmark_archive.json` appended with M28 record
- ✅ M28 report written

## Files

| Type | Path |
|------|------|
| Server module | `src/lib/ai/inference/joint.server.ts` |
| Route | `src/routes/api/eeg/embed/foundation.ts` |
| Channels | `src/lib/eeg/channels.ts` |
| Migration | `supabase/migrations/20260817000001_joint_embeddings_2312.sql` |
| Unit test | `src/lib/ai/inference/__tests__/joint-fusion-2312.test.ts` |
| E2E test | `src/lib/ai/inference/__tests__/joint-server.test.ts` |
| Archive script | `scripts/tmp/m28_joint_2312_production.py` |
| Archive | `reports/benchmark_archive.json` |
| Report | `reports/MISSION28_JOINT_2312_PRODUCTION_REPORT.md` |
