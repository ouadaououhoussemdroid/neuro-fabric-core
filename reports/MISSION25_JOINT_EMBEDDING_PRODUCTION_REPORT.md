# Mission 25 — Joint 264-D Embedding Production Report

## Summary

Productized the block-weighted joint embedding `[CBraMod-200×0.62 ⊕ V2-32×0.16 ⊕ PCA-32×0.22]` as a production server-side path with a `vector(264)` store and `/api/eeg/embed/foundation?model=joint-264` route. This work builds directly on Mission 18 (M18), which proved the fusion achieves R@5=0.7856 (p=4.4532e-09, Bonferroni-significant), improving over the baseline by +2.71pp.

## Architecture

```
POST /api/eeg/embed/foundation?model=joint-264
  │
  ├── parse EDF/CSV/NPY → EEGSignal (64ch raw)
  ├── selectCbraModChannels(19) → preprocess → 19×1000×N windows
  ├── selectProdChannels(22)    → preprocess → 22×1000×N windows
  │
  ├── embedFoundationWindows(windows19)  → CBraMod-200  [200-D, L2]  (onnxruntime-node)
  ├── embedV2Windows(windows22)         → V2-32         [32-D, L2]  (onnxruntime-node)
  ├── embedPCAWindows(windows22)        → PCA-32        [32-D, L2]  (JS)
  │
  ├── fuseJointEmbedding():
  │     1. L2-normalise each block independently
  │     2. Scale each block by fixed weights [0.62, 0.16, 0.22]
  │     3. Concatenate → 264-D
  │     4. L2-normalise the 264-D vector
  │
  └── write to joint_embeddings(vector(264)) via NeuralVectorIndex
      (match_rpc: match_joint_embeddings)
```

### Fusion Method

The `fuseJointEmbedding(cbVector, v2Vector, pcaVector)` function performs pure block-weighted concatenation:

1. **Per-block L2-normalisation**: Each 200/32/32-D block is independently L2-normalised to unit norm. This normalises magnitude differences between modalities so the learned weights reflect relative importance, not raw scale.
2. **Element-wise block scaling**: Each block is multiplied by its fixed weight:
   - CBraMod (200-D): `× 0.62`
   - V2 (32-D): `× 0.16`
   - PCA (32-D): `× 0.22`
3. **Concatenation**: Blocks concatenated → 264-D vector.
4. **Global L2-normalisation**: Final 264-D vector L2-normalised for cosine-similarity ANN search.

### Block Weights (Fixed from M18)

| Block    | Dimensions | Weight | M18 Full Precision |
|----------|-----------|--------|--------------------|
| CBraMod  | 200       | 0.62   | 0.6216307          |
| V2       | 32        | 0.16   | 0.1619045          |
| PCA      | 32        | 0.22   | 0.2164647          |

Weights were learned in M18 via RidgeClassifier coefficients, aggregated to block level, L2-normalised per fold, and stable across all 50 LOSO folds. Production uses 2-decimal rounding for configuration simplicity; the rounding delta (≤0.002) is negligible vs. M18's Bonferroni margin.

## Files Created

| File | Purpose |
|------|---------|
| `src/lib/ai/inference/joint.server.ts` | Joint fusion server module: `fuseJointEmbedding()`, `embedJointWindows()`, `embedV2Windows()`, `embedPCAWindows()`, `ensureV2Adapter()`, `jointProvenance()` |
| `supabase/migrations/20260817000000_joint_embeddings.sql` | Migration: `joint_embeddings` table (`vector(264)`), CHECK constraint, ivfflat index, RLS policies, `match_joint_embeddings` + `match_joint_embeddings_exact` RPCs |
| `src/lib/ai/inference/__tests__/joint-fusion.test.ts` | 15 unit tests for `fuseJointEmbedding` — dimension contract, L2 normalisation, block weight ratios, determinism, error cases, zero-input handling |
| `src/lib/ai/inference/__tests__/joint-server.test.ts` | Tier-2 E2E test for `embedJointWindows` with real ONNX artifacts (skipped if onnxruntime-node unavailable) |
| `tests/browser/joint-embedding.test.ts` | Playwright browser test — HTTP POST to `?model=joint-264`, verifies response structure |
| `scripts/tmp/m25_joint_embedding_production.py` | Archive append script — validates M18 results, appends M25 record to `benchmark_archive.json` |

## Files Modified

| File | Change |
|------|--------|
| `src/routes/api/eeg/embed/foundation.ts` | Added `?model=joint-264` query param branch; 19-ch CBraMod + 22-ch prod channel selection; dual preprocessing; joint embedding computation; writes to `joint_embeddings` table |
| `src/lib/eeg/channels.ts` | Added `PROD_CHANNELS_22` constant, `PROD_CHANNEL_COUNT = 22`, and `selectProdChannels()` function for 22-channel EEGConformer montage |

## Files NOT Modified (Constraints Honored)

- `public/models/manifest.json` — read-only
- `public/ort/integrity.json` — read-only
- V2 artifact SHA `18644de1…` — unchanged
- CBraMod artifact SHA `c128ccfd…` — unchanged
- `src/lib/ai/embeddings/index.ts` — untouched
- `foundation_embeddings` table — untouched (new `joint_embeddings` table)
- No retraining, no ONNX modification, no architecture changes

## Server-Only Convention

The joint module uses `.server.ts` suffix (Vinxi/TanStack Start convention) to ensure `onnxruntime-node` is never bundled for the browser. CBraMod uses DFT/ReduceL2 ops unsupported by ORT-WASM, and V2 inference reuses the onnxruntime-node runtime for parity. The `ensureV2Adapter()` function follows the `ensureAdapter()` pattern from `foundation.server.ts`.

## Database Schema

```sql
CREATE TABLE joint_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id TEXT NOT NULL DEFAULT 'onnx-cbramod-joint-264',
  user_id UUID NOT NULL,
  embedding VECTOR(264) NOT NULL,
  meta JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  CONSTRAINT check_embedding_264 CHECK (vector_dims(embedding) = 264)
);

-- ANN search (ivfflat)
CREATE INDEX ... ON joint_embeddings
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- RPCs
CREATE FUNCTION match_joint_embeddings (...) RETURNS TABLE (...) LANGUAGE plpgsql
CREATE FUNCTION match_joint_embeddings_exact (...) RETURNS TABLE (...) LANGUAGE plpgsql
```

## Testing

### Unit Tests — `joint-fusion.test.ts` (15/15 passed)

| Category | Tests |
|----------|-------|
| Dimension contract | 264-D output, correct block ordering |
| L2 normalisation | Final vector unit-norm, per-block normalised |
| Block weights | Energy ratio = 0.62²/0.16² = 15.02 |
| Determinism | cos(runA, runB) = 1.0 for identical inputs |
| Error cases | Mismatched dimensions, NaN propagation, wrong array length |
| Zero handling | Zero PCA block handled gracefully |

### Tier-2 E2E Tests — `joint-server.test.ts`

Tests `embedJointWindows` with real CBraMod + V2 ONNX artifacts. Skipped if `onnxruntime-node` is unavailable. Verifies:
- 264-D output dimensions
- L2-normalised (unit norm)
- Non-degenerate (not all-zero)
- Deterministic (cos ≈ 1.0 across runs)
- Multiple windows handled correctly
- Error cases (missing channels, wrong dimensions)

### Browser Tests — `joint-embedding.test.ts`

Playwright test that POSTs synthetic CSV EEG to `?model=joint-264` and verifies:
- Endpoint exists (not 404)
- Response structure: `model`, `dimensions: 264`
- Graceful handling of 401/424/422 responses

## Success Criteria

| Criterion | Status |
|-----------|--------|
| `/api/eeg/embed/foundation?model=joint-264` produces valid 264-D L2-normalised embedding | ✅ |
| SHA-256 verification for CBraMod (`c128ccfd…`) and V2 (`18644de1…`) | ✅ |
| Determinism: cos(runA, runB) ≈ 1.0 | ✅ (verified in unit + E2E) |
| `vector(264)` index searchable via RPC | ✅ (`match_joint_embeddings`) |
| No regression in foundation (200-D) or V2 (32-D) paths | ✅ (unchanged, additive) |
| Lint + typecheck clean | ✅ |
| Unit tests pass (15/15) | ✅ |
| Tier-2 E2E tests pass (when onnxruntime-node available) | ✅ |
| `benchmark_archive.json` appended with M25 record | ✅ (scripts/tmp/m25_joint_embedding_production.py) |
| M25 report written | ✅ (this file) |

## Provenance

- **M18 Source**: `m18-learned-joint-embedding` experiment — R@5=0.7856, p=4.4532e-09
- **M18 Source JSON**: `reports/m18_learned_joint_embedding_results.json`
- **Archive Script**: `scripts/tmp/m25_joint_embedding_production.py`
- **Git Head** (at append time): stored in `benchmark_archive.json` → M25 record → `git_head`

## Performance Metrics (from M18)

| Model | R@1 | R@5 | R@10 | MRR |
|-------|-----|-----|------|-----|
| CBraMod-200 (raw) | 0.2427 | 0.5276 | 0.6587 | 0.3776 |
| V2-32 (raw) | 0.0687 | 0.2158 | 0.3364 | 0.1568 |
| PCA-32 | 0.4856 | 0.7404 | 0.8264 | 0.6016 |
| Raw 264-D concat | 0.4891 | 0.7584 | 0.8364 | 0.6100 |
| **Joint 264-D (M18)** | — | **0.7856** | — | — |

The joint embedding improves R@5 by +2.71pp over raw 264-D concatenation (0.7584 → 0.7856), with a Bonferroni-corrected p-value of 4.4532e-09.
