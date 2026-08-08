# EEGConformer — Compatibility Assessment

- Date: 2026-06-17
- Goal: confirm that the planned Track-A artefact drops into the existing
  Neuro-Fabric AI Foundation Layer with **zero** changes to preprocessing,
  vector search, or benchmarking.

## Contract recap

| Field        | Value                   | Source                                       |
| ------------ | ----------------------- | -------------------------------------------- |
| Channels     | 22                      | `registerBraindecodeEEGConformer` defaults   |
| Sample rate  | 250 Hz                  | same                                         |
| Window       | 1000 samples (4 s)      | same                                         |
| Input shape  | `[B, 22, 1000]` float32 | exporter wrapper                             |
| Output name  | `embedding`             | `EEGConformerExportWrapper`                  |
| Output shape | `[B, 32]` float32       | attention-pooled features                    |
| Opset        | 17                      | `scripts/export_braindecode_eegconformer.py` |

## Layer-by-layer compatibility

### Preprocessing (`src/lib/eeg/preprocessing/*`)

| Stage                   | Current behaviour                          | EEGConformer needs          | Status               |
| ----------------------- | ------------------------------------------ | --------------------------- | -------------------- |
| `filters.ts`            | Configurable band-pass                     | 0.5–40 Hz                   | ✅ already supported |
| `artifact-rejection.ts` | Variance / amplitude thresholds            | Same                        | ✅                   |
| `normalize.ts`          | Exponential moving standardisation         | Matches Braindecode default | ✅                   |
| `segment.ts`            | Sliding window, configurable size + stride | 1000 samples, 50 % overlap  | ✅                   |

Verdict: **no code change**. Upstream callers must hand `embedEEG()` a
22-channel @ 250 Hz tensor; the adapter throws a precise error otherwise
(see `BraindecodeAdapter` shape validation).

### Embedding pipeline (`src/lib/ai/embeddings`, `src/lib/ai/inference`)

| Concern        | Current                                       | EEGConformer                            | Status |
| -------------- | --------------------------------------------- | --------------------------------------- | ------ |
| Routing        | `embed()` facade selects adapter by `modelId` | Honours `braindecode-eegconformer-prod` | ✅     |
| Validation     | NaN/Inf/dim/zero check + L2 normalise         | Dim = 32; non-zero by construction      | ✅     |
| Fallback chain | model → ONNX → PCA                            | Already wired in `embed-eeg.ts`         | ✅     |
| SSR safety     | ORT lazy-imported behind capability probe     | Same path                               | ✅     |

Only change needed: flip `DEFAULT_PREFERRED` in `embed-eeg.ts` to
`"braindecode-eegconformer-prod"` (Phase 2 of the roadmap).

### Vector search (`src/lib/vector-search`, `src/lib/ai/vector-bridge`)

- `VectorIndex` is dimension-agnostic and cosine-based; a 32-D vector
  works without changes.
- `NeuralVectorIndex` tags each entry with `modelId`, so PCA (64-D) and
  EEGConformer (32-D) vectors coexist safely — queries filter by
  `modelId` and never cross dimensions. ✅

### Benchmark framework (`src/lib/ai/benchmark`)

- `benchmarkAll(modelIds, input, iters)` iterates over registered ids;
  EEGConformer is already a row in the published benchmark report. ✅
- No assumption on embedding dim. ✅

### Artefact registry (`src/lib/ai/artifacts/index.ts`)

- `braindecode-eegconformer-prod` entry exists; today its `source.kind`
  is `"inline"` placeholder. Only edit: switch to
  `{ kind: "url", url, sha256, bytes }`. ✅

## Cross-cutting checks

| Check                  | Result                                        |
| ---------------------- | --------------------------------------------- |
| Bundle size impact     | 0 — artefact fetched at runtime               |
| New runtime deps       | 0 — `onnxruntime-web` already installed       |
| New env vars / secrets | 0                                             |
| Migration required     | None — vectors are model-tagged               |
| Tests requiring update | None — existing 26-test suite covers the path |

## Open compatibility questions

1. **Browser support of opset 17 attention ops** — verified in
   `onnxruntime-web>=1.18` (WASM + WASM-SIMD). Pin in `package.json`
   to avoid surprise downgrades.
2. **Mobile heap budget** — 19 MB delta is fine on desktop; revisit for
   low-end Android (< 2 GB RAM) where capability probe should prefer
   PCA. Already handled by the fallback chain; document in the rollout
   flag spec.
3. **Future EEGConformer variants** — registry key is suffixed `-prod`;
   variant ids (`-v2`, `-bciiv2a-v1`, etc.) can be registered in
   parallel without collisions.

## Verdict

The Track-A artefact is a **plug-in deployment**: one artefact-registry
edit, one bootstrap call, one default-id flip. Every other layer of the
platform is already EEGConformer-shaped.
