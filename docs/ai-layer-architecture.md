# AI Foundation Layer — Phase 3 (Neural Embeddings)

_Date: 2026-06-17_

## Pipeline

```
EEG signal ─▶ Preprocessing ─▶ Windows ─▶ AI Layer (embed facade)
                                              │
                                              ├─ resolve modelId
                                              ├─ load adapter (LRU cache)
                                              ├─ run inference (ONNX / PCA / …)
                                              ├─ validate (NaN, dim, zero)
                                              ├─ L2-normalise (default on)
                                              └─ return EmbedResult
                                              │
                                              ▼
                                     VectorIndex / NeuralVectorIndex
                                     (cosine similarity, top-k search)
```

PCA remains the **default embedder** (`pca-legacy-v1`) and the **automatic
fallback** whenever a non-PCA adapter fails to load or run. No existing call
site needs to change.

## New modules (`src/lib/ai/*`)

| Module           | Purpose                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| `artifacts/`     | Declarative metadata: id, kind, runtime, input/output schema, source, provenance. |
| `validation/`    | `validateEmbedding`, `l2Normalize`, `isUnitNorm`.                                 |
| `benchmark/`     | `benchmarkAdapter`, `benchmarkAll` — latency p50/p95, dim, heap delta.            |
| `vector-bridge/` | `NeuralVectorIndex` — embed + upsert + cosine search in one call.                 |
| `embeddings/`    | `embed()` facade — observability, validation, normalisation, fallback.            |

## Embedding contract

`embed(input, { modelId?, fallbackToPCA?, normalize?, expectedDim? })` returns

```ts
interface EmbedResult {
  vector: number[]; // L2-normalised by default
  dim: number;
  modelId: string; // actual model used (may be pca-legacy-v1 after fallback)
  durationMs: number;
  fellBack: boolean;
  fallbackReason?: string;
  normalized: boolean;
}
```

Validation runs before normalisation and rejects NaN/Infinity, empty vectors,
all-zero vectors, dim mismatches, and out-of-range values.

## Observability

Every call emits structured JSON log lines:

- `ai.embed.start` — modelId, normalize, fallback
- `ai.embed.load` — adapter load began
- `ai.embed.fail` — adapter threw (warn)
- `ai.embed.fallback` — fallback to PCA triggered (warn)
- `ai.embed.invalid` — validation failure (error)
- `ai.embed.done` — modelId, dim, durationMs, fellBack, normalized

## Vector search integration

The existing `src/lib/vector-search` (brute-force cosine, in-memory) is
unchanged. `NeuralVectorIndex` is a thin wrapper that embeds inputs through
the AI facade before delegating to `VectorIndex`, and stores `{ modelId,
fellBack }` on each entry so neural and PCA vectors can be filtered apart
downstream.

## Migration safety

| Concern                      | Status                                                    |
| ---------------------------- | --------------------------------------------------------- |
| Legacy `embedSignal` callers | Untouched — `src/lib/embeddings` unchanged.               |
| PCA adapter                  | Unchanged. Still the default.                             |
| ONNX adapter / runtime       | Unchanged. Phase-2 capability probe + fallback preserved. |
| Vector search API            | Unchanged. New wrapper added alongside.                   |
| EEG preprocessing            | Unchanged.                                                |
| Database schema              | Unchanged.                                                |
| Tests                        | 18 / 18 passing (was 8).                                  |

No files were removed. No public exports were renamed. Every new export is
additive.

## Benchmarking

`benchmarkAdapter(modelId, input, iterations)` returns
`{ latencyMsMean, latencyMsP50, latencyMsP95, embeddingDim, fellBack,
heapDeltaBytes, error }`. Used by the test suite to verify PCA stays in the
sub-10 ms range and by the inference dashboard to compare adapters
head-to-head.

## Out of scope (next phases)

- EEGPT integration
- Braindecode integration
- pgvector-backed persistent neural store
- Server-side ONNX inference
