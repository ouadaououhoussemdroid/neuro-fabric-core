# AI Foundation Layer

Migration-safe abstraction layer enabling Neuro-Fabric to evolve from PCA-based
embeddings toward foundation models (ONNX, PyTorch exports, Braindecode, EEGPT)
without breaking existing pipelines.

## Layout

- `models/`      — model registry, capability descriptors, version metadata.
- `embeddings/`  — embedder interface + PCA legacy adapter (default fallback).
- `adapters/`    — runtime adapters: PCA (live, default), ONNX (live via
                   onnxruntime-web), PyTorch export (stub), Braindecode
                   (stub), EEGPT (stub).
- `inference/`   — runtime-agnostic inference engine + routing.

## Contract

All adapters implement `EEGModelAdapter` (see `adapters/types.ts`). The
`InferenceEngine` selects an adapter from the registry by `modelId` and
delegates `embed()` / `predict()` calls. Unimplemented adapters throw
`NotImplementedError` at load time so callers can fall back to PCA.

## Backwards compatibility

- `src/lib/embeddings/*` continues to work unchanged.
- The PCA adapter wraps `embedSignal` from `src/lib/embeddings`; nothing in the
  existing pipeline is removed or rerouted.
- The legacy `src/lib/model-registry` constant remains the source of truth for
  the *active* runtime decoder until adapters are wired into the API surface.

## ONNX adapter

`ONNXAdapter` wraps `onnxruntime-web`'s `InferenceSession`. Construct with an
artifact (URL, `ArrayBuffer`, or `Uint8Array`) and an `inputShape`:

```ts
import { ONNXAdapter } from "@/lib/ai/adapters";
import { registerModel } from "@/lib/ai/models/registry";

registerModel(() => new ONNXAdapter({
  id: "eeg-onnx-embed-v1",
  name: "EEG ONNX Embedder",
  version: "1.0.0",
  description: "Band-power features -> 64-d embedding",
  artifact: "/models/eeg-embed.onnx",
  task: "embedding",
  inputShape: { kind: "features", dim: 95 },
  embeddingDim: 64,
  executionProviders: ["wasm"],
}));
```

Input shapes:
- `{ kind: "features", dim }` — `[1, dim]` from pooled band-power features
  (auto-computed from `signal` or `windows`).
- `{ kind: "raw", channels, samples }` — `[1, C, T]` from the first window.

### Capability detection and PCA fallback

`isONNXRuntimeAvailable()` probes `onnxruntime-web` once and memoises. The
`embed()` facade in `src/lib/ai/embeddings` automatically falls back to the
PCA adapter when the requested adapter is unknown, lacks `embed()`, or
throws during `load()` / `embed()`. The result object carries
`fellBack: boolean` and an optional `fallbackReason`. Pass
`fallbackToPCA: false` to surface the underlying error.

### Tests

Run with `bunx vitest run src/lib/ai`. The ONNX adapter is tested against a
pluggable fake runtime so tests do not require WASM.
