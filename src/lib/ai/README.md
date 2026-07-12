# AI Foundation Layer

Migration-safe abstraction layer enabling Neuro-Fabric to evolve from PCA-based
embeddings toward foundation models (ONNX, PyTorch exports, Braindecode, EEGPT)
without breaking existing pipelines.

## Current state (post-T-028)

| Adapter                                   | Status                  | Description                                                                                        |
| ----------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| **PCA**                                   | Live (default fallback) | `PCAEmbeddingAdapter` wraps `embedSignal` from `src/lib/embeddings`                                |
| **ONNX**                                  | Live                    | `ONNXAdapter` wraps `onnxruntime-web`'s `InferenceSession` with self-hosted WASM                   |
| **Braindecode (ONNX)**                    | Live (production)       | `braindecode-eegconformer-prod` — real EEGConformer ONNX artefact via `braindecode-onnx-bridge.ts` |
| **Braindecode (Pyodide)**                 | Scaffold                | `BraindecodeAdapter` with pluggable bridge; default bridge returns `isAvailable() === false`       |
| **EEGNetv4 / ShallowFBCSPNet / Deep4Net** | Registered (untrained)  | In the model zoo for ablation; no trained ONNX artefacts shipped                                   |
| **EEGPT**                                 | Scheduled (stub)        | `implemented: false`, throws `NotImplementedError`. Blocked on license-clear weights (see T-016)   |
| **PyTorch export**                        | Stub                    | `implemented: false`, throws `NotImplementedError`                                                 |

The production embedding path is: `braindecode-eegconformer-prod` (ONNX) →
generic ONNX → `pca-legacy-v1` (terminal fallback). Fallbacks are never silent
— the `embed()` facade emits `EMBED_FALLBACK_EVENT`, logs `ai.embed.fallback.loud`,
and sets `fellBack: true` on the result.

## Layout

- `models/` — model registry, capability descriptors, version metadata, MLflow sync.
- `embeddings/` — embedder interface + PCA legacy adapter (default fallback).
- `adapters/` — runtime adapters (see table above) + WebGPU EP feature flag.
- `inference/` — runtime-agnostic inference engine + routing.
- `artefacts/` — content-hashed ONNX artefact loader (SHA-256 manifest, verify-at-load).
- `benchmark/` — latency harness + embedding validation metrics (recall@k, cosine).
- `explainability/` — Captum saliency sidecar loader (T-018).
- `validation/` — `validateEmbedding` (rejects NaN/Inf/zero/dim-mismatch), L2 normalization.
- `vector-bridge/` — `NeuralVectorIndex` embeds via the AI facade and upserts into the vector store.

## Contract

All adapters implement `EEGModelAdapter` (see `adapters/types.ts`). The
`InferenceEngine` selects an adapter from the registry by `modelId` and
delegates `embed()` / `predict()` calls. Unimplemented adapters throw
`NotImplementedError` at load time so callers can fall back to PCA.

## ONNX adapter

`ONNXAdapter` wraps `onnxruntime-web`'s `InferenceSession`. Construct with an
artifact (URL, `ArrayBuffer`, or `Uint8Array`) and an `inputShape`:

```ts
import { ONNXAdapter } from "@/lib/ai/adapters";
import { registerModel } from "@/lib/ai/models/registry";

registerModel(
  () =>
    new ONNXAdapter({
      id: "eeg-onnx-embed-v1",
      name: "EEG ONNX Embedder",
      version: "1.0.0",
      description: "Band-power features -> 64-d embedding",
      artifact: "/models/eeg-embed.onnx",
      task: "embedding",
      inputShape: { kind: "features", dim: 95 },
      embeddingDim: 64,
      executionProviders: ["wasm"],
    }),
);
```

Input shapes:

- `{ kind: "features", dim }` — `[1, dim]` from pooled band-power features
  (auto-computed from `signal` or `windows`).
- `{ kind: "raw", channels, samples }` — `[1, C, T]` from the first window.

### Self-hosted ORT WASM (T-008)

The ONNX Runtime WASM bundle is self-hosted under `/ort/` via the
`ortWasmSelfHostPlugin` Vite plugin (see `vite-plugins/ort-wasm-self-host.ts`).
The plugin copies `ort-wasm-simd-threaded.{wasm,mjs}` from `node_modules` at
build time and generates `integrity.json` with SHA-384 hashes. The adapter
sets `ort.env.wasm.wasmPaths = "/ort/"` — no jsdelivr CDN dependency.

### WebGPU EP (T-024)

The WebGPU execution provider is opt-in via the `webgpu-flag.ts` module. It
defaults to WASM-only and can be enabled at runtime via `setWebGPUEnabled(true)`
or at build time via `VITE_ORT_WEBGPU=true`. The adapter requests
`["webgpu", "wasm"]` when enabled (ORT falls back to WASM if WebGPU is
unavailable in the browser).

### Capability detection and PCA fallback

`isONNXRuntimeAvailable()` probes `onnxruntime-web` once and memoises. The
`embed()` facade in `src/lib/ai/embeddings` automatically falls back to the
PCA adapter when the requested adapter is unknown, lacks `embed()`, or
throws during `load()` / `embed()`. The result object carries
`fellBack: boolean` and an optional `fallbackReason`. Pass
`fallbackToPCA: false` to surface the underlying error.

### Tests

Run with `bun run test src/lib/ai`. The ONNX adapter is tested against a
pluggable fake runtime so tests do not require WASM.
