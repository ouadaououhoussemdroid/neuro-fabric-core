# AI Foundation Layer

Migration-safe abstraction layer enabling Neuro-Fabric to evolve from PCA-based
embeddings toward foundation models (ONNX, PyTorch exports, Braindecode, EEGPT)
without breaking existing pipelines.

## Current state (trained EEGConformer deployed)

| Adapter                                   | Status                  | Description                                                                                        |
| ----------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| **PCA**                                   | Live (default fallback) | `PCAEmbeddingAdapter` wraps `embedSignal` from `src/lib/embeddings`                                |
| **ONNX**                                  | Live                    | `ONNXAdapter` wraps `onnxruntime-web`'s `InferenceSession` with self-hosted WASM                   |
| **Braindecode (ONNX)**                    | Live (production)       | `braindecode-eegconformer-prod` — trained EEGConformer ONNX artefact via `braindecode-onnx-bridge.ts`. Holdout accuracy 0.578, recall@10 0.941. |
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

The WebGPU execution provider (EP) is an opt-in feature for accelerating ONNX model inference using the browser's WebGPU API. 
It is controlled via the `webgpu-flag.ts` module and defaults to **disabled** (WASM-only) to ensure compatibility with browsers that do not yet support WebGPU.

#### Enabling WebGPU EP

WebGPU EP can be enabled in two ways:
1. **Build-time**: Set the environment variable `VITE_ORT_WEBGPU=true` when building the application.
2. **Runtime**: Call `setWebGPUEnabled(true)` from JavaScript (e.g., from a settings UI).

The adapter will request the execution providers `["webgpu", "wasm"]` when WebGPU EP is enabled. 
ONNX Runtime Web automatically falls back to WASM if WebGPU is unavailable or unsupported in the current browser.

#### Capability Detection

The `isWebGPUAvailable()` function checks for the presence of `navigator.gpu` to determine if the browser supports WebGPU.
The `isWebGPUEnabled()` function returns `true` only if:
- Either the build-time flag (`VITE_ORT_WEBGPU`) is set to `"true"` **or** the runtime toggle (`setWebGPUEnabled(true)`) has been called, **AND**
- The browser supports WebGPU (`isWebGPUAvailable()` returns `true`).

#### Execution Providers

When WebGPU EP is enabled, the ONNX adapter uses the execution provider array `["webgpu", "wasm"]`. 
This allows ONNX Runtime Web to attempt to use WebGPU for acceleration, falling back to WASM if WebGPU fails to initialize or is not supported.

#### Cold-Start Benchmark (T-025)

Measuring the cold-start time (the time to load the model and initialize the runtime) is important for understanding the performance impact of enabling WebGPU EP. 
The cold-start time includes:
- Fetching the ONNX model artifact (if not cached)
- Initializing the ONNX Runtime Web with the selected execution providers
- Loading and validating the model
- Allocating resources (e.g., GPU memory for WebGPU)

To benchmark cold-start with and without WebGPU EP:

1. **Enable WebGPU EP**: Set `VITE_ORT_WEBGPU=true` and/or call `setWebGPUEnabled(true)` before initializing the adapter.
2. **Disable WebGPU EP**: Ensure `VITE_ORT_WEBGPU` is not set to `"true"` and call `setWebGPUEnabled(false)` (the default).
3. **Measure the time** to create an `ONNXAdapter` instance and load a model (e.g., the trained EEGConformer model) using the `performance.now()` API in the browser.
4. **Compare the times**: The difference indicates the overhead or improvement introduced by WebGPU EP initialization.

> **Note**: Actual cold-start times vary based on the model size, device capabilities, and network conditions. 
> For accurate results, run the benchmark in a production-like environment with the target model and device.

Example benchmark code (to be run in the browser console or a test script):
```ts
import { ONNXAdapter } from "@/lib/ai/adapters";
import { setWebGPUEnabled } from "@/lib/ai/adapters/webgpu-flag";

async function benchmarkColdStart(modelPath: string, inputShape: { kind: "features"; dim: number }): Promise<{ wasmTime: number; webgpuTime: number }> {
  // Measure WASM-only cold-start
  setWebGPUEnabled(false);
  const wasmStart = performance.now();
  const wasmAdapter = new ONNXAdapter({
    artifact: modelPath,
    inputShape,
    embeddingDim: 32, // Example: adjust to your model
    executionProviders: ["wasm"], // Explicitly request WASM (default)
  });
  await wasmAdapter.load(); // Assuming a load() method exists; adapt to actual API
  const wasmTime = performance.now() - wasmStart;

  // Measure WebGPU EP cold-start
  setWebGPUEnabled(true);
  const webgpuStart = performance.now();
  const webgpuAdapter = new ONNXAdapter({
    artifact: modelPath,
    inputShape,
    embeddingDim: 32,
    // executionProviders: ["webgpu", "wasm"], // Default when WebGPU EP is enabled
  });
  await webgpuAdapter.load();
  const webgpuTime = performance.now() - webgpuStart;

  return { wasmTime, webgpuTime };
}

// Usage:
// benchmarkColdStart("/models/eegconformer.onnx", { kind: "features", dim: 32 })
//   .then(({ wasmTime, webgpuTime }) => {
//     console.log(`WASM cold-start: ${wasmTime.toFixed(2)} ms`);
//     console.log(`WebGPU EP cold-start: ${webgpuTime.toFixed(2)} ms`);
//     console.log(`Difference: ${(webgpuTime - wasmTime).toFixed(2)} ms`);
//   });
```

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
