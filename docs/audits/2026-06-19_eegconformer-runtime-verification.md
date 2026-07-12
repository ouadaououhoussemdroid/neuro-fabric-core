# EEGConformer Runtime Verification + WASM Path Fix

- Date: 2026-06-19
- Scope: end-to-end runtime audit of `embedEEG()` in the live preview.
  Temporary diagnostic route added, executed in the browser, then removed.

## Method

A throwaway route `src/routes/diag-eegconformer.tsx` ran on mount:

1. Listed `registry.listModels()` ids.
2. Asserted `hasModel("braindecode-eegconformer-prod")`.
3. Logged the descriptor capabilities.
4. Probed `isONNXRuntimeAvailable()`.
5. `HEAD /models/eegconformer.onnx` to confirm the artefact is served.
6. Called `embedEEG(input)` with synthetic 22 ch × 1000-sample window
   (no `preferredModelId` override → exercise the real default routing).
7. Reported `modelId`, `dim`, `fellBack`, `fallbackReason`, L2 norm.

The route was executed via the live preview browser, then deleted.

## Findings

### First run — FAIL (silent PCA fallback)

```
registry ids: pca-legacy-v1, pytorch-export-placeholder, eegpt-placeholder,
              braindecode-eegnetv4-default, braindecode-eegconformer-prod
hasModel(braindecode-eegconformer-prod) = true
descriptor: {channels:22, sampleRate:250, windowSamples:1000, embeddingDim:32}
onnxruntime-web available: true
HEAD /models/eegconformer.onnx → 200 3360306 bytes
embedEEG done in 542.4 ms
  modelId       = pca-legacy-v1
  dim           = 110
  fellBack      = true
  fallbackReason= no available backend found.
    ERR: [wasm] RuntimeError: Aborted(both async and sync fetching of
    the wasm failed). Build with -sASSERTIONS for more info.
VERDICT: FAIL — silent fallback to PCA
```

Routing logic was correct (start id = `braindecode-eegconformer-prod`,
artefact reachable, ORT module imported). The failure was inside
`InferenceSession.create`: the ORT WASM backend could not fetch its
`.wasm` binary. `onnxruntime-web` resolves the binary via
`import.meta.url`, which is rewritten by Vite's dev server (`?v=<hash>`
versioned module proxies) into a path the binary loader cannot resolve
— both async fetch and synchronous fallback aborted, no backend
available, the embed() facade walked the chain to PCA.

### Fix

`src/lib/ai/adapters/onnx-adapter.ts` — `defaultRuntime()` now sets
`ort.env.wasm.wasmPaths` to the matching jsdelivr release (pinned to
the installed `1.26.0`). One change, scoped to the adapter, no public
API impact.

```ts
if (mod?.env?.wasm && mod.env.wasm.wasmPaths == null) {
  mod.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
}
```

### Second run — PASS

```
registry ids: pca-legacy-v1, pytorch-export-placeholder, eegpt-placeholder,
              braindecode-eegnetv4-default, braindecode-eegconformer-prod
hasModel(braindecode-eegconformer-prod) = true
descriptor: {channels:22, sampleRate:250, windowSamples:1000, embeddingDim:32}
onnxruntime-web available: true
HEAD /models/eegconformer.onnx → 200 3360306 bytes
embedEEG done in 3525.3 ms          (cold session create + first inference)
  modelId       = braindecode-eegconformer-prod
  dim           = 32
  fellBack      = false
  fallbackReason= (none)
  normalized    = true
  L2 norm       = 1.0000
VERDICT: PASS — EEGConformer ONNX live, 32-D, no fallback
```

## Verification matrix

| Check                                                | Result                  |
| ---------------------------------------------------- | ----------------------- |
| `public/models/eegconformer.onnx` served at runtime  | ✅ 200, 3 360 306 bytes |
| `embedEEG()` selects `braindecode-eegconformer-prod` | ✅                      |
| ONNX Runtime executes in the browser (WASM EP)       | ✅ after wasmPaths fix  |
| Returned embedding dim = 32                          | ✅                      |
| No silent PCA fallback                               | ✅ `fellBack: false`    |
| L2-normalised, finite                                | ✅ ‖v‖ = 1.0000         |

## Files modified

- `src/lib/ai/adapters/onnx-adapter.ts` — pin `ort.env.wasm.wasmPaths`
  to jsdelivr CDN for `onnxruntime-web@1.26.0` to fix in-browser
  WASM-binary resolution.

## Files added then removed (diagnostic only)

- `src/routes/diag-eegconformer.tsx` — created for this audit, deleted
  after passing run. No production traces remain.

## Follow-ups (not required for this audit)

- Optional: vendor the ORT `.wasm` files into `public/ort/` and point
  `wasmPaths` there to remove the runtime CDN dependency.
- Cold session create dominated the 3.5 s first-call cost. Subsequent
  inferences in the same session are not measured here; record P95
  with the existing benchmark harness against this artefact.
