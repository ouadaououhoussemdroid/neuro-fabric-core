# Self-hosted ONNX Runtime WASM

This directory hosts the `onnxruntime-web` WASM bundle from our own origin
(T-008). The files are copied here at build time by the
`ortWasmSelfHostPlugin` Vite plugin (see `vite-plugins/ort-wasm-self-host.ts`),
which reads from `node_modules/onnxruntime-web/dist/` and writes:

- `ort-wasm-simd-threaded.wasm` (~12 MB) — the SIMD + threaded WASM binary
- `ort-wasm-simd-threaded.mjs` — the JS loader for the above
- `ort-wasm-simd-threaded.jsep.wasm` (~25 MB) — the JSEP (WebGPU) variant
- `ort-wasm-simd-threaded.jsep.mjs` — the JS loader for the above
- `integrity.json` — SHA-384 hashes + sizes for every artefact

The ONNX adapter (`src/lib/ai/adapters/onnx-adapter.ts`) sets
`ort.env.wasm.wasmPaths = "/ort/"` so the runtime fetches WASM from here
instead of jsdelivr.

## Integrity verification

`integrity.json` is regenerated on every build. Each entry has the form:

```json
{
  "file": "ort-wasm-simd-threaded.wasm",
  "sha384": "sha384-<base64>",
  "size": 13022405
}
```

The adapter loads this manifest at runtime and can verify the fetched
artefact matches the expected hash before instantiating the runtime.

## Overriding the path

Set `VITE_ORT_WASM_PATHS` to point at a different origin (e.g. a CDN
bucket) if self-hosting under `/ort/` is not desired:

```bash
VITE_ORT_WASM_PATHS=https://cdn.example.com/ort/ vite build
```

## Why self-host?

Removes the hard dependency on `cdn.jsdelivr.net` (audit item D5). A CDN
outage would otherwise cause silent PCA fallback with no user-visible
signal. Self-hosting makes the WASM availability a first-class ops concern.
