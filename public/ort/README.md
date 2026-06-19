# Self-hosted ONNX Runtime WASM (planned)

This directory is reserved for the same-origin `onnxruntime-web` WASM bundle.
When populated, set `VITE_ORT_WASM_PATHS=/ort/` and the ONNX adapter
(`src/lib/ai/adapters/onnx-adapter.ts`) will load WASM from here instead of
the pinned jsdelivr CDN.

## Why it is currently empty

The two artefacts shipped by `onnxruntime-web@1.26.0`
(`ort-wasm-simd-threaded.wasm` ≈ 12 MB and `ort-wasm-simd-threaded.jsep.wasm`
≈ 25 MB) exceed the platform's per-file repository commit limit (10 MB) and
the asset CDN currently rejects both the `.wasm` extension and the
`application/wasm` content-type. Until one of those constraints is lifted
(large-file repo support, or asset-CDN `application/wasm` allow-listing),
`wasmPaths` defaults to the pinned jsdelivr release matching the installed
runtime version.

## Recommended migration once unblocked

1. Copy from `node_modules/onnxruntime-web/dist/`:
   - `ort-wasm-simd-threaded.wasm`
   - `ort-wasm-simd-threaded.mjs`
   - `ort-wasm-simd-threaded.jsep.wasm`
   - `ort-wasm-simd-threaded.jsep.mjs`
2. Set `VITE_ORT_WASM_PATHS=/ort/` in the production environment.
3. Verify with the existing diagnostic harness in
   `docs/audits/2026-06-19_eegconformer-runtime-verification.md` that the
   pipeline reports `fellBack: false` and a 32-D embedding.