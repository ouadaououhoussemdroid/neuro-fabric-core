# AI Foundation Layer

Migration-safe abstraction layer enabling Neuro-Fabric to evolve from PCA-based
embeddings toward foundation models (ONNX, PyTorch exports, Braindecode, EEGPT)
without breaking existing pipelines.

## Layout

- `models/`      — model registry, capability descriptors, version metadata.
- `embeddings/`  — embedder interface + PCA legacy adapter (default fallback).
- `adapters/`    — runtime adapters: PCA (live), ONNX (stub), PyTorch export
                   (stub), Braindecode (stub), EEGPT (stub).
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
