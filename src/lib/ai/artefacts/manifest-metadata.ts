/**
 * T-016 — Sidecar metadata that enriches the auto-generated ONNX artefact
 * manifest with registry IDs, WASM compatibility, and WASM blocker op lists.
 *
 * The Vite `artefactManifestPlugin` merges this into `public/models/manifest.json`
 * at build time. Each key must match the artifact filename (without `.onnx`).
 */
import type { ManifestMetadata } from "./hashed-artefact";

export const TIER4_MANIFEST_METADATA: ManifestMetadata = {
  // Braindecode EEGConformer — production default
  eegconformer: {
    registryId: "braindecode-eegconformer-prod",
    wasmCompatible: true,
  },
  // EEGPT — ViT transformer, INT8-quantised
  "eegpt-encoder-int8": {
    registryId: "onnx-eegpt",
    wasmCompatible: true,
  },
  // FEMBA-tiny — Mamba, FP32 with graph-surgery adapter (reshape)
  "femba-tiny-encoder-adapter": {
    registryId: "onnx-femba-tiny",
    wasmCompatible: true,
  },
  // FEMBA-tiny — raw FP32 (benchmark reference)
  "femba-tiny-encoder": {
    registryId: "onnx-femba-tiny-raw",
    wasmCompatible: true,
  },
  // FEMBA-tiny — FP16 browser variant
  "femba-tiny-encoder-fp16": {
    registryId: "onnx-femba-tiny-fp16",
    wasmCompatible: true,
  },
  // LaBraM — ViT with channel patching, graph-surgery reshape
  "labram-encoder": {
    registryId: "onnx-labram",
    wasmCompatible: true,
  },
  // CBraMod — Conv+Transformer, DFT blocker (server-only)
  "cbramod-encoder": {
    registryId: "onnx-cbramod",
    wasmCompatible: false,
    wasmBlockers: ["DFT", "ReduceL2"],
  },
};
