/**
 * T-016 — Sidecar metadata that enriches the auto-generated ONNX artefact
 * manifest with registry IDs, WASM compatibility, and WASM blocker op lists.
 *
 * The Vite `artefactManifestPlugin` merges this into `public/models/manifest.json`
 * at build time. Each key must match the artifact filename (without `.onnx`).
 */
import type { ManifestMetadata } from "./hashed-artefact";

export const TIER4_MANIFEST_METADATA: ManifestMetadata = {
  // Braindecode EEGConformer v1 — rollback-only after V2 GA promotion
  eegconformer: {
    registryId: "braindecode-eegconformer-prod",
    wasmCompatible: true,
  },
  // EEGConformer v2 — GA default model (fine-tuned on PhysioNet EEGMMIDB, 20 subjects)
  // External data merged into ONNX for WASM compatibility (no external data file)
  eegconformer_finetuned: {
    registryId: "braindecode-eegconformer-prod-v2",
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
  // M43 trained sleep staging probe (2312→5)
  "staging-probe-joint2312-v1": {
    registryId: "sleep-staging-v1",
    wasmCompatible: true,
    trained: true,
  },
  // M43 trained sleep quality probe (2312→1)
  "quality-probe-joint2312-v1": {
    registryId: "sleep-quality-v1",
    wasmCompatible: true,
    trained: true,
  },
  // M44 trained sleep staging V2-32 browser probe (32→5)
  "staging-probe-v2-32d-v1": {
    wasmCompatible: true,
  },
  // M44 trained sleep quality V2-32 browser probe (32→1)
  "quality-probe-v2-32d-v1": {
    wasmCompatible: true,
  },
};
