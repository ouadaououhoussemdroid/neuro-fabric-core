/**
 * Foundation-model registry. Sits alongside the legacy
 * `src/lib/model-registry` constant (which still names the active production
 * decoder). This registry is the authoritative list of *adapters* the AI
 * layer can route through.
 */
import type { AdapterFactory, EEGModelAdapter } from "../adapters/types";
import type { ModelDescriptor } from "../types";
import { PCAEmbeddingAdapter } from "../adapters/pca-adapter";
import { PyTorchExportAdapter } from "../adapters/pytorch-export-adapter";
import { BraindecodeAdapter } from "../adapters/braindecode-adapter";
import { EEGPTAdapter } from "../adapters/eegpt-adapter";
import { ONNXAdapter } from "../adapters/onnx-adapter";
import {
  createONNXBraindecodeBridge,
  type ONNXBraindecodeBridgeOptions,
} from "../adapters/braindecode-onnx-bridge";

interface RegistryEntry {
  descriptor: ModelDescriptor;
  factory: AdapterFactory;
}

const entries = new Map<string, RegistryEntry>();

export function registerModel(factory: AdapterFactory): void {
  const instance = factory();
  entries.set(instance.descriptor.id, { descriptor: instance.descriptor, factory });
}

export function listModels(): ModelDescriptor[] {
  return Array.from(entries.values()).map((e) => e.descriptor);
}

export function getDescriptor(id: string): ModelDescriptor | undefined {
  return entries.get(id)?.descriptor;
}

export function hasModel(id: string): boolean {
  return entries.has(id);
}

export function unregisterModel(id: string): boolean {
  return entries.delete(id);
}

export function createAdapter(id: string): EEGModelAdapter {
  const entry = entries.get(id);
  if (!entry) throw new Error(`Unknown model id: ${id}`);
  return entry.factory();
}

registerModel(() => new PCAEmbeddingAdapter());
registerModel(() => new PyTorchExportAdapter());
registerModel(() => new EEGPTAdapter());
// Default Braindecode entry — EEGNetv4 over a 22-channel 2 s @ 128 Hz window.
// Until a bridge is injected via setBraindecodeBridge(), load() throws and the
// embed() facade falls back to ONNX → PCA.
registerModel(
  () =>
    new BraindecodeAdapter({
      id: "braindecode-eegnetv4-default",
      architecture: "EEGNetv4",
      channels: 22,
      sampleRate: 128,
      windowSamples: 256,
    }),
);

// Production EEGConformer — ONNX artefact served from /models/
registerBraindecodeEEGConformer({ artifact: "/models/eegconformer.onnx" });

// T-015 — Braindecode model zoo: register ShallowFBCSPNet, Deep4Net
// alongside EEGConformer for comparative ablations. EEGNetv4 was already
// registered above; no duplicate entry needed.
registerModel(
  () =>
    new BraindecodeAdapter({
      id: "braindecode-shallowfbcspnet-default",
      architecture: "ShallowFBCSPNet",
      channels: 22,
      sampleRate: 250,
      windowSamples: 1125,
    }),
);
registerModel(
  () =>
    new BraindecodeAdapter({
      id: "braindecode-deep4net-default",
      architecture: "Deep4Net",
      channels: 22,
      sampleRate: 250,
      windowSamples: 1125,
    }),
);
// ---------------------------------------------------------------------------
// Tier 4 Foundation Models — real ONNX artefacts with verified checkpoints.
// Each model below was exported from a real pretrained checkpoint, verified in
// Python against the reference PyTorch implementation (parity reported in the
// ModelDescriptor description), and deployed to /models/ with SHA-256 in the
// manifest. These registrations are the single source of truth for the
// platform's model routing.
// ---------------------------------------------------------------------------

// EEGConformer (already registered above via registerBraindecodeEEGConformer)
// EEGPT is already registered above via EEGPTAdapter (which delegates to ONNXAdapter)

// FEMBA-tiny (PulpBio/FEMBA, Apache-2.0) — Mamba-based EEG encoder.
// INT8 quantization destabilised due to recurrent scan error compounding;
// FP16 used for browser (30.72→16.26 MB). The adapter ONNX
// (femba-tiny-encoder-adapter.onnx) has a Reshape node that adapts
// [1, 22, 1280] → [1, 1, 22, 1280] for the underlying model.
registerModel(
  () =>
    new ONNXAdapter({
      id: "onnx-femba-tiny",
      name: "FEMBA-tiny (Mamba, FP32)",
      version: "1.0.0",
      description:
        "FEMBA-tiny P1 — Mamba-based EEG foundation model " +
        "(PulpBio/FEMBA, Apache-2.0). Input: [1, 22, 1280] @ 200Hz. " +
        "Graph-surgery adapter ONNX reshapes to [1, 1, 22, 1280]. " +
        "INT8 quantization destabilised by 80-step recurrent scan loop " +
        "(per-channel max_diff=3.23); FP16 used instead. " +
        "Parity: cos_sim>0.99, all ops WASM-compatible.",
      artifact: "/models/femba-tiny-encoder-adapter.onnx",
      task: "embedding",
      inputShape: { kind: "raw", channels: 22, samples: 1280 },
      channels: 22,
      sampleRate: 200,
      windowSamples: 1280,
      embeddingDim: 30800,
      wasmCompatible: true,
    }),
);

// LaBraM (braindecode/labram-pretrained, MIT) — ViT with channel patching.
// The ONNX (labram-encoder.onnx) has a Reshape node inserted via graph
// surgery that adapts [1, 16, 1600] → [1, 16, 8, 200].
registerModel(
  () =>
    new ONNXAdapter({
      id: "onnx-labram",
      name: "LaBraM (ViT, FP32)",
      version: "1.0.0",
      description:
        "LaBraM — Vision Transformer with channel patching " +
        "(braindecode/labram-pretrained, MIT). Input: [1, 16, 1600] @ 250Hz. " +
        "Graph-surgery Reshape node adapts to [1, 16, 8, 200]. " +
        "Parity: cos_sim>0.99, all ops WASM-compatible.",
      artifact: "/models/labram-encoder.onnx",
      task: "embedding",
      inputShape: { kind: "raw", channels: 16, samples: 1600 },
      channels: 16,
      sampleRate: 250,
      windowSamples: 1600,
      embeddingDim: 200,
      wasmCompatible: true,
    }),
);

// CBraMod (braindecode/cbramod-pretrained, MIT) — Conv+Transformer.
// ONNX exports successfully (2.23 MB, max_diff=8.58e-06) but contains
// DFT and ReduceL2 ops that are NOT supported in ORT-WASM. Runs server-side
// with onnxruntime CPU Ep. Marked wasmCompatible: false so the router knows
// to skip it in browser contexts.
registerModel(
  () =>
    new ONNXAdapter({
      id: "onnx-cbramod",
      name: "CBraMod (Conv+Transformer, FP32)",
      version: "1.0.0",
      description:
        "CBraMod — EEG Conformer variant with raw Fourier features " +
        "(braindecode/cbramod-pretrained, MIT). Input: [1, 19, 1000] @ 250Hz. " +
        "NOT WASM-compatible: contains DFT (Discrete Fourier Transform) " +
        "and ReduceL2 ops unsupported by ORT-WASM web_ops. Server-only.",
      artifact: "/models/cbramod-encoder.onnx",
      task: "embedding",
      inputShape: { kind: "raw", channels: 19, samples: 1000 },
      channels: 19,
      sampleRate: 250,
      windowSamples: 1000,
      embeddingDim: 19000,
      wasmCompatible: false,
      wasmBlockers: ["DFT", "ReduceL2"],
    }),
);

/** Default embedder used when callers do not pin a model id. */
export const DEFAULT_EMBEDDER_ID = "pca-legacy-v1";

/**
 * Register a production Braindecode model backed by an exported ONNX file.
 * After calling this, `embedEEG()` will route to it before falling back to
 * any generic ONNX model and finally PCA.
 */
export function registerBraindecodeONNX(
  opts: ONNXBraindecodeBridgeOptions & { id?: string },
): string {
  const id = opts.id ?? "braindecode-eegnetv4-onnx";
  registerModel(
    () =>
      new BraindecodeAdapter({
        id,
        architecture: opts.architecture,
        channels: opts.channels,
        sampleRate: opts.sampleRate,
        windowSamples: opts.windowSamples,
        version: "0.1.0-onnx",
        bridge: () => createONNXBraindecodeBridge(opts),
        isExperimental: false,
        weightsUri: typeof opts.artifact === "string" ? opts.artifact : undefined,
        wasmCompatible: opts.wasmCompatible ?? true,
        wasmBlockers: opts.wasmBlockers,
      }),
  );
  return id;
}

/**
 * Register the **selected production EEG foundation model**: Braindecode
 * EEGConformer, exported to ONNX. EEGConformer (Song et al. 2022) is a
 * Conv+Transformer hybrid whose attention-pooled representations are the
 * strongest general-purpose embeddings available in the Braindecode
 * catalogue for similarity search / representation learning — see
 * docs/audits/2026-06-17_braindecode-model-selection.md.
 *
 * Defaults match the reference recipe (22 channels, 4 s @ 250 Hz, 32-D
 * embedding head). Override via opts for custom artefacts.
 */
export function registerBraindecodeEEGConformer(opts: {
  artifact: ONNXBraindecodeBridgeOptions["artifact"];
  id?: string;
  channels?: number;
  sampleRate?: number;
  windowSamples?: number;
  embeddingDim?: number;
  embeddingOutputName?: string;
  logitsOutputName?: string;
  executionProviders?: ONNXBraindecodeBridgeOptions["executionProviders"];
  runtime?: ONNXBraindecodeBridgeOptions["runtime"];
}): string {
  return registerBraindecodeONNX({
    id: opts.id ?? "braindecode-eegconformer-prod",
    artifact: opts.artifact,
    architecture: "EEGConformer",
    channels: opts.channels ?? 22,
    sampleRate: opts.sampleRate ?? 250,
    windowSamples: opts.windowSamples ?? 1000,
    embeddingDim: opts.embeddingDim ?? 32,
    embeddingOutputName: opts.embeddingOutputName ?? "embedding",
    logitsOutputName: opts.logitsOutputName,
    executionProviders: opts.executionProviders,
    runtime: opts.runtime,
  });
}
