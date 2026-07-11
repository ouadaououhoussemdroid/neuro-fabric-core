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
