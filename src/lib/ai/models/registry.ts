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