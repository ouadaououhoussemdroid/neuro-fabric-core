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
registerModel(() => new BraindecodeAdapter());
registerModel(() => new EEGPTAdapter());

/** Default embedder used when callers do not pin a model id. */
export const DEFAULT_EMBEDDER_ID = "pca-legacy-v1";