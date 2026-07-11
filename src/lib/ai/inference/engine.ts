/**
 * Runtime-agnostic inference engine. Holds a small LRU of loaded adapters so
 * repeated calls against the same model id reuse the runtime (ONNX sessions,
 * Pyodide instances) without reloading weights on every request.
 */
import { createAdapter, getDescriptor } from "../models/registry";
import type { EEGModelAdapter } from "../adapters/types";
import type { EmbeddingOutput, ModelDescriptor, ModelInput, PredictionOutput } from "../types";

export class InferenceEngine {
  private cache = new Map<string, EEGModelAdapter>();

  constructor(private readonly maxLoaded = 2) {}

  describe(modelId: string): ModelDescriptor | undefined {
    return getDescriptor(modelId);
  }

  private async acquire(modelId: string): Promise<EEGModelAdapter> {
    const existing = this.cache.get(modelId);
    if (existing) return existing;
    const adapter = createAdapter(modelId);
    await adapter.load();
    this.cache.set(modelId, adapter);
    if (this.cache.size > this.maxLoaded) {
      const oldest = this.cache.keys().next().value as string | undefined;
      if (oldest && oldest !== modelId) {
        const a = this.cache.get(oldest);
        this.cache.delete(oldest);
        if (a) await a.unload();
      }
    }
    return adapter;
  }

  async embed(modelId: string, input: ModelInput): Promise<EmbeddingOutput> {
    const adapter = await this.acquire(modelId);
    if (!adapter.embed) throw new Error(`Adapter "${modelId}" has no embed()`);
    return adapter.embed(input);
  }

  async predict(modelId: string, input: ModelInput): Promise<PredictionOutput> {
    const adapter = await this.acquire(modelId);
    if (!adapter.predict) throw new Error(`Adapter "${modelId}" has no predict()`);
    return adapter.predict(input);
  }

  async dispose(): Promise<void> {
    for (const a of this.cache.values()) await a.unload();
    this.cache.clear();
  }
}

/** Process-wide default engine. */
export const inferenceEngine = new InferenceEngine();
