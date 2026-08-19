/**
 * Runtime-agnostic inference engine. Holds a small LRU of loaded adapters so
 * repeated calls against the same model id reuse the runtime (ONNX sessions,
 * Pyodide instances) without reloading weights on every request.
 *
 * P3 enhancement: added per-model async mutex (`withLock`) so concurrent forwards
 * on the same cached session are serialized — ORT-Web WASM `session.run()` is not
 * reentrant. Also added `getAdapter` (internal alias), `disposeModel`, `cacheSize`,
 * and `pending` promise dedup for concurrent first-loads.
 */
import { createAdapter, getDescriptor } from "../models/registry";
import type { EEGModelAdapter } from "../adapters/types";
import type { EmbeddingOutput, ModelDescriptor, ModelInput, PredictionOutput } from "../types";

export class InferenceEngine {
  private cache = new Map<string, EEGModelAdapter>();
  /** In-flight load promises — coalesces concurrent first-loads for the same id. */
  private pending = new Map<string, Promise<EEGModelAdapter>>();
  /** Per-model async mutex for serializing non-reentrant session.run(). */
  private mutexes = new Map<string, { tail: Promise<void>; count: number }>();

  constructor(private readonly maxLoaded = 2) {}

  describe(modelId: string): ModelDescriptor | undefined {
    return getDescriptor(modelId);
  }

  /** Returns the cached adapter if loaded, otherwise loads + caches it. */
  private async acquire(modelId: string): Promise<EEGModelAdapter> {
    const existing = this.cache.get(modelId);
    if (existing) return existing;
    const inFlight = this.pending.get(modelId);
    if (inFlight) return inFlight;

    const promise = (async () => {
      const adapter = createAdapter(modelId);
      await adapter.load();
      this.cache.set(modelId, adapter);
      this.evictIfOversized();
      return adapter;
    })();
    this.pending.set(modelId, promise);
    try {
      return await promise;
    } finally {
      this.pending.delete(modelId);
    }
  }

  /** Public alias used by embedEEG() and tests. */
  async getAdapter(modelId: string): Promise<EEGModelAdapter> {
    return this.acquire(modelId);
  }

  private evictIfOversized(): void {
    if (this.cache.size <= this.maxLoaded) return;
    const oldest = this.cache.keys().next().value as string | undefined;
    if (oldest !== undefined) {
      const victim = this.cache.get(oldest);
      this.cache.delete(oldest);
      this.mutexes.delete(oldest);
      if (victim) void victim.unload();
    }
  }

  /** Drop + unload a single model from the LRU cache (used on forward failure). */
  async disposeModel(modelId: string): Promise<void> {
    const adapter = this.cache.get(modelId);
    this.cache.delete(modelId);
    this.mutexes.delete(modelId);
    if (adapter) await adapter.unload();
  }

  /** Number of models currently cached. Test/observability hook. */
  cacheSize(): number {
    return this.cache.size;
  }

  /**
   * Per-model async mutex. Chains concurrent forwards behind the tail of the
   * previous forward so ORT-Web WASM session.run() is never called reentrantly.
   */
  private withLock<T>(modelId: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.mutexes.get(modelId)?.tail ?? Promise.resolve();
    const cur = (async () => {
      await prev;
      return fn();
    })();
    // The mutex tail is void — we only care about sequencing, not the return value.
    this.mutexes.set(modelId, { tail: cur.then(() => {}, () => {}), count: 0 });
    return cur;
  }

  async embed(modelId: string, input: ModelInput): Promise<EmbeddingOutput> {
    const adapter = await this.acquire(modelId);
    if (!adapter.embed) throw new Error(`Adapter "${modelId}" has no embed()`);
    return this.withLock(modelId, () => adapter.embed!(input));
  }

  async predict(modelId: string, input: ModelInput): Promise<PredictionOutput> {
    const adapter = await this.acquire(modelId);
    if (!adapter.predict) throw new Error(`Adapter "${modelId}" has no predict()`);
    return this.withLock(modelId, () => adapter.predict!(input));
  }

  async dispose(): Promise<void> {
    for (const a of this.cache.values()) await a.unload();
    this.cache.clear();
    this.pending.clear();
    this.mutexes.clear();
  }
}

/** Process-wide default engine. */
export const inferenceEngine = new InferenceEngine();
