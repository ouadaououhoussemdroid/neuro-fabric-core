/**
 * P3 — InferenceEngine lifecycle: the process-wide LRU + shutdown contract.
 *
 * These are the Node-side complements to the browser V3 smoke tests. They cover
 * exactly the invariants the persistent-session production wiring depends on,
 * without loading any real ONNX weights (an inert stub adapter is registered
 * per-test into the model registry, mirroring the tier4-broken-* pattern):
 *
 *   - SSR/server shutdown: `dispose()` is a safe no-op before the first request
 *     (the server may never serve an embedEEG request), and fully unwinds the
 *     LRU cache after requests (calls `unload()` on each cached adapter, then
 *     resets the cache). src/server.ts wires SIGTERM/SIGINT to `inferenceEngine.
 *     dispose()`.
 *   - LRU bound: with maxLoaded=1, loading a second model evicts (unload + drop)
 *     the least-recently-inserted victim so the process-wide cache never grows
 *     unbounded on Nitro SSR.
 *   - Concurrent first-load dedup: N simultaneous `getAdapter(id)` for a cold
 *     model coalesce onto a single `load()` (pending-promise dedup), so racing
 *     first requests never each spin up a session.
 *   - disposeModel(): drops + unloads a single model (the path embedEEG uses on a
 *     failed forward so the next request retries a fresh session).
 *   - embed() routing: forwards are invoked as methods on the adapter instance
 *     (preserving `this`, the regression that broke the browser path) and are
 *     serialized per model by `withLock` (ORT-Web WASM session.run is not
 *     reentrant — concurrent forwards must queue behind a single session).
 *
 * No network / no fetch stubbing needed: the stub loads synchronously and embeds
 * in-memory, so these are fast and deterministic.
 */
import { describe, it, expect, afterEach } from "vitest";
import { InferenceEngine } from "../engine";
import { registerModel, hasModel, unregisterModel } from "../../models/registry";
import type { EEGModelAdapter } from "../../adapters/types";
import type { ModelDescriptor, ModelInput, EmbeddingOutput } from "../../types";

/** Inert adapter: real load/unload lifecycle + a deterministic 8-D embed. */
class StubAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor;
  loadCount = 0;
  unloadCount = 0;
  embedCount = 0;
  private loaded = false;

  constructor(id: string) {
    this.descriptor = {
      id,
      kind: "braindecode",
      name: `stub-${id}`,
      version: "0.0.0-test",
      description: "Inert stub for InferenceEngine lifecycle tests",
      isExperimental: false,
      capabilities: {
        task: "embedding",
        channels: 1,
        sampleRate: 1,
        windowSamples: 1,
        embeddingDim: 8,
        runtime: "js",
        implemented: true,
      },
      createdAt: "2026-08-13T00:00:00Z",
    };
  }

  async load(): Promise<void> {
    this.loadCount++;
    this.loaded = true;
  }

  async unload(): Promise<void> {
    this.unloadCount++;
    this.loaded = false;
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async embed(_input: ModelInput): Promise<EmbeddingOutput> {
    this.embedCount++;
    const vector = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25];
    return { vector, dim: 8, modelId: this.descriptor.id, durationMs: 0 };
  }
}

const tempIds: string[] = [];

/** Register a singleton stub under `id`; returns the instance to spy on. */
function registerStub(id: string): StubAdapter {
  const stub = new StubAdapter(id);
  registerModel(() => stub);
  tempIds.push(id);
  return stub;
}

afterEach(() => {
  for (const id of tempIds) {
    if (hasModel(id)) unregisterModel(id);
  }
  tempIds.length = 0;
});

describe("InferenceEngine lifecycle (P3 persistent-session contract)", () => {
  it("dispose() is a safe no-op on a fresh engine — SSR shutdown before any request", async () => {
    const engine = new InferenceEngine();
    expect(engine.cacheSize()).toBe(0);
    // Server may receive SIGTERM/SIGINT without ever serving an embedEEG request.
    await expect(engine.dispose()).resolves.toBeUndefined();
    expect(engine.cacheSize()).toBe(0);
    // No models were registered, so there's nothing to dispose — must not throw.
  });

  it("dispose() unloads cached adapters and resets the LRU — SSR shutdown after requests", async () => {
    const engine = new InferenceEngine();
    const stub = registerStub("engine-t2-stub");
    await engine.getAdapter("engine-t2-stub");
    expect(engine.cacheSize()).toBe(1);
    expect(stub.loadCount).toBe(1);

    await engine.dispose();
    expect(stub.unloadCount).toBe(1);
    expect(engine.cacheSize()).toBe(0);
  });

  it("evicts the least-recently-inserted adapter when over capacity (LRU bound, maxLoaded=1)", async () => {
    const engine = new InferenceEngine(1);
    const a = registerStub("engine-t3-a");
    const b = registerStub("engine-t3-b");

    await engine.getAdapter("engine-t3-a");
    await engine.getAdapter("engine-t3-b");

    expect(engine.cacheSize()).toBe(1); // bounded at maxLoaded
    expect(a.unloadCount).toBe(1); // victim: evicted + unloaded
    expect(b.unloadCount).toBe(0); // kept — newest
  });

  it("coalesces concurrent first-loads for the same model id onto a single load()", async () => {
    const engine = new InferenceEngine();
    const stub = registerStub("engine-t4-stub");

    const [r1, r2, r3] = await Promise.all([
      engine.getAdapter("engine-t4-stub"),
      engine.getAdapter("engine-t4-stub"),
      engine.getAdapter("engine-t4-stub"),
    ]);

    expect(r1).toBe(stub);
    expect(r2).toBe(stub);
    expect(r3).toBe(stub);
    expect(stub.loadCount).toBe(1); // deduped — one session, not three
  });

  it("disposeModel() unloads and drops a single model", async () => {
    const engine = new InferenceEngine(); // maxLoaded=2 (default)
    const a = registerStub("engine-t5-a");
    const b = registerStub("engine-t5-b");

    await engine.getAdapter("engine-t5-a");
    await engine.getAdapter("engine-t5-b");
    expect(engine.cacheSize()).toBe(2);

    await engine.disposeModel("engine-t5-a");
    expect(engine.cacheSize()).toBe(1);
    expect(a.unloadCount).toBe(1); // dropped + unloaded
  });

  it("embed() routes through the cached adapter (this-binding preserved) and serializes forwards", async () => {
    const engine = new InferenceEngine();
    const stub = registerStub("engine-t6-stub");

    const out = await engine.embed("engine-t6-stub", {
      kind: "features",
      features: [[1]],
    });
    expect(out.modelId).toBe("engine-t6-stub");
    expect(out.dim).toBe(8);
    expect(stub.embedCount).toBe(1);
  });

  it("embed() serializes 8 concurrent forwards behind one cached session (reentrant guard)", async () => {
    const engine = new InferenceEngine();
    const stub = registerStub("engine-t6b-stub");
    await engine.getAdapter("engine-t6b-stub"); // warm the cache (1 session)

    // ORT-Web WASM session.run() is NOT reentrant — withLock must queue these.
    const results = await Promise.all(
      Array.from({ length: 8 }, () =>
        engine.embed("engine-t6b-stub", { kind: "features", features: [[1]] }),
      ),
    );
    expect(results).toHaveLength(8);
    expect(results.every((r) => r.dim === 8)).toBe(true);
    expect(stub.embedCount).toBe(8); // all ran, serialized, single session reused
    expect(stub.loadCount).toBe(1); // no reload — session persists
  });
});
