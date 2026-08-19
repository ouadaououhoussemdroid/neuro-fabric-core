/**
 * T-036 unit tests for the server-native CBraMod foundation service.
 *
 * These tests exercise the wiring without running a native forward:
 *   - `onnxruntime-node` is mocked, so `foundationRuntime()` returns a synthetic
 *     `OrtRuntime` and `embedFoundationWindows` drives a fake ONNX session.
 *   - The real manifest + 22 MB artifact are read on disk through `ensureAdapter`,
 *     so SHA-256 provenance is verified against the actual c128ccfd… digest;
 *     only the ONNX *session* is mocked (no native forward runs here).
 *
 * The REAL end-to-end native forward (22 MB artifact + onnxruntime-node CPU EP)
 * lives in `foundation-e2e.test.ts` (skipped when onnxruntime-node cannot load).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { EEGWindow } from "@/lib/eeg/types";

vi.mock("onnxruntime-node", () => {
  // Faux ONNX session: input "eeg" [1,19,1000], output "embedding" [1,19,5,200].
  const fakeOutput = new Float32Array(19 * 5 * 200);
  for (let i = 0; i < fakeOutput.length; i++) fakeOutput[i] = Math.random();
  const session = {
    inputNames: ["eeg"],
    outputNames: ["embedding"],
    run: async (_feeds: unknown) => ({
      embedding: { data: fakeOutput, dims: [1, 19, 5, 200], type: "float32" },
    }),
    release: async () => {},
  };
  return {
    InferenceSession: {
      create: async (_path: unknown, _opts: unknown) => session,
    },
    Tensor: class {
      type: string;
      data: Float32Array;
      dims: readonly number[];
      constructor(type: string, data: Float32Array, dims: readonly number[]) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    },
    env: { backend: {} },
  };
});

const {
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
  FOUNDATION_ARTIFACT_ID,
  FoundationUnavailableError,
  foundationRuntime,
  foundationProvenance,
  embedFoundationWindows,
  resetFoundationAdapter,
} = await import("../foundation.server");

function syntheticWindow(): EEGWindow {
  return {
    data: Array.from({ length: 19 }, () =>
      Array.from({ length: 1000 }, () => Math.random()),
    ),
    sampleRate: 250,
    start: 0,
    end: 1000,
  };
}

describe("foundation.server (unit, mocked ort runtime)", () => {
  beforeEach(() => resetFoundationAdapter());
  afterEach(() => resetFoundationAdapter());

  it("exposes the Tier-2 id/dim/artifact constants", () => {
    expect(FOUNDATION_MODEL_ID).toBe("onnx-cbramod-foundation-200d");
    expect(FOUNDATION_EMBEDDING_DIM).toBe(200);
    expect(FOUNDATION_ARTIFACT_ID).toBe("cbramod-encoder");
  });

  it("foundationRuntime() conforms to the OrtRuntime structural contract", async () => {
    const rt = await foundationRuntime();
    expect(typeof rt.InferenceSession.create).toBe("function");
    expect(typeof rt.Tensor).toBe("function");
    // ONNXAdapter calls these getters on the created session.
    const sess = await rt.InferenceSession.create("ignored", { executionProviders: ["cpu"] });
    expect(sess.inputNames).toEqual(["eeg"]);
    expect(sess.outputNames).toEqual(["embedding"]);
  });

  it("foundationProvenance() reports the SHA-verified CBraMod digest", () => {
    const prov = foundationProvenance();
    expect(prov.artifact_id).toBe("cbramod-encoder");
    // T-016 provenance: the exact digest Mission-11 validated at rest.
    expect(prov.sha256).toBe("c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47");
    expect(prov.embedding_dim).toBe(200);
    expect(prov.output_pooling).toBe("mean-tokens");
    expect(prov.runtime).toBe("onnxruntime-node cpu");
    expect(prov.sample_rate_hz).toBe(250);
    expect(prov.window_samples).toBe(1000);
  });

  it("FoundationUnavailableError names its reason", () => {
    const e = new FoundationUnavailableError("mock-unavailable");
    expect(e.name).toBe("FoundationUnavailableError");
    expect(e.reason).toBe("mock-unavailable");
    expect(e.message).toMatch(/mock-unavailable/);
  });

  it("embedFoundationWindows produces 200-D L2-normalised vectors (no fallback)", async () => {
    const out = await embedFoundationWindows([syntheticWindow(), syntheticWindow()]);
    expect(out).toHaveLength(2);
    for (const r of out) {
      expect(r.dim).toBe(200);
      expect(r.vector).toHaveLength(200);
      expect(r.modelId).toBe(FOUNDATION_MODEL_ID);
      expect(r.fellBack).toBe(false);
      // L2 unit norm (finalize normalizes).
      const norm = Math.sqrt(r.vector.reduce((s, v) => s + v * v, 0));
      expect(norm).toBeCloseTo(1, 4);
    }
  });
});
