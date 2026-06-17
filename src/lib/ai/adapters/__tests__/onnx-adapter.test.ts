import { describe, it, expect, beforeEach } from "vitest";
import {
  ONNXAdapter,
  isONNXRuntimeAvailable,
  __resetONNXCapabilityProbe,
  type OrtRuntime,
  type OrtSessionLike,
  type OrtTensorLike,
} from "../onnx-adapter";

function makeFakeRuntime(opts: { outputDim?: number; failLoad?: boolean } = {}) {
  const calls = { runs: 0, released: 0 };
  const session: OrtSessionLike = {
    inputNames: ["input"],
    outputNames: ["embedding"],
    async run(feeds) {
      calls.runs++;
      const t = feeds["input"];
      const sum = Array.from(t.data as ArrayLike<number>).reduce(
        (a, b) => a + Number(b),
        0,
      );
      const dim = opts.outputDim ?? 4;
      const data = new Float32Array(dim).map((_, i) => sum + i);
      return { embedding: { data, dims: [1, dim] } satisfies OrtTensorLike };
    },
    async release() {
      calls.released++;
    },
  };
  const runtime: OrtRuntime = {
    InferenceSession: {
      async create() {
        if (opts.failLoad) throw new Error("fake load failure");
        return session;
      },
    },
    Tensor: class {
      constructor(
        public type: "float32",
        public data: Float32Array,
        public dims: readonly number[],
      ) {}
    } as unknown as OrtRuntime["Tensor"],
  };
  return { runtime, calls };
}

describe("ONNXAdapter", () => {
  beforeEach(() => __resetONNXCapabilityProbe());

  it("loads, runs, and unloads against a feature input", async () => {
    const { runtime, calls } = makeFakeRuntime({ outputDim: 8 });
    const adapter = new ONNXAdapter({
      id: "test-onnx",
      name: "Test",
      version: "0.0.1",
      description: "fake",
      artifact: "/fake.onnx",
      task: "embedding",
      inputShape: { kind: "features", dim: 3 },
      runtime: async () => runtime,
    });
    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);
    const out = await adapter.embed({
      kind: "features",
      features: [[1, 2, 3]],
    });
    expect(out.dim).toBe(8);
    expect(out.vector).toHaveLength(8);
    expect(calls.runs).toBe(1);
    await adapter.unload();
    expect(adapter.isLoaded()).toBe(false);
    expect(calls.released).toBe(1);
  });

  it("rejects mismatched feature dim", async () => {
    const { runtime } = makeFakeRuntime();
    const adapter = new ONNXAdapter({
      id: "t",
      name: "t",
      version: "0",
      description: "",
      artifact: "/fake.onnx",
      task: "embedding",
      inputShape: { kind: "features", dim: 5 },
      runtime: async () => runtime,
    });
    await adapter.load();
    await expect(
      adapter.embed({ kind: "features", features: [[1, 2, 3]] }),
    ).rejects.toThrow(/expected 5 features/);
  });

  it("builds [1,C,T] tensor for raw windows", async () => {
    const { runtime, calls } = makeFakeRuntime({ outputDim: 2 });
    const adapter = new ONNXAdapter({
      id: "t",
      name: "t",
      version: "0",
      description: "",
      artifact: "/fake.onnx",
      task: "embedding",
      inputShape: { kind: "raw", channels: 2, samples: 4 },
      runtime: async () => runtime,
    });
    await adapter.load();
    const out = await adapter.embed({
      kind: "windows",
      windows: [
        {
          data: [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ],
          sampleRate: 100,
          start: 0,
          end: 4,
        },
      ],
    });
    expect(out.vector).toHaveLength(2);
    expect(calls.runs).toBe(1);
  });

  it("isONNXRuntimeAvailable returns true for a working runtime", async () => {
    const { runtime } = makeFakeRuntime();
    await expect(isONNXRuntimeAvailable(async () => runtime)).resolves.toBe(
      true,
    );
  });

  it("isONNXRuntimeAvailable returns false when import fails", async () => {
    await expect(
      isONNXRuntimeAvailable(async () => {
        throw new Error("no wasm");
      }),
    ).resolves.toBe(false);
  });
});