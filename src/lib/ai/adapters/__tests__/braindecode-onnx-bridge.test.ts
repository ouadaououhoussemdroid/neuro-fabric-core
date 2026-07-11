import { describe, it, expect } from "vitest";
import { createONNXBraindecodeBridge } from "../braindecode-onnx-bridge";
import { BraindecodeAdapter } from "../braindecode-adapter";
import { __resetONNXCapabilityProbe, type OrtRuntime } from "../onnx-adapter";
import { embedEEG } from "../../inference/embed-eeg";
import { registerModel, unregisterModel } from "../../models/registry";
import type { ModelInput } from "../../types";

function fakeRuntime(embeddingDim: number): () => Promise<OrtRuntime> {
  return async () => ({
    InferenceSession: {
      async create() {
        return {
          inputNames: ["input"],
          outputNames: ["embedding"],
          async run() {
            return {
              embedding: {
                data: Float32Array.from(Array.from({ length: embeddingDim }, (_, i) => i + 1)),
                dims: [1, embeddingDim],
              },
            };
          },
          async release() {},
        };
      },
    },
    Tensor: function (_t: string, data: Float32Array, dims: readonly number[]) {
      return { data, dims };
    } as unknown as OrtRuntime["Tensor"],
  });
}

function makeInput(C: number, T: number, sr: number): ModelInput {
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / T);
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: T }] };
}

describe("ONNX-backed Braindecode bridge", () => {
  it("loads, forwards, and returns an embedding through ONNX", async () => {
    __resetONNXCapabilityProbe();
    const bridge = createONNXBraindecodeBridge({
      artifact: new Uint8Array([1, 2, 3]),
      architecture: "EEGNetv4",
      channels: 4,
      sampleRate: 128,
      windowSamples: 256,
      embeddingDim: 8,
      runtime: fakeRuntime(8),
    });
    expect(await bridge.isAvailable()).toBe(true);
    await bridge.load({
      architecture: "EEGNetv4",
      channels: 4,
      sampleRate: 128,
      windowSamples: 256,
    });
    const out = await bridge.forward(
      Array.from({ length: 4 }, () => Array.from({ length: 256 }, (_, i) => i)),
    );
    expect(out.embedding).toHaveLength(8);
    await bridge.unload();
  });

  it("embedEEG cascades through ONNX-Braindecode → PCA when bridge fails", async () => {
    __resetONNXCapabilityProbe();
    registerModel(
      () =>
        new BraindecodeAdapter({
          id: "braindecode-eegnetv4-onnx",
          architecture: "EEGNetv4",
          channels: 4,
          sampleRate: 128,
          windowSamples: 256,
          bridge: () => ({
            async isAvailable() {
              return true;
            },
            async load() {},
            async forward() {
              throw new Error("simulated ONNX failure");
            },
            async unload() {},
          }),
        }),
    );
    try {
      const res = await embedEEG(makeInput(4, 256, 128), {
        preferredModelId: "braindecode-eegnetv4-onnx",
      });
      expect(res.fellBack).toBe(true);
      expect(res.modelId).toBe("pca-legacy-v1");
      expect(res.fallbackReason).toMatch(/simulated ONNX failure/);
      expect(res.normalized).toBe(true);
    } finally {
      unregisterModel("braindecode-eegnetv4-onnx");
    }
  });

  it("embedEEG uses PCA directly when no foundation model is registered", async () => {
    const res = await embedEEG(makeInput(2, 256, 128));
    expect(res.modelId).toBe("pca-legacy-v1");
    expect(res.vector.length).toBeGreaterThan(0);
  });
});
