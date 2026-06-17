import { describe, it, expect } from "vitest";
import { embed } from "../index";
import { registerModel, unregisterModel } from "../../models/registry";
import { ONNXAdapter } from "../../adapters/onnx-adapter";

function makeWindowInput() {
  const C = 2;
  const T = 256;
  const sampleRate = 128;
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * (8 + c) * t) / sampleRate);
    data.push(ch);
  }
  return {
    kind: "windows" as const,
    windows: [{ data, sampleRate, start: 0, end: T }],
  };
}

describe("embed() facade", () => {
  it("falls back to PCA when an unknown model id is requested", async () => {
    const out = await embed(makeWindowInput(), { modelId: "does-not-exist" });
    expect(out.fellBack).toBe(true);
    expect(out.modelId).toBe("pca-legacy-v1");
    expect(out.vector.length).toBeGreaterThan(0);
  });

  it("falls back to PCA when an ONNX adapter fails to load", async () => {
    registerModel(
      () =>
        new ONNXAdapter({
          id: "broken-onnx",
          name: "Broken",
          version: "0",
          description: "",
          artifact: "/missing.onnx",
          task: "embedding",
          inputShape: { kind: "features", dim: 5 },
          runtime: async () => {
            throw new Error("runtime unavailable");
          },
        }),
    );
    try {
      const out = await embed(makeWindowInput(), { modelId: "broken-onnx" });
      expect(out.fellBack).toBe(true);
      expect(out.fallbackReason).toMatch(/runtime unavailable/);
      expect(out.modelId).toBe("pca-legacy-v1");
    } finally {
      unregisterModel("broken-onnx");
    }
  });

  it("propagates errors when fallbackToPCA is false", async () => {
    await expect(
      embed(makeWindowInput(), { modelId: "still-unknown", fallbackToPCA: false }),
    ).rejects.toThrow(/Unknown model id/);
  });
});