import { describe, it, expect } from "vitest";
import { embed } from "../index";
import { registerModel, unregisterModel } from "../../models/registry";
import { ONNXAdapter } from "../../adapters/onnx-adapter";

const FAKE_FEATURES = [[0.1, 0.2, 0.3, 0.4, 0.5]];

describe("embed() facade", () => {
  it("falls back to PCA when an unknown model id is requested", async () => {
    const out = await embed(
      { kind: "features", features: FAKE_FEATURES },
      { modelId: "does-not-exist" },
    );
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
      const out = await embed(
        { kind: "features", features: FAKE_FEATURES },
        { modelId: "broken-onnx" },
      );
      expect(out.fellBack).toBe(true);
      expect(out.fallbackReason).toMatch(/runtime unavailable/);
      expect(out.modelId).toBe("pca-legacy-v1");
    } finally {
      unregisterModel("broken-onnx");
    }
  });

  it("propagates errors when fallbackToPCA is false", async () => {
    await expect(
      embed(
        { kind: "features", features: FAKE_FEATURES },
        { modelId: "still-unknown", fallbackToPCA: false },
      ),
    ).rejects.toThrow(/Unknown model id/);
  });
});