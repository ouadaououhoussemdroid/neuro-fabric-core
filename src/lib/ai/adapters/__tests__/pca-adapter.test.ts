/**
 * T-014 — Regression tests for the PCA legacy embedder adapter.
 *
 * Verifies Fix 2: the PCA adapter emits exactly 32-dim vectors (matching
 * vector(32)), padding with zeros when the underlying embedSignal util
 * produces fewer dims (e.g. low-channel inputs where featureDim < 32).
 */
import { describe, it, expect } from "vitest";
import { PCAEmbeddingAdapter } from "../pca-adapter";
import type { EEGWindow } from "../../../eeg/types";

function makeWindows(channels: number, count: number, sr = 250, samples = 500): EEGWindow[] {
  return Array.from({ length: count }, (_, i) => ({
    data: Array.from({ length: channels }, (_, c) =>
      Array.from({ length: samples }, (_, t) => Math.sin((2 * Math.PI * (10 + c) * t) / sr)),
    ),
    sampleRate: sr,
    start: 0,
    end: samples,
  }));
}

describe("PCAEmbeddingAdapter (canonical 32-D)", () => {
  it("descriptor declares embeddingDim=32", () => {
    const a = new PCAEmbeddingAdapter();
    expect(a.descriptor.capabilities.embeddingDim).toBe(32);
  });

  it("pads low-channel output to exactly 32 dims", async () => {
    // 3 channels → featureDim = 15. PCA returns k=min(32,15)=15 dims.
    // The adapter must pad to 32.
    const a = new PCAEmbeddingAdapter();
    const windows = makeWindows(3, 5); // enough for PCA path (>=4 windows)
    const out = await a.embed({ kind: "windows", windows });
    expect(out.dim).toBe(32);
    expect(out.vector).toHaveLength(32);
    expect(out.modelId).toBe("pca-legacy-v1");
  });

  it("truncates high-channel output to exactly 32 dims", async () => {
    // 8 channels → featureDim = 40 > 32. AE path produces 32 (latentDim=32).
    const a = new PCAEmbeddingAdapter();
    const windows = makeWindows(8, 40); // enough for AE path
    const out = await a.embed({ kind: "windows", windows });
    expect(out.dim).toBe(32);
    expect(out.vector).toHaveLength(32);
  });
});
