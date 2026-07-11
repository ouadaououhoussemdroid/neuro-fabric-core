import { describe, it, expect } from "vitest";
import { embedSignal, extractFeatureMatrix } from "../index";
import type { EEGWindow } from "../../eeg/types";

const FS = 250;
const N = 500;

function makeWindow(seed: number, sampleRate = FS, n = N): EEGWindow {
  return {
    data: [
      Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * (10 + seed) * i) / sampleRate)),
      Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * (20 + seed) * i) / sampleRate)),
    ],
    sampleRate,
    start: 0,
    end: n,
  };
}

describe("extractFeatureMatrix", () => {
  it("maps each window to a feature row via bandPowerFeatures", () => {
    const windows = [makeWindow(0), makeWindow(1), makeWindow(2)];
    const matrix = extractFeatureMatrix(windows);
    expect(matrix).toHaveLength(3);
    matrix.forEach((row) => expect(row).toHaveLength(10)); // 2 channels x 5 bands
  });
});

describe("embedSignal", () => {
  it("throws on an empty windows array", () => {
    expect(() => embedSignal([])).toThrow("no windows");
  });

  it("falls back to raw-bandpower when there are too few windows for a stable PCA fit", () => {
    // Only 2 windows, well under the default latentDim=64 threshold
    const windows = [makeWindow(0), makeWindow(1)];
    const result = embedSignal(windows, 64);
    expect(result.model).toBe("raw-bandpower");
    expect(result.dimensions).toBe(result.featureDim);
    expect(result.vector).toHaveLength(10); // featureDim for 2 channels
  });

  it("mean-pools the raw-bandpower fallback vector across windows", () => {
    // Two windows with the same content -> pooled vector should equal
    // either window's own feature vector.
    const w = makeWindow(0);
    const result = embedSignal([w, w], 64);
    expect(result.model).toBe("raw-bandpower");
    // Recompute the single-window features independently and compare
    const singleFeatures = extractFeatureMatrix([w])[0];
    result.vector.forEach((v, i) => expect(v).toBeCloseTo(singleFeatures[i], 8));
  });

  it("uses the linear-ae (PCA) path once there are enough windows and latentDim < featureDim", () => {
    // featureDim = 10 (2ch x 5 bands); need featureDim > latentDim and
    // enough windows (>= max(latentDim, 4)) to take the PCA branch.
    const latentDim = 3;
    const windows = Array.from({ length: 10 }, (_, i) => makeWindow(i));
    const result = embedSignal(windows, latentDim);
    expect(result.model).toBe("linear-ae");
    expect(result.dimensions).toBe(latentDim);
    expect(result.vector).toHaveLength(latentDim);
    expect(result.featureDim).toBe(10);
  });

  it("reports a non-negative durationMs", () => {
    const result = embedSignal([makeWindow(0), makeWindow(1)], 64);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });
});
