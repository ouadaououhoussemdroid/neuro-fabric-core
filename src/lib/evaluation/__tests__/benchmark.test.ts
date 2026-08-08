/**
 * T-028 — Tests for the Benchmark Comparison module.
 *
 * Validates: Fisher's Linear Discriminant, model comparison, and
 * benchmark result structure.
 */
import { describe, it, expect } from "vitest";
import {
  fisherLinearDiscriminant,
  compareModels,
  type BenchmarkResult,
  FOUNDATION_MODELS,
  type FoundationModelSpec,
} from "../benchmark";

/** Generate a synthetic embedding vector of the given dimension, seeded. */
function makeEmbedding(dim: number, seed: number): number[] {
  return Array.from({ length: dim }, (_, i) => {
    let s = seed * 101 + i * 37;
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return ((s / 0x7fffffff) * 2 - 1) / Math.sqrt(dim);
  });
}

describe("fisherLinearDiscriminant", () => {
  it("returns 0 for empty input", () => {
    const result = fisherLinearDiscriminant([], []);
    expect(result.score).toBe(0);
    expect(result.nClasses).toBe(0);
    expect(result.n).toBe(0);
  });

  it("returns 0 for single class", () => {
    const emb = [makeEmbedding(4, 1), makeEmbedding(4, 2), makeEmbedding(4, 3)];
    const result = fisherLinearDiscriminant(emb, [0, 0, 0]);
    expect(result.nClasses).toBe(1);
    expect(result.score).toBe(0);
  });

  it("higher score for well-separated classes", () => {
    // Class 0: all-positive embeddings; Class 1: all-negative.
    const emb0 = Array.from({ length: 10 }, (_, i) => makeEmbedding(8, i + 1).map((v) => v + 1.0));
    const emb1 = Array.from({ length: 10 }, (_, i) =>
      makeEmbedding(8, i + 100).map((v) => v - 1.0),
    );
    const result = fisherLinearDiscriminant(
      [...emb0, ...emb1],
      [...Array(10).fill(0), ...Array(10).fill(1)],
    );
    expect(result.nClasses).toBe(2);
    expect(result.n).toBe(20);
    expect(result.score).toBeGreaterThan(1);
  });

  it("lower score for overlapping classes", () => {
    // Both classes use random embeddings → no class structure.
    const emb = Array.from({ length: 20 }, (_, i) => makeEmbedding(8, i + 1));
    const result = fisherLinearDiscriminant(emb, Array(10).fill(0).concat(Array(10).fill(1)));
    expect(result.nClasses).toBe(2);
    expect(result.score).toBeLessThan(1);
  });
});

describe("compareModels", () => {
  it("computes paired comparisons across benchmark folds", () => {
    // Simulate 5 folds of benchmark results for two models.
    const resultsA: BenchmarkResult[] = Array.from({ length: 5 }, (_, i) => ({
      dataset: `ds-${i}`,
      modelId: "model-a",
      nSamples: 10,
      nClasses: 4,
      embeddingDim: 32,
      accuracy: 0.7 + i * 0.02,
      f1: 0.68 + i * 0.02,
      recallAtK: 0.75 + i * 0.01,
      pcaBaselineRecallAt1: 0.4,
      beatsPca: true,
      fisherScore: 2.5 + i * 0.1,
      cosineSeparation: 0.3,
      intraClassCosine: { mean: 0.5, std: 0.1 },
      interClassCosine: { mean: 0.1, std: 0.05 },
      latencyMs: 50,
    }));
    const resultsB: BenchmarkResult[] = Array.from({ length: 5 }, (_, i) => ({
      dataset: `ds-${i}`,
      modelId: "model-b",
      nSamples: 10,
      nClasses: 4,
      embeddingDim: 32,
      accuracy: 0.55 + i * 0.01,
      f1: 0.52 + i * 0.01,
      recallAtK: 0.6 + i * 0.01,
      pcaBaselineRecallAt1: 0.4,
      beatsPca: true,
      fisherScore: 1.5 + i * 0.05,
      cosineSeparation: 0.2,
      intraClassCosine: { mean: 0.4, std: 0.1 },
      interClassCosine: { mean: 0.2, std: 0.05 },
      latencyMs: 35,
    }));

    const comparisons = compareModels(resultsA, resultsB, "test-dataset");
    expect(comparisons.length).toBe(4); // accuracy, recallAtK, f1, fisherScore

    // model-a should be significantly better than model-b on accuracy.
    const accComparison = comparisons.find((c) => c.metric === "accuracy");
    expect(accComparison).toBeDefined();
    expect(accComparison!.modelA).toBe("model-a");
    expect(accComparison!.modelB).toBe("model-b");
    expect(accComparison!.meanDiff).toBeGreaterThan(0.1);
    expect(accComparison!.cohensD.d).toBeGreaterThan(0.8);
  });

  it("handles identical results (d=0, no significance)", () => {
    const results: BenchmarkResult[] = Array.from({ length: 5 }, (_, i) => ({
      dataset: `ds-${i}`,
      modelId: "m1",
      nSamples: 10,
      nClasses: 2,
      embeddingDim: 32,
      accuracy: 0.5,
      f1: 0.5,
      recallAtK: 0.5,
      pcaBaselineRecallAt1: 0.5,
      beatsPca: false,
      fisherScore: 1.0,
      cosineSeparation: 0,
      intraClassCosine: { mean: 0, std: 0 },
      interClassCosine: { mean: 0, std: 0 },
      latencyMs: 50,
    }));
    const comparisons = compareModels(results, JSON.parse(JSON.stringify(results)), "ds");
    expect(comparisons.length).toBe(4);
    for (const c of comparisons) {
      expect(c.meanDiff).toBeCloseTo(0, 10);
      expect(c.tTest.pValue).toBeGreaterThan(0);
    }
  });
});

describe("FOUNDATION_MODELS", () => {
  it("registers all four foundation models", () => {
    const ids = FOUNDATION_MODELS.map((m) => m.id);
    expect(ids).toContain("braindecode-eegconformer-prod");
    expect(ids).toContain("onnx-eegpt");
    expect(ids).toContain("onnx-femba-tiny");
    expect(ids).toContain("onnx-labram");
    expect(ids).toContain("onnx-cbramod");
  });

  it("every spec has required fields", () => {
    for (const spec of FOUNDATION_MODELS) {
      expect(spec.id).toBeTruthy();
      expect(spec.name).toBeTruthy();
      expect(spec.repo).toMatch(/^https:\/\//);
      expect(spec.paper).toMatch(/^https:\/\/arxiv\.org/);
      expect(spec.license).toBeTruthy();
      expect(spec.checkpointUrl).toBeTruthy();
      expect(spec.embeddingDim).toBeGreaterThan(0);
      expect(spec.channels).toBeGreaterThan(0);
      expect(spec.sampleRate).toBeGreaterThan(0);
      expect(spec.windowSamples).toBeGreaterThan(0);
      expect(spec.onnxPath).toBeTruthy();
      expect(spec.onnxOpset).toBeGreaterThanOrEqual(17);
      expect(spec.modelSizeMB).toBeGreaterThan(0);
    }
  });

  it("marks CBraMod as non-WASM compatible due to DFT", () => {
    const cbramod = FOUNDATION_MODELS.find((m) => m.id === "onnx-cbramod");
    expect(cbramod).toBeDefined();
    expect(cbramod!.wasmCompatible).toBe(false);
    expect(cbramod!.wasmBlockers).toContain(
      "DFT (Discrete Fourier Transform) — not supported in ORT-WASM",
    );
  });

  it("marks FEMBA-tiny as FP16 (INT8 blocked by recurrent scan)", () => {
    const femba = FOUNDATION_MODELS.find((m) => m.id === "onnx-femba-tiny");
    expect(femba).toBeDefined();
    expect(femba!.quantizeFormat).toBe("fp16");
    expect(femba!.wasmCompatible).toBe(true);
  });

  it("marks EEGPT as INT8 (ViT, no recurrent scan)", () => {
    const eegpt = FOUNDATION_MODELS.find((m) => m.id === "onnx-eegpt");
    expect(eegpt).toBeDefined();
    expect(eegpt!.quantizeFormat).toBe("int8");
    expect(eegpt!.wasmCompatible).toBe(true);
  });
});
