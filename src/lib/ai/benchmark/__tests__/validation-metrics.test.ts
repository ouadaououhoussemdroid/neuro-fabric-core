import { describe, it, expect } from "vitest";
import {
  cosineMatrix,
  recallAtK,
  intraInterClassCosine,
  validateEmbeddings,
} from "../validation-metrics";

// Build embeddings where class A and B are clearly separated.
function buildEmbeddings(): { emb: number[][]; labels: number[] } {
  const emb: number[][] = [];
  const labels: number[] = [];
  // Class 0: vectors near [1, 0, 0, 0]
  for (let i = 0; i < 10; i++) {
    emb.push([1 + i * 0.01, 0.01 * i, 0, 0]);
    labels.push(0);
  }
  // Class 1: vectors near [0, 1, 0, 0]
  for (let i = 0; i < 10; i++) {
    emb.push([0.01 * i, 1 + i * 0.01, 0, 0]);
    labels.push(1);
  }
  return { emb, labels };
}

describe("cosineMatrix", () => {
  it("produces a symmetric matrix with diagonal ≈ 1", () => {
    const { emb } = buildEmbeddings();
    const sim = cosineMatrix(emb);
    expect(sim).toHaveLength(20);
    expect(sim[0]).toHaveLength(20);
    expect(sim[0][0]).toBeCloseTo(1, 5);
    expect(sim[0][1]).toBeCloseTo(sim[1][0], 10);
  });
});

describe("recallAtK", () => {
  it("returns 1.0 for well-separated clusters with k=1", () => {
    const { emb, labels } = buildEmbeddings();
    const r = recallAtK(emb, labels, 1);
    expect(r).toBeGreaterThan(0.9);
  });

  it("returns lower recall for k=1 on random embeddings", () => {
    const emb = Array.from({ length: 20 }, () => Array.from({ length: 4 }, () => Math.random()));
    const labels = Array.from({ length: 20 }, (_, i) => (i < 10 ? 0 : 1));
    const r = recallAtK(emb, labels, 1);
    expect(r).toBeLessThan(0.9);
  });

  it("returns 0 for an empty array", () => {
    expect(recallAtK([], [], 10)).toBe(0);
  });
});

describe("intraInterClassCosine", () => {
  it("intra-class cosine > inter-class cosine for separated clusters", () => {
    const { emb, labels } = buildEmbeddings();
    const stats = intraInterClassCosine(emb, labels);
    expect(stats.intraMean).toBeGreaterThan(stats.interMean);
    expect(stats.separationMargin).toBeGreaterThan(0);
    expect(stats.nIntraPairs).toBeGreaterThan(0);
    expect(stats.nInterPairs).toBeGreaterThan(0);
  });
});

describe("validateEmbeddings", () => {
  it("produces a full report with beatsPca=true for clearly separated data", () => {
    const { emb, labels } = buildEmbeddings();
    const report = validateEmbeddings(emb, labels, 5);
    expect(report.n).toBe(20);
    expect(report.nClasses).toBe(2);
    expect(report.embeddingDim).toBe(4);
    expect(report.cosineAnalysis.separationMargin).toBeGreaterThan(0);
    expect(report.normMean).toBeGreaterThan(0);
    expect(typeof report.beatsPca).toBe("boolean");
    expect(report.pcaBaseline.pcaDim).toBeLessThanOrEqual(4);
  });

  it("recallAtK key matches the k argument", () => {
    const { emb, labels } = buildEmbeddings();
    const report = validateEmbeddings(emb, labels, 7);
    expect(report.recallAtK).toHaveProperty("7");
  });
});
