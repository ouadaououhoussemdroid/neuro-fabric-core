/**
 * T-028 — Tests for the Ground Truth Annotation module.
 *
 * Validates: correlation/concordance metrics, label grouping/summarization,
 * and edge-case handling.
 */
import { describe, it, expect } from "vitest";
import {
  correlatePredictions,
  groupLabels,
  summarizeAnnotations,
  type GroundTruthLabel,
  type LabelType,
} from "../ground-truth";

function makeLabel(
  analysisId: string,
  subjectId: string,
  type: LabelType,
  value: number,
): GroundTruthLabel {
  return {
    id: `label-${analysisId}-${type}`,
    analysisId,
    subjectId,
    type,
    value,
    confidence: 0.9,
    annotatorUserId: "annotator-1",
    createdAt: new Date().toISOString(),
  };
}

describe("correlatePredictions", () => {
  it("returns undefined for fewer than 2 samples", () => {
    expect(correlatePredictions([0.5], [0.6])).toBeUndefined();
    expect(correlatePredictions([], [])).toBeUndefined();
  });

  it("perfect correlation yields r=1, ccc=1, low error", () => {
    const pred = [0.1, 0.3, 0.5, 0.7, 0.9];
    const truth = [0.1, 0.3, 0.5, 0.7, 0.9];
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    expect(result!.pearsonR).toBeCloseTo(1, 5);
    expect(result!.ccc).toBeCloseTo(1, 5);
    expect(result!.mae).toBeCloseTo(0, 10);
    expect(result!.rmse).toBeCloseTo(0, 10);
  });

  it("perfect anti-correlation yields r=-1", () => {
    const pred = [0.1, 0.3, 0.5, 0.7, 0.9];
    const truth = [0.9, 0.7, 0.5, 0.3, 0.1];
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    expect(result!.pearsonR).toBeCloseTo(-1, 5);
  });

  it("independent sequences have low correlation", () => {
    // Use many data points so correlation is genuinely near zero.
    const n = 50;
    const pred: number[] = [];
    const truth: number[] = [];
    for (let i = 0; i < n; i++) {
      pred.push((i * 0.017) % 1);
      truth.push((i * 0.037 + 0.5) % 1);
    }
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    expect(Math.abs(result!.pearsonR)).toBeLessThan(0.3);
  });

  it("computes MAE and RMSE correctly", () => {
    const pred = [1.0, 2.0, 3.0];
    const truth = [1.5, 2.0, 2.5];
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    // MAE = (|0.5| + 0 + |0.5|) / 3 = 1/3
    expect(result!.mae).toBeCloseTo(1 / 3, 5);
    // RMSE = sqrt((0.25 + 0 + 0.25) / 3) = sqrt(1/6)
    expect(result!.rmse).toBeCloseTo(Math.sqrt(1 / 6), 5);
  });

  it("bias test detects systematic offset", () => {
    // Predicted values consistently 0.1 above ground truth.
    const pred = [0.5, 0.6, 0.7, 0.8, 0.9];
    const truth = [0.4, 0.5, 0.6, 0.7, 0.8];
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    expect(result!.differenceCI.mean).toBeCloseTo(0.1, 5);
    expect(result!.biasTest.significant).toBe(true);
  });

  it("bias test not significant when predictions are unbiased", () => {
    // Perfectly correlated with small noise.
    const pred = [0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9];
    const truth = [0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9];
    const result = correlatePredictions(pred, truth);
    expect(result).toBeDefined();
    expect(result!.biasTest.pValue).toBeGreaterThanOrEqual(0);
    expect(result!.biasTest.pValue).toBeLessThanOrEqual(1);
  });
});

describe("groupLabels", () => {
  it("groups by subject then by type", () => {
    const labels = [
      makeLabel("a1", "sub-1", "attention", 0.7),
      makeLabel("a1", "sub-1", "workload", 0.5),
      makeLabel("a2", "sub-2", "attention", 0.8),
      makeLabel("a3", "sub-1", "attention", 0.6),
    ];
    const grouped = groupLabels(labels);
    expect(grouped.size).toBe(2);
    expect(grouped.get("sub-1")!.size).toBe(2);
    expect(grouped.get("sub-1")!.get("attention")!.length).toBe(2);
    expect(grouped.get("sub-1")!.get("workload")!.length).toBe(1);
    expect(grouped.get("sub-2")!.get("attention")!.length).toBe(1);
  });

  it("handles empty input", () => {
    const grouped = groupLabels([]);
    expect(grouped.size).toBe(0);
  });
});

describe("summarizeAnnotations", () => {
  it("computes correct summary statistics", () => {
    const labels = [
      makeLabel("a1", "sub-1", "attention", 0.7),
      makeLabel("a2", "sub-1", "attention", 0.5),
      makeLabel("a3", "sub-1", "attention", 0.9),
      makeLabel("a4", "sub-1", "workload", 0.4),
      makeLabel("a5", "sub-1", "workload", 0.6),
    ];
    const summaries = summarizeAnnotations(labels);
    expect(summaries.length).toBe(2);
    const attention = summaries.find((s) => s.type === "attention");
    expect(attention).toBeDefined();
    expect(attention!.nLabels).toBe(3);
    expect(attention!.meanValue).toBeCloseTo(0.7, 5);
    expect(attention!.stdValue).toBeCloseTo(0.2, 1);
    expect(attention!.meanConfidence).toBeCloseTo(0.9, 5);

    const workload = summaries.find((s) => s.type === "workload");
    expect(workload).toBeDefined();
    expect(workload!.nLabels).toBe(2);
    expect(workload!.meanValue).toBeCloseTo(0.5, 5);
  });

  it("handles empty input", () => {
    const summaries = summarizeAnnotations([]);
    expect(summaries).toEqual([]);
  });

  it("computes time span for temporal annotations", () => {
    const labels: GroundTruthLabel[] = [
      {
        id: "l1",
        analysisId: "a1",
        subjectId: "sub-1",
        type: "attention",
        value: 0.7,
        startSample: 0,
        endSample: 100,
        confidence: 0.9,
        annotatorUserId: "anno-1",
        createdAt: new Date().toISOString(),
      },
      {
        id: "l2",
        analysisId: "a2",
        subjectId: "sub-1",
        type: "attention",
        value: 0.5,
        startSample: 50,
        endSample: 200,
        confidence: 0.8,
        annotatorUserId: "anno-1",
        createdAt: new Date().toISOString(),
      },
    ];
    const summaries = summarizeAnnotations(labels);
    expect(summaries[0].timeSpanSamples).toEqual({ start: 0, end: 200 });
  });

  it("handles annotations without temporal range", () => {
    const labels = [
      makeLabel("a1", "sub-1", "attention", 0.7),
      makeLabel("a2", "sub-1", "attention", 0.5),
    ];
    const summaries = summarizeAnnotations(labels);
    expect(summaries[0].timeSpanSamples).toBeUndefined();
  });
});
