import { describe, it, expect } from "vitest";
import { runRecallSLO, formatSLOAlert, DEFAULT_SLO_CONFIG } from "../recall-slo";
import type { SLOSample } from "../recall-slo";

function makeSamples(): SLOSample[] {
  const samples: SLOSample[] = [];
  // Class 0: clustered near [1, 0, 0, 0]
  for (let i = 0; i < 20; i++) {
    samples.push({
      id: `a${i}`,
      embedding: [1 + i * 0.01, 0, 0, 0],
      label: 0,
      modelId: "test-model",
    });
  }
  // Class 1: clustered near [0, 1, 0, 0]
  for (let i = 0; i < 20; i++) {
    samples.push({
      id: `b${i}`,
      embedding: [0, 1 + i * 0.01, 0, 0],
      label: 1,
      modelId: "test-model",
    });
  }
  return samples;
}

describe("runRecallSLO", () => {
  it("passes when ANN recall meets the threshold and ratio floor", () => {
    const samples = makeSamples();
    // ANN recall matches brute-force → ratio = 1.0
    const report = runRecallSLO(samples, 1.0);
    expect(report.passed).toBe(true);
    expect(report.alert).toBeUndefined();
    expect(report.bruteForceRecall).toBe(1.0);
    expect(report.annRecallRatio).toBe(1.0);
  });

  it("fails when ANN recall is below the threshold", () => {
    const samples = makeSamples();
    const report = runRecallSLO(samples, 0.5, { threshold: 0.85 });
    expect(report.passed).toBe(false);
    expect(report.alert).toContain("SLO failed");
  });

  it("fails when ANN recall ratio is below the floor even if recall is high", () => {
    const samples = makeSamples();
    // ANN recall = 0.8, brute-force = 1.0 → ratio = 0.8 < 0.95 floor
    const report = runRecallSLO(samples, 0.8, { annRecallRatioFloor: 0.95 });
    expect(report.passed).toBe(false);
    expect(report.annRecallRatio).toBeCloseTo(0.8, 2);
  });

  it("handles empty samples gracefully", () => {
    const report = runRecallSLO([], 0);
    expect(report.n).toBe(0);
    expect(report.passed).toBe(false);
  });

  it("produces a per-model breakdown", () => {
    const samples = makeSamples();
    // Add a second model with its own samples.
    samples.push(
      ...Array.from({ length: 10 }, (_, i) => ({
        id: `c${i}`,
        embedding: [0, 0, 1 + i * 0.01, 0],
        label: 2,
        modelId: "other-model",
      })),
    );
    const report = runRecallSLO(samples, 0.9);
    expect(report.perModel).toHaveLength(2);
    expect(report.perModel[0].modelId).toBe("test-model");
    expect(report.perModel[1].modelId).toBe("other-model");
  });

  it("uses the default SLO config when none provided", () => {
    expect(DEFAULT_SLO_CONFIG.k).toBe(10);
    expect(DEFAULT_SLO_CONFIG.threshold).toBe(0.85);
    expect(DEFAULT_SLO_CONFIG.annRecallRatioFloor).toBe(0.95);
  });
});

describe("formatSLOAlert", () => {
  it("includes the key metrics in the alert string", () => {
    const samples = makeSamples();
    const report = runRecallSLO(samples, 0.5, { threshold: 0.85 });
    const alert = formatSLOAlert(report);
    expect(alert).toContain("pgvector=");
    expect(alert).toContain("bruteForce=");
    expect(alert).toContain("ratio=");
    expect(alert).toContain("threshold=");
  });
});
