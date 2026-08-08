/**
 * T-028 — Tests for the statistical reporting module.
 *
 * Validates the mathematical primitives (incomplete beta, t-distribution),
 * descriptive statistics, confidence intervals, Cohen's d, t-tests, and
 * publication table generation against known closed-form values and
 * scipy-compatible results.
 */
import { describe, it, expect } from "vitest";
import {
  logGamma,
  incompleteBeta,
  inverseIncompleteBeta,
  tTestPValue,
  tCriticalValue,
  describe as describeStats,
  confidenceInterval,
  cohensD,
  interpretCohensD,
  tTestOneSample,
  tTestTwoSample,
  formatPValue,
  publicationTable,
} from "../index";

describe("logGamma", () => {
  it("matches known values", () => {
    // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(5) = 24
    expect(Math.exp(logGamma(1))).toBeCloseTo(1, 10);
    expect(Math.exp(logGamma(2))).toBeCloseTo(1, 10);
    expect(Math.exp(logGamma(3))).toBeCloseTo(2, 10);
    expect(Math.exp(logGamma(5))).toBeCloseTo(24, 10);
  });

  it("Γ(0.5) = sqrt(π)", () => {
    expect(Math.exp(logGamma(0.5))).toBeCloseTo(Math.sqrt(Math.PI), 10);
  });

  it("satisfies Γ(x+1) = x·Γ(x)", () => {
    // Γ(4.5) = 3.5 * Γ(3.5) = 3.5 * 2.5 * Γ(2.5) = ...
    const g45 = Math.exp(logGamma(4.5));
    const g35 = Math.exp(logGamma(3.5));
    expect(g45).toBeCloseTo(3.5 * g35, 10);
  });
});

describe("incompleteBeta", () => {
  it("returns 0 for x=0 and 1 for x=1", () => {
    expect(incompleteBeta(1, 1, 0)).toBe(0);
    expect(incompleteBeta(1, 1, 1)).toBe(1);
    expect(incompleteBeta(2, 3, 0)).toBe(0);
    expect(incompleteBeta(2, 3, 1)).toBe(1);
  });

  // I_x(a, b) for a=b=1 is just x (uniform distribution).
  it("I_x(1,1) = x (uniform)", () => {
    for (const x of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(incompleteBeta(1, 1, x)).toBeCloseTo(x, 10);
    }
  });

  // I_0.5(1,1) = 0.5
  it("symmetric case I_0.5(a,a) = 0.5", () => {
    expect(incompleteBeta(2, 2, 0.5)).toBeCloseTo(0.5, 10);
    expect(incompleteBeta(5, 5, 0.5)).toBeCloseTo(0.5, 10);
  });
});

describe("inverseIncompleteBeta", () => {
  it("round-trips with incompleteBeta", () => {
    for (const { a, b, x } of [
      { a: 1, b: 1, x: 0.3 },
      { a: 2, b: 3, x: 0.5 },
      { a: 5, b: 2, x: 0.2 },
      { a: 10, b: 10, x: 0.7 },
    ]) {
      const p = incompleteBeta(a, b, x);
      const xBack = inverseIncompleteBeta(p, a, b);
      expect(xBack).toBeCloseTo(x, 8);
    }
  });
});

describe("t-distribution functions", () => {
  it("t-test p-value for t=0 should be 1.0", () => {
    expect(tTestPValue(0, 10)).toBeCloseTo(1, 10);
    expect(tTestPValue(0, 100)).toBeCloseTo(1, 10);
  });

  it("two-tailed p-value is symmetric", () => {
    expect(tTestPValue(2.5, 20)).toBeCloseTo(tTestPValue(-2.5, 20), 10);
  });

  // t=2.228 with df=10 → two-tailed p ≈ 0.05 (critical value for 0.05 two-tailed)
  it("critical t-value for 95% CI with df=10 matches known value", () => {
    const tCrit = tCriticalValue(0.95, 10);
    expect(tCrit).toBeCloseTo(2.228, 2);
  });

  // t=1.984 with df=100 → two-tailed p ≈ 0.05
  it("critical t-value for 95% CI with df=100 matches known value", () => {
    const tCrit = tCriticalValue(0.95, 100);
    expect(tCrit).toBeCloseTo(1.984, 2);
  });

  // t=2.776 with df=4 → two-tailed p ≈ 0.05
  it("critical t-value for 95% CI with df=4 matches known value", () => {
    const tCrit = tCriticalValue(0.95, 4);
    expect(tCrit).toBeCloseTo(2.776, 2);
  });

  it("p-value matches scipy for t=3.2, df=15", () => {
    // scipy: 2 * (1 - t.cdf(3.2, df=15)) ≈ 0.00596
    expect(tTestPValue(3.2, 15)).toBeCloseTo(0.00596, 4);
  });
});

describe("describe (descriptive statistics)", () => {
  it("computes correct mean, std, and sum", () => {
    const values = [1, 2, 3, 4, 5];
    const stats = describeStats(values);
    expect(stats.n).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.sum).toBe(15);
    expect(stats.median).toBe(3);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
    // Sample std (n-1 denominator) = sqrt(2.5) ≈ 1.5811
    expect(stats.std).toBeCloseTo(Math.sqrt(2.5), 5);
    expect(stats.variance).toBeCloseTo(2.5, 5);
    expect(stats.stdErr).toBeCloseTo(Math.sqrt(2.5 / 5), 5);
  });

  it("returns zeros for single element", () => {
    const stats = describeStats([42]);
    expect(stats.n).toBe(1);
    expect(stats.mean).toBe(42);
    expect(stats.std).toBe(0);
  });

  it("handles empty arrays", () => {
    const stats = describeStats([]);
    expect(stats.n).toBe(0);
    expect(stats.mean).toBeNaN();
  });

  it("computes correct median for even-length array", () => {
    const stats = describeStats([1, 2, 3, 4]);
    expect(stats.median).toBe(2.5);
    expect(stats.n).toBe(4);
  });
});

describe("confidenceInterval", () => {
  it("CI brackets the mean and has correct width", () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const ci = confidenceInterval(values, 0.95);
    expect(ci.mean).toBe(5.5);
    expect(ci.lower).toBeLessThan(ci.mean);
    expect(ci.upper).toBeGreaterThan(ci.mean);
    expect(ci.confidence).toBe(0.95);
    expect(ci.margin).toBeGreaterThan(0);
  });

  it("CI width decreases with larger sample size", () => {
    const small = Array.from({ length: 10 }, () => Math.random());
    const large = Array.from({ length: 1000 }, () => Math.random());
    const ciSmall = confidenceInterval(small, 0.95);
    const ciLarge = confidenceInterval(large, 0.95);
    expect(ciLarge.margin).toBeLessThan(ciSmall.margin);
  });

  it("95% CI for normal data with known mean matches expected range", () => {
    // Generate data with mean 50, std 10
    const rng = (seed: number) => {
      let s = seed;
      return () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return (s / 0x7fffffff) * 2 - 1;
      };
    };
    const rand = rng(42);
    const values = Array.from({ length: 1000 }, () => 50 + rand() * 10);
    const ci = confidenceInterval(values, 0.95);
    expect(ci.mean).toBeCloseTo(50, 0);
    expect(ci.lower).toBeCloseTo(49.4, 0);
    expect(ci.upper).toBeCloseTo(50.6, 0);
  });
});

describe("cohensD", () => {
  it("returns d=0 for identical distributions", () => {
    const g1 = [1, 2, 3, 4, 5];
    const g2 = [1, 2, 3, 4, 5];
    const result = cohensD(g1, g2);
    expect(result.d).toBeCloseTo(0, 10);
    expect(result.interpretation).toBe("negligible");
  });

  it("returns large d for well-separated distributions", () => {
    const g1 = [10, 11, 12, 13, 14, 15];
    const g2 = [1, 2, 3, 4, 5, 6];
    const result = cohensD(g1, g2);
    expect(result.d).toBeGreaterThan(0.8);
    expect(result.interpretation).toBe("large");
  });

  it("interpretCohensD applies conventional thresholds", () => {
    expect(interpretCohensD(0.1)).toBe("negligible");
    expect(interpretCohensD(0.3)).toBe("small");
    expect(interpretCohensD(0.6)).toBe("medium");
    expect(interpretCohensD(1.0)).toBe("large");
    // Negative d should be interpreted by absolute value
    expect(interpretCohensD(-0.6)).toBe("medium");
  });
});

describe("tTestOneSample", () => {
  it("detects significant difference from chance (0.5)", () => {
    // Data clearly above 0.5
    const values = [0.7, 0.8, 0.75, 0.82, 0.78, 0.81, 0.77, 0.79, 0.8, 0.76];
    const result = tTestOneSample(values, 0.5);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.significant).toBe(true);
    expect(result.t).toBeGreaterThan(0);
  });

  it("does not reject when mean equals mu0", () => {
    // Data centered exactly at mu0 with variance
    const values = [0.4, 0.6, 0.5, 0.5, 0.4, 0.6, 0.5, 0.5];
    const result = tTestOneSample(values, 0.5);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.significant).toBe(false);
  });

  it("p-value matches known scipy result", () => {
    // scipy.stats.ttest_1samp([5, 6, 7, 8, 9], 6)
    // mean=7, sample std=sqrt(2.5), SEM=sqrt(2.5/5)=0.7071, t=1/0.7071=1.4142
    // p ≈ 0.2302 (df=4, two-tailed)
    const result = tTestOneSample([5, 6, 7, 8, 9], 6);
    expect(result.meanDiff).toBeCloseTo(1, 10);
    expect(result.t).toBeCloseTo(1.4142, 3);
    expect(result.pValue).toBeCloseTo(0.2302, 3);
    expect(result.significant).toBe(false);
  });
});

describe("tTestTwoSample", () => {
  it("detects significant difference between groups", () => {
    const group1 = Array.from({ length: 50 }, () => 0.8 + Math.random() * 0.1);
    const group2 = Array.from({ length: 50 }, () => 0.5 + Math.random() * 0.1);
    const result = tTestTwoSample(group1, group2);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.significant).toBe(true);
  });

  it("does not detect difference when groups are identical", () => {
    const group1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const group2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = tTestTwoSample(group1, group2);
    expect(result.t).toBeCloseTo(0, 5);
  });
});

describe("formatPValue", () => {
  it("formats p-values with proper thresholds", () => {
    expect(formatPValue(0.0001)).toBe("<0.001");
    expect(formatPValue(0.001)).toBe("<0.01");
    expect(formatPValue(0.01)).toBe("<0.05");
    expect(formatPValue(0.049)).toBe("<0.05");
    expect(formatPValue(0.06)).toBe("0.060");
    expect(formatPValue(0.5)).toBe("0.500");
    expect(formatPValue(1.0)).toBe("1.000");
  });
});

describe("publicationTable", () => {
  it("generates a markdown table with expected columns", () => {
    const metrics = [
      {
        name: "Accuracy",
        n: 10,
        mean: 0.72,
        std: 0.15,
        ciLower: 0.68,
        ciUpper: 0.76,
        pValue: 0.0001,
        effectSize: { d: 0.85, interpretation: "large" as const, pooledStd: 1.0 },
      },
      {
        name: "F1-Score",
        n: 10,
        mean: 0.68,
        std: 0.12,
        ciLower: 0.62,
        ciUpper: 0.74,
        pValue: 0.045,
        effectSize: { d: 0.43, interpretation: "small" as const, pooledStd: 1.0 },
      },
    ];
    const table = publicationTable(metrics);
    expect(table).toContain("Metric");
    expect(table).toContain("Accuracy");
    expect(table).toContain("F1-Score");
    expect(table).toContain("<0.001");
    expect(table).toContain("d=0.85");
    expect(table).toContain("large");
    expect(table).toContain("d=0.43");
    expect(table).toContain("small");
    expect(table).toContain("|");
  });

  it("handles missing p-value and effect size gracefully", () => {
    const metrics = [
      {
        name: "Latency",
        n: 50,
        mean: 0.35,
        std: 0.08,
        ciLower: 0.33,
        ciUpper: 0.37,
      },
    ];
    const table = publicationTable(metrics);
    expect(table).toContain("N/A");
  });
});
