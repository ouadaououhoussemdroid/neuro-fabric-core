/**
 * T-028 — Statistical Reporting module.
 *
 * Provides inferential statistics necessary for scientific credibility
 * (Tier 3): Cohen's d effect sizes, t-tests, confidence intervals,
 * p-value formatting, and publication-ready table generation.
 *
 * All functions are dependency-free (only uses `Math.*`) so they run
 * identically in the browser, Node, and Cloudflare Workers.
 *
 * References:
 * - Numerical Recipes in C (2nd ed.), §6.2–6.4 (incomplete beta, t-distribution)
 * - Lanczos (1964) approximation for log-gamma
 * - Cohen, J. (1988) "Statistical Power Analysis for the Behavioral Sciences"
 */

// ── Mathematical primitives ──────────────────────────────────────────────

/** Lanczos coefficient set (g=5, n=7) from Numerical Recipes. */
const LANCZOS_G = 5;
const LANCZOS_COEFFS = [
  1.000000000190015, 76.18009172947146, -86.50532032941678, 24.01409824083091, -1.231739572450155,
  0.120865097386618e-2, -0.5395239384953e-5,
];

/** Natural log of the gamma function via the Lanczos approximation. */
export function logGamma(x: number): number {
  if (x <= 0) return Number.NaN;
  const g = LANCZOS_G;
  const c = LANCZOS_COEFFS;
  if (x < 0.5) {
    // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }
  x -= 1;
  let a = c[0];
  for (let i = 1; i < c.length; i++) a += c[i] / (x + i);
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularized incomplete beta function I_x(a, b). */
export function incompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x),
  );
  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betacf(a, b, x)) / a;
  }
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/** Continued fraction expansion for the incomplete beta function. */
function betacf(a: number, b: number, x: number): number {
  const MAXIT = 200;
  const EPS = 3e-16;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < EPS) break;
  }
  return h;
}

/**
 * Inverse of the regularized incomplete beta function.
 * Finds x such that I_x(a, b) = p.
 *
 * Uses bisection — `I_x(a, b)` is monotonically increasing in x for
 * fixed a, b > 0 — which guarantees convergence regardless of the
 * difficulty of the parameter combination (the Newton-Raphson approach
 * with a closed-form initial guess was unreliable for certain (p, a, b)
 * combinations, producing NaN via log(0) when the guess fell outside
 * [0, 1]).
 */
export function inverseIncompleteBeta(p: number, a: number, b: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (incompleteBeta(a, b, mid) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// ── t-distribution functions ─────────────────────────────────────────────

/**
 * Two-tailed p-value for a t-statistic and degrees of freedom.
 * Uses the relationship: p = I_{df/(df+t^2)}(df/2, 1/2).
 */
export function tTestPValue(t: number, df: number): number {
  if (df <= 0) return Number.NaN;
  const x = df / (df + t * t);
  return incompleteBeta(df / 2, 0.5, x);
}

/**
 * Critical t-value for a given two-tailed confidence level and df.
 * Returns t_crit such that P(|T| <= t_crit) = confidence.
 */
export function tCriticalValue(confidence: number, df: number): number {
  // For confidence C, we want I_x(df/2, 1/2) = 1 - C
  const alpha = 1 - confidence;
  const x = inverseIncompleteBeta(alpha, df / 2, 0.5);
  return Math.sqrt(df * (1 / x - 1));
}

// ── Descriptive statistics ───────────────────────────────────────────────

export interface DescriptiveStats {
  n: number;
  mean: number;
  median: number;
  std: number;
  stdErr: number;
  min: number;
  max: number;
  sum: number;
  variance: number;
}

/** Compute descriptive statistics for a sample. */
export function describe(values: number[]): DescriptiveStats {
  const n = values.length;
  if (n === 0) {
    return {
      n: 0,
      mean: NaN,
      median: NaN,
      std: NaN,
      stdErr: NaN,
      min: NaN,
      max: NaN,
      sum: 0,
      variance: NaN,
    };
  }
  const sum = values.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const variance = n > 1 ? values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (n - 1) : 0;
  const std = Math.sqrt(variance);
  const sorted = [...values].sort((a, b) => a - b);
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
  return {
    n,
    mean,
    median,
    std,
    stdErr: std / Math.sqrt(n),
    min: Math.min(...values),
    max: Math.max(...values),
    sum,
    variance,
  };
}

// ── Confidence Interval ──────────────────────────────────────────────────

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  mean: number;
  confidence: number;
  margin: number;
  stdErr: number;
  df: number;
}

/**
 * Compute a confidence interval for the mean using the t-distribution.
 * @param values     Sample data.
 * @param confidence  Confidence level (default 0.95).
 */
export function confidenceInterval(values: number[], confidence = 0.95): ConfidenceInterval {
  const stats = describe(values);
  const df = stats.n - 1;
  if (df <= 0) {
    return {
      lower: NaN,
      upper: NaN,
      mean: stats.mean,
      confidence,
      margin: NaN,
      stdErr: NaN,
      df,
    };
  }
  const tCrit = tCriticalValue(confidence, df);
  const margin = tCrit * stats.stdErr;
  return {
    lower: stats.mean - margin,
    upper: stats.mean + margin,
    mean: stats.mean,
    confidence,
    margin,
    stdErr: stats.stdErr,
    df,
  };
}

// ── Cohen's d ────────────────────────────────────────────────────────────

export type EffectSizeInterpretation = "negligible" | "small" | "medium" | "large";

export interface CohensDResult {
  d: number;
  interpretation: EffectSizeInterpretation;
  pooledStd: number;
}

/**
 * Interpret Cohen's d using conventional thresholds (Cohen, 1988).
 * |d| < 0.2 → negligible, 0.2–0.5 → small, 0.5–0.8 → medium, ≥0.8 → large.
 */
export function interpretCohensD(d: number): EffectSizeInterpretation {
  const abs = Math.abs(d);
  if (abs < 0.2) return "negligible";
  if (abs < 0.5) return "small";
  if (abs < 0.8) return "medium";
  return "large";
}

/**
 * Compute Cohen's d for two independent groups (pooled std denominator).
 */
export function cohensD(group1: number[], group2: number[]): CohensDResult {
  const s1 = describe(group1);
  const s2 = describe(group2);
  if (s1.n < 2 || s2.n < 2) {
    return { d: NaN, interpretation: "negligible", pooledStd: NaN };
  }
  const pooledVar = ((s1.n - 1) * s1.variance + (s2.n - 1) * s2.variance) / (s1.n + s2.n - 2);
  const pooledStd = Math.sqrt(pooledVar);
  // Guard: if there is no variance, Cohen's d is undefined (0/0 or x/0).
  // Return 0 when means are equal, Infinity when they differ.
  const d =
    pooledStd === 0
      ? s1.mean === s2.mean
        ? 0
        : Math.sign(s1.mean - s2.mean) * Infinity
      : (s1.mean - s2.mean) / pooledStd;
  return { d, interpretation: interpretCohensD(d), pooledStd };
}

// ── t-tests ──────────────────────────────────────────────────────────────

export interface TTestResult {
  t: number;
  pValue: number;
  df: number;
  mean: number;
  meanDiff: number;
  ci: ConfidenceInterval;
  significant: boolean;
  alpha: number;
}

/**
 * One-sample t-test: tests whether the sample mean differs from `mu0`.
 * @param values  Sample data.
 * @param mu0     Null hypothesis mean (e.g., chance level 0.25 for 4-class).
 */
export function tTestOneSample(values: number[], mu0: number, alpha = 0.05): TTestResult {
  const stats = describe(values);
  const df = stats.n - 1;
  if (df <= 0) {
    return {
      t: NaN,
      pValue: NaN,
      df,
      mean: stats.mean,
      meanDiff: stats.mean - mu0,
      ci: confidenceInterval(values),
      significant: false,
      alpha,
    };
  }
  // When std=0 (no variation): if mean == mu0 → t=0, p=1 (not significant).
  // If mean ≠ mu0 → t=∞, p=0 (significant).
  if (stats.std === 0) {
    const meanDiff = stats.mean - mu0;
    const t = meanDiff === 0 ? 0 : Math.sign(meanDiff) * Infinity;
    const pValue = meanDiff === 0 ? 1 : 0;
    return {
      t,
      pValue,
      df,
      mean: stats.mean,
      meanDiff,
      ci: confidenceInterval(values),
      significant: pValue < alpha,
      alpha,
    };
  }
  const t = (stats.mean - mu0) / stats.stdErr;
  const pValue = tTestPValue(t, df);
  return {
    t,
    pValue,
    df,
    mean: stats.mean,
    meanDiff: stats.mean - mu0,
    ci: confidenceInterval(values, 1 - alpha),
    significant: pValue < alpha,
    alpha,
  };
}

/**
 * Two-sample Welch's t-test (unequal variances).
 */
export function tTestTwoSample(group1: number[], group2: number[], alpha = 0.05): TTestResult {
  const s1 = describe(group1);
  const s2 = describe(group2);
  if (s1.n < 2 || s2.n < 2) {
    return {
      t: NaN,
      pValue: NaN,
      df: NaN,
      mean: s1.mean,
      meanDiff: s1.mean - s2.mean,
      ci: confidenceInterval(group1, 1 - alpha),
      significant: false,
      alpha,
    };
  }
  const se1 = s1.variance / s1.n;
  const se2 = s2.variance / s2.n;
  const se = Math.sqrt(se1 + se2);
  const t = (s1.mean - s2.mean) / se;
  // Welch-Satterthwaite degrees of freedom
  const df = (se1 + se2) ** 2 / (se1 ** 2 / (s1.n - 1) + se2 ** 2 / (s2.n - 1));
  const pValue = tTestPValue(t, df);
  const pooledCI = confidenceInterval(group1, 1 - alpha);
  return {
    t,
    pValue,
    df,
    mean: s1.mean,
    meanDiff: s1.mean - s2.mean,
    ci: pooledCI,
    significant: pValue < alpha,
    alpha,
  };
}

// ── p-value formatting ───────────────────────────────────────────────────

/**
 * Format a p-value for publication: <0.001, <0.01, <0.05, or precise value.
 */
export function formatPValue(p: number): string {
  if (p <= 0) return "<0.001";
  if (p < 0.001) return "<0.001";
  if (p < 0.01) return "<0.01";
  if (p < 0.05) return "<0.05";
  return p.toFixed(3);
}

// ── Publication-ready table ──────────────────────────────────────────────

export interface PublicationMetric {
  name: string;
  n: number;
  mean: number;
  std: number;
  ciLower: number;
  ciUpper: number;
  pValue?: number;
  effectSize?: CohensDResult;
}

/**
 * Generate a markdown table of metrics suitable for publication.
 *
 * ```markdown
 * | Metric    | Mean | Std  | 95% CI      | p-value | Effect Size |
 * | --------- | ---- | ---- | ----------- | ------- | ----------- |
 * | Accuracy  | 0.72 | 0.15 | (0.68-0.76) | <0.001  | d=0.85 large |
 * ```
 */
export function publicationTable(metrics: PublicationMetric[]): string {
  const header = "| Metric       | N  | Mean  | Std   | 95% CI          | p-value | Effect Size |";
  const separator =
    "|--------------|----|-------|-------|-----------------|---------|-------------|";
  const rows = metrics.map(
    (m) =>
      `| ${m.name.padEnd(12)} | ${m.n.toString().padEnd(2)} | ${m.mean
        .toFixed(3)
        .padEnd(5)} | ${m.std
        .toFixed(3)
        .padEnd(
          5,
        )} | (${m.ciLower.toFixed(3)}-${m.ciUpper.toFixed(3)}) | ${m.pValue !== undefined ? formatPValue(m.pValue).padEnd(7) : "N/A".padEnd(7)} | ${
        m.effectSize
          ? `d=${m.effectSize.d.toFixed(2)} ${m.effectSize.interpretation}`.padEnd(10)
          : "N/A".padEnd(10)
      } |`,
  );
  return [header, separator, ...rows].join("\n");
}

// ── Re-exports for convenience ──────────────────────────────────────────

export { describe as meanStd };
