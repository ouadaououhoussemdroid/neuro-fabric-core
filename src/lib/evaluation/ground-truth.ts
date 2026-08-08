/**
 * T-028 — Ground Truth Annotation infrastructure.
 *
 * Types and utilities for recording, querying, and correlating ground-truth
 * cognitive-state labels against system predictions.  Ground-truth labels
 * (e.g., experimenter-annotated attention levels, behavioural response times,
 * or physiological reference measures) are essential for validating the
 * heuristic and trained cognitive decoders.
 *
 * Used by:
 *   - `POST /api/annotations` — submit a new annotation.
 *   - `GET /api/annotations?analysisId=...` — retrieve annotations.
 *   - `POST /api/evaluate/cross-subject` — ground truth fed into LOSO.
 */

import { describe, confidenceInterval, tTestPValue } from "../stats";

/** The type of cognitive state being annotated. */
export type LabelType = "attention" | "workload" | "arousal" | "class";

/** A single ground-truth annotation for one analysis. */
export interface GroundTruthLabel {
  /** Unique identifier (UUID). */
  id: string;
  /** Foreign key to `eeg_analyses.id`. */
  analysisId: string;
  /** Subject identifier. */
  subjectId: string;
  /** What cognitive state this annotation measures. */
  type: LabelType;
  /**
   * The annotated value.
   * - For continuous states (attention/workload/arousal): a float in [0, 1].
   * - For discrete classes: the integer class index.
   */
  value: number;
  /** Start sample (inclusive) for temporal annotations. */
  startSample?: number;
  /** End sample (exclusive) for temporal annotations. */
  endSample?: number;
  /** UUID of the user who created this annotation. */
  annotatorUserId: string;
  /** Annotator's confidence in this label (0–1). Defaults to 1. */
  confidence: number;
  /** ISO timestamp. */
  createdAt: string;
}

/** A collection of ground-truth labels for one subject + label type. */
export interface GroundTruthSet {
  subjectId: string;
  type: LabelType;
  labels: GroundTruthLabel[];
}

/** Summary statistics for a set of annotations. */
export interface AnnotationSummary {
  subjectId: string;
  type: LabelType;
  nLabels: number;
  meanValue: number;
  stdValue: number;
  meanConfidence: number;
  timeSpanSamples?: { start: number; end: number };
}

/**
 * Correlation result between predicted and ground-truth values.
 * Uses Pearson r (linear correlation) and Concordance Correlation
 * Coefficient (CCC, which accounts for both correlation and agreement).
 */
export interface CorrelationResult {
  n: number;
  pearsonR: number;
  /** Concordance Correlation Coefficient ρ_c = 2ρσ_xy / (σ_x² + σ_y² + (μ_x - μ_y)²). */
  ccc: number;
  /** Mean Absolute Error between predicted and ground truth. */
  mae: number;
  /** Root Mean Square Error. */
  rmse: number;
  /** 95 % CI for the mean difference (predicted - ground truth). */
  differenceCI: { lower: number; upper: number; mean: number };
  /** t-test of the differences against 0 (tests for systematic bias). */
  biasTest: { t: number; pValue: number; significant: boolean };
}

/**
 * Compute Pearson correlation + concordance + error metrics between two
 * sequences of equal length.  Returns `undefined` if fewer than 2 paired
 * values are available.
 */
export function correlatePredictions(
  predicted: number[],
  groundTruth: number[],
): CorrelationResult | undefined {
  const n = Math.min(predicted.length, groundTruth.length);
  if (n < 2) return undefined;

  const x = predicted.slice(0, n);
  const y = groundTruth.slice(0, n);

  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  // Covariance and variances.
  let covXY = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    covXY += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  covXY /= n;
  varX /= n;
  varY /= n;

  const pearsonR = varX > 0 && varY > 0 ? covXY / Math.sqrt(varX * varY) : 0;

  // Concordance Correlation Coefficient.
  const ccc = varX + varY > 0 ? (2 * covXY) / (varX + varY + (meanX - meanY) ** 2) : 0;

  // Error metrics.
  const diffs = x.map((xi, i) => xi - y[i]);
  const mae = diffs.reduce((a, b) => a + Math.abs(b), 0) / n;
  const rmse = Math.sqrt(diffs.reduce((a, b) => a + b * b, 0) / n);

  // CI for mean difference (using t-distribution via stats module).
  const diffStats = describe(diffs);
  const diffCI = confidenceInterval(diffs, 0.95);

  // t-test: differences ≠ 0 (tests for systematic bias).
  const stdErr = diffStats.stdErr;
  const t = stdErr > 0 ? diffStats.mean / stdErr : 0;
  const df = n - 1;
  const pValue = df > 0 && stdErr > 0 ? tTestPValue(t, df) : 1;

  return {
    n,
    pearsonR,
    ccc,
    mae,
    rmse,
    differenceCI: { lower: diffCI.lower, upper: diffCI.upper, mean: diffStats.mean },
    biasTest: {
      t,
      pValue,
      significant: pValue < 0.05,
    },
  };
}

/** Group labels by subject, then by type. */
export function groupLabels(
  labels: GroundTruthLabel[],
): Map<string, Map<LabelType, GroundTruthLabel[]>> {
  const bySubject = new Map<string, Map<LabelType, GroundTruthLabel[]>>();
  for (const label of labels) {
    if (!bySubject.has(label.subjectId)) {
      bySubject.set(label.subjectId, new Map());
    }
    const byType = bySubject.get(label.subjectId)!;
    if (!byType.has(label.type)) {
      byType.set(label.type, []);
    }
    byType.get(label.type)!.push(label);
  }
  return bySubject;
}

/** Compute summary statistics for each subject+type combination. */
export function summarizeAnnotations(labels: GroundTruthLabel[]): AnnotationSummary[] {
  const grouped = groupLabels(labels);
  const summaries: AnnotationSummary[] = [];

  for (const [subjectId, byType] of grouped) {
    for (const [type, typeLabels] of byType) {
      const values = typeLabels.map((l) => l.value);
      const confidences = typeLabels.map((l) => l.confidence);
      const stats = describe(values);

      let timeSpan: { start: number; end: number } | undefined;
      const withRange = typeLabels.filter(
        (l) => l.startSample !== undefined && l.endSample !== undefined,
      );
      if (withRange.length > 0) {
        timeSpan = {
          start: Math.min(...withRange.map((l) => l.startSample!)),
          end: Math.max(...withRange.map((l) => l.endSample!)),
        };
      }

      summaries.push({
        subjectId,
        type,
        nLabels: typeLabels.length,
        meanValue: stats.mean,
        stdValue: stats.std,
        meanConfidence: describe(confidences).mean,
        timeSpanSamples: timeSpan,
      });
    }
  }

  return summaries;
}
