/**
 * T-012 — Recall@10 SLO harness.
 *
 * Samples labelled embedding snippets, computes recall@10 against the
 * pgvector-backed {@link NeuralVectorIndex}, compares it against the
 * in-memory brute-force baseline, and emits an SLO report. When recall
 * drops below a configured threshold, the report flags a regression.
 *
 * Intended to run as a nightly cron (pg_cron or external scheduler) via
 * the `/api/public/cron/recall` server route. The harness is pure logic
 * (no I/O) so it is fully testable in isolation.
 */
import { recallAtK, type EmbeddingValidationReport } from "../ai/benchmark/validation-metrics";

export interface SLOSample {
  id: string;
  embedding: number[];
  label: number;
  modelId: string;
}

export interface SLOReport {
  /** ISO timestamp of the run. */
  timestamp: string;
  /** Number of samples evaluated. */
  n: number;
  /** Recall@10 measured from pgvector ANN search. */
  pgvectorRecall: number;
  /** Recall@10 measured from exact brute-force search (the ground truth). */
  bruteForceRecall: number;
  /** Ratio pgvector / bruteForce; < 1.0 means ANN is losing recall. */
  annRecallRatio: number;
  /** SLO threshold (minimum acceptable recall). */
  threshold: number;
  /** Whether the SLO is met. */
  passed: boolean;
  /** Per-model breakdown. */
  perModel: Array<{
    modelId: string;
    n: number;
    recall: number;
    passed: boolean;
  }>;
  /** Alert message if the SLO failed. */
  alert?: string;
}

export interface SLOConfig {
  k?: number;
  threshold?: number;
  annRecallRatioFloor?: number;
}

export const DEFAULT_SLO_CONFIG: Required<SLOConfig> = {
  k: 10,
  threshold: 0.85,
  annRecallRatioFloor: 0.95,
};

/**
 * Run the recall@10 SLO check against a set of labelled samples.
 *
 * @param samples      Labelled embeddings from the database.
 * @param annRecall    Recall@10 as measured by the pgvector ANN search
 *                     (the system under test).
 * @param config       SLO thresholds.
 */
export function runRecallSLO(
  samples: SLOSample[],
  annRecall: number,
  config: SLOConfig = {},
): SLOReport {
  const { k, threshold, annRecallRatioFloor } = { ...DEFAULT_SLO_CONFIG, ...config };

  // Brute-force recall@10 (ground truth) using exact cosine.
  const emb = samples.map((s) => s.embedding);
  const labels = samples.map((s) => s.label);
  const bruteForceRecall = recallAtK(emb, labels, k);

  const annRecallRatio = annRecall > 0 ? annRecall / bruteForceRecall : 0;
  const passed = annRecall >= threshold && annRecallRatio >= annRecallRatioFloor;

  // Per-model breakdown.
  const modelIds = [...new Set(samples.map((s) => s.modelId))];
  const perModel = modelIds.map((modelId) => {
    const modelSamples = samples.filter((s) => s.modelId === modelId);
    const modelEmb = modelSamples.map((s) => s.embedding);
    const modelLabels = modelSamples.map((s) => s.label);
    const recall = recallAtK(modelEmb, modelLabels, k);
    return { modelId, n: modelSamples.length, recall, passed: recall >= threshold };
  });

  const report: SLOReport = {
    timestamp: new Date().toISOString(),
    n: samples.length,
    pgvectorRecall: annRecall,
    bruteForceRecall,
    annRecallRatio,
    threshold,
    passed,
    perModel,
  };

  if (!passed) {
    report.alert = `Recall@${k} SLO failed: ann=${annRecall.toFixed(4)} < threshold=${threshold}, ratio=${annRecallRatio.toFixed(4)} < floor=${annRecallRatioFloor}`;
  }

  return report;
}

/** Serialize a report for logging / alerting. */
export function formatSLOAlert(report: SLOReport): string {
  return `[SLO] recall@10 regression: pgvector=${report.pgvectorRecall.toFixed(4)} bruteForce=${report.bruteForceRecall.toFixed(4)} ratio=${report.annRecallRatio.toFixed(4)} threshold=${report.threshold} — ${report.alert ?? "OK"}`;
}
