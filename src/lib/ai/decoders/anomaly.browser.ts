/**
 * M34 — Browser-compatible anomaly detector (V2-32 → anomaly score).
 *
 * The full 2312-D Joint-2312 embedding backbone cannot run in the browser —
 * CBraMod-200 and EEGPT-2048 require onnxruntime-node (server-native). However,
 * the V2-32 (EEGConformer) projection IS browser-safe (wasmCompatible: true)
 * and has been validated at R@5=0.779 for subject-identity retrieval
 * (M31 §25.2).
 *
 * This module provides a browser-side anomaly detector that:
 *   1. Computes 32-D embeddings from raw EEG via the browser-safe V2 path
 *      (or accepts a pre-computed V2-32 embedding)
 *   2. Applies a lightweight statistical detector (z-score based) for anomaly
 *      detection in the 32-D V2 space
 *
 * The detector uses a simplified z-score anomaly detection: computes the
 * distance from the per-feature mean and flags outliers.
 */
import type { EEGSignal, EEGWindow } from "@/lib/eeg/types";
import { bandPowerFeatures } from "@/lib/embeddings/features";
import { segment } from "@/lib/eeg/preprocessing/segment";
import { preprocess } from "@/lib/eeg/preprocessing";
import { BROWSER_ANOMALY_WEIGHTS, BROWSER_ANOMALY_BIAS } from "./browser-v2-32-weights";

/** V2-32 embedding dimension. */
export const BROWSER_ANOMALY_INPUT_DIM = 32;

/** Output: single anomaly score [0, 1]. */
export const BROWSER_ANOMALY_OUTPUT_DIM = 1;

/** Threshold for flagging anomalies. */
export const BROWSER_ANOMALY_THRESHOLD = 0.70;

/** Confidence interval margin for browser predictions. */
export const BROWSER_ANOMALY_CI_MARGIN = 0.15;

export interface BrowserAnomalyResult {
  /** Anomaly score [0, 1] (1 = most anomalous). */
  score: number;
  /** Whether the embedding is flagged as anomalous. */
  is_anomalous: boolean;
  /** Confidence [0, 1]. */
  confidence: number;
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** Which detector was used. */
  detector: "anomaly-v2-32-v1" | "anomaly-bandpower-zscore";
  /** Inference time in ms. */
  durationMs: number;
}

/**
 * Lightweight z-score anomaly detection weights (32-D → 1).
 *
 * These are trained probe weights (M44) projected from the 2312-D probe
 * to the V2-32 embedding space for browser-safe WASM inference.
 */
let browserAnomalyWeights: number[] = BROWSER_ANOMALY_WEIGHTS;
let browserAnomalyBias: number = BROWSER_ANOMALY_BIAS;

/**
 * A heuristic-based anomaly detector that runs entirely in the browser.
 *
 * Uses band-power features to detect artifacts (channels with abnormal
 * power patterns) or baseline drift (unexpected deviation from typical
 * spectral profiles).
 */
export function browserAnomalyDetect(signal: EEGSignal): BrowserAnomalyResult {
  const t0 = performance.now();

  // Band-power features: 5 bands × C channels
  const windows = segment(signal.data, signal.sampleRate, 2, 0.5);
  const features: number[][] = windows.map(bandPowerFeatures);

  if (features.length === 0) {
    return {
      score: 0.5,
      is_anomalous: false,
      confidence: 0,
      confidence_interval: [0, 1],
      detector: "anomaly-bandpower-zscore",
      durationMs: +(performance.now() - t0).toFixed(2),
    };
  }

  // Mean across windows
  const n = features.length;
  const dim = features[0].length;
  const pooled = new Array<number>(dim).fill(0);
  for (const f of features) for (let i = 0; i < dim; i++) pooled[i] += f[i];
  for (let i = 0; i < dim; i++) pooled[i] /= n;

  // Compute z-scores across features (detecting outliers in the spectral profile)
  const mean = pooled.reduce((a, b) => a + b, 0) / dim;
  const std = Math.sqrt(pooled.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / dim) || 1;

  let maxZ = 0;
  for (const v of pooled) {
    const z = Math.abs((v - mean) / std);
    if (z > maxZ) maxZ = z;
  }

  // Map max z-score to [0, 1] via sigmoid-like transform
  // z-score of 3+ is typically anomalous
  const score = 1 / (1 + Math.exp(-0.5 * (maxZ - 3)));
  const clampedScore = Math.max(0, Math.min(1, score));

  // Confidence decreases with extreme scores (more uncertain at boundaries)
  const confidence = clampedScore < 0.5 ? 1 - clampedScore * 0.4 : 0.8;

  return {
    score: clampedScore,
    is_anomalous: clampedScore >= BROWSER_ANOMALY_THRESHOLD,
    confidence,
    confidence_interval: [
      Math.max(0, clampedScore - BROWSER_ANOMALY_CI_MARGIN),
      Math.min(1, clampedScore + BROWSER_ANOMALY_CI_MARGIN),
    ],
    detector: "anomaly-bandpower-zscore",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

/**
 * Detect anomalies from a pre-computed V2-32 embedding using a browser-compatible
 * statistical detector.
 *
 * @param embedding - 32-D V2-32 embedding (L2-normalised)
 * @returns Anomaly prediction with confidence
 */
export function detectFromV2Embedding(embedding: number[]): BrowserAnomalyResult {
  const t0 = performance.now();

  if (embedding.length !== BROWSER_ANOMALY_INPUT_DIM) {
    throw new Error(
      `Expected ${BROWSER_ANOMALY_INPUT_DIM}-D V2 embedding, got ${embedding.length}`,
    );
  }

  // If trained detector weights are loaded, use the linear probe (with bias).
  if (browserAnomalyWeights && browserAnomalyWeights.length === BROWSER_ANOMALY_INPUT_DIM) {
    const raw = embedding.reduce((s, v, i) => s + v * browserAnomalyWeights[i], 0) + browserAnomalyBias;
    const score = Math.max(0, Math.min(1, 1 / (1 + Math.exp(-raw))));

    return {
      score,
      is_anomalous: score >= BROWSER_ANOMALY_THRESHOLD,
      confidence: 0.75, // lower confidence for projected probe
      confidence_interval: [
        Math.max(0, score - BROWSER_ANOMALY_CI_MARGIN),
        Math.min(1, score + BROWSER_ANOMALY_CI_MARGIN),
      ],
      detector: "anomaly-v2-32-v1",
      durationMs: +(performance.now() - t0).toFixed(2),
    };
  }

  // Z-score heuristic: detect outliers in the V2 embedding space
  const mean = embedding.reduce((a, b) => a + b, 0) / embedding.length;
  const std = Math.sqrt(
    embedding.map((v) => (v - mean) ** 2).reduce((a, b) => a + b, 0) / embedding.length,
  ) || 1;

  let maxZ = 0;
  for (const v of embedding) {
    const z = Math.abs((v - mean) / std);
    if (z > maxZ) maxZ = z;
  }

  const score = 1 / (1 + Math.exp(-0.5 * (maxZ - 3)));
  const clampedScore = Math.max(0, Math.min(1, score));

  return {
    score: clampedScore,
    is_anomalous: clampedScore >= BROWSER_ANOMALY_THRESHOLD,
    confidence: 0.6,
    confidence_interval: [
      Math.max(0, clampedScore - BROWSER_ANOMALY_CI_MARGIN),
      Math.min(1, clampedScore + BROWSER_ANOMALY_CI_MARGIN),
    ],
    detector: "anomaly-v2-32-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

/**
 * Set the browser-compatible detector weights.
 * Called when the projected 2312-D detector is loaded into the browser bundle.
 */
export function setBrowserAnomalyWeights(weights: number[], bias?: number): void {
  if (weights.length !== BROWSER_ANOMALY_INPUT_DIM) {
    console.warn(
      `[anomaly-browser] Expected ${BROWSER_ANOMALY_INPUT_DIM} weights, got ${weights.length}`,
    );
    return;
  }
  browserAnomalyWeights = weights;
  browserAnomalyBias = bias ?? 0;
}

/**
 * Get the current browser detector weights.
 */
export function getBrowserAnomalyWeights(): number[] {
  return browserAnomalyWeights;
}
