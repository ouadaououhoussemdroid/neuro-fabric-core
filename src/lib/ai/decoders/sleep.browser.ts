/**
 * M39 + M40 — Browser-compatible sleep decoders (V2-32 → sleep stages and quality).
 *
 * The full 2312-D Joint-2312 embedding backbone cannot run in the browser —
 * CBraMod-200 and EEGPT-2048 require onnxruntime-node (server-native). However,
 * the V2-32 (EEGConformer) projection IS browser-safe (wasmCompatible: true)
 * and has been validated at R@5=0.779 for subject-identity retrieval
 * (M31 §25.2).
 *
 * This module provides browser-side sleep decoders that:
 *   1. Accepts a pre-computed V2-32 embedding (or raw EEG signal for band-power
 *      feature extraction)
 *   2. Applies a lightweight linear probe (32→5) for sleep staging (softmax)
 *      or (32→1) for sleep quality (regression, clamped to [0, 1])
 *
 * The probe weights are a projected subset of the 2312-D probe (M44 trained
 * V2-32 probes). The trained weights are loaded by default and can be
 * overridden at runtime via `setBrowserSleepWeights` / `setBrowserSleepQualityWeights`.
 */
import type { EEGSignal } from "@/lib/eeg/types";
import {
  BROWSER_SLEEP_STAGING_WEIGHTS,
  BROWSER_SLEEP_STAGING_BIAS,
  BROWSER_SLEEP_QUALITY_WEIGHTS,
  BROWSER_SLEEP_QUALITY_BIAS,
} from "./browser-v2-32-weights";
import { bandPowerFeatures } from "@/lib/embeddings/features";
import { segment } from "@/lib/eeg/preprocessing/segment";

/** V2-32 embedding dimension. */
export const BROWSER_SLEEP_INPUT_DIM = 32;

/** Output: 5 sleep stages (W, N1, N2, N3, REM). */
export const BROWSER_SLEEP_OUTPUT_DIM = 5;

/** Sleep stage labels in output order (index 0-4). */
export const BROWSER_SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"];

/** Confidence interval margin for browser predictions (larger than server due to dimensionality loss). */
export const BROWSER_SLEEP_CI_MARGIN = 0.15;

/** Browser sleep quality result. */
export interface BrowserSleepQualityResult {
  /** Normalized sleep quality score [0, 1]. */
  score: number;
  /** Quality band label. */
  band: "poor" | "fair" | "good" | "excellent";
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** Confidence [0, 1]. */
  confidence: number;
  /** Which decoder was used. */
  decoder: "sleep-v2-32-v1" | "sleep-quality-heuristic";
  /** Inference time in ms. */
  durationMs: number;
}

export interface BrowserSleepResult {
  /** Predicted sleep stage index (0=W, 1=N1, 2=N2, 3=N3, 4=REM). */
  stage_id: number;
  /** Predicted sleep stage label. */
  stage: string;
  /** Softmax probabilities for each stage [W, N1, N2, N3, REM]. */
  probabilities: [number, number, number, number, number];
  /** Confidence [0, 1] (max probability). */
  confidence: number;
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** Which decoder was used. */
  decoder: "sleep-v2-32-v1" | "sleep-bandpower-heuristic";
  /** Inference time in ms. */
  durationMs: number;
}

/**
 * Lightweight linear probe weights (32-D → 5), initialized from trained M44 probe.
 *
 * These are the trained RidgeClassifier weights exported from
 * `models/sleep/staging-probe-v2-32d-v1.onnx`. When a different probe is
 * loaded via `setBrowserSleepWeights`, it overrides these defaults.
 */
let browserSleepWeights: number[][] = BROWSER_SLEEP_STAGING_WEIGHTS;
let browserSleepBias: number[] | null = BROWSER_SLEEP_STAGING_BIAS ?? null;

/**
 * Softmax function (stable implementation).
 */
function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exp = logits.map((l) => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((e) => e / sum);
}

/**
 * A heuristic-based sleep stage classifier that runs entirely in the browser.
 *
 * Uses band-power spectral features to classify sleep stages:
 * - W (Wake): high alpha/beta (12-30 Hz), low delta/theta
 * - N1: increasing theta (4-8 Hz), decreasing alpha
 * - N2: sleep spindles (12-14 Hz), K-complexes
 * - N3 (SWS): dominant delta (0.5-4 Hz)
 * - REM: mixed theta + beta, similar to wake but with atonia
 *
 * This is the browser-fallback path: when a 2312-D embedding is not available,
 * the browser computes band-power features and applies the heuristic.
 */
export function browserSleepStage(signal: EEGSignal): BrowserSleepResult {
  const t0 = performance.now();

  // Band-power features: 5 bands × C channels
  const windows = segment(signal.data, signal.sampleRate, 4, 0.5); // 4s windows
  const features: number[][] = windows.map(bandPowerFeatures);

  if (features.length === 0) {
    return defaultResult(t0);
  }

  // Mean across windows
  const n = features.length;
  const dim = features[0].length;
  const pooled = new Array<number>(dim).fill(0);
  for (const f of features) for (let i = 0; i < dim; i++) pooled[i] += f[i];
  for (let i = 0; i < dim; i++) pooled[i] /= n;

  // 5 bands: δ, θ, α, β, γ (per channel, averaged across channels)
  const bands = 5;
  const chans = dim / bands;
  const bandPowers = new Array<number>(bands).fill(0);
  for (let b = 0; b < bands; b++) {
    for (let c = 0; c < chans; c++) bandPowers[b] += pooled[c * bands + b];
    bandPowers[b] /= chans;
  }

  const [delta, theta, alpha, beta, gamma] = bandPowers;
  const total = bandPowers.reduce((a, b) => a + b, 0) || 1;

  // Heuristic logits for each sleep stage based on spectral patterns
  const logits = [
    // W (Wake): high alpha + beta
    (alpha / total) * 2.0 + (beta / total) * 1.5 - (delta / total) * 0.5,
    // N1: high theta
    (theta / total) * 2.5 - (alpha / total) * 1.0 - (delta / total) * 0.5,
    // N2: moderate theta + alpha (transition)
    (theta / total) * 1.5 + (alpha / total) * 0.5 - (delta / total) * 0.5,
    // N3 (SWS): dominant delta
    (delta / total) * 3.0 - (theta / total) * 0.5 - (alpha / total) * 0.5,
    // REM: mixed theta + beta (like wake but with theta)
    (theta / total) * 1.5 + (beta / total) * 1.0 - (delta / total) * 0.3,
  ];

  const probs = softmax(logits);
  const [pW, pN1, pN2, pN3, pREM] = probs;
  const probabilities: [number, number, number, number, number] = [pW, pN1, pN2, pN3, pREM];
  const confidence = Math.max(...probs);
  const stageId = probs.indexOf(confidence);
  const stage = BROWSER_SLEEP_STAGES[stageId];

  return {
    stage_id: stageId,
    stage,
    probabilities,
    confidence,
    confidence_interval: [
      Math.max(0, confidence - BROWSER_SLEEP_CI_MARGIN),
      Math.min(1, confidence + BROWSER_SLEEP_CI_MARGIN),
    ],
    decoder: "sleep-bandpower-heuristic",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

/**
 * Classify sleep stage from a pre-computed V2-32 embedding using a
 * browser-compatible linear probe.
 *
 * @param embedding - 32-D V2-32 embedding (L2-normalised)
 * @returns Sleep stage prediction with probabilities and confidence
 */
export function detectSleepFromV2Embedding(embedding: number[]): BrowserSleepResult {
  const t0 = performance.now();

  if (embedding.length !== BROWSER_SLEEP_INPUT_DIM) {
    throw new Error(
      `Expected ${BROWSER_SLEEP_INPUT_DIM}-D V2 embedding, got ${embedding.length}`,
    );
  }

  // If trained probe weights are loaded, use the linear probe (with optional bias).
  if (
    browserSleepWeights &&
    browserSleepWeights.length === BROWSER_SLEEP_OUTPUT_DIM &&
    browserSleepWeights[0]?.length === BROWSER_SLEEP_INPUT_DIM
  ) {
    const logits = browserSleepWeights.map((w, idx) => {
      const dot = embedding.reduce((s, v, i) => s + v * w[i], 0);
      return browserSleepBias ? dot + browserSleepBias[idx] : dot;
    });
    const probs = softmax(logits);
    const [pW, pN1, pN2, pN3, pREM] = probs;
    const probabilities: [number, number, number, number, number] = [pW, pN1, pN2, pN3, pREM];
    const confidence = Math.max(...probs);
    const stageId = probs.indexOf(confidence);
    const stage = BROWSER_SLEEP_STAGES[stageId];

    return {
      stage_id: stageId,
      stage,
      probabilities,
      confidence,
      confidence_interval: [
        Math.max(0, confidence - BROWSER_SLEEP_CI_MARGIN),
        Math.min(1, confidence + BROWSER_SLEEP_CI_MARGIN),
      ],
      decoder: "sleep-v2-32-v1",
      durationMs: +(performance.now() - t0).toFixed(2),
    };
  }

  // Fallback: use band-power heuristic on the embedding
  return heuristicFromEmbedding(embedding, t0);
}

/**
 * Apply a band-power heuristic directly on V2-32 embedding for sleep staging.
 * The V2 embedding encodes spectral-temporal features; indices 0-4 (delta),
 * 5-9 (theta), 10-14 (alpha), 15-19 (beta), 20-24 (gamma) correlate with
 * sleep-relevant spectral bands.
 */
function heuristicFromEmbedding(embedding: number[], t0: number): BrowserSleepResult {
  const bands = 5;
  const perBand = Math.floor(embedding.length / bands);
  const bandAvgs = new Array<number>(bands).fill(0);
  for (let b = 0; b < bands; b++) {
    for (let i = 0; i < perBand; i++) {
      bandAvgs[b] += Math.abs(embedding[b * perBand + i]);
    }
    bandAvgs[b] /= perBand;
  }
  const total = bandAvgs.reduce((a, b) => a + b, 0) || 1;
  const [delta, theta, alpha, beta, gamma] = bandAvgs;

  const logits = [
    (alpha / total) * 2.0 + (beta / total) * 1.5 - (delta / total) * 0.5,
    (theta / total) * 2.5 - (alpha / total) * 1.0 - (delta / total) * 0.5,
    (theta / total) * 1.5 + (alpha / total) * 0.5 - (delta / total) * 0.5,
    (delta / total) * 3.0 - (theta / total) * 0.5 - (alpha / total) * 0.5,
    (theta / total) * 1.5 + (beta / total) * 1.0 - (delta / total) * 0.3,
  ];

  const probs = softmax(logits);
  const [pW, pN1, pN2, pN3, pREM] = probs;
  const probabilities: [number, number, number, number, number] = [pW, pN1, pN2, pN3, pREM];
  const confidence = Math.max(...probs);
  const stageId = probs.indexOf(confidence);
  const stage = BROWSER_SLEEP_STAGES[stageId];

  return {
    stage_id: stageId,
    stage,
    probabilities,
    confidence,
    confidence_interval: [
      Math.max(0, confidence - BROWSER_SLEEP_CI_MARGIN),
      Math.min(1, confidence + BROWSER_SLEEP_CI_MARGIN),
    ],
    decoder: "sleep-v2-32-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

function defaultResult(t0: number): BrowserSleepResult {
  const probabilities: [number, number, number, number, number] = [0.2, 0.2, 0.2, 0.2, 0.2];
  return {
    stage_id: -1,
    stage: "UNKNOWN",
    probabilities,
    confidence: 0,
    confidence_interval: [0, 1],
    decoder: "sleep-bandpower-heuristic",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

/**
 * Set the browser-compatible linear probe weights (32-D → 5 logits).
 * Called when the projected 2312-D probe is loaded into the browser bundle.
 *
 * @param weights - 5 arrays of 32 values each (one per sleep stage)
 * @param bias - optional 5-element bias vector (one per sleep stage)
 */
export function setBrowserSleepWeights(weights: number[][] | null, bias?: number[]): void {
  if (weights === null) {
    // Reset to trained default weights
    browserSleepWeights = BROWSER_SLEEP_STAGING_WEIGHTS;
    browserSleepBias = BROWSER_SLEEP_STAGING_BIAS ?? null;
    return;
  }
  if (weights.length !== BROWSER_SLEEP_OUTPUT_DIM) {
    console.warn(
      `[sleep-browser] Expected ${BROWSER_SLEEP_OUTPUT_DIM} weight rows, got ${weights.length}`,
    );
    return;
  }
  if (weights[0]?.length !== BROWSER_SLEEP_INPUT_DIM) {
    console.warn(
      `[sleep-browser] Expected ${BROWSER_SLEEP_INPUT_DIM} weights per row, got ${weights[0]?.length}`,
    );
    return;
  }
  browserSleepWeights = weights;
  browserSleepBias = bias ?? null;
}

/**
 * Get the current browser sleep probe weights.
 */
export function getBrowserSleepWeights(): number[][] {
  return browserSleepWeights;
}

// ─────────────────────────────────────────────────────────────────────
// Sleep Quality (M40)
// ─────────────────────────────────────────────────────────────────────

/**
 * Lightweight linear quality probe weights (32-D → 1), initialized from
 * the trained M44 probe (`models/sleep/quality-probe-v2-32d-v1.onnx`).
 */
let browserSleepQualityWeights: number[] = BROWSER_SLEEP_QUALITY_WEIGHTS;
let browserSleepQualityBias: number | null = BROWSER_SLEEP_QUALITY_BIAS ?? null;

/**
 * Classify sleep quality from a pre-computed V2-32 embedding.
 *
 * @param embedding - 32-D V2-32 embedding (L2-normalised)
 * @returns Sleep quality prediction [0, 1] with band and confidence
 */
export function browserSleepQuality(embedding: number[]): BrowserSleepQualityResult {
  const t0 = performance.now();

  if (embedding.length !== BROWSER_SLEEP_INPUT_DIM) {
    throw new Error(
      `Expected ${BROWSER_SLEEP_INPUT_DIM}-D V2 embedding, got ${embedding.length}`,
    );
  }

  let score: number;

  // If trained probe weights are loaded, use the linear probe (with optional bias).
  if (browserSleepQualityWeights && browserSleepQualityWeights.length === BROWSER_SLEEP_INPUT_DIM) {
    score = embedding.reduce((s, v, i) => s + v * browserSleepQualityWeights![i], 0);
    if (browserSleepQualityBias !== null) score += browserSleepQualityBias;
  } else {
    // Fallback: band-power heuristic on the V2 embedding.
    // Indices 0-4 (delta), 5-9 (theta), 10-14 (alpha), 15-19 (beta), 20-24 (gamma)
    const bands = 5;
    const perBand = Math.floor(embedding.length / bands);
    const bandAvgs = new Array<number>(bands).fill(0);
    for (let b = 0; b < bands; b++) {
      for (let i = 0; i < perBand; i++) {
        bandAvgs[b] += Math.abs(embedding[b * perBand + i]);
      }
      bandAvgs[b] /= perBand;
    }
    const total = bandAvgs.reduce((a, b) => a + b, 0) || 1;
    const [delta, theta, alpha] = bandAvgs;

    // Heuristic: high delta + balanced theta → deep sleep → good quality.
    // High alpha/beta (wake-like) → poor quality.
    score = (delta / total) * 0.6 + (theta / total) * 0.3 + (alpha / total) * (-0.2);
    score = Math.min(1, Math.max(0, score + 0.5)); // center around 0.5
  }

  // Clamp to [0, 1] and derive quality band.
  score = Math.max(0, Math.min(1, score));
  const band: "poor" | "fair" | "good" | "excellent" =
    score < 0.4
      ? "poor"
      : score < 0.6
        ? "fair"
        : score < 0.8
          ? "good"
          : "excellent";

  // Confidence: distance from nearest boundary.
  const distances = [0.0, 0.4, 0.6, 0.8, 1.0].map((b) => Math.abs(score - b));
  const confidence = 1 - Math.min(...distances);

  return {
    score,
    band,
    confidence_interval: [Math.max(0, score - 0.1), Math.min(1, score + 0.1)],
    confidence,
    decoder: "sleep-quality-heuristic",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

/**
 * Set the browser-compatible linear quality probe weights (32-D → 1 scalar).
 * Called when the projected 2312-D quality probe is loaded into the browser bundle.
 *
 * @param weights - 32 values, one per V2-32 dimension
 * @param bias - optional scalar bias
 */
export function setBrowserSleepQualityWeights(weights: number[] | null, bias?: number): void {
  if (weights === null) {
    // Reset to trained default weights
    browserSleepQualityWeights = BROWSER_SLEEP_QUALITY_WEIGHTS;
    browserSleepQualityBias = BROWSER_SLEEP_QUALITY_BIAS ?? null;
    return;
  }
  if (weights.length !== BROWSER_SLEEP_INPUT_DIM) {
    console.warn(
      `[sleep-browser] Expected ${BROWSER_SLEEP_INPUT_DIM} quality weights, got ${weights.length}`,
    );
    return;
  }
  browserSleepQualityWeights = weights;
  browserSleepQualityBias = bias ?? null;
}

/**
 * Get the current browser sleep quality probe weights.
 */
export function getBrowserSleepQualityWeights(): number[] {
  return browserSleepQualityWeights;
}
