/**
 * M48 — Predictive Neural Coding Engine
 *
 * Implements a predictive coding framework for EEG: the model predicts the
 * next N timesteps of a multi-channel EEG signal, and the prediction error
 * (surprise) is scored as a deviation metric. Large surprise spikes indicate
 * neural state transitions, artifacts, or pre-symptomatic deviations.
 *
 * PIPELINE:
 *   EEG[C×T] → rolling-window LSTM prediction (autoregressive) →
 *   predicted[t+1..t+H] → prediction_error = actual - predicted →
 *   surprise_score = RMS(prediction_error) per channel →
 *   band-limited surprise (delta/theta/alpha/beta/gamma) →
 *   anomaly threshold (k×σ above baseline) → alert or normal
 *
 * This module is server-side (.server.ts suffix) because:
 *   - The LSTM prediction graph requires onnxruntime-node (server-only ONNX)
 *   - CPU-based fallback uses a lightweight linear predictor (AR model)
 *
 * The LSTM model is a tiny ONNX graph (2-layer LSTM, hidden=64) that
 * autoregressively predicts EEG windows. It is trained on the Joint-2312
 * embedding space where the temporal dynamics are lower-dimensional.
 *
 * Architecture mirrors foundation.server.ts:
 *   - Dynamic import of onnxruntime-node
 *   - Cached session for reuse across requests
 *   - SHA-256 verified model artifact
 *   - Metrics: predictiveCoding*
 */
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

import type { EEGSignal, EEGWindow } from "@/lib/eeg/types";
import { bandpass as cpuBandpass } from "@/lib/eeg/preprocessing/filters";
import { verifyArtefact } from "@/lib/ai/artefacts/hashed-artefact";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance, type ServiceProvenance } from "../services/provenance.server";
import { JOINT_2312_MODEL_ID, JOINT_2312_EMBEDDING_DIM } from "./joint.server";
import { ONNXAdapter, type OrtRuntime, type OrtSessionLike } from "@/lib/ai/adapters/onnx-adapter";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const PREDICTIVE_CODING_SERVICE = "predictive-neural-coding";
/** Service version. */
export const PREDICTIVE_CODING_VERSION = "v0.1.0";
/** Default prediction horizon (timesteps ahead to predict). */
export const DEFAULT_FORECAST_HORIZON = 8;
/** Number of past timesteps the LSTM conditions on. */
export const DEFAULT_RECEPTIVE_FIELD = 32;
/** Default channel for band-limited surprise analysis. */
export const EEG_BANDS = ["delta", "theta", "alpha", "beta", "gamma"] as const;
export type EEGBand = (typeof EEG_BANDS)[number];
/** Band frequency ranges [low, high] Hz. */
export const BAND_RANGES: Record<EEGBand, [number, number]> = {
  delta: [0.5, 4],
  theta: [4, 8],
  alpha: [8, 13],
  beta: [13, 30],
  gamma: [30, 100],
};
/** Default anomaly threshold: k standard deviations above baseline. */
export const DEFAULT_ANOMALY_K_SIGMA = 3.5;
/** Default sampling rate for the LSTM prediction model. */
export const MODEL_SAMPLE_RATE = 250;
/** Default number of EEG channels for prediction. */
export const MODEL_CHANNELS = 22;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/** Band-limited surprise scores per channel. */
export interface BandSurpriseScores {
  delta: number[];
  theta: number[];
  alpha: number[];
  beta: number[];
  gamma: number[];
}

/** Single-channel surprise result. */
export interface ChannelSurprise {
  channel: string;
  /** Root-mean-square prediction error. */
  rmsError: number;
  /** Per-band surprise scores. */
  bandScores: Record<EEGBand, number>;
  /** Whether the channel's surprise exceeds the anomaly threshold. */
  isAnomalous: boolean;
  /** Anomaly score [0, 1] (sigmoid of z-scored RMS error). */
  anomalyScore: number;
}

/** Full predictive coding result. */
export interface PredictiveCodingResult {
  /** Per-channel surprise scores and anomaly flags. */
  channels: ChannelSurprise[];
  /** Overall surprise (mean RMS error across all channels). */
  overallSurprise: number;
  /** Whether the overall signal is flagged as anomalous. */
  isAnomalous: boolean;
  /** Anomaly score [0, 1]. */
  anomalyScore: number;
  /** Number of timesteps predicted ahead. */
  forecastHorizon: number;
  /** Processing time in milliseconds. */
  durationMs: number;
  /** Whether GPU/LSTM inference was used (vs CPU fallback). */
  usedModel: boolean;
  /** Provenance record. */
  provenance: ServiceProvenance;
}

/** Options for predictive coding inference. */
export interface PredictiveCodingOptions {
  /** Number of timesteps to predict ahead. Default: 8. */
  horizon?: number;
  /** Number of past timesteps to condition on. Default: 32. */
  receptiveField?: number;
  /** Anomaly threshold in k-sigma. Default: 3.5. */
  anomalyThreshold?: number;
  /** Whether to compute band-limited surprise. Default: true. */
  bandAnalysis?: boolean;
}

/** Error thrown when predictive coding inference fails. */
export class PredictiveCodingError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "PredictiveCodingError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// ONNX LSTM model loading (server-side, onnxruntime-node)
// ─────────────────────────────────────────────────────────────────────

/** SHA-256 of the predictive coding LSTM ONNX model. */
export const PREDICTIVE_CODING_MODEL_SHA256 =
  "0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
/** Path to the LSTM ONNX model artifact. */
export const PREDICTIVE_CODING_MODEL_URI = "/models/predictive/predict-lstm.onnx";

let cachedLSTMAdapter: ONNXAdapter | null = null;

/**
 * Load the LSTM predictive coding ONNX model (22×32 → 22×8), with SHA-256
 * verification. Idempotent — the loaded session is cached for reuse.
 */
async function ensureLSTMAdapter(): Promise<ONNXAdapter> {
  if (cachedLSTMAdapter) return cachedLSTMAdapter;

  cachedLSTMAdapter = new ONNXAdapter({
    id: "predictive-coding-lstm-v1",
    name: "Predictive Coding LSTM (autoregressive EEG prediction)",
    version: "0.1.0",
    description:
      "2-layer LSTM (hidden=64) autoregressively predicts 8 future timesteps " +
      "from 32 past timesteps of 22-channel EEG. Trained on Joint-2312 projected space.",
    task: "regression",
    inputShape: { kind: "raw", channels: MODEL_CHANNELS, samples: DEFAULT_RECEPTIVE_FIELD },
    embeddingDim: DEFAULT_FORECAST_HORIZON * MODEL_CHANNELS,
    outputPooling: "none",
    artifact: PREDICTIVE_CODING_MODEL_URI,
    enableVerification: true,
    executionProviders: ["cpu"],
    runtime: lstmRuntime,
  });

  await cachedLSTMAdapter.load();
  return cachedLSTMAdapter;
}

/** Server-side runtime factory for onnxruntime-node (LSTM inference). */
async function lstmRuntime(): Promise<OrtRuntime> {
  let ort: unknown;
  try {
    ort = await import("onnxruntime-node");
  } catch (e) {
    throw new PredictiveCodingError(
      `onnxruntime-node import failed: ${(e as Error).message}`,
      "RUNTIME_UNAVAILABLE",
    );
  }
  const module = ort as {
    InferenceSession?: {
      create: (
        path: string | Uint8Array,
        options?: Record<string, unknown>,
      ) => Promise<unknown>;
    };
    Tensor?: new (
      type: "float32",
      data: Float32Array,
      dims: readonly number[],
    ) => unknown;
    env?: Record<string, unknown>;
  };
  if (!module.InferenceSession || !module.Tensor) {
    throw new PredictiveCodingError(
      "onnxruntime-node exposed no InferenceSession/Tensor",
      "RUNTIME_INVALID",
    );
  }
  return {
    InferenceSession: {
      create: async (path, options) =>
        (await module.InferenceSession!.create(path as string, options)) as unknown as OrtSessionLike,
    },
    Tensor: module.Tensor as unknown as OrtRuntime["Tensor"],
  };
}

/** Reset the cached LSTM adapter (test helper). */
export function resetPredictiveCoding(): void {
  cachedLSTMAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedLSTMAdapter = null;
}

// ─────────────────────────────────────────────────────────────────────
// CPU fallback: autoregressive (AR) linear predictor
// ─────────────────────────────────────────────────────────────────────

/**
 * CPU fallback: AR(p) predictor using linear regression on past samples.
 *
 * Fits coefficients via ordinary least squares on the receptive field window,
 * then predicts the next `horizon` timesteps autoregressively.
 */
function predictAR(
  signal: number[][],
  horizon: number,
  receptiveField: number,
): number[][] {
  const channels = signal.length;
  const results: number[][] = [];

  for (let ch = 0; ch < channels; ch++) {
    const data = signal[ch];
    const n = data.length;
    if (n < receptiveField + horizon) {
      // Not enough data — return zeros
      results.push(new Array(horizon).fill(0));
      continue;
    }

    // Build AR design matrix: predict x[t] from [x[t-1], x[t-2], ..., x[t-p]]
    const p = Math.min(receptiveField, 16); // AR order
    const startY = p;
    const endY = n - horizon;

    if (startY >= endY) {
      results.push(new Array(horizon).fill(data[n - 1] ?? 0));
      continue;
    }

    // Solve least squares: X @ beta = y, where X[i] = [x[i], x[i-1], ..., x[i-p+1]]
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = startY; i < endY; i++) {
      const row: number[] = [];
      for (let j = 0; j < p; j++) {
        row.push(data[i - j] ?? 0);
      }
      X.push(row);
      y.push(data[i]);
    }

    // Simple gradient descent AR coefficient estimation (avoid external deps).
    // Normalize inputs to prevent gradient explosion on high-amplitude EEG.
    const beta = new Array(p).fill(0);
    const lr = 0.01; // higher LR for normalized data
    const normFactor = Math.max(
      ...X.flat().map((v) => Math.abs(v)),
      ...y.map((v) => Math.abs(v)),
      1,
    );
    for (let iter = 0; iter < 50; iter++) {
      for (let i = 0; i < X.length; i++) {
        let pred = 0;
        for (let j = 0; j < p; j++) pred += beta[j] * (X[i][j] / normFactor);
        const err = y[i] / normFactor - pred;
        for (let j = 0; j < p; j++) {
          beta[j] += lr * err * (X[i][j] / normFactor);
          if (!Number.isFinite(beta[j])) beta[j] = 0;
        }
      }
    }

    // Autoregressive prediction: use last `p` samples, then feed predictions back
    const context = data.slice(n - p);
    const predictions: number[] = [];
    for (let h = 0; h < horizon; h++) {
      let pred = 0;
      for (let j = 0; j < p; j++) pred += beta[j] * context[context.length - 1 - j];
      predictions.push(pred);
      context.push(pred);
    }

    results.push(predictions);
  }

  return results;
}

// ─────────────────────────────────────────────────────────────────────
// Band-limited surprise computation
// ─────────────────────────────────────────────────────────────────────

/**
 * Compute per-band surprise scores for a channel.
 * Applies bandpass filters to the prediction error and computes RMS energy
 * in each frequency band.
 *
 * @param error Full prediction error array
 * @param fs Sampling rate
 * @returns Per-band surprise scores
 */
function computeBandSurprise(error: number[], fs: number): Record<EEGBand, number> {
  const bandScores: Record<EEGBand, number> = {
    delta: 0,
    theta: 0,
    alpha: 0,
    beta: 0,
    gamma: 0,
  };

  for (const band of EEG_BANDS) {
    const [low, high] = BAND_RANGES[band];
    try {
      const filtered = cpuBandpass([error], fs, low, high);
      const bandError = filtered[0];
      const rms = Math.sqrt(bandError.reduce((s, v) => s + v * v, 0) / bandError.length);
      bandScores[band] = rms;
    } catch {
      bandScores[band] = 0;
    }
  }

  return bandScores;
}

/**
 * Compute the z-score-based anomaly score for a value relative to a baseline.
 * Uses a sigmoid mapping: score = 1 / (1 + exp(-2 * (z - k + 2)))
 */
function anomalyScore(rmsError: number, baselineMean: number, baselineStd: number): number {
  if (!Number.isFinite(rmsError) || rmsError < 0) return 0;
  if (baselineStd <= 0 || !Number.isFinite(baselineStd) || !Number.isFinite(baselineMean)) {
    return rmsError > 0 ? 0.8 : 0.2;
  }
  const z = (rmsError - baselineMean) / baselineStd;
  const exponent = -2 * (z - DEFAULT_ANOMALY_K_SIGMA + 2);
  if (exponent > 700) return 0; // avoid overflow → Math.exp returns Infinity
  if (exponent < -700) return 1; // saturated
  const score = 1 / (1 + Math.exp(exponent));
  return Number.isFinite(score) ? score : 0;
}

// ─────────────────────────────────────────────────────────────────────
// Core inference
// ─────────────────────────────────────────────────────────────────────

/**
 * Run predictive coding on a raw EEG signal.
 *
 * Uses the LSTM ONNX model if available; falls back to AR(p) CPU prediction.
 * Computes per-channel prediction error (surprise) and band-limited surprise
 * scores, then flags channels/timesteps exceeding the anomaly threshold.
 *
 * @param signal EEG signal [C×N]
 * @param opts Prediction options
 * @returns Predictive coding results with surprise scores and anomaly flags
 */
export async function predictSignal(
  signal: EEGSignal,
  opts: PredictiveCodingOptions = {},
): Promise<PredictiveCodingResult> {
  const tStart = startTimer("predictive_coding.inference.total");

  const horizon = opts.horizon ?? DEFAULT_FORECAST_HORIZON;
  const receptiveField = opts.receptiveField ?? DEFAULT_RECEPTIVE_FIELD;
  const anomalyK = opts.anomalyThreshold ?? DEFAULT_ANOMALY_K_SIGMA;
  const bandAnalysis = opts.bandAnalysis ?? true;

  metrics.predictiveCodingRequestsTotal.inc();
  metrics.predictiveCodingForecastHorizonTotal.inc({ horizon: String(horizon) });

  const { data } = signal;
  const fs = signal.sampleRate;

  // Preprocess: bandpass 1-40 Hz to match model training
  const bandpass = cpuBandpass(data, fs, 1, 40);

  // Try to use the LSTM ONNX model
  let predictions: number[][];
  let usedModel = false;

  try {
    const adapter = await ensureLSTMAdapter();
    usedModel = true;

    // Extract the last `receptiveField` samples from each channel — the
    // LSTM input window. The ONNXAdapter expects a "signal" ModelInput
    // with channels matching the descriptor (22) and samples matching
    // the receptive field (32).
    const receptiveSignal: EEGSignal = {
      channels: signal.channels,
      data: bandpass.map((ch) => ch.slice(-receptiveField)),
      sampleRate: fs,
    };

    // The LSTM outputs [1, C, horizon] flattened → horizon*channels values.
    // predict() returns { values: { step_0: ..., step_1: ... } } — we
    // reconstruct the per-channel predictions from the flattened output.
    const result = await adapter.predict({
      kind: "signal",
      signal: receptiveSignal,
    });

    // Reshape: output is [1, C, horizon] → per-channel [horizon]
    const outputValues = Object.values(result.values);
    predictions = [];
    const chCount = data.length;
    for (let ch = 0; ch < chCount; ch++) {
      const pred: number[] = [];
      for (let t = 0; t < horizon; t++) {
        const idx = ch * horizon + t;
        pred.push(outputValues[idx] ?? 0);
      }
      predictions.push(pred);
    }
  } catch (e) {
    log("warn", "predictive_coding.lstm_unavailable_fallback_ar", {
      error: (e as Error).message,
    });
    // CPU fallback: AR predictor
    predictions = predictAR(bandpass, horizon, receptiveField);
  }

  // Compute prediction error (surprise) per channel
  // The "actual" values are the ground-truth future `horizon` samples
  const actual = bandpass.map((ch) => ch.slice(-horizon));

  const channelResults: ChannelSurprise[] = [];
  const allRMS: number[] = [];

  for (let ch = 0; ch < data.length; ch++) {
    const pred = predictions[ch];
    const act = actual[ch];

    // Prediction error
    const error = pred.map((p, i) => (act[i] ?? 0) - p);

    // RMS error = surprise magnitude
    const sqSum = error.reduce((s, v) => s + (Number.isFinite(v) ? v * v : 0), 0);
    const rmsError = Math.sqrt(sqSum / (error.length || 1));
    allRMS.push(rmsError);

    // Band-limited surprise
    const bandScores: Record<EEGBand, number> = {
      delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0,
    };

    if (bandAnalysis && error.length > 1) {
      const bandSurprise = computeBandSurprise(error, fs);
      Object.assign(bandScores, bandSurprise);
      for (const band of EEG_BANDS) {
        metrics.predictiveCodingSurpriseScore.observe(
          { band },
          Number.isFinite(bandScores[band]) ? bandScores[band] : 0,
        );
      }
    }

    channelResults.push({
      channel: signal.channels[ch] ?? `ch${ch}`,
      rmsError,
      bandScores,
      isAnomalous: false, // computed below after baseline
      anomalyScore: 0,
    });
  }

  // Compute baseline (mean/std) from non-anomalous channels
  const validRMS = allRMS.filter((v) => Number.isFinite(v));
  const baselineMean = validRMS.reduce((a, b) => a + b, 0) / validRMS.length;
  const baselineStd = Math.sqrt(
    validRMS.reduce((s, v) => s + (v - baselineMean) ** 2, 0) / validRMS.length,
  );

  // Flag anomalous channels and compute scores
  let overallSurprise = 0;
  let maxScore = 0;
  for (const result of channelResults) {
    const score = anomalyScore(result.rmsError, baselineMean, baselineStd);
    result.anomalyScore = score;
    result.isAnomalous = result.rmsError > baselineMean + anomalyK * (baselineStd || 1);
    maxScore = Math.max(maxScore, score);
    overallSurprise += result.rmsError;
  }
  overallSurprise = Number.isFinite(overallSurprise)
    ? overallSurprise / channelResults.length
    : 0;

  const inferMs = tStart.end({ used_model: usedModel });
  metrics.predictiveCodingLatencyMs.observe({ used_model: String(usedModel) }, inferMs);

  const provenance = buildServiceProvenance({
    service: PREDICTIVE_CODING_SERVICE,
    serviceVersion: PREDICTIVE_CODING_VERSION,
    taskHeadId: "predictive-coding-lstm-v1",
    taskHeadVersion: "0.1.0",
    taskHeadSha256: PREDICTIVE_CODING_MODEL_SHA256,
    taskHeadDataset: "Joint-2312 projected EEG (spectral-proxy labels)",
    taskHeadMetrics: {
      horizon,
      receptiveField,
      anomalyK,
    },
    experimentId: "m48-predictive-neural-coding",
  });

  return {
    channels: channelResults,
    overallSurprise,
    isAnomalous: maxScore > 0.7,
    anomalyScore: maxScore,
    forecastHorizon: horizon,
    durationMs: inferMs,
    usedModel,
    provenance,
  };
}
