/**
 * T-025 — Trained cognitive decoder v0.
 *
 * Replaces the heuristic ratio-based `baselineDecode` with a calibrated
 * logistic regression exported to ONNX (see
 * `scripts/train_cognitive_decoder.py`). The decoder loads the ONNX model
 * via onnxruntime-web and returns calibrated probabilities for attention,
 * workload, and arousal, with confidence intervals.
 *
 * When the ONNX model is unavailable (no runtime, model not loaded), it
 * falls back to the heuristic baseline so the upload pipeline never breaks.
 */
import type { EEGSignal } from "../eeg/types";
import { bandStats } from "./features";
import { bandPowerFeatures } from "../embeddings/features";
import { segment } from "../eeg/preprocessing/segment";
import { defaultRuntime, type OrtRuntime, type OrtSessionLike } from "../ai/adapters/onnx-adapter";

/** Output of the trained cognitive decoder. */
export interface TrainedCognitiveReport {
  attention: number;
  workload: number;
  arousal: number;
  /** Confidence interval [lower, upper] for each metric. */
  confidence: {
    attention: [number, number];
    workload: [number, number];
    arousal: [number, number];
  };
  bandStats: ReturnType<typeof bandStats>;
  decoder: "trained-logistic-v0" | "baseline-spectral-v1";
  durationMs: number;
  /** Whether the trained model was used (false = heuristic fallback). */
  trained: boolean;
}

/** Feature vector: 5 band powers (δ, θ, α, β, γ), averaged across channels. */
function extractFeatures(signal: EEGSignal): number[] {
  const windows = segment(signal.data, signal.sampleRate, 2, 0.5);
  if (windows.length === 0) return [0, 0, 0, 0, 0];

  const acc = [0, 0, 0, 0, 0];
  const C = signal.channels.length;
  for (const w of windows) {
    const f = bandPowerFeatures(w); // [C*5]
    for (let c = 0; c < C; c++) {
      for (let b = 0; b < 5; b++) {
        acc[b] += f[c * 5 + b];
      }
    }
  }
  const total = acc.reduce((a, b) => a + b, 0) || 1;
  return acc.map((v) => (v / total) * 100); // percentage, matching training script
}

/** Heuristic baseline (same as the existing baselineDecode). */
function heuristicDecode(signal: EEGSignal): TrainedCognitiveReport {
  const t0 = performance.now();
  const b = bandStats(signal);
  const attention = squash(b.beta / Math.max(1e-9, b.alpha + b.theta));
  const workload = squash(b.theta / Math.max(1e-9, b.alpha));
  const arousal = Math.min(1, Math.max(0, b.beta + b.gamma));
  const ci = (v: number): [number, number] => [Math.max(0, v - 0.1), Math.min(1, v + 0.1)];
  return {
    attention,
    workload,
    arousal,
    confidence: {
      attention: ci(attention),
      workload: ci(workload),
      arousal: ci(arousal),
    },
    bandStats: b,
    decoder: "baseline-spectral-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
    trained: false,
  };
}

function squash(x: number): number {
  if (!Number.isFinite(x) || x <= 0) return 0;
  const z = Math.log(x);
  return 1 / (1 + Math.exp(-z));
}

/** Compute a confidence interval from a probability + a fixed margin. */
function confidenceInterval(prob: number, margin: number): [number, number] {
  return [Math.max(0, prob - margin), Math.min(1, prob + margin)];
}

/**
 * Create an ONNX-backed decoder function that loads the trained cognitive
 * decoder model and runs inference on 5-element band-power feature vectors.
 *
 * The session is cached for reuse across calls. If the model cannot be
 * loaded (missing file, no WASM runtime, etc.), the returned function
 * throws — the caller (decodeWithTrainedModel) catches this and falls
 * back to the heuristic baseline.
 *
 * @param modelUrl  URL of the ONNX model (defaults to the shipped artefact).
 */
let cachedSession: OrtSessionLike | null = null;
let cachedRuntime: OrtRuntime | null = null;

export async function createONNXDecoder(
  modelUrl = "/models/cognitive-decoder-v0.onnx",
): Promise<(features: number[]) => Promise<[number, number, number]>> {
  if (cachedSession) {
    return runInference;
  }

  cachedRuntime = await defaultRuntime();
  if (!cachedRuntime?.InferenceSession || !cachedRuntime?.Tensor) {
    throw new Error("ONNX runtime unavailable for cognitive decoder");
  }

  cachedSession = await cachedRuntime.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });

  return runInference;
}

/** Reset the cached ONNX session (test helper). */
export function __resetCognitiveDecoderCache(): void {
  cachedSession = null;
  cachedRuntime = null;
}

async function runInference(features: number[]): Promise<[number, number, number]> {
  if (!cachedSession || !cachedRuntime) {
    throw new Error("Cognitive decoder session not loaded");
  }

  const inputName = cachedSession.inputNames[0];
  const outputName = cachedSession.outputNames[0];
  const tensor = new cachedRuntime.Tensor("float32", Float32Array.from(features), [1, 5]);
  const out = await cachedSession.run({ [inputName]: tensor });
  const output = out[outputName];
  if (!output) throw new Error(`Cognitive decoder: output "${outputName}" missing`);

  const values = Array.from(output.data as ArrayLike<number>, Number);
  return [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0];
}

/**
 * Decode cognitive state using the trained logistic regression (ONNX) with
 * a heuristic fallback.
 *
 * @param signal     The EEG signal to decode.
 * @param onnxDecoder Optional function that runs the ONNX model on the
 *                    5-element feature vector and returns
 *                    `[attention, workload, arousal]` probabilities.
 */
export function decodeWithTrainedModel(
  signal: EEGSignal,
  onnxDecoder?: (features: number[]) => Promise<[number, number, number]>,
): TrainedCognitiveReport | Promise<TrainedCognitiveReport> {
  if (!onnxDecoder) {
    return heuristicDecode(signal);
  }

  const t0 = performance.now();
  const features = extractFeatures(signal);
  const b = bandStats(signal);

  return onnxDecoder(features).then(
    ([attention, workload, arousal]) => {
      // The logistic regression outputs class probabilities; use 0.08 as
      // the CI margin (calibrated on the synthetic training set's std).
      return {
        attention,
        workload,
        arousal,
        confidence: {
          attention: confidenceInterval(attention, 0.08),
          workload: confidenceInterval(workload, 0.08),
          arousal: confidenceInterval(arousal, 0.08),
        },
        bandStats: b,
        decoder: "trained-logistic-v0",
        durationMs: +(performance.now() - t0).toFixed(2),
        trained: true,
      };
    },
    () => heuristicDecode(signal),
  );
}
