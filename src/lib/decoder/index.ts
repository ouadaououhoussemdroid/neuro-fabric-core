import type { EEGSignal } from "../eeg/types";
import { bandStats, type BandStats } from "./features";
import { tfjsDecode, type TFJSDecoderResult } from "./tfjs-decoder";
import { decodeWithTrainedModel, createONNXDecoder } from "./trained-decoder";

export { bandStats, tfjsDecode };
export type { BandStats, TFJSDecoderResult };

export interface CognitiveStateReport {
  attention: number;
  workload: number;
  arousal: number;
  bandStats: BandStats;
  decoder: "baseline-spectral-v1" | "tfjs-eeg-v1" | "trained-logistic-v0";
  durationMs: number;
  /** Mean confidence (0–1). Present when the trained decoder is used. */
  confidence?: number;
  /** Whether the trained model was used (false = heuristic fallback). */
  trained?: boolean;
  /** Per-metric confidence intervals [lower, upper]. Present when trained. */
  confidenceIntervals?: TrainedCognitiveReport["confidence"];
}

function squash(x: number): number {
  if (!Number.isFinite(x) || x <= 0) return 0;
  const z = Math.log(x);
  return 1 / (1 + Math.exp(-z));
}

function baselineDecode(signal: EEGSignal): CognitiveStateReport {
  const t0 = performance.now();
  const b = bandStats(signal);
  return {
    attention: squash(b.beta / Math.max(1e-9, b.alpha + b.theta)),
    workload: squash(b.theta / Math.max(1e-9, b.alpha)),
    arousal: Math.min(1, Math.max(0, b.beta + b.gamma)),
    bandStats: b,
    decoder: "baseline-spectral-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

export async function decodeCognitiveState(signal: EEGSignal): Promise<CognitiveStateReport> {
  // Try the trained cognitive decoder (ONNX logistic regression) first.
  // Falls back to the heuristic spectral baseline if the model is
  // unavailable or inference fails.
  try {
    const onnxDecoder = await createONNXDecoder();
    const report = await decodeWithTrainedModel(signal, onnxDecoder);
    if (report.trained) {
      return {
        attention: report.attention,
        workload: report.workload,
        arousal: report.arousal,
        bandStats: report.bandStats,
        decoder: "trained-logistic-v0",
        durationMs: report.durationMs,
        confidence: (report.confidence.attention[0] + report.confidence.attention[1]) / 2,
        trained: true,
        confidenceIntervals: report.confidence,
      };
    }
    // decodeWithTrainedModel fell back to heuristic internally
    return {
      attention: report.attention,
      workload: report.workload,
      arousal: report.arousal,
      bandStats: report.bandStats,
      decoder: "baseline-spectral-v1",
      durationMs: report.durationMs,
      trained: false,
      confidenceIntervals: report.confidence,
    };
  } catch (e) {
    console.warn("[decoder] trained-logistic-v0 failed, falling back to baseline:", e);
    const baseline = baselineDecode(signal);
    return { ...baseline, trained: false };
  }
}
