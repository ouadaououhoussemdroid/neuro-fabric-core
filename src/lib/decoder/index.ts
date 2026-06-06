import type { EEGSignal } from "../eeg/types";
import { bandStats, type BandStats } from "./features";
import { tfjsDecode, type TFJSDecoderResult } from "./tfjs-decoder";
import { ACTIVE_DECODER } from "../model-registry";

export { bandStats, tfjsDecode };
export type { BandStats, TFJSDecoderResult };

export interface CognitiveStateReport {
  attention: number;
  workload:  number;
  arousal:   number;
  bandStats: BandStats;
  decoder: "baseline-spectral-v1" | "tfjs-eeg-v1";
  durationMs: number;
  confidence?: number;
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
    workload:  squash(b.theta / Math.max(1e-9, b.alpha)),
    arousal:   Math.min(1, Math.max(0, b.beta + b.gamma)),
    bandStats: b,
    decoder: "baseline-spectral-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}

export function decodeCognitiveState(signal: EEGSignal): CognitiveStateReport {
  if (ACTIVE_DECODER === "tfjs-eeg-v1") {
    try {
      return tfjsDecode(signal);
    } catch (e) {
      console.warn("[decoder] tfjs-eeg-v1 failed, falling back to baseline:", e);
      return baselineDecode(signal);
    }
  }
  return baselineDecode(signal);
}
