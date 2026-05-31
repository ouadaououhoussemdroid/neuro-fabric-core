import type { EEGSignal } from "../eeg/types";
import { bandStats, type BandStats } from "./features";

export { bandStats };
export type { BandStats };

/**
 * Baseline cognitive-state decoders. These are intentionally simple,
 * spectrally-grounded heuristics derived from the same band-power features
 * the embedding engine uses. They are NOT trained classifiers; they expose
 * a stable surface so a learned model can be swapped in later without
 * changing callers.
 *
 * Each score is a probability in [0,1] derived from real spectral content
 * of the input signal. No randomness, no mocked percentages.
 */

export interface CognitiveStateReport {
  attention: number;  // beta / (alpha + theta) — Pope index style
  workload:  number;  // theta(frontal-ish proxy) / alpha
  arousal:   number;  // (beta + gamma) / total
  bandStats: BandStats;
  decoder: "baseline-spectral-v1";
  durationMs: number;
}

function squash(x: number): number {
  // Map (0, +∞) → (0, 1) via logistic on log-ratio
  if (!Number.isFinite(x) || x <= 0) return 0;
  const z = Math.log(x);
  return 1 / (1 + Math.exp(-z));
}

export function decodeCognitiveState(signal: EEGSignal): CognitiveStateReport {
  const t0 = performance.now();
  const b = bandStats(signal);
  const attentionRatio = b.beta / Math.max(1e-9, b.alpha + b.theta);
  const workloadRatio = b.theta / Math.max(1e-9, b.alpha);
  const arousalFrac = b.beta + b.gamma; // already normalized to total
  return {
    attention: squash(attentionRatio),
    workload:  squash(workloadRatio),
    arousal:   Math.min(1, Math.max(0, arousalFrac)),
    bandStats: b,
    decoder: "baseline-spectral-v1",
    durationMs: +(performance.now() - t0).toFixed(2),
  };
}