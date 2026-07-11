import type { EEGSignal } from "../eeg/types";
import { bandPowerFeatures } from "../embeddings/features";
import { segment } from "../eeg/preprocessing/segment";

export interface BandStats {
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

/** Average per-band power over the whole signal, normalized to unit sum. */
export function bandStats(signal: EEGSignal, windowSec = 2): BandStats {
  const windows = segment(signal.data, signal.sampleRate, windowSec, 0.5);
  if (windows.length === 0) return { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };

  const bandsPerCh = 5;
  const C = signal.channels.length;
  const acc = [0, 0, 0, 0, 0];
  for (const w of windows) {
    const f = bandPowerFeatures(w); // [C*5]
    for (let c = 0; c < C; c++) {
      for (let b = 0; b < bandsPerCh; b++) {
        acc[b] += f[c * bandsPerCh + b];
      }
    }
  }
  const total = acc.reduce((a, b) => a + b, 0) || 1;
  return {
    delta: acc[0] / total,
    theta: acc[1] / total,
    alpha: acc[2] / total,
    beta: acc[3] / total,
    gamma: acc[4] / total,
  };
}
