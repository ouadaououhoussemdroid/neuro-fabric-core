import { describe, it, expect } from "vitest";
import { decodeCognitiveState, bandStats } from "../index";
import type { EEGSignal } from "../../eeg/types";

const FS = 250;
const N = 2000;

function sineSignal(freqs: number[], fs = FS, n = N): EEGSignal {
  return {
    channels: freqs.map((_, i) => `ch${i}`),
    data: freqs.map((f) =>
      Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / fs)),
    ),
    sampleRate: fs,
  };
}

// squash(x) = 1/(1+e^-ln(x)) = x/(x+1) for x>0, else 0. Verified algebraically
// against src/lib/decoder/index.ts's private squash() — used here as an
// independent closed-form re-implementation, not a copy of the source.
function squash(x: number): number {
  return Number.isFinite(x) && x > 0 ? x / (x + 1) : 0;
}

describe("decodeCognitiveState", () => {
  it("uses the baseline-spectral-v1 decoder (the active production path)", () => {
    const result = decodeCognitiveState(sineSignal([10]));
    expect(result.decoder).toBe("baseline-spectral-v1");
  });

  it("implements attention = squash(beta / (alpha + theta)) against independently-recomputed bandStats", () => {
    const signal = sineSignal([10, 20, 6]); // mix of alpha, beta, theta
    const stats = bandStats(signal);
    const expectedAttention = squash(stats.beta / Math.max(1e-9, stats.alpha + stats.theta));
    const result = decodeCognitiveState(signal);
    expect(result.attention).toBeCloseTo(expectedAttention, 10);
  });

  it("implements workload = squash(theta / alpha) against independently-recomputed bandStats", () => {
    const signal = sineSignal([10, 6]); // alpha + theta
    const stats = bandStats(signal);
    const expectedWorkload = squash(stats.theta / Math.max(1e-9, stats.alpha));
    const result = decodeCognitiveState(signal);
    expect(result.workload).toBeCloseTo(expectedWorkload, 10);
  });

  it("implements arousal = clamp(beta + gamma, 0, 1) against independently-recomputed bandStats", () => {
    const signal = sineSignal([20, 35]); // beta + gamma
    const stats = bandStats(signal);
    const expectedArousal = Math.min(1, Math.max(0, stats.beta + stats.gamma));
    const result = decodeCognitiveState(signal);
    expect(result.arousal).toBeCloseTo(expectedArousal, 10);
  });

  it("all three outputs stay within [0, 1] regardless of input", () => {
    for (const freqs of [[2], [10], [20], [35], [2, 10, 20, 35]]) {
      const result = decodeCognitiveState(sineSignal(freqs));
      expect(result.attention).toBeGreaterThanOrEqual(0);
      expect(result.attention).toBeLessThanOrEqual(1);
      expect(result.workload).toBeGreaterThanOrEqual(0);
      expect(result.workload).toBeLessThanOrEqual(1);
      expect(result.arousal).toBeGreaterThanOrEqual(0);
      expect(result.arousal).toBeLessThanOrEqual(1);
    }
  });

  it("returns squashed 0 outputs (not NaN/Infinity) for a signal too short to produce any bandStats", () => {
    const signal = sineSignal([10], FS, 10); // shorter than the default window
    const result = decodeCognitiveState(signal);
    expect(result.attention).toBe(0);
    expect(result.workload).toBe(0);
    expect(Number.isFinite(result.arousal)).toBe(true);
  });

  it("includes the underlying bandStats and a non-negative durationMs in the report", () => {
    const result = decodeCognitiveState(sineSignal([10]));
    expect(result.bandStats).toBeDefined();
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });
});
