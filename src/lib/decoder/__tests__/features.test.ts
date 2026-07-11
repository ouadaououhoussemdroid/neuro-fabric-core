import { describe, it, expect } from "vitest";
import { bandStats } from "../features";
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

describe("bandStats", () => {
  it("normalizes to sum to 1 across the five bands", () => {
    const stats = bandStats(sineSignal([10]));
    const sum = stats.delta + stats.theta + stats.alpha + stats.beta + stats.gamma;
    expect(sum).toBeCloseTo(1, 6);
  });

  it("concentrates almost all normalized power in the alpha band for a 10Hz signal", () => {
    const stats = bandStats(sineSignal([10]));
    expect(stats.alpha).toBeGreaterThan(0.95);
    expect(stats.delta).toBeLessThan(0.05);
    expect(stats.theta).toBeLessThan(0.05);
    expect(stats.beta).toBeLessThan(0.05);
    expect(stats.gamma).toBeLessThan(0.05);
  });

  it("averages across channels and across windows (two channels in different bands split roughly evenly)", () => {
    // ch0 = pure alpha tone, ch1 = pure beta tone, equal amplitude
    const stats = bandStats(sineSignal([10, 20]));
    expect(stats.alpha).toBeGreaterThan(0.4);
    expect(stats.beta).toBeGreaterThan(0.4);
    expect(stats.alpha + stats.beta).toBeGreaterThan(0.9);
  });

  it("returns all zeros when the signal is too short to segment into a single window", () => {
    const signal = sineSignal([10], FS, 10); // far shorter than the default 2s window
    expect(bandStats(signal)).toEqual({ delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 });
  });

  it("respects a custom windowSec parameter", () => {
    // Should not throw and should still normalize to 1 with a shorter window
    const signal = sineSignal([10], FS, 500);
    const stats = bandStats(signal, 1);
    const sum = stats.delta + stats.theta + stats.alpha + stats.beta + stats.gamma;
    expect(sum).toBeCloseTo(1, 6);
  });
});
