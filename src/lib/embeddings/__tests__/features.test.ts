import { describe, it, expect } from "vitest";
import { bandPowerFeatures } from "../features";
import type { EEGWindow } from "../../eeg/types";

const FS = 250;
const N = 500;

function sineWindow(freqs: number[], fs = FS, n = N): EEGWindow {
  return {
    data: freqs.map((f) =>
      Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / fs)),
    ),
    sampleRate: fs,
    start: 0,
    end: n,
  };
}

// Band order: delta(0.5-4) theta(4-8) alpha(8-13) beta(13-30) gamma(30-45)
describe("bandPowerFeatures", () => {
  it("returns C * 5 features (5 bands per channel)", () => {
    const w = sineWindow([10, 20, 2]);
    expect(bandPowerFeatures(w)).toHaveLength(15);
  });

  it("concentrates power for a 10Hz tone almost entirely in the alpha band", () => {
    const w = sineWindow([10]);
    const [delta, theta, alpha, beta, gamma] = bandPowerFeatures(w);
    const total = delta + theta + alpha + beta + gamma;
    expect(alpha / total).toBeGreaterThan(0.99);
  });

  it("routes tones to the correct band across all five bands", () => {
    const cases: [number, number][] = [
      [2, 0], // delta
      [6, 1], // theta
      [10, 2], // alpha
      [20, 3], // beta
      [35, 4], // gamma
    ];
    for (const [freq, expectedIdx] of cases) {
      const bands = bandPowerFeatures(sineWindow([freq]));
      const total = bands.reduce((a, b) => a + b, 0);
      expect(bands[expectedIdx] / total).toBeGreaterThan(0.9);
    }
  });

  it("computes each channel independently and preserves channel order", () => {
    const w = sineWindow([10, 20]); // ch0=alpha tone, ch1=beta tone
    const feats = bandPowerFeatures(w);
    const ch0 = feats.slice(0, 5);
    const ch1 = feats.slice(5, 10);
    const ch0Total = ch0.reduce((a, b) => a + b, 0);
    const ch1Total = ch1.reduce((a, b) => a + b, 0);
    expect(ch0[2] / ch0Total).toBeGreaterThan(0.9); // ch0 alpha-dominant
    expect(ch1[3] / ch1Total).toBeGreaterThan(0.9); // ch1 beta-dominant
  });

  it("returns near-zero power for a silent (all-zero) channel", () => {
    const w: EEGWindow = { data: [new Array(N).fill(0)], sampleRate: FS, start: 0, end: N };
    const feats = bandPowerFeatures(w);
    feats.forEach((v) => expect(v).toBeCloseTo(0, 10));
  });
});
