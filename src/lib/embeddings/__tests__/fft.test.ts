import { describe, it, expect } from "vitest";
import { fftPowerSpectrum, bandPowerFeatures } from "../features";
import type { EEGWindow } from "../../eeg/types";

const FS = 250;

/** Generate a pure-tone signal. */
function tone(freq: number, n: number, fs: number): number[] {
  return Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * freq * i) / fs));
}

describe("fftPowerSpectrum", () => {
  it("detects a single tone at the correct frequency", () => {
    const freq = 10;
    const x = tone(freq, 512, FS);
    const spec = fftPowerSpectrum(x, FS);
    // Find the peak bin.
    let peakIdx = 0;
    let peakPower = 0;
    for (let i = 0; i < spec.length; i++) {
      if (spec[i].power > peakPower) {
        peakPower = spec[i].power;
        peakIdx = i;
      }
    }
    expect(spec[peakIdx].freq).toBeCloseTo(freq, 0);
  });

  it("returns near-zero power for a silent signal", () => {
    const x = new Array(512).fill(0);
    const spec = fftPowerSpectrum(x, FS);
    const total = spec.reduce((s, b) => s + b.power, 0);
    expect(total).toBeCloseTo(0, 20);
  });

  it("handles non-power-of-two lengths (zero-pads internally)", () => {
    const x = tone(10, 500, FS); // 500 → pads to 512
    const spec = fftPowerSpectrum(x, FS);
    expect(spec.length).toBeGreaterThan(0);
    // Still detects the tone.
    const peak = spec.reduce((m, b) => (b.power > m.power ? b : m), spec[0]);
    expect(peak.freq).toBeCloseTo(10, 0);
  });

  it("is O(N log N): completes quickly for a large input", () => {
    const x = tone(10, 8192, 1000);
    const t0 = performance.now();
    fftPowerSpectrum(x, 1000);
    const elapsed = performance.now() - t0;
    // Generous bound — the old DFT would be ~16× slower at this size.
    expect(elapsed).toBeLessThan(100);
  });

  it("produces consistent results across repeated calls", () => {
    const x = tone(10, 512, FS);
    const a = fftPowerSpectrum(x, FS);
    const b = fftPowerSpectrum(x, FS);
    expect(a).toHaveLength(b.length);
    for (let i = 0; i < a.length; i++) {
      expect(a[i].power).toBeCloseTo(b[i].power, 10);
    }
  });
});

describe("bandPowerFeatures (FFT-backed)", () => {
  it("still routes a 10 Hz tone to the alpha band", () => {
    const w: EEGWindow = {
      data: [tone(10, 512, FS)],
      sampleRate: FS,
      start: 0,
      end: 512,
    };
    const bands = bandPowerFeatures(w);
    const total = bands.reduce((a, b) => a + b, 0);
    expect(bands[2] / total).toBeGreaterThan(0.9); // alpha
  });

  it("routes a 35 Hz tone to the gamma band", () => {
    const w: EEGWindow = {
      data: [tone(35, 512, FS)],
      sampleRate: FS,
      start: 0,
      end: 512,
    };
    const bands = bandPowerFeatures(w);
    const total = bands.reduce((a, b) => a + b, 0);
    expect(bands[4] / total).toBeGreaterThan(0.85); // gamma
  });
});
