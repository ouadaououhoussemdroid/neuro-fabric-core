import { describe, it, expect } from "vitest";
import { bandpass, notch } from "../filters";

const FS = 250;
const N = 1000; // 4s — long enough that IIR settling transients are a small fraction of total energy

function sine(freq: number, amp = 1, n = N, fs = FS): number[] {
  return Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / fs));
}

function rms(x: number[]): number {
  return Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);
}

describe("bandpass", () => {
  it("preserves an in-band signal's amplitude (>95%)", () => {
    const input = sine(5);
    const output = bandpass([input], FS, 1, 40)[0];
    expect(rms(output) / rms(input)).toBeGreaterThan(0.95);
  });

  it("strongly attenuates a signal above the passband (<5%)", () => {
    const input = sine(100);
    const output = bandpass([input], FS, 1, 40)[0];
    expect(rms(output) / rms(input)).toBeLessThan(0.05);
  });

  it("substantially removes a DC offset (highpass below the low cutoff)", () => {
    const input = new Array(N).fill(5);
    const output = bandpass([input], FS, 1, 40)[0];
    const mean = output.reduce((a, b) => a + b, 0) / output.length;
    expect(Math.abs(mean)).toBeLessThan(1); // well below the original offset of 5
  });

  it("preserves array shape (same length, same channel count)", () => {
    const output = bandpass([sine(5), sine(10)], FS, 1, 40);
    expect(output).toHaveLength(2);
    expect(output[0]).toHaveLength(N);
    expect(output[1]).toHaveLength(N);
  });

  it("filters each channel independently", () => {
    const dcChannel = new Array(N).fill(5);
    const inBandChannel = sine(5);
    const [dcOut, sineOut] = bandpass([dcChannel, inBandChannel], FS, 1, 40);
    // DC channel should be attenuated; the sine channel should not be
    // affected by what happened to the DC channel.
    expect(rms(dcOut)).toBeLessThan(1);
    expect(rms(sineOut) / rms(inBandChannel)).toBeGreaterThan(0.95);
  });

  it("throws for an invalid range (low <= 0)", () => {
    expect(() => bandpass([sine(5)], FS, 0, 40)).toThrow("invalid range");
  });

  it("throws for an invalid range (high >= Nyquist)", () => {
    expect(() => bandpass([sine(5)], FS, 1, 130)).toThrow("invalid range");
  });

  it("throws for an invalid range (low >= high)", () => {
    expect(() => bandpass([sine(5)], FS, 40, 1)).toThrow("invalid range");
  });
});

describe("notch", () => {
  it("strongly attenuates a signal at the notch frequency (<15%)", () => {
    const input = sine(50);
    const output = notch([input], FS, 50)[0];
    expect(rms(output) / rms(input)).toBeLessThan(0.15);
  });

  it("leaves a signal well away from the notch frequency largely unaffected (>95%)", () => {
    const input = sine(10);
    const output = notch([input], FS, 50)[0];
    expect(rms(output) / rms(input)).toBeGreaterThan(0.95);
  });

  it("throws for an invalid fc (<= 0)", () => {
    expect(() => notch([sine(10)], FS, 0)).toThrow("invalid fc");
  });

  it("throws for an invalid fc (>= Nyquist)", () => {
    expect(() => notch([sine(10)], FS, 130)).toThrow("invalid fc");
  });
});
