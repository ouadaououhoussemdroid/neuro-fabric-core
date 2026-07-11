import { describe, it, expect } from "vitest";
import { zscore, demean } from "../normalize";

function mean(x: number[]): number {
  return x.reduce((a, b) => a + b, 0) / x.length;
}
function std(x: number[]): number {
  const m = mean(x);
  return Math.sqrt(x.reduce((s, v) => s + (v - m) ** 2, 0) / x.length);
}

describe("zscore", () => {
  it("produces mean ~0 and std ~1 per channel", () => {
    const data = [
      [1, 2, 3, 4, 5],
      [10, 20, 30, 40, 100],
    ];
    const out = zscore(data);
    for (const ch of out) {
      expect(mean(ch)).toBeCloseTo(0, 10);
      expect(std(ch)).toBeCloseTo(1, 10);
    }
  });

  it("normalizes each channel independently", () => {
    const data = [
      [0, 0, 0, 10], // very different distribution
      [5, 5, 5, 5.1],
    ];
    const [ch0, ch1] = zscore(data);
    // Different input scales should not leak into each other's output scale
    expect(std(ch0)).toBeCloseTo(1, 8);
    expect(std(ch1)).toBeCloseTo(1, 8);
  });

  it("guards against divide-by-zero for a constant (zero-variance) channel", () => {
    const out = zscore([[5, 5, 5, 5]]);
    expect(out[0].every((v) => Number.isFinite(v))).toBe(true);
    // std guard is `|| 1`, so a constant channel becomes all zeros (x - mean = 0)
    expect(out[0]).toEqual([0, 0, 0, 0]);
  });

  it("returns an empty array unchanged for an empty channel", () => {
    expect(zscore([[]])).toEqual([[]]);
  });

  it("preserves the number of channels and samples", () => {
    const data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const out = zscore(data);
    expect(out).toHaveLength(3);
    out.forEach((ch) => expect(ch).toHaveLength(3));
  });
});

describe("demean", () => {
  it("produces mean ~0 per channel", () => {
    const out = demean([[1, 2, 3, 4, 5]]);
    expect(mean(out[0])).toBeCloseTo(0, 10);
  });

  it("preserves relative differences between samples (only removes DC, doesn't rescale)", () => {
    const input = [10, 20, 30];
    const out = demean([input])[0];
    for (let i = 1; i < input.length; i++) {
      expect(out[i] - out[i - 1]).toBeCloseTo(input[i] - input[i - 1], 10);
    }
  });

  it("normalizes each channel independently", () => {
    const [ch0, ch1] = demean([
      [100, 102, 104],
      [-5, -3, -1],
    ]);
    expect(mean(ch0)).toBeCloseTo(0, 10);
    expect(mean(ch1)).toBeCloseTo(0, 10);
  });

  it("does not divide by anything, so a constant channel becomes all zeros", () => {
    expect(demean([[7, 7, 7]])[0]).toEqual([0, 0, 0]);
  });
});
