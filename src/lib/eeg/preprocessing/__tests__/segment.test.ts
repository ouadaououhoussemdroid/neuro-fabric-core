import { describe, it, expect } from "vitest";
import { segment } from "../segment";

describe("segment", () => {
  it("returns an empty array for empty input", () => {
    expect(segment([], 128)).toEqual([]);
  });

  it("returns an empty array when the window is longer than the signal", () => {
    const data = [Array.from({ length: 10 }, (_, i) => i)];
    expect(segment(data, 128, 2, 0.5)).toEqual([]); // 2s @ 128Hz = 256 samples > 10
  });

  it("produces windows of the exact requested length", () => {
    const N = 1000;
    const data = [Array.from({ length: N }, (_, i) => i)];
    const windows = segment(data, 250, 2, 0.5); // W = 500
    expect(windows.length).toBeGreaterThan(0);
    for (const w of windows) {
      expect(w.data[0]).toHaveLength(500);
      expect(w.end - w.start).toBe(500);
    }
  });

  it("computes the correct window count and step for a known overlap", () => {
    // fs=100, windowSec=1 -> W=100; overlap=0.5 -> step=50
    const N = 300;
    const data = [Array.from({ length: N }, (_, i) => i)];
    const windows = segment(data, 100, 1, 0.5);
    // starts: 0, 50, 100, 150, 200 (200+100=300 <= 300 OK), 250 would need 350 > 300 -> stop
    expect(windows.map((w) => w.start)).toEqual([0, 50, 100, 150, 200]);
  });

  it("produces non-overlapping windows when overlap=0", () => {
    const N = 300;
    const data = [Array.from({ length: N }, (_, i) => i)];
    const windows = segment(data, 100, 1, 0);
    expect(windows.map((w) => w.start)).toEqual([0, 100, 200]);
  });

  it("slices data correctly (window content matches the source range)", () => {
    const data = [Array.from({ length: 20 }, (_, i) => i * 10)];
    const windows = segment(data, 10, 1, 0); // W=10, step=10 -> starts 0, 10
    expect(windows[0].data[0]).toEqual(data[0].slice(0, 10));
    expect(windows[1].data[0]).toEqual(data[0].slice(10, 20));
  });

  it("slices all channels consistently at the same start/end", () => {
    const data = [
      Array.from({ length: 20 }, (_, i) => i),
      Array.from({ length: 20 }, (_, i) => i * 100),
    ];
    const windows = segment(data, 10, 1, 0);
    expect(windows[0].data[0]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(windows[0].data[1]).toEqual([0, 100, 200, 300, 400, 500, 600, 700, 800, 900]);
  });

  it("carries the sampleRate through to each window", () => {
    const data = [Array.from({ length: 50 }, (_, i) => i)];
    const windows = segment(data, 25, 1, 0);
    windows.forEach((w) => expect(w.sampleRate).toBe(25));
  });

  it("returns an empty array when windowSec resolves to <= 0 samples", () => {
    const data = [Array.from({ length: 100 }, (_, i) => i)];
    expect(segment(data, 100, 0, 0.5)).toEqual([]);
  });

  it("uses the default window (2s) and overlap (0.5) when not specified", () => {
    const N = 1000;
    const data = [Array.from({ length: N }, (_, i) => i)];
    const withDefaults = segment(data, 250);
    const explicit = segment(data, 250, 2, 0.5);
    expect(withDefaults).toEqual(explicit);
  });
});
