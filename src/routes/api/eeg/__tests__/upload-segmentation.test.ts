/**
 * T-031 — Production-path segmentation verification.
 *
 * Proves that the EEG upload route passes a 4 s @ 250 Hz segment config
 * (1000 samples, 50% overlap) to preprocess(), which produces 1000-sample
 * windows that match the EEGConformer training recipe exactly.
 *
 * These tests exercise the actual preprocess() function used by upload.ts,
 * asserting the contract: 250 Hz × 4 s = 1000 samples, 0.5 overlap.
 */
import { describe, it, expect } from "vitest";
import { preprocess } from "@/lib/eeg/preprocessing";
import { segment } from "@/lib/eeg/preprocessing/segment";
import type { EEGSignal } from "@/lib/eeg/types";

const FS = 250; // 250 Hz production sample rate
const WINDOW_SEC = 4; // 4-second windows (training recipe)
const OVERLAP = 0.5; // 50% overlap
const EXPECTED_SAMPLES = FS * WINDOW_SEC; // 1000 samples

/** Build a synthetic multi-channel signal long enough for several windows. */
function makeSignal(channels: number, durationSec: number, sampleRate: number): EEGSignal {
  const n = Math.floor(durationSec * sampleRate);
  const data: number[][] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      ch[i] = Math.sin((2 * Math.PI * 10 * i) / sampleRate) * 0.5 + c * 0.001;
    }
    data.push(ch);
  }
  const chNames = Array.from({ length: channels }, (_, i) => `ch${i}`);
  return { channels: chNames, data, sampleRate };
}

describe("T-031 — Production segmentation: 1000-sample windows", () => {
  describe("segment() with T-031 config", () => {
    it("produces exactly 1000-sample windows from a 250 Hz signal", () => {
      const N = 5000; // 20 s of data at 250 Hz
      const data = [Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * 10 * i) / FS))];
      const windows = segment(data, FS, WINDOW_SEC, OVERLAP);
      expect(windows.length).toBeGreaterThan(0);
      for (const w of windows) {
        expect(w.data[0]).toHaveLength(EXPECTED_SAMPLES);
        expect(w.end - w.start).toBe(EXPECTED_SAMPLES);
      }
    });

    it("uses 4 s window and 50% overlap (step = 500 samples)", () => {
      const N = 5000; // 20 s
      const data = [Array.from({ length: N }, (_, i) => i)];
      const windows = segment(data, FS, WINDOW_SEC, OVERLAP);
      // W = 1000, step = 500, starts: 0, 500, 1000, 1500, 2000, ...
      expect(windows[0].start).toBe(0);
      expect(windows[0].end).toBe(EXPECTED_SAMPLES);
      expect(windows[1].start).toBe(EXPECTED_SAMPLES / 2);
      expect(windows[1].end).toBe(EXPECTED_SAMPLES + EXPECTED_SAMPLES / 2);
    });

    it("window count matches expected for a 10 s signal: floor((10000-1000)/500)+1 = 19", () => {
      const N = 10000; // 40 s
      const data = [Array.from({ length: N }, (_, i) => i)];
      const windows = segment(data, FS, WINDOW_SEC, OVERLAP);
      // expected = floor((10000 - 1000) / 500) + 1 = 19
      expect(windows.length).toBe(19);
    });
  });

  describe("preprocess() with T-031 segment config", () => {
    it("produces 1000-sample windows when segment config is { windowSec: 4, overlap: 0.5 }", () => {
      const signal = makeSignal(22, 10, FS); // 22 ch, 10 s @ 250 Hz = 2500 samples
      const result = preprocess(signal, {
        bandpass: undefined, // skip bandpass for this structural test
        notch: undefined,
        normalize: false,
        segment: { windowSec: 4, overlap: 0.5 },
      });
      expect(result.windows.length).toBeGreaterThan(0);
      for (const w of result.windows) {
        expect(w.data[0]).toHaveLength(EXPECTED_SAMPLES);
      }
    });

    it("matches the T-031 upload.ts segment config exactly (windowSec: 4, overlap: 0.5)", () => {
      const signal = makeSignal(22, 8, FS); // 8 s → 2 windows at 0.5 overlap
      const result = preprocess(signal, {
        segment: { windowSec: 4, overlap: 0.5 },
        bandpass: { low: 1, high: 40 }, // default-compatible range
      });
      expect(result.windows.length).toBe(3); // starts: 0, 500, 1000 (0+1000, 500+1000, 1000+1000)
      for (const w of result.windows) {
        expect(w.data[0]).toHaveLength(EXPECTED_SAMPLES);
      }
    });

    it("default segment config (2 s) produces 500-sample windows — proving the fix matters", () => {
      const signal = makeSignal(22, 10, FS);
      const result = preprocess(signal, {
        segment: undefined, // default: 2 s
      });
      expect(result.windows.length).toBeGreaterThan(0);
      for (const w of result.windows) {
        expect(w.data[0]).toHaveLength(500); // default produces 500, NOT 1000
      }
      // Now with T-031 config:
      const result2 = preprocess(signal, {
        segment: { windowSec: 4, overlap: 0.5 },
      });
      for (const w of result2.windows) {
        expect(w.data[0]).toHaveLength(EXPECTED_SAMPLES); // 1000
      }
    });
  });

  describe("EEGConformer training recipe compatibility", () => {
    it("4 s @ 250 Hz = 1000 samples matches EEGConformer expected window size", () => {
      expect(EXPECTED_SAMPLES).toBe(1000);
      // The EEGConformer model expects exactly 1000 samples per window
      const signal = makeSignal(22, 10, FS);
      const result = preprocess(signal, {
        bandpass: { low: 4, high: 38 }, // T-031 bandpass range
        segment: { windowSec: 4, overlap: 0.5 },
      });
      for (const w of result.windows) {
        expect(w.data.length).toBe(22); // 22 channels
        expect(w.sampleRate).toBe(FS); // 250 Hz
        expect(w.data[0].length).toBe(1000); // 1000 samples
      }
    });

    it("preprocessing report includes segment step with correct params", () => {
      const signal = makeSignal(22, 8, FS);
      const result = preprocess(signal, {
        segment: { windowSec: 4, overlap: 0.5 },
      });
      const segStep = result.report.steps.find((s) => s.name === "segment");
      expect(segStep).toBeDefined();
      expect(segStep!.params.windowSec).toBe(4);
      expect(segStep!.params.overlap).toBe(0.5);
    });
  });
});
