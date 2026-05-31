import type { EEGSignal, EEGWindow, PreprocessingReport } from "../types";
import { bandpass, notch } from "./filters";
import { zscore } from "./normalize";
import { segment } from "./segment";

export { bandpass, notch, zscore, segment };

export interface PreprocessOptions {
  bandpass?: { low: number; high: number } | false;
  notch?: { fc: 50 | 60; q?: number } | false;
  normalize?: boolean;
  segment?: { windowSec: number; overlap: number } | false;
}

export interface PreprocessResult {
  signal: EEGSignal;
  windows: EEGWindow[];
  report: PreprocessingReport;
}

const DEFAULTS: Required<Omit<PreprocessOptions, "bandpass" | "notch" | "segment">> & {
  bandpass: { low: number; high: number };
  notch: { fc: 50 | 60; q: number };
  segment: { windowSec: number; overlap: number };
} = {
  bandpass: { low: 1, high: 40 },
  notch: { fc: 60, q: 30 },
  normalize: true,
  segment: { windowSec: 2, overlap: 0.5 },
};

/** Run the full preprocessing pipeline and produce timing report + windows. */
export function preprocess(input: EEGSignal, opts: PreprocessOptions = {}): PreprocessResult {
  const steps: PreprocessingReport["steps"] = [];
  const t0 = performance.now();
  let data = input.data;
  const fs = input.sampleRate;

  const bp = opts.bandpass === false ? null : { ...DEFAULTS.bandpass, ...(opts.bandpass ?? {}) };
  if (bp) {
    const s = performance.now();
    data = bandpass(data, fs, bp.low, bp.high);
    steps.push({ name: "bandpass", params: bp, durationMs: +(performance.now() - s).toFixed(2) });
  }

  const nt = opts.notch === false ? null : { ...DEFAULTS.notch, ...(opts.notch ?? {}) };
  if (nt) {
    const s = performance.now();
    data = notch(data, fs, nt.fc, nt.q);
    steps.push({ name: "notch", params: nt, durationMs: +(performance.now() - s).toFixed(2) });
  }

  if (opts.normalize !== false) {
    const s = performance.now();
    data = zscore(data);
    steps.push({ name: "zscore", params: {}, durationMs: +(performance.now() - s).toFixed(2) });
  }

  const seg = opts.segment === false ? null : { ...DEFAULTS.segment, ...(opts.segment ?? {}) };
  let windows: EEGWindow[] = [];
  if (seg) {
    const s = performance.now();
    windows = segment(data, fs, seg.windowSec, seg.overlap);
    steps.push({ name: "segment", params: { ...seg, count: windows.length }, durationMs: +(performance.now() - s).toFixed(2) });
  }

  const signal: EEGSignal = { ...input, data };
  const report: PreprocessingReport = {
    channels: data.length,
    samples: data[0]?.length ?? 0,
    sampleRate: fs,
    steps,
    totalDurationMs: +(performance.now() - t0).toFixed(2),
  };
  return { signal, windows, report };
}