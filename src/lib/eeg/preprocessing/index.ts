import type { EEGSignal, EEGWindow, PreprocessingReport } from "../types";
import { bandpass, notch } from "./filters";
import { zscore } from "./normalize";
import { segment } from "./segment";
import { rejectArtifacts } from "./artifact-rejection";
import type { ArtifactThresholds, ArtifactReport } from "./artifact-rejection";

export { bandpass, notch, zscore, segment, rejectArtifacts };
export type { ArtifactThresholds, ArtifactReport };

export interface PreprocessOptions {
  bandpass?: { low: number; high: number } | false;
  notch?: { fc: 50 | 60; q?: number } | false;
  normalize?: boolean;
  segment?: { windowSec: number; overlap: number } | false;
  artifactRejection?:
    | {
        enabled?: boolean;
        thresholds?: Partial<ArtifactThresholds>;
        maxContaminationPercent?: number;
      }
    | false;
}

export interface PreprocessResult {
  signal: EEGSignal;
  windows: EEGWindow[];
  report: PreprocessingReport;
  artifactReport?: ArtifactReport;
}

const DEFAULTS = {
  bandpass: { low: 1, high: 40 },
  notch: { fc: 60 as const, q: 30 },
  normalize: true,
  segment: { windowSec: 2, overlap: 0.5 },
  artifactRejection: { enabled: true, maxContaminationPercent: 40 },
};

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
    steps.push({
      name: "segment",
      params: { ...seg, count: windows.length },
      durationMs: +(performance.now() - s).toFixed(2),
    });
  }

  let artifactReport: ArtifactReport | undefined;
  const arOpts = opts.artifactRejection;
  const arEnabled = arOpts !== false && arOpts?.enabled !== false;

  if (arEnabled && windows.length > 0) {
    const s = performance.now();
    const arConfig: { thresholds?: Partial<ArtifactThresholds>; maxContaminationPercent?: number } =
      arOpts && typeof arOpts === "object" ? arOpts : {};
    const { windows: cleanWindows, report: arReport } = rejectArtifacts(windows, {
      thresholds: arConfig?.thresholds,
      maxContaminationPercent:
        arConfig?.maxContaminationPercent ?? DEFAULTS.artifactRejection.maxContaminationPercent,
    });
    artifactReport = arReport;
    windows = cleanWindows;
    steps.push({
      name: "artifact-rejection",
      params: {
        totalWindows: arReport.totalWindows,
        rejectedWindows: arReport.rejectedWindows,
        rejectedPercent: +arReport.rejectedPercent.toFixed(1),
        keptWindows: arReport.keptWindows,
      },
      durationMs: +(performance.now() - s).toFixed(2),
    });
  }

  const signal: EEGSignal = { ...input, data };
  const report: PreprocessingReport = {
    channels: data.length,
    samples: data[0]?.length ?? 0,
    sampleRate: fs,
    steps,
    totalDurationMs: +(performance.now() - t0).toFixed(2),
  };

  return { signal, windows, report, artifactReport };
}
