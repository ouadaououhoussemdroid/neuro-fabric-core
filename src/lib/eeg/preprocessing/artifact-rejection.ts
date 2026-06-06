import type { EEGWindow } from "../types";

export interface ArtifactReport {
  totalWindows: number;
  rejectedWindows: number;
  rejectedPercent: number;
  keptWindows: number;
  reasons: Record<string, number>;
  thresholds: ArtifactThresholds;
}

export interface ArtifactThresholds {
  amplitudeStd: number;
  flatlineStd: number;
  jumpThreshold: number;
}

const DEFAULT_THRESHOLDS: ArtifactThresholds = {
  amplitudeStd: 5,
  flatlineStd: 1e-6,
  jumpThreshold: 8,
};

export function detectWindowArtifacts(win: EEGWindow, globalMean: number[], globalStd: number[], thresholds: ArtifactThresholds = DEFAULT_THRESHOLDS): string[] {
  const reasons: string[] = [];
  const C = win.data.length;
  for (let c = 0; c < C; c++) {
    const ch = win.data[c];
    if (!ch || ch.length === 0) continue;
    const mean = globalMean[c] ?? 0;
    const std = Math.max(globalStd[c] ?? 1, 1e-9);
    const limit = thresholds.amplitudeStd * std;
    if (ch.some((v) => Math.abs(v - mean) > limit)) { reasons.push(`ch${c}:amplitude`); break; }
    const variance = ch.reduce((s, v) => s + (v - mean) ** 2, 0) / ch.length;
    if (variance < thresholds.flatlineStd) { reasons.push(`ch${c}:flatline`); break; }
    const jumpLimit = thresholds.jumpThreshold * std;
    for (let i = 1; i < ch.length; i++) {
      if (Math.abs(ch[i] - ch[i - 1]) > jumpLimit) { reasons.push(`ch${c}:jump`); break; }
    }
    if (reasons.length > 0) break;
  }
  return reasons;
}

function computeGlobalStats(windows: EEGWindow[]): { mean: number[]; std: number[] } {
  if (windows.length === 0) return { mean: [], std: [] };
  const C = windows[0].data.length;
  const mean = new Array<number>(C).fill(0);
  const std = new Array<number>(C).fill(1);
  let totalSamples = 0;
  for (const w of windows) {
    for (let c = 0; c < C; c++) {
      const ch = w.data[c];
      if (!ch) continue;
      totalSamples += ch.length;
      mean[c] += ch.reduce((s, v) => s + v, 0);
    }
  }
  if (totalSamples > 0) {
    const samplesPerChannel = totalSamples / windows.length;
    for (let c = 0; c < C; c++) mean[c] /= samplesPerChannel;
  }
  const variance = new Array<number>(C).fill(0);
  let varSamples = 0;
  for (const w of windows) {
    for (let c = 0; c < C; c++) {
      const ch = w.data[c];
      if (!ch) continue;
      varSamples += ch.length;
      variance[c] += ch.reduce((s, v) => s + (v - mean[c]) ** 2, 0);
    }
  }
  if (varSamples > 0) {
    const samplesPerChannel = varSamples / windows.length;
    for (let c = 0; c < C; c++) std[c] = Math.sqrt(variance[c] / samplesPerChannel) || 1;
  }
  return { mean, std };
}

export function rejectArtifacts(windows: EEGWindow[], options: { thresholds?: Partial<ArtifactThresholds>; maxContaminationPercent?: number } = {}): { windows: EEGWindow[]; report: ArtifactReport } {
  const thresholds: ArtifactThresholds = { ...DEFAULT_THRESHOLDS, ...options.thresholds };
  const maxContamination = options.maxContaminationPercent ?? 40;
  if (windows.length === 0) return { windows: [], report: { totalWindows: 0, rejectedWindows: 0, rejectedPercent: 0, keptWindows: 0, reasons: {}, thresholds } };
  const { mean, std } = computeGlobalStats(windows);
  const clean: EEGWindow[] = [];
  const reasons: Record<string, number> = {};
  let rejected = 0;
  for (const win of windows) {
    const r = detectWindowArtifacts(win, mean, std, thresholds);
    if (r.length === 0) { clean.push(win); }
    else { rejected++; for (const reason of r) reasons[reason] = (reasons[reason] ?? 0) + 1; }
  }
  const rejectedPercent = (rejected / windows.length) * 100;
  if (rejectedPercent > maxContamination) console.warn(`[artifact-rejection] ${rejectedPercent.toFixed(1)}% of windows rejected.`);
  return { windows: clean, report: { totalWindows: windows.length, rejectedWindows: rejected, rejectedPercent, keptWindows: clean.length, reasons, thresholds } };
                                    }
