import type { EEGSignal } from "../eeg/types";

export type QualityLevel = "good" | "warning" | "bad";

export interface ChannelQuality {
  channel: string;
  level: QualityLevel;
  issues: string[];
  rms: number;
  flatPercent: number;
  clippedPercent: number;
  nanPercent: number;
}

export interface SignalQualityReport {
  overall: QualityLevel;
  score: number;
  channels: ChannelQuality[];
  warnings: string[];
  errors: string[];
}

const FLAT_THRESHOLD = 1e-6;
const MAX_RMS_UV = 200;
const MIN_RMS_UV = 0.05;

export function checkSignalQuality(signal: EEGSignal): SignalQualityReport {
  const warnings: string[] = [];
  const errors: string[] = [];
  const channelReports: ChannelQuality[] = [];

  for (let c = 0; c < signal.channels.length; c++) {
    const ch = signal.channels[c];
    const samples = signal.data[c];
    if (!samples || samples.length === 0) {
      errors.push(`Channel ${ch}: no data`);
      channelReports.push({ channel: ch, level: "bad", issues: ["No data"], rms: 0, flatPercent: 100, clippedPercent: 0, nanPercent: 100 });
      continue;
    }

    const issues: string[] = [];
    const N = samples.length;

    const nanCount = samples.filter((v) => !Number.isFinite(v)).length;
    const nanPercent = (nanCount / N) * 100;
    if (nanPercent > 5) issues.push(`${nanPercent.toFixed(1)}% missing values`);

    const rms = Math.sqrt(samples.reduce((s, v) => s + v * v, 0) / N);

    let flatCount = 0;
    for (let i = 1; i < N; i++) {
      if (Math.abs(samples[i] - samples[i - 1]) < FLAT_THRESHOLD) flatCount++;
    }
    const flatPercent = (flatCount / (N - 1)) * 100;
    if (flatPercent > 30) issues.push(`Flat line (${flatPercent.toFixed(0)}% of samples)`);

    const absMax = Math.max(...samples.map(Math.abs));
    const clippedCount = samples.filter((v) => Math.abs(v) >= absMax * 0.999 && absMax > 1).length;
    const clippedPercent = (clippedCount / N) * 100;
    if (clippedPercent > 1) issues.push(`Clipping detected (${clippedPercent.toFixed(1)}%)`);

    if (rms > MAX_RMS_UV) issues.push(`High amplitude RMS=${rms.toFixed(1)} µV`);
    if (rms < MIN_RMS_UV && flatPercent < 10) issues.push(`Very low amplitude RMS=${rms.toFixed(4)} µV`);

    const level: QualityLevel = issues.length === 0 ? "good" : issues.length <= 1 ? "warning" : "bad";
    channelReports.push({ channel: ch, level, issues, rms, flatPercent, clippedPercent, nanPercent });
  }

  const badCount = channelReports.filter((c) => c.level === "bad").length;
  const warnCount = channelReports.filter((c) => c.level === "warning").length;
  const total = channelReports.length || 1;
  const score = Math.max(0, Math.round(100 - (badCount / total) * 60 - (warnCount / total) * 20));
  const overall: QualityLevel = badCount > total * 0.3 ? "bad" : badCount > 0 || warnCount > total * 0.3 ? "warning" : "good";

  if (signal.sampleRate < 128) warnings.push(`Low sample rate (${signal.sampleRate} Hz)`);
  if (signal.channels.length < 2) warnings.push("Single-channel signal");
  if ((signal.data[0]?.length ?? 0) / signal.sampleRate < 2) warnings.push("Recording too short (< 2s)");

  const nanMeta = signal.meta?.nan_percent as number | undefined;
  if (nanMeta && nanMeta > 5) errors.push(`File contained ${nanMeta.toFixed(1)}% non-finite values`);

  return { overall, score, channels: channelReports, warnings, errors };
}

export function qualityColor(level: QualityLevel): string {
  return level === "good" ? "#22c55e" : level === "warning" ? "#f59e0b" : "#ef4444";
}

export function qualityLabel(level: QualityLevel): string {
  return level === "good" ? "Good" : level === "warning" ? "Warning" : "Poor";
                                       }
