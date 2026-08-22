/**
 * M33 — SEED dataset loader.
 *
 * SEED (Scientific Electroenceural Database) is a publicly available EEG dataset
 * for emotion/affective-state research (Zheng & Lu, 2013). Each subject watches
 * 15 movie clips (5 positive, 5 negative, 5 neutral) across 3 sessions, with
 * 62-channel EEG at 200 Hz. Per-trial valence/arousal/dominance ratings (1-9 scale)
 * and NASA-TLX workload scores are provided in the annotation CSV.
 *
 * License: CC-BY-NC-SA 4.0 (non-commercial).
 *
 * This loader parses SEED EDF files (62-channel standard 10-20 montage at 200 Hz)
 * and returns an EEGSignal compatible with the Joint-2312 pipeline
 * (selectEEGPTChannels accepts all 62 channels directly).
 *
 * Reference: https://seed-dataset-2013.se.ee.tsinghua.edu.cn/
 */
import type { EEGSignal, EEGWindow } from "../eeg/types";
import { canonicalizeChannel, EEGPT_CHANNELS_62 } from "../eeg/channels";
import { resampleSignal } from "../eeg/preprocessing/resample";
import { preprocess } from "../eeg/preprocessing";
import { segment } from "../eeg/preprocessing/segment";

/** SEED dataset metadata matching the manifest schema. */
export const seedDataset = {
  name: "SEED",
  license: "CC-BY-NC-SA-4.0",
  sourceUrl: "https://seed-dataset-2013.se.ee.tsinghua.edu.cn/",
  nSubjects: 15,
  nChannels: 62,
  sampleRate: 200,
  nClasses: 9, // valence/arousal/dominance rated 1-9
  metadata: {
    paradigm: "emotional_stimuli",
    sessions: 3,
    trials: 15,
    tasks: ["positive", "negative", "neutral", "SEED-IV"],
  },
} as const;

/** Per-trial cognitive label from SEED annotation CSV. */
export interface SEEDCognitiveLabel {
  /** Trial index (1-15). */
  trial: number;
  /** Movie clip name / description. */
  video: string;
  /** Valence rating (1-9). */
  valence: number;
  /** Arousal rating (1-9). */
  arousal: number;
  /** Dominance rating (1-9). */
  dominance: number;
  /** Session index (1-3). */
  session: number;
  /** Subject id (e.g. "S01"). */
  subjectId: string;
}

/**
 * Derive a continuous workload proxy from SEED valence/arousal labels.
 *
 * SEED does not have direct NASA-TLX workload scores for all trials, so we
 * derive a workload proxy using the standard approach from the SEED-IV paper:
 *   workload = 0.6 * arousal + 0.2 * valence + 0.2 * dominance
 * Normalised to [0, 1].
 *
 * This is a proxy, not ground truth — M33 will train on this and validate
 * against real NASA-TLX scores when available.
 */
export function deriveWorkloadFromLabels(label: SEEDCognitiveLabel): number {
  const raw = 0.6 * label.arousal + 0.2 * label.valence + 0.2 * label.dominance;
  // Map from [1, 9] → [0, 1]
  return Math.max(0, Math.min(1, (raw - 1) / 8));
}

/**
 * Parse a SEED annotation CSV (exported from the .csv label files).
 *
 * Expected columns:
 *   subject_id, session, trial, video, valence, arousal, dominance
 *
 * Returns labels sorted by (subject_id, session, trial).
 */
export function parseSEEDAnnotations(csvText: string): SEEDCognitiveLabel[] {
  const lines = csvText.trim().split("\n");
  const header = lines[0].split(",").map((s) => s.trim());
  const idx = (col: string) => header.indexOf(col);

  const labels: SEEDCognitiveLabel[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(",").map((s) => s.trim());
    if (row.length < header.length) continue;
    labels.push({
      trial: parseInt(row[idx("trial")], 10) || 0,
      video: row[idx("video")] ?? "",
      valence: parseFloat(row[idx("valence")]) || 0,
      arousal: parseFloat(row[idx("arousal")]) || 0,
      dominance: parseFloat(row[idx("dominance")]) || 0,
      session: parseInt(row[idx("session")], 10) || 1,
      subjectId: row[idx("subject_id")] ?? "",
    });
  }
  return labels.sort((a, b) => {
    if (a.subjectId !== b.subjectId) return a.subjectId.localeCompare(b.subjectId);
    if (a.session !== b.session) return a.session - b.session;
    return a.trial - b.trial;
  });
}

/**
 * Parse a SEED EDF file into an EEGSignal.
 *
 * SEED uses standard 10-20 channel naming (62 channels at 200 Hz).
 * The returned signal's channels are validated against the EEGPT 62-channel
 * montage — SEED is a direct superset, so all 62 channels map cleanly.
 */
export function parseSEEDEDF(buffer: ArrayBuffer): EEGSignal {
  // Reuse the existing EDF parser — SEED files are standard EDF+ format.
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { parseEDF } = require("../eeg/parsers/edf");
  const signal = parseEDF(buffer);

  // Validate all 62 EEGPT channels are present (SEED uses standard 10-20).
  const byName = new Map<string, number>();
  signal.channels.forEach((ch, i) => {
    byName.set(canonicalizeChannel(ch), i);
  });
  const missing: string[] = [];
  for (const wanted of EEGPT_CHANNELS_62) {
    if (!byName.has(canonicalizeChannel(wanted))) missing.push(wanted);
  }
  if (missing.length > 0) {
    throw new Error(
      `SEED EDF: missing required channels (${missing.join(", ")})`,
    );
  }

  return signal;
}

/**
 * Resample SEED signal (200 Hz) to the EEGPT/CBraMod/V2 target rate (250 Hz).
 *
 * SEED runs at 200 Hz; the Joint-2312 pipeline expects 250 Hz
 * (FOUNDATION_SAMPLE_RATE_HZ). This is a thin wrapper around the shared
 * resampleSignal utility.
 */
export function resampleSEEDSignal(signal: EEGSignal, targetHz: number): EEGSignal {
  return resampleSignal(signal, targetHz);
}

/**
 * Preprocess a SEED signal for the Joint-2312 pipeline.
 *
 * Mirrors the foundation.ts joint-2312 preprocessing:
 *   - Bandpass: 1-40 Hz (EEGPT band)
 *   - No notch (SEED data is already clean)
 *   - Segment: 4s windows, 50% overlap
 *   - Z-score normalisation
 */
export function preprocessSEED(signal: EEGSignal, targetSampleRateHz: number = 250) {
  const resampled = resampleSEEDSignal(signal, targetSampleRateHz);
  return preprocess(resampled, {
    bandpass: { low: 1, high: 40 },
    notch: false,
    normalize: true,
    segment: { windowSec: 4, overlap: 0.5 },
  });
}

/**
 * Extract the 4 channel sets needed for Joint-2312 from a preprocessed SEED signal.
 *
 * Returns windowed signals for CBraMod (19-ch), V2/PCA (22-ch), and EEGPT (62-ch),
 * ready for `embedJoint2312Windows()`.
 */
export function extractJoint2312Channels(signal: EEGSignal): {
  /** 19-channel CBraMod signal (subset of 22). */
  cbramod19: EEGSignal;
  /** 22-channel V2/PCA signal (PROD_CHANNELS_22). */
  v2_22: EEGSignal;
  /** 62-channel EEGPT signal. */
  eegpt62: EEGSignal;
} {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { selectCbraModChannels, selectProdChannels, selectEEGPTChannels } = require("../eeg/channels");
  return {
    cbramod19: selectCbraModChannels(signal),
    v2_22: selectProdChannels(signal),
    eegpt62: selectEEGPTChannels(signal),
  };
}

/**
 * Full SEED preprocessing pipeline: parse → resample → preprocess → segment → extract channels.
 *
 * Returns per-window EEGWindows for each of the 4 Joint-2312 blocks,
 * aligned so that window[i] in each block corresponds to the same 4-second
 * time segment.
 */
export function preprocessSEEDForJoint2312(
  edfBuffer: ArrayBuffer,
  targetSampleRateHz: number = 250,
): {
  cbramodWindows: EEGWindow[];
  v2Windows: EEGWindow[];
  eegptWindows: EEGWindow[];
  sampleRate: number;
  preprocessReport: ReturnType<typeof preprocess>["report"];
} {
  const signal = parseSEEDEDF(edfBuffer);
  const pre = preprocessSEED(signal, targetSampleRateHz);

  // Extract aligned channel sets from the preprocessed signal.
  const { cbramod19, v2_22, eegpt62 } = extractJoint2312Channels(pre.signal);

  // Re-segment each channel set into 4s windows (already done in preprocessing,
  // but we need per-block windowing for the embedding functions).
  const cbramodWindows = segment(cbramod19.data, pre.signal.sampleRate, 4, 0.5);
  const v2Windows = segment(v2_22.data, pre.signal.sampleRate, 4, 0.5);
  const eegptWindows = segment(eegpt62.data, pre.signal.sampleRate, 4, 0.5);

  if (
    cbramodWindows.length === 0 ||
    v2Windows.length === 0 ||
    eegptWindows.length === 0 ||
    cbramodWindows.length !== v2Windows.length ||
    v2Windows.length !== eegptWindows.length
  ) {
    throw new Error(
      `SEED preprocessing: window count mismatch (CBraMod=${cbramodWindows.length}, ` +
        `V2=${v2Windows.length}, EEGPT=${eegptWindows.length})`,
    );
  }

  return {
    cbramodWindows,
    v2Windows,
    eegptWindows,
    sampleRate: pre.signal.sampleRate,
    preprocessReport: pre.report,
  };
}
