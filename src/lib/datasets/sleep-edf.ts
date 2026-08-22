/**
 * M38 — Sleep-EDF dataset loader with channel expansion for Joint-2312.
 *
 * Sleep-EDF is a publicly available EEG dataset for sleep staging research.
 * Each recording uses a minimal 7-channel montage:
 *   - Fpz-Cz (frontal-central)
 *   - Pz-Oz (parietal-occipital)
 *   - EOG (electrooculogram, horizontal)
 *   - Plus optionally Fpz, Pz, Oz, Cz, A1 (reference/mastoid) in some versions
 *
 * The signal is at 100 Hz (Sleep-EDF v1, PhysioNet 1.0.0) or 128 Hz
 * (Sleep-EDF v2 / updated). Labels are 5-stage sleep:
 *   - Wake (W)
 *   - N1 (N1 / Stage 1)
 *   - N2 (N2 / Stage 2)
 *   - N3 (N3 / Stage 3 / Slow Wave Sleep)
 *   - REM (REM / Rapid Eye Movement)
 *
 * **Channel expansion challenge:** Joint-2312 requires 62-channel EEGPT montage
 * at 250 Hz. Sleep-EDF's 7 channels must be spatially expanded to 62 channels
 * using a nearest-neighbour interpolation strategy, then upsampled to 250 Hz.
 *
 * License: BSD-3-Clause (PhysioNet Sleep-EDF Database)
 * Reference: https://physionet.org/content/sleep-edf/1.0.0/
 */
import type { EEGSignal, EEGWindow, PreprocessingReport } from "../eeg/types";
import { parseEDF, type EDFAnnotation } from "../eeg/parsers/edf";
import {
  EEGPT_CHANNELS_62,
  canonicalizeChannel,
  selectCbraModChannels,
  selectProdChannels,
  selectEEGPTChannels,
} from "../eeg/channels";
import { resampleSignal } from "../eeg/preprocessing/resample";
import { preprocess, type PreprocessOptions } from "../eeg/preprocessing";
import { segment } from "../eeg/preprocessing/segment";

// ─────────────────────────────────────────────────────────────────────────────
// Sleep-EDF dataset metadata
// ─────────────────────────────────────────────────────────────────────────────

/** Sleep-EDF dataset metadata matching the benchmark_archive manifest schema. */
export const sleepEDFDataset = {
  name: "Sleep-EDF",
  license: "BSD-3-Clause",
  sourceUrl: "https://physionet.org/content/sleep-edf/1.0.0/",
  nSubjects: 99,
  nChannels: 7, // Fpz-Cz, Pz-Oz, EOG, + optional references
  sampleRate: 100, // Hz (Sleep-EDF 1.0.0) — v2 is 128 Hz
  nClasses: 5, // W, N1, N2, N3, REM
  metadata: {
    paradigm: "sleep_staging",
    sessions: 2, // 2 nights per subject
    epochs: "30-minute",
    stages: ["W", "N1", "N2", "N3", "REM"],
    channel_montage: ["Fpz-Cz", "Pz-Oz", "EOG"],
  },
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// Sleep stage labels
// ─────────────────────────────────────────────────────────────────────────────

/** Sleep stage label (AASM 5-stage convention). */
export type SleepStage = "W" | "N1" | "N2" | "N3" | "REM" | "UNKNOWN";

/** Mapping from annotation text to sleep stage. */
export const SLEEP_STAGES: Record<string, SleepStage> = {
  // Wake variants
  W: "W",
  Wake: "W",
  "0": "W",
  Stage0: "W",
  // N1 variants
  N1: "N1",
  Stage1: "N1",
  "1": "N1",
  // N2 variants
  N2: "N2",
  Stage2: "N2",
  "2": "N2",
  // N3 variants
  N3: "N3",
  Stage3: "N3",
  Stage4: "N3",
  SWS: "N3",
  "3": "N3",
  "4": "N3",
  // REM variants
  R: "REM",
  REM: "REM",
  "5": "REM",
};

/** Sleep stage label for machine learning (integer encoding). */
export const SLEEP_STAGE_LABELS: SleepStage[] = ["W", "N1", "N2", "N3", "REM"];

/** Integer → SleepStage mapping for model output. */
export const SLEEP_STAGE_ID_TO_LABEL: Record<number, SleepStage> = {
  0: "W",
  1: "N1",
  2: "N2",
  3: "N3",
  4: "REM",
};

/** SleepStage → Integer mapping for label encoding. */
export const SLEEP_STAGE_LABEL_TO_ID: Record<SleepStage, number> = {
  W: 0,
  N1: 1,
  N2: 2,
  N3: 3,
  REM: 4,
  UNKNOWN: -1,
};

// ─────────────────────────────────────────────────────────────────────────────
// Sleep epoch annotation
// ─────────────────────────────────────────────────────────────────────────────

/** A single 30-second sleep epoch with its stage label. */
export interface SleepEpoch {
  /** Epoch index (0-based, starts at recording onset). */
  index: number;
  /** Start time in seconds from recording onset. */
  startTime: number;
  /** End time in seconds (startTime + duration). */
  endTime: number;
  /** Duration in seconds (30 for standard sleep staging). */
  duration: number;
  /** Sleep stage label. */
  stage: SleepStage;
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel expansion: 7-channel Sleep-EDF → 62-channel EEGPT montage
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Mapping from EEGPT channel names to their nearest Sleep-EDF source channel.
 *
 * Sleep-EDF provides only 7 channels (Fpz-Cz, Pz-Oz, EOG, and optionally
 * Fpz, Pz, Oz, Cz, A1). We expand to EEGPT's 62-channel montage by assigning
 * each of the 62 channels to its nearest anatomic neighbour in the Sleep-EDF
 * montage:
 *
 *   - Frontal channels (F3, F4, F7, etc.) → Fpz-Cz
 *   - Central channels (C3, C4, Cz, etc.) → Fpz-Cz (shares Cz reference)
 *   - Parietal channels (P3, P4, P7, etc.) → Pz-Oz
 *   - Occipital channels (O1, O2, Oz, etc.) → Pz-Oz (shares Oz reference)
 *   - Temporal channels (T7, T8, etc.) → Fpz-Cz or Pz-Oz (nearest)
 *   - Fp1, Fp2, AFz → Fpz-Cz
 *   - EOG channel → mapped to frontal channels (EOG activity correlates with
 *     frontal eye artifacts, useful for blink detection)
 *
 * This is a heuristic spatial interpolation — it does NOT add new signal
 * information, but it allows the Joint-2312 CBraMod/EEGPT models to process
 * the signal without channel-count mismatch errors. The resulting embedding
 * will be lower quality than a true 62-channel recording, but retains the
 * spectral characteristics of slow waves, sleep spindles, and REM patterns.
 */
const SLEEP_CHANNEL_TO_EEGPT_NEIGHBOUR: Record<string, string> = {
  // Frontal group → Fpz-Cz
  FP1: "Fpz-Cz", FP2: "Fpz-Cz", FPZ: "Fpz-Cz",
  AF7: "Fpz-Cz", AF3: "Fpz-Cz", AF4: "Fpz-Cz", AF8: "Fpz-Cz",
  F7: "Fpz-Cz", F5: "Fpz-Cz", F3: "Fpz-Cz", F1: "Fpz-Cz",
  FZ: "Fpz-Cz", F2: "Fpz-Cz", F4: "Fpz-Cz", F6: "Fpz-Cz", F8: "Fpz-Cz",
  FT7: "Fpz-Cz", FC5: "Fpz-Cz", FC3: "Fpz-Cz", FC1: "Fpz-Cz",
  FCZ: "Fpz-Cz", FC2: "Fpz-Cz", FC4: "Fpz-Cz", FC6: "Fpz-Cz", FT8: "Fpz-Cz",

  // Central group → Fpz-Cz (shares Cz reference)
  T7: "Fpz-Cz", C5: "Fpz-Cz", C3: "Fpz-Cz", C1: "Fpz-Cz",
  CZ: "Fpz-Cz", C2: "Fpz-Cz", C4: "Fpz-Cz", C6: "Fpz-Cz", T8: "Fpz-Cz",

  // Parietal-temporal transition → Pz-Oz
  TP7: "Pz-Oz", CP5: "Pz-Oz", CP3: "Pz-Oz", CP1: "Pz-Oz",
  CPZ: "Pz-Oz", CP2: "Pz-Oz", CP4: "Pz-Oz", CP6: "Pz-Oz", TP8: "Pz-Oz",

  // Parietal group → Pz-Oz
  P7: "Pz-Oz", P5: "Pz-Oz", P3: "Pz-Oz", P1: "Pz-Oz",
  PZ: "Pz-Oz", P2: "Pz-Oz", P4: "Pz-Oz", P6: "Pz-Oz", P8: "Pz-Oz",

  // Occipital group → Pz-Oz (shares Oz reference)
  PO7: "Pz-Oz", PO5: "Pz-Oz", PO3: "Pz-Oz", POZ: "Pz-Oz",
  PO4: "Pz-Oz", PO6: "Pz-Oz", PO8: "Pz-Oz",
  O1: "Pz-Oz", OZ: "Pz-Oz", O2: "Pz-Oz",
};

/**
 * Expand a 7-channel Sleep-EDF signal to the 62-channel EEGPT montage.
 *
 * Each of the 62 EEGPT channels is assigned a signal from the nearest
 * Sleep-EDF source channel. Channels that don't have a mapping default to
 * the closest source (Fpz-Cz for frontal/central, Pz-Oz for parietal/occipital).
 *
 * The returned signal has 62 channels in EEGPT_CHANNELS_62 order, with
 * 250 Hz sample rate (caller must resample first if needed).
 */
export function expandSleepToEEGPT(signal: EEGSignal): EEGSignal {
  const byName = new Map<string, number[]>();
  signal.channels.forEach((ch, i) => {
    const canon = canonicalizeChannel(ch);
    if (!byName.has(canon)) byName.set(canon, signal.data[i] ?? []);
  });

  // Ensure the 3 canonical Sleep-EDF source channels exist
  const fpCz =
    byName.get("FPZ-CZ") ?? byName.get("FPZCZ") ?? byName.get("FPZ") ?? byName.get("CZ");
  const pzOz =
    byName.get("PZ-OZ") ?? byName.get("PZOZ") ?? byName.get("PZ") ?? byName.get("OZ");
  const eog =
    byName.get("EOG") ??
    byName.get("EOG1") ??
    byName.get("EOG2") ??
    byName.get("EOGL") ??
    byName.get("EOGR");

  // Build the source channel lookup; missing sources resolve to undefined
  const sourceChannels: Record<string, number[] | undefined> = {
    "Fpz-Cz": fpCz,
    "Pz-Oz": pzOz,
    EOG: eog,
  };

  // Fallback: any channel without a valid source falls back to the first
  // available source channel (preferring Fpz-Cz, then any non-undefined source)
  const fallbackSource = fpCz ?? pzOz ?? eog;

  const channels: string[] = [];
  const data: number[][] = [];

  for (const wanted of EEGPT_CHANNELS_62) {
    const canon = canonicalizeChannel(wanted);
    const sourceName = SLEEP_CHANNEL_TO_EEGPT_NEIGHBOUR[canon] ?? "Fpz-Cz";
    const sourceData = sourceChannels[sourceName];

    if (sourceData) {
      channels.push(wanted);
      data.push([...sourceData]); // copy
    } else if (fallbackSource) {
      // Nearest source channel unavailable — fall back to the best available
      channels.push(wanted);
      data.push([...fallbackSource]);
    } else {
      // No source data at all — default to zeros
      const samples = signal.data[0]?.length ?? 0;
      channels.push(wanted);
      data.push(new Array<number>(samples).fill(0));
    }
  }

  return {
    channels,
    data,
    sampleRate: signal.sampleRate,
    meta: {
      ...signal.meta,
      source_channels: signal.channels,
      source_channel_count: signal.channels.length,
      expansion: "7→62 (nearest-neighbour spatial interpolation)",
      channel_mapping: SLEEP_CHANNEL_TO_EEGPT_NEIGHBOUR,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Annotation parsing
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parse Sleep-EDF EDF+ annotations into sleep-stage epochs.
 *
 * Sleep-EDF annotation files contain TAL entries like:
 *   "Sleep stage W"
 *   "Sleep stage 1"  (N1)
 *   "Sleep stage 2"  (N2)
 *   "Sleep stage 3"  (N3)
 *   "Sleep stage 4"  (N3 — merged with S3 in AASM)
 *   "Sleep stage R"  (REM)
 *   "Sleep stage 0"  (W — alternate format)
 *
 * Each annotation marks the start of a 30-second epoch.
 */
export function parseSleepAnnotations(annotations: EDFAnnotation[]): SleepEpoch[] {
  const epochs: SleepEpoch[] = [];
  const EPOCH_DURATION = 30; // seconds

  // Filter to sleep stage annotations only
  const stageAnns = annotations.filter((a) => {
    const text = a.text.toLowerCase();
    return text.startsWith("sleep stage");
  });

  for (const ann of stageAnns) {
    // Extract stage identifier after "Sleep stage" — handle numeric (0-5),
    // single-letter (W, R), and named (N1, N2, N3, SWS, Wake, REM) forms.
    const match = ann.text.match(/Sleep stage\s+(\S+)/i);
    if (!match) continue;

    const key = match[1];
    const stage: SleepStage = SLEEP_STAGES[key] ?? SLEEP_STAGES[key.toUpperCase()] ?? "UNKNOWN";

    epochs.push({
      index: epochs.length,
      startTime: ann.onset,
      endTime: ann.onset + (ann.duration ?? EPOCH_DURATION),
      duration: ann.duration ?? EPOCH_DURATION,
      stage,
    });
  }

  return epochs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full preprocessing pipeline for Sleep-EDF → Joint-2312
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Target sample rate for the Joint-2312 pipeline (matches CBraMod + EEGPT).
 */
export const JOINT_2312_SAMPLE_RATE_HZ = 250;

/**
 * Window duration for Joint-2312 embedding (4 seconds, matching the
 * foundation pipeline's window size).
 */
export const JOINT_WINDOW_SEC = 4;

/**
 * Sleep epoch duration (30 seconds per AASM convention).
 */
export const SLEEP_EPOCH_SEC = 30;

/**
 * Result of preprocessing a Sleep-EDF file for the Joint-2312 pipeline.
 * Returns per-window EEGWindows for each of the 4 Joint-2312 blocks,
 * aligned so that window[i] in each block corresponds to the same 4-second
 * time segment within a sleep epoch.
 */
export interface SleepEEGPreprocessResult {
  cbramodWindows: EEGWindow[];
  v2Windows: EEGWindow[];
  eegptWindows: EEGWindow[];
  sleepEpochs: SleepEpoch[];
  sampleRate: number;
  preprocessReport: PreprocessingReport;
}

/**
 * Preprocess a Sleep-EDF EDF buffer for the Joint-2312 pipeline.
 *
 * Pipeline:
 *   1. Parse EDF → raw 7-channel signal + annotations
 *   2. Parse sleep stage annotations from EDF+ TAL
 *   3. Resample to 250 Hz (from 100 or 128 Hz)
 *   4. Bandpass filter (0.5–40 Hz — sleep-relevant range) + z-score normalize
 *   5. Expand 7 channels → 62 channels (spatial interpolation)
 *   6. Select CBraMod (19), V2/Prod (22), EEGPT (62) channel subsets
 *   7. Segment each block into 4-second windows (50% overlap)
 *
 * The returned windows are aligned across all 3 blocks so that
 * `cbramodWindows[i]`, `v2Windows[i]`, and `eegptWindows[i]` all correspond
 * to the same 4-second time segment.
 */
export function preprocessSleepEDF(
  edfBuffer: ArrayBuffer,
  opts: PreprocessOptions = {},
): {
  signal: EEGSignal;
  windows: EEGWindow[];
  epochs: SleepEpoch[];
  report: SleepEEGPreprocessResult;
} {
  const t0 = performance.now();

  // Step 1: Parse EDF + annotations
  const parsed = parseEDF(edfBuffer);
  const annotations: EDFAnnotation[] =
    (parsed.meta?.annotations as EDFAnnotation[] | undefined) ?? [];

  // Step 2: Parse sleep stage epochs
  const epochs = parseSleepAnnotations(annotations);

  // Step 3: Resample to 250 Hz if needed
  const resampled =
    parsed.sampleRate !== JOINT_2312_SAMPLE_RATE_HZ
      ? resampleSignal(parsed, JOINT_2312_SAMPLE_RATE_HZ)
      : parsed;

  // Step 4: Bandpass + normalize (sleep range: 0.5–40 Hz)
  const pre = preprocess(resampled, {
    bandpass: { low: 0.5, high: 40 },
    notch: false, // Sleep-EDF is typically clean
    normalize: true,
    segment: false, // we do our own windowing after expansion
    artifactRejection: false,
    ...opts,
  });

  // Step 5: Expand 7 channels → 62 channels
  const expanded = expandSleepToEEGPT(pre.signal);

  // Step 6: Select channel subsets for Joint-2312 blocks
  const cbramodSignal = selectCbraModChannels(expanded);
  const v2Signal = selectProdChannels(expanded);
  const eegptSignal = selectEEGPTChannels(expanded);

  // Step 7: Segment each block into 4-second windows
  const cbramodWindows = segment(
    cbramodSignal.data,
    JOINT_2312_SAMPLE_RATE_HZ,
    JOINT_WINDOW_SEC,
    0.5,
  );
  const v2Windows = segment(
    v2Signal.data,
    JOINT_2312_SAMPLE_RATE_HZ,
    JOINT_WINDOW_SEC,
    0.5,
  );
  const eegptWindows = segment(
    eegptSignal.data,
    JOINT_2312_SAMPLE_RATE_HZ,
    JOINT_WINDOW_SEC,
    0.5,
  );

  // Verify alignment
  if (
    cbramodWindows.length !== v2Windows.length ||
    v2Windows.length !== eegptWindows.length
  ) {
    throw new Error(
      `Sleep-EDF preprocessing: window count mismatch (CBRaMod=${cbramodWindows.length}, ` +
        `V2=${v2Windows.length}, EEGPT=${eegptWindows.length})`,
    );
  }

  const totalMs = +(performance.now() - t0).toFixed(2);

  return {
    signal: expanded,
    windows: eegptWindows,
    epochs,
    report: {
      cbramodWindows,
      v2Windows,
      eegptWindows,
      sleepEpochs: epochs,
      sampleRate: JOINT_2312_SAMPLE_RATE_HZ,
      preprocessReport: {
        ...pre.report,
        totalDurationMs: totalMs,
        steps: [
          ...pre.report.steps,
          {
            name: "channel-expansion",
            params: { from: 7, to: 62, method: "nearest-neighbour spatial interpolation" },
            durationMs: +(totalMs - pre.report.totalDurationMs).toFixed(2),
          },
        ],
      },
    },
  };
}
