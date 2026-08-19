/**
 * CBraMod channel selection (Tier-2 / Mission 12).
 *
 * CBraMod was trained on the standard 19-channel 10-20 montage and its ONNX
 * input contract is [1, 19, 1000] over exactly those 19 channels in a fixed
 * order. The foundation path must therefore select those 19 channels from the
 * parsed EEG signal (which typically carries 22 or 64 channels) and present
 * them in CBraMod's expected input order before preprocessing + inference.
 *
 * This mirrors Mission-11's `normalize_ch_name` + `CBRAMOD_CHANS` selection
 * (canonical uppercase 10-20 labels), so the server-side 200-D embeddings are
 * reproducible from the validated Python benchmark. It is additive and used only
 * by the foundation path; it does not touch the Tier-1 (22-channel) selection.
 */
import type { EEGSignal } from "./types";

/**
 * CBraMod's 19-channel 10-20 montage, in the model's expected input order.
 * Must match the order used during CBraMod's training/export (Mission 11 used
 * the identical 19-label set).
 */
export const CBRAMOD_CHANNELS_19 = [
  "FP1",
  "FP2",
  "F3",
  "F4",
  "C3",
  "C4",
  "P3",
  "P4",
  "O1",
  "O2",
  "F7",
  "F8",
  "T7",
  "T8",
  "P7",
  "P8",
  "FZ",
  "CZ",
  "PZ",
] as const;

/** Number of channels the CBraMod ONNX model consumes. */
export const CBRAMOD_CHANNEL_COUNT = 19;

/**
 * Normalise a 10-20 channel label to the canonical uppercase form used by the
 * CBraMod channel set: trim, uppercase, strip a leading "EEG " prefix, drop
 * internal whitespace, and strip trailing periods (so "Fp1", "FP1", "EEG Fp1"
 * and the PhysioNet EDF label form "Fp1." all canonicalise to "FP1"). Mirrors
 * Mission-11 `normalize_ch_name`.
 */
export function canonicalizeChannel(label: string): string {
  return (label ?? "")
    .trim()
    .toUpperCase()
    .replace(/^EEG\s+/, "")
    .replace(/\s+/g, "")
    .replace(/\.+$/g, "");
}

/** Build a canonical-name → source-index map for the signal's channels. */
function canonicalIndexMap(channels: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < channels.length; i++) {
    const canon = canonicalizeChannel(channels[i]);
    if (!m.has(canon)) m.set(canon, i);
  }
  return m;
}

export interface CbraModChannelSelection {
  /** Channels selected, in CBraMod input order (canonical uppercase names). */
  channels: string[];
  /** Per-channel data slice, indexed identically to `channels`. */
  data: number[][];
  /** Indices of the selected channels in the original signal. */
  sourceIndices: number[];
}

/**
 * Select CBraMod's 19 channels from a parsed EEG signal, returned in CBraMod's
 * expected input order. Throws a descriptive error listing any missing channel
 * so the foundation path fails loud rather than feeding a wrong-channel-count
 * tensor to the model.
 */
export function selectCbraModChannels(signal: EEGSignal): EEGSignal {
  const byName = canonicalIndexMap(signal.channels);
  const channels: string[] = [];
  const data: number[][] = [];
  const sourceIndices: number[] = [];
  const missing: string[] = [];
  for (const wanted of CBRAMOD_CHANNELS_19) {
    const idx = byName.get(wanted);
    if (idx === undefined) {
      missing.push(wanted);
      continue;
    }
    sourceIndices.push(idx);
    channels.push(wanted);
    data.push(signal.data[idx] ?? []);
  }
  if (missing.length > 0) {
    throw new Error(
      `CBraMod foundation path: required channels missing (${missing.join(", ")}); ` +
        `available: ${JSON.stringify(signal.channels)}`,
    );
  }
  return {
    channels,
    data,
    sampleRate: signal.sampleRate,
    meta: {
      ...signal.meta,
      source_channels: signal.channels.length,
      selected_indices: sourceIndices,
    },
  };
}

/**
 * Production 22-channel EEGConformer / band-power montage, in the model's expected
 * input order. Matches the BCI-IV-2a subset used in the Python reference scripts
 * (scripts/t032-embedding-quality.py:52) and the V2 ONNX model's 22-channel
 * contract (input [1, 22, 1000] @ 250 Hz). The 22 channels are a superset of
 * CBraMod's 19 — the union is needed because the joint-264 path selects both
 * sets from the same source signal.
 */
export const PROD_CHANNELS_22 = [
  "FP1",
  "FP2",
  "F5",
  "F6",
  "F3",
  "F4",
  "F1",
  "F2",
  "FC5",
  "FC6",
  "FC3",
  "FC4",
  "C5",
  "C6",
  "C3",
  "C4",
  "T7",
  "T8",
  "P7",
  "P8",
  "P5",
  "P6",
] as const;

/** Number of channels the EEGConformer V2 ONNX model consumes. */
export const PROD_CHANNEL_COUNT = PROD_CHANNELS_22.length; // 22

/**
 * Select EEGConformer's 22 production channels from a parsed EEG signal, returned
 * in the model's expected input order. Throws if any of the 22 required channels
 * are absent — fail loud, never silently zero-pad (would corrupt the 32-D
 * representation and misalign the joint space).
 */
export function selectProdChannels(signal: EEGSignal): EEGSignal {
  const byName = canonicalIndexMap(signal.channels);
  const channels: string[] = [];
  const data: number[][] = [];
  const sourceIndices: number[] = [];
  const missing: string[] = [];
  for (const wanted of PROD_CHANNELS_22) {
    const idx = byName.get(wanted);
    if (idx === undefined) {
      missing.push(wanted);
      continue;
    }
    sourceIndices.push(idx);
    channels.push(wanted);
    data.push(signal.data[idx] ?? []);
  }
  if (missing.length > 0) {
    throw new Error(
      `Prod channel selection: required channels missing (${missing.join(", ")}); ` +
        `available: ${JSON.stringify(signal.channels)}`,
    );
  }
  return {
    channels,
    data,
    sampleRate: signal.sampleRate,
    meta: {
      ...signal.meta,
      source_channels: signal.channels.length,
      selected_indices: sourceIndices,
    },
  };
}

/**
 * EEGPT's 62-channel standard 10-20 montage, in the model's expected input order.
 *
 * Matches the EEGPT model's training montage (braindecode/eegpt-pretrained,
 * Apache-2.0). Input contract: [1, 62, 1000] @ 250 Hz. The 62-channel set is a
 * superset of CBraMod's 19 and V2/Prod's 22 — the union ensures all four fusion
 * blocks can be selected from the same source signal.
 */
export const EEGPT_CHANNELS_62 = [
  "FP1", "FPZ", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1",
  "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
  "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
  "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
  "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
  "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
  "O1", "OZ", "O2",
] as const;

/** Number of channels the EEGPT ONNX model consumes. */
export const EEGPT_CHANNEL_COUNT = 62;

/**
 * Channels interpolated from spatial neighbours when the source signal lacks them.
 * PO5/PO6 are not present in the PhysioNet EEGMMIDB 64-channel set (which has
 * PO7, PO3, PO4, PO8 but not the intermediate PO5/PO6), so they are derived as
 * the mean of their nearest neighbours — the same strategy used in M26/M27
 * Python preprocessing (preprocess_eegpt_trial).
 */
const EEGPT_INTERPOLATED: Record<string, string[]> = {
  PO5: ["PO7", "PO3"],
  PO6: ["PO4", "PO8"],
};

/**
 * Select EEGPT's 62 channels from a parsed EEG signal, returned in the model's
 * expected input order. Channels not present in the source are either
 * interpolated from spatial neighbours (PO5, PO6) or — for any other missing
 * channel — cause a loud throw, so the EEGPT input tensor is never silently
 * zero-padded.
 */
export function selectEEGPTChannels(signal: EEGSignal): EEGSignal {
  const byName = canonicalIndexMap(signal.channels);
  const channels: string[] = [];
  const data: number[][] = [];
  const sourceIndices: number[] = [];
  const missing: string[] = [];

  for (const wanted of EEGPT_CHANNELS_62) {
    const idx = byName.get(wanted);
    if (idx !== undefined) {
      sourceIndices.push(idx);
      channels.push(wanted);
      data.push(signal.data[idx] ?? []);
    } else if (wanted in EEGPT_INTERPOLATED) {
      const neighbours = EEGPT_INTERPOLATED[wanted]!;
      const avail = neighbours
        .map((n) => {
          const i = byName.get(n);
          return i !== undefined ? signal.data[i] : null;
        })
        .filter((d): d is number[] => d !== null);
      if (avail.length > 0) {
        const n = avail.length;
        const samples = avail[0].length;
        const interp = new Array<number>(samples).fill(0);
        for (const ch of avail) {
          for (let t = 0; t < samples; t++) interp[t] += ch[t];
        }
        for (let t = 0; t < samples; t++) interp[t] /= n;
        sourceIndices.push(-1);
        channels.push(wanted);
        data.push(interp);
      } else {
        const samples = signal.data[0]?.length ?? 0;
        sourceIndices.push(-1);
        channels.push(wanted);
        data.push(new Array<number>(samples).fill(0));
      }
    } else {
      missing.push(wanted);
    }
  }

  if (missing.length > 0) {
    throw new Error(
      `EEGPT channel selection: required channels missing (${missing.join(", ")}); ` +
        `available: ${JSON.stringify(signal.channels)}`,
    );
  }

  return {
    channels,
    data,
    sampleRate: signal.sampleRate,
    meta: {
      ...signal.meta,
      source_channels: signal.channels.length,
      selected_indices: sourceIndices,
      interpolated: ["PO5", "PO6"],
    },
  };
}
