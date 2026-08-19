/**
 * Resampling helpers for EEG signals (Tier-2 / CBraMod foundation path).
 *
 * The CBraMod ONNX model expects [1, 19, 1000] @ 250 Hz. The shared
 * `preprocess()` pipeline does bandpass → z-score → segmentation but does NOT
 * perform rate conversion (it assumes the caller has already selected the target
 * rate) — this matches Mission-11's preprocessing, where PhysioNet EEGMMIDB
 * EDFs (160 Hz) are resampled to 250 Hz before the 4 s / 1000-sample window is
 * extracted. No resampling utility previously existed in the codebase, so this
 * is an additive helper used solely by the foundation path; it does not alter
 * the shared `preprocess()` behaviour or the Tier-1 V2 path.
 *
 * Implementation: per-channel linear interpolation (monophonic), which preserves
 * the relative time alignment of events (onset indices are re-derived from the
 * resampled sample count). Pure TS, browser-compatible (no Node built-ins),
 * though currently consumed only server-side.
 */
import type { EEGSignal } from "../types";

/** Linearly interpolate a single channel onto `nOut` uniformly-spaced samples. */
export function resampleChannel(input: ArrayLike<number>, nOut: number): number[] {
  const nIn = input.length;
  if (nOut <= 0) return [];
  if (nIn === 0) return new Array<number>(nOut).fill(0);
  if (nIn === nOut) return Array.from(input as ArrayLike<number>);
  const out = new Array<number>(nOut);
  const scale = (nIn - 1) / (nOut - 1);
  for (let i = 0; i < nOut; i++) {
    const t = i * scale;
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, nIn - 1);
    const frac = t - lo;
    const a = input[lo] as number;
    const b = input[hi] as number;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

/**
 * Resample an EEGSignal to `targetHz`, preserving channel order and labels.
 * No-op (same ref) when the signal is already at the target rate.
 */
export function resampleSignal(signal: EEGSignal, targetHz: number): EEGSignal {
  if (signal.sampleRate === targetHz) return signal;
  const factor = targetHz / signal.sampleRate;
  const n0 = signal.data[0]?.length ?? 0;
  const nOut = Math.max(1, Math.round(n0 * factor));
  const data = new Array<number[]>(signal.channels.length);
  for (let c = 0; c < signal.channels.length; c++) {
    data[c] = resampleChannel(signal.data[c] ?? [], nOut);
  }
  return {
    channels: signal.channels,
    data,
    sampleRate: targetHz,
    meta: signal.meta,
  };
}
