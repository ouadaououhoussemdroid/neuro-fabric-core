import type { EEGWindow } from "../types";

/**
 * Slice a multi-channel signal into fixed-length overlapping windows.
 * @param data       [C][N]
 * @param sampleRate Hz
 * @param windowSec  window length in seconds
 * @param overlap    fraction [0,1)
 */
export function segment(
  data: number[][],
  sampleRate: number,
  windowSec = 2,
  overlap = 0.5,
): EEGWindow[] {
  if (data.length === 0) return [];
  const N = data[0].length;
  const W = Math.floor(windowSec * sampleRate);
  if (W <= 0 || W > N) return [];
  const step = Math.max(1, Math.floor(W * (1 - overlap)));
  const windows: EEGWindow[] = [];
  for (let start = 0; start + W <= N; start += step) {
    const slice = data.map((ch) => ch.slice(start, start + W));
    windows.push({ data: slice, sampleRate, start, end: start + W });
  }
  return windows;
}
