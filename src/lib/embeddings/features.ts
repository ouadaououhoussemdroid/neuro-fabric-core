import type { EEGWindow } from "../eeg/types";

/** Compute per-band power per channel using a naive DFT on each window.
 *  Returns a flat feature vector [C * 5] in band order δ θ α β γ.
 *  Bands (Hz): δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45.
 */
const BANDS: Array<[number, number]> = [
  [0.5, 4],
  [4, 8],
  [8, 13],
  [13, 30],
  [30, 45],
];

function dftPowerSpectrum(x: number[], fs: number): { freq: number; power: number }[] {
  const N = x.length;
  // Cap resolution for speed: use min(N, 512) points
  const M = Math.min(N, 512);
  const step = N / M;
  const sampled = new Array<number>(M);
  for (let i = 0; i < M; i++) sampled[i] = x[Math.floor(i * step)];
  // Hann window
  for (let i = 0; i < M; i++) {
    sampled[i] *= 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (M - 1));
  }
  const out: { freq: number; power: number }[] = [];
  const half = M / 2;
  for (let k = 1; k < half; k++) {
    let re = 0,
      im = 0;
    const c = (2 * Math.PI * k) / M;
    for (let n = 0; n < M; n++) {
      const a = c * n;
      re += sampled[n] * Math.cos(a);
      im -= sampled[n] * Math.sin(a);
    }
    out.push({ freq: (k * fs) / M, power: (re * re + im * im) / (M * M) });
  }
  return out;
}

export function bandPowerFeatures(window: EEGWindow): number[] {
  const out: number[] = [];
  for (const ch of window.data) {
    const spec = dftPowerSpectrum(ch, window.sampleRate);
    for (const [lo, hi] of BANDS) {
      let p = 0;
      for (const s of spec) if (s.freq >= lo && s.freq < hi) p += s.power;
      out.push(p);
    }
  }
  return out;
}
