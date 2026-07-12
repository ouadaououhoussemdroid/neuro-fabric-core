import type { EEGWindow } from "../eeg/types";

/**
 * Compute per-band power per channel using an FFT-based spectrum.
 *
 * T-007: replaces the original O(N²) naive DFT with an O(N log N) radix-2
 * FFT. The public API (`bandPowerFeatures`) and the band-power contract are
 * unchanged; only the spectrum computation is replaced.
 *
 * Returns a flat feature vector [C * 5] in band order δ θ α β γ.
 * Bands (Hz): δ 0.5–4, θ 4–8, α 8–13, β 13–30, γ 30–45.
 */
const BANDS: Array<[number, number]> = [
  [0.5, 4],
  [4, 8],
  [8, 13],
  [13, 30],
  [30, 45],
];

/** Next power of two ≥ n. */
function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

interface FreqBin {
  freq: number;
  power: number;
}

/**
 * Compute the one-sided power spectrum of a signal using a radix-2 FFT.
 *
 * Steps:
 *   1. Apply a Hann window (same as the old DFT path).
 *   2. Zero-pad to the next power of two (radix-2 requirement).
 *   3. In-place FFT (iterative Cooley-Tukey).
 *   4. Return magnitude-squared power for bins 1..N/2-1, normalised by N².
 */
function fftPowerSpectrum(x: number[], fs: number): FreqBin[] {
  const N = x.length;
  if (N < 2) return [];

  // Hann window — identical to the old implementation.
  const windowed = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
    windowed[i] = x[i] * w;
  }

  // Pad to next power of two for radix-2.
  const M = nextPow2(N);
  const re = new Float64Array(M);
  const im = new Float64Array(M);
  for (let i = 0; i < N; i++) re[i] = windowed[i];

  fftInPlace(re, im);

  const out: FreqBin[] = [];
  const half = M >> 1;
  for (let k = 1; k < half; k++) {
    const real = re[k];
    const imag = im[k];
    // Normalise by N² (we windowed the N-length signal, not the padded M).
    out.push({ freq: (k * fs) / M, power: (real * real + imag * imag) / (N * N) });
  }
  return out;
}

/**
 * Iterative in-place radix-2 Cooley-Tukey FFT.
 *
 * Operates on separate real/imaginary Float64Arrays of equal length, which
 * must be a power of two. Bit-reversal permutation first, then butterflies.
 */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  // Butterfly stages.
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half];
        const bIm = im[i + k + half];
        // t = b * w
        const tRe = bRe * curRe - bIm * curIm;
        const tIm = bRe * curIm + bIm * curRe;
        re[i + k] = aRe + tRe;
        im[i + k] = aIm + tIm;
        re[i + k + half] = aRe - tRe;
        im[i + k + half] = aIm - tIm;
        // Advance twiddle by multiplication (avoids repeated cos/sin calls).
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

export function bandPowerFeatures(window: EEGWindow): number[] {
  const out: number[] = [];
  for (const ch of window.data) {
    const spec = fftPowerSpectrum(ch, window.sampleRate);
    for (const [lo, hi] of BANDS) {
      let p = 0;
      for (const s of spec) if (s.freq >= lo && s.freq < hi) p += s.power;
      out.push(p);
    }
  }
  return out;
}

/** Exported for testing / reuse (e.g. the MNE parity harness). */
export { fftPowerSpectrum };
