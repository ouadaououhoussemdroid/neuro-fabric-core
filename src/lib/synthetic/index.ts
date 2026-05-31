import type { EEGSignal } from "../eeg/types";

/**
 * Synthetic EEG generator using 1/f spectral priors plus band-specific
 * oscillatory components. The output is real numeric data with realistic
 * spectral structure (pink noise + δ/θ/α/β bumps), compatible with the same
 * preprocessing → embedding → decoder pipeline as recorded EEG.
 */

export interface SyntheticOptions {
  channels?: number;
  sampleRate?: number;
  durationSec?: number;
  /** Relative weights for δ θ α β γ bands; defaults are eyes-closed resting. */
  bandWeights?: { delta?: number; theta?: number; alpha?: number; beta?: number; gamma?: number };
  /** RNG seed for reproducibility. */
  seed?: number;
}

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  // Box-Muller
  const u = Math.max(1e-9, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Pink (1/f) noise via Voss-McCartney with 16 octaves. */
function pinkNoise(n: number, rng: () => number): number[] {
  const octaves = 16;
  const rows = new Array<number>(octaves).fill(0);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    // Update the lowest set bit's row
    let k = 0;
    let mask = i ^ (i - 1);
    while ((mask & 1) === 0 && k < octaves - 1) { mask >>>= 1; k++; }
    rows[k] = gaussian(rng) * 0.5;
    let s = 0;
    for (let r = 0; r < octaves; r++) s += rows[r];
    out[i] = s / Math.sqrt(octaves);
  }
  return out;
}

const BAND_CENTER = { delta: 2, theta: 6, alpha: 10, beta: 20, gamma: 38 } as const;

export function generateSyntheticEEG(opts: SyntheticOptions = {}): EEGSignal {
  const C = opts.channels ?? 16;
  const fs = opts.sampleRate ?? 256;
  const dur = opts.durationSec ?? 10;
  const N = Math.floor(fs * dur);
  const w = {
    delta: 1.0,
    theta: 0.7,
    alpha: 1.4,
    beta: 0.6,
    gamma: 0.3,
    ...(opts.bandWeights ?? {}),
  };
  const seedBase = opts.seed ?? 0xC0FFEE;

  const channels: string[] = [];
  const data: number[][] = [];

  for (let c = 0; c < C; c++) {
    const rng = mulberry32(seedBase + c * 9973);
    const pink = pinkNoise(N, rng);
    const out = new Array<number>(N);
    // Random per-band phase / slight detune per channel
    const phases = {
      delta: rng() * 2 * Math.PI,
      theta: rng() * 2 * Math.PI,
      alpha: rng() * 2 * Math.PI,
      beta:  rng() * 2 * Math.PI,
      gamma: rng() * 2 * Math.PI,
    };
    const detune = {
      delta: BAND_CENTER.delta * (1 + (rng() - 0.5) * 0.1),
      theta: BAND_CENTER.theta * (1 + (rng() - 0.5) * 0.1),
      alpha: BAND_CENTER.alpha * (1 + (rng() - 0.5) * 0.05),
      beta:  BAND_CENTER.beta  * (1 + (rng() - 0.5) * 0.1),
      gamma: BAND_CENTER.gamma * (1 + (rng() - 0.5) * 0.1),
    };
    for (let i = 0; i < N; i++) {
      const t = i / fs;
      const sig =
        w.delta * Math.sin(2 * Math.PI * detune.delta * t + phases.delta) +
        w.theta * Math.sin(2 * Math.PI * detune.theta * t + phases.theta) +
        w.alpha * Math.sin(2 * Math.PI * detune.alpha * t + phases.alpha) +
        w.beta  * Math.sin(2 * Math.PI * detune.beta  * t + phases.beta) +
        w.gamma * Math.sin(2 * Math.PI * detune.gamma * t + phases.gamma);
      // Scale to ~microvolt range; pink dominates broadband noise floor
      out[i] = 10 * (0.6 * sig + 1.2 * pink[i]);
    }
    channels.push(`SYN${c.toString().padStart(2, "0")}`);
    data.push(out);
  }

  return {
    channels,
    data,
    sampleRate: fs,
    meta: { format: "synthetic", seed: seedBase, bandWeights: w, durationSec: dur },
  };
}