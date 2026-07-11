/**
 * IIR biquad filters (Robert Bristow-Johnson cookbook).
 * Implemented as zero-phase by forward-backward (filtfilt) for offline preprocessing.
 */

type Biquad = { b0: number; b1: number; b2: number; a1: number; a2: number };

function lowpassCoeffs(fs: number, fc: number, q = Math.SQRT1_2): Biquad {
  const w0 = (2 * Math.PI * fc) / fs;
  const cosw = Math.cos(w0);
  const sinw = Math.sin(w0);
  const alpha = sinw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cosw) / 2 / a0,
    b1: (1 - cosw) / a0,
    b2: (1 - cosw) / 2 / a0,
    a1: (-2 * cosw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function highpassCoeffs(fs: number, fc: number, q = Math.SQRT1_2): Biquad {
  const w0 = (2 * Math.PI * fc) / fs;
  const cosw = Math.cos(w0);
  const sinw = Math.sin(w0);
  const alpha = sinw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 + cosw) / 2 / a0,
    b1: -(1 + cosw) / a0,
    b2: (1 + cosw) / 2 / a0,
    a1: (-2 * cosw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function notchCoeffs(fs: number, fc: number, q = 30): Biquad {
  const w0 = (2 * Math.PI * fc) / fs;
  const cosw = Math.cos(w0);
  const sinw = Math.sin(w0);
  const alpha = sinw / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: 1 / a0,
    b1: (-2 * cosw) / a0,
    b2: 1 / a0,
    a1: (-2 * cosw) / a0,
    a2: (1 - alpha) / a0,
  };
}

function applyBiquad(x: number[], c: Biquad): number[] {
  const n = x.length;
  const y = new Array<number>(n);
  let x1 = 0,
    x2 = 0,
    y1 = 0,
    y2 = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = c.b0 * xi + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
    y[i] = yi;
    x2 = x1;
    x1 = xi;
    y2 = y1;
    y1 = yi;
  }
  return y;
}

/** Zero-phase forward-backward filtering. */
function filtfilt(x: number[], c: Biquad): number[] {
  const fwd = applyBiquad(x, c);
  fwd.reverse();
  const back = applyBiquad(fwd, c);
  back.reverse();
  return back;
}

function applyToChannels(data: number[][], c: Biquad): number[][] {
  return data.map((ch) => filtfilt(ch, c));
}

/** Bandpass = highpass then lowpass, applied per channel with zero-phase. */
export function bandpass(data: number[][], fs: number, low: number, high: number): number[][] {
  if (low <= 0 || high >= fs / 2 || low >= high) {
    throw new Error(`bandpass: invalid range ${low}-${high} Hz at fs=${fs}`);
  }
  const hp = highpassCoeffs(fs, low);
  const lp = lowpassCoeffs(fs, high);
  const afterHp = applyToChannels(data, hp);
  return applyToChannels(afterHp, lp);
}

/** Narrowband notch at fc Hz (e.g. mains 50 or 60). */
export function notch(data: number[][], fs: number, fc: number, q = 30): number[][] {
  if (fc <= 0 || fc >= fs / 2) {
    throw new Error(`notch: invalid fc=${fc} at fs=${fs}`);
  }
  const c = notchCoeffs(fs, fc, q);
  return applyToChannels(data, c);
}
