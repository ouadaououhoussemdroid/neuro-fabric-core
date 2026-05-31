/** Per-channel z-score normalization. */
export function zscore(data: number[][]): number[][] {
  return data.map((ch) => {
    const n = ch.length;
    if (n === 0) return ch;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += ch[i];
    const mean = sum / n;
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const d = ch[i] - mean;
      sq += d * d;
    }
    const std = Math.sqrt(sq / n) || 1;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = (ch[i] - mean) / std;
    return out;
  });
}

/** Subtract per-channel mean only (DC removal). */
export function demean(data: number[][]): number[][] {
  return data.map((ch) => {
    const n = ch.length;
    let s = 0;
    for (let i = 0; i < n; i++) s += ch[i];
    const m = s / n;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = ch[i] - m;
    return out;
  });
}