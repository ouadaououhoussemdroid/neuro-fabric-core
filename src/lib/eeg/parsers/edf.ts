import type { EEGSignal } from "../types";

/**
 * Minimal EDF / EDF+ reader (16-bit signed little-endian).
 * Spec: https://www.edfplus.info/specs/edf.html
 *
 * Returns one EEGSignal using the sample rate of the FIRST signal.
 * Channels with a different sample rate are skipped (warned via meta.warnings).
 */
export function parseEDF(buffer: ArrayBuffer): EEGSignal {
  const bytes = new Uint8Array(buffer);
  const dec = new TextDecoder("ascii");
  const ascii = (off: number, len: number) => dec.decode(bytes.subarray(off, off + len)).trim();

  const version = ascii(0, 8);
  if (version !== "0" && !version.startsWith("0")) {
    // Some files have version "0" padded
  }
  const headerBytes = parseInt(ascii(184, 8), 10);
  const numDataRecords = parseInt(ascii(236, 8), 10);
  const recordDuration = parseFloat(ascii(244, 8));
  const ns = parseInt(ascii(252, 4), 10);
  if (!Number.isFinite(headerBytes) || !Number.isFinite(ns) || ns <= 0) {
    throw new Error("EDF: invalid header");
  }

  // Per-signal header section (each field length * ns)
  const labels: string[] = [];
  const physMin: number[] = [];
  const physMax: number[] = [];
  const digMin: number[] = [];
  const digMax: number[] = [];
  const samplesPerRecord: number[] = [];

  let off = 256;
  for (let i = 0; i < ns; i++) labels.push(ascii(off + i * 16, 16));
  off += 16 * ns;
  off += 80 * ns; // transducer
  off += 8 * ns;  // physical dimension
  for (let i = 0; i < ns; i++) physMin.push(parseFloat(ascii(off + i * 8, 8)));
  off += 8 * ns;
  for (let i = 0; i < ns; i++) physMax.push(parseFloat(ascii(off + i * 8, 8)));
  off += 8 * ns;
  for (let i = 0; i < ns; i++) digMin.push(parseInt(ascii(off + i * 8, 8), 10));
  off += 8 * ns;
  for (let i = 0; i < ns; i++) digMax.push(parseInt(ascii(off + i * 8, 8), 10));
  off += 8 * ns;
  off += 80 * ns; // prefiltering
  for (let i = 0; i < ns; i++) samplesPerRecord.push(parseInt(ascii(off + i * 8, 8), 10));
  // off += 8 * ns; off += 32 * ns; // reserved

  if (samplesPerRecord.some((s) => !Number.isFinite(s) || s <= 0)) {
    throw new Error("EDF: invalid samples-per-record");
  }
  const fs0 = samplesPerRecord[0] / recordDuration;

  const view = new DataView(buffer);
  const channels: string[] = [];
  const data: number[][] = [];
  const scales: number[] = [];
  const offsets: number[] = [];
  const keepIdx: number[] = [];
  const warnings: string[] = [];

  for (let i = 0; i < ns; i++) {
    if (samplesPerRecord[i] / recordDuration !== fs0) {
      warnings.push(`channel ${labels[i]} skipped (fs mismatch)`);
      continue;
    }
    // EDF Annotations channels start with "EDF Annotations"
    if (labels[i].startsWith("EDF Annotations")) continue;
    keepIdx.push(i);
    channels.push(labels[i]);
    const scale = (physMax[i] - physMin[i]) / (digMax[i] - digMin[i]);
    scales.push(scale);
    offsets.push(physMin[i] - scale * digMin[i]);
    data.push(new Array<number>(samplesPerRecord[i] * numDataRecords));
  }

  const recordSize = samplesPerRecord.reduce((a, b) => a + b, 0) * 2;
  for (let r = 0; r < numDataRecords; r++) {
    let cursor = headerBytes + r * recordSize;
    for (let i = 0; i < ns; i++) {
      const spr = samplesPerRecord[i];
      const keepPos = keepIdx.indexOf(i);
      if (keepPos === -1) {
        cursor += spr * 2;
        continue;
      }
      const scale = scales[keepPos];
      const ofs = offsets[keepPos];
      const out = data[keepPos];
      const base = r * spr;
      for (let s = 0; s < spr; s++) {
        const raw = view.getInt16(cursor + s * 2, true);
        out[base + s] = raw * scale + ofs;
      }
      cursor += spr * 2;
    }
  }

  return {
    channels,
    data,
    sampleRate: fs0,
    meta: { format: "edf", numDataRecords, recordDuration, warnings },
  };
}