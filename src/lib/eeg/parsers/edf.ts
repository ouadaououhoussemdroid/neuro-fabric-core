import type { EEGSignal } from "../types";

/**
 * EDF / EDF+ / BDF reader (16-bit or 24-bit signed little-endian).
 * Spec: https://www.edfplus.info/specs/edf.html
 *       https://www.teuniz.net/edfbrowser/bdf%20format%20description.html
 *
 * Returns one EEGSignal using the sample rate of the FIRST signal.
 * Channels with a different sample rate are skipped (warned via meta.warnings).
 *
 * EDF+ / BDF annotation channels ("EDF Annotations" / "BDF Annotations") are
 * parsed into `meta.annotations` as `{ onset: number, duration: number | null,
 * text: string }[]` rather than silently dropped.
 *
 * GDF is a distinct header layout; detection throws a clear error so callers
 * can fall back to the Pyodide+MNE path rather than getting a corrupt parse.
 */

/** A single parsed EDF+/BDF annotation (Time-Stamped Annotation List entry). */
export interface EDFAnnotation {
  /** Onset in seconds from the start of the recording. */
  onset: number;
  /** Duration in seconds, or null if not specified. */
  duration: number | null;
  /** Free-text annotation (may contain multiple comma-separated terms). */
  text: string;
}

type SampleWidth = 2 | 3;

export function parseEDF(buffer: ArrayBuffer): EEGSignal {
  const bytes = new Uint8Array(buffer);
  const dec = new TextDecoder("ascii");
  const ascii = (off: number, len: number) => dec.decode(bytes.subarray(off, off + len)).trim();

  const versionRaw = ascii(0, 8);
  const isBDF = versionRaw.charCodeAt(0) === 0xff && versionRaw.slice(1).startsWith("BIOSEMI");
  const sampleWidth: SampleWidth = isBDF ? 3 : 2;

  const headerBytes = parseInt(ascii(184, 8), 10);
  const numDataRecords = parseInt(ascii(236, 8), 10);
  const recordDuration = parseFloat(ascii(244, 8));
  const ns = parseInt(ascii(252, 4), 10);
  if (!Number.isFinite(headerBytes) || !Number.isFinite(ns) || ns <= 0) {
    throw new Error("EDF: invalid header");
  }
  if (!Number.isFinite(recordDuration) || recordDuration <= 0) {
    throw new Error("EDF: invalid record duration");
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
  off += 8 * ns; // physical dimension
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
  const annotations: EDFAnnotation[] = [];
  const annIdx: number[] = [];

  for (let i = 0; i < ns; i++) {
    // EDF+ / BDF annotation channels — must be checked before the fs-mismatch
    // guard, because annotation channels intentionally carry a different
    // samples-per-record (they are time-stamped, not regularly sampled).
    if (labels[i].startsWith("EDF Annotations") || labels[i].startsWith("BDF Annotations")) {
      annIdx.push(i);
      continue;
    }
    if (samplesPerRecord[i] / recordDuration !== fs0) {
      warnings.push(`channel ${labels[i]} skipped (fs mismatch)`);
      continue;
    }
    keepIdx.push(i);
    channels.push(labels[i]);
    const scale = (physMax[i] - physMin[i]) / (digMax[i] - digMin[i]);
    scales.push(scale);
    offsets.push(physMin[i] - scale * digMin[i]);
    data.push(new Array<number>(samplesPerRecord[i] * numDataRecords));
  }

  const recordSize = samplesPerRecord.reduce((a, b) => a + b, 0) * sampleWidth;

  // Per-record start timestamp carried by EDF+ TALs; for files without
  // annotations this stays 0 and annotation onsets are absolute.
  let recordStartOffset = 0;

  for (let r = 0; r < numDataRecords; r++) {
    let cursor = headerBytes + r * recordSize;
    for (let i = 0; i < ns; i++) {
      const spr = samplesPerRecord[i];
      if (annIdx.includes(i)) {
        // Annotation channel: raw bytes form a TAL stream.
        const raw = bytes.subarray(cursor, cursor + spr * sampleWidth);
        recordStartOffset = parseTAL(raw, recordStartOffset, annotations);
        cursor += spr * sampleWidth;
        continue;
      }
      const keepPos = keepIdx.indexOf(i);
      if (keepPos === -1) {
        cursor += spr * sampleWidth;
        continue;
      }
      const scale = scales[keepPos];
      const ofs = offsets[keepPos];
      const out = data[keepPos];
      const base = r * spr;
      for (let s = 0; s < spr; s++) {
        out[base + s] = readSample(view, cursor + s * sampleWidth, sampleWidth) * scale + ofs;
      }
      cursor += spr * sampleWidth;
    }
  }

  const meta: Record<string, unknown> = {
    format: isBDF ? "bdf" : "edf",
    numDataRecords,
    recordDuration,
    warnings,
  };
  if (annotations.length > 0) meta.annotations = annotations;

  return { channels, data, sampleRate: fs0, meta };
}

/** Read a signed little-endian sample of 2 (int16) or 3 (int24) bytes. */
function readSample(view: DataView, offset: number, width: SampleWidth): number {
  if (width === 2) return view.getInt16(offset, true);
  // BDF 24-bit: little-endian, sign-extended from 24 bits.
  const b0 = view.getUint8(offset);
  const b1 = view.getUint8(offset + 1);
  const b2 = view.getUint8(offset + 2);
  let val = b0 | (b1 << 8) | (b2 << 16);
  if (val & 0x800000) val |= ~0xffffff; // sign-extend
  return val;
}

/**
 * Parse a Time-Stamped Annotation List (EDF+ spec) from a raw byte slice of
 * one data record's annotation channel.
 *
 * TAL grammar (per EDF+ spec §3.1):
 *   onset \x14 [duration] \x15 annotation_text \x00
 *   onset \x14 \x15 annotation_text \x00     (no duration)
 *   onset \x14 \x15 \x00                      (record start time, empty text)
 *
 * Byte 0x14 (20) separates onset from duration; 0x15 (21) separates
 * duration from annotation text; 0x00 (0) terminates each TAL. One data
 * record's annotation channel may contain multiple concatenated TALs.
 *
 * `recordStart` carries the timestamp of the current record (from the
 * leading empty-text TAL if present) so annotation onsets are absolute.
 * Returns the updated record start for the caller to thread forward.
 */
function parseTAL(raw: Uint8Array, recordStart: number, out: EDFAnnotation[]): number {
  const text = new TextDecoder("latin1").decode(raw);
  let i = 0;
  const len = text.length;
  let recStart = recordStart;

  while (i < len) {
    // Read onset: everything up to 0x14 or 0x00.
    const onsetStart = i;
    while (i < len && text.charCodeAt(i) !== 0x14 && text.charCodeAt(i) !== 0x00) i++;
    if (i >= len) break;
    const onsetStr = text.slice(onsetStart, i).trim();
    const onset = Number(onsetStr);
    if (!Number.isFinite(onset)) {
      // Not a valid TAL onset; skip to next null terminator.
      while (i < len && text.charCodeAt(i) !== 0x00) i++;
      if (i < len) i++;
      continue;
    }

    // Expect 0x14 (onset→duration separator). If missing, malformed — skip.
    if (text.charCodeAt(i) === 0x14) {
      i++;
    }

    // Read optional duration until 0x15.
    const durStart = i;
    while (i < len && text.charCodeAt(i) !== 0x15 && text.charCodeAt(i) !== 0x00) i++;
    const durStr = text.slice(durStart, i).trim();
    let duration: number | null = null;
    if (durStr.length > 0) {
      const d = Number(durStr);
      if (Number.isFinite(d)) duration = d;
    }

    // Skip 0x15 if present.
    if (text.charCodeAt(i) === 0x15) {
      i++;
    }

    // Read annotation text until 0x00.
    const textStart = i;
    while (i < len && text.charCodeAt(i) !== 0x00) i++;
    const annText = text.slice(textStart, i).trim();
    if (i < len) i++; // consume 0x00

    if (annText.length === 0) {
      // Empty-text TAL: this is the record's start time.
      recStart = onset;
      continue;
    }
    out.push({ onset, duration, text: annText });
  }
  return recStart;
}
