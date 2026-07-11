import type { EEGSignal } from "../types";

/**
 * Minimal .npy v1.0 / v2.0 reader.
 * Spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 *
 * Supports float32 / float64 / int16 / int32 little-endian, 2-D arrays only.
 * Shape convention: [C, N] (channels x samples). Pass sampleRate explicitly.
 */
export function parseNPY(buffer: ArrayBuffer, sampleRate: number): EEGSignal {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("NPY: sampleRate is required and must be > 0");
  }
  const bytes = new Uint8Array(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < magic.length; i++) {
    if (bytes[i] !== magic[i]) throw new Error("NPY: bad magic");
  }
  const major = bytes[6];
  const headerLen =
    major >= 2 ? new DataView(buffer).getUint32(8, true) : new DataView(buffer).getUint16(8, true);
  const headerStart = major >= 2 ? 12 : 10;
  const header = new TextDecoder("ascii").decode(
    bytes.subarray(headerStart, headerStart + headerLen),
  );

  // Header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (64, 2560), }
  const descrMatch = header.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !shapeMatch) throw new Error("NPY: bad header");
  const descr = descrMatch[1];
  const fortran = fortranMatch?.[1] === "True";
  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  if (shape.length !== 2) throw new Error("NPY: only 2-D arrays supported");

  const [d0, d1] = shape;
  const dataStart = headerStart + headerLen;
  const view = new DataView(buffer, dataStart);

  const read = (i: number): number => {
    switch (descr) {
      case "<f4":
      case "|f4":
        return view.getFloat32(i * 4, true);
      case "<f8":
      case "|f8":
        return view.getFloat64(i * 8, true);
      case "<i2":
      case "|i2":
        return view.getInt16(i * 2, true);
      case "<i4":
      case "|i4":
        return view.getInt32(i * 4, true);
      default:
        throw new Error(`NPY: unsupported dtype ${descr}`);
    }
  };

  // Heuristic: assume channels are the smaller of the two dims.
  const channelsAxis0 = d0 <= d1;
  const C = channelsAxis0 ? d0 : d1;
  const N = channelsAxis0 ? d1 : d0;
  const data: number[][] = Array.from({ length: C }, () => new Array<number>(N));

  for (let i = 0; i < d0; i++) {
    for (let j = 0; j < d1; j++) {
      const linear = fortran ? j * d0 + i : i * d1 + j;
      const v = read(linear);
      const c = channelsAxis0 ? i : j;
      const n = channelsAxis0 ? j : i;
      data[c][n] = v;
    }
  }

  const channels = Array.from({ length: C }, (_, i) => `ch${i}`);
  return { channels, data, sampleRate, meta: { format: "npy", dtype: descr, shape } };
}
