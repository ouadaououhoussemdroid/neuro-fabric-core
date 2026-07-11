import { describe, it, expect } from "vitest";
import { parseNPY } from "../npy";

/** Builds a minimal, spec-compliant .npy v1.0 buffer (float32, 2-D). */
function buildNPY(opts: {
  shape: [number, number];
  data: number[]; // row-major (C-order) or column-major (fortran) per `fortran`
  fortran?: boolean;
  dtype?: "<f4" | "<f8" | "<i2" | "<i4";
}): ArrayBuffer {
  const { shape, data, fortran = false, dtype = "<f4" } = opts;
  const headerDict = `{'descr': '${dtype}', 'fortran_order': ${fortran ? "True" : "False"}, 'shape': (${shape[0]}, ${shape[1]}), }`;
  // Total header (magic[6] + version[2] + headerLen[2] + dict) must be a
  // multiple of 64 bytes per spec; pad the dict with spaces + trailing \n.
  const prefixLen = 6 + 2 + 2;
  const unpadded = headerDict.length + 1; // +1 for trailing \n
  const total = Math.ceil((prefixLen + unpadded) / 64) * 64;
  const padLen = total - prefixLen - unpadded;
  const paddedDict = headerDict + " ".repeat(padLen) + "\n";
  const headerLen = paddedDict.length;

  const bytesPerElem = dtype === "<f8" ? 8 : dtype === "<i2" ? 2 : 4;
  const dataBytes = data.length * bytesPerElem;
  const buf = new ArrayBuffer(prefixLen + headerLen + dataBytes);
  const bytes = new Uint8Array(buf);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  bytes.set(magic, 0);
  bytes[6] = 1; // major version
  bytes[7] = 0; // minor version
  new DataView(buf).setUint16(8, headerLen, true);
  bytes.set(new TextEncoder().encode(paddedDict), 10);

  const view = new DataView(buf, prefixLen + headerLen);
  for (let i = 0; i < data.length; i++) {
    if (dtype === "<f4") view.setFloat32(i * 4, data[i], true);
    else if (dtype === "<f8") view.setFloat64(i * 8, data[i], true);
    else if (dtype === "<i2") view.setInt16(i * 2, data[i], true);
    else if (dtype === "<i4") view.setInt32(i * 4, data[i], true);
  }
  return buf;
}

describe("parseNPY", () => {
  it("throws if sampleRate is missing or invalid", () => {
    const buf = buildNPY({ shape: [2, 3], data: [1, 2, 3, 4, 5, 6] });
    expect(() => parseNPY(buf, 0)).toThrow("sampleRate is required");
  });

  it("throws on bad magic bytes", () => {
    const buf = new ArrayBuffer(16);
    expect(() => parseNPY(buf, 128)).toThrow("bad magic");
  });

  it("reads a C-order float32 array with shape [channels, samples]", () => {
    // 2 channels x 3 samples, C-order (row-major): ch0=[1,2,3], ch1=[4,5,6]
    const buf = buildNPY({ shape: [2, 3], data: [1, 2, 3, 4, 5, 6] });
    const signal = parseNPY(buf, 250);
    expect(signal.sampleRate).toBe(250);
    expect(signal.channels).toEqual(["ch0", "ch1"]);
    expect(signal.data[0]).toEqual([1, 2, 3]);
    expect(signal.data[1]).toEqual([4, 5, 6]);
  });

  it("reads a Fortran-order (column-major) array correctly", () => {
    // 2 channels x 3 samples, Fortran-order (column-major): the flat buffer
    // is [ch0[0], ch1[0], ch0[1], ch1[1], ch0[2], ch1[2]]
    const flat = [1, 4, 2, 5, 3, 6];
    const buf = buildNPY({ shape: [2, 3], data: flat, fortran: true });
    const signal = parseNPY(buf, 128);
    expect(signal.data[0]).toEqual([1, 2, 3]);
    expect(signal.data[1]).toEqual([4, 5, 6]);
  });

  it("picks the smaller dimension as channels (heuristic) when shape is [samples, channels]", () => {
    // 5 samples x 2 channels -> channels axis is d1 (the smaller dim)
    const flat = [1, 10, 2, 20, 3, 30, 4, 40, 5, 50]; // row-major [5,2]
    const buf = buildNPY({ shape: [5, 2], data: flat });
    const signal = parseNPY(buf, 128);
    expect(signal.channels).toHaveLength(2);
    expect(signal.data[0]).toEqual([1, 2, 3, 4, 5]);
    expect(signal.data[1]).toEqual([10, 20, 30, 40, 50]);
  });

  it("throws for a non-2-D shape", () => {
    const headerDict = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }";
    const prefixLen = 10;
    const total = Math.ceil((prefixLen + headerDict.length + 1) / 64) * 64;
    const padded = headerDict + " ".repeat(total - prefixLen - headerDict.length - 1) + "\n";
    const buf = new ArrayBuffer(prefixLen + padded.length + 12);
    const bytes = new Uint8Array(buf);
    bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0);
    bytes[6] = 1;
    bytes[7] = 0;
    new DataView(buf).setUint16(8, padded.length, true);
    bytes.set(new TextEncoder().encode(padded), 10);
    expect(() => parseNPY(buf, 128)).toThrow("only 2-D arrays supported");
  });

  it("supports int16 and int32 dtypes", () => {
    const i2 = buildNPY({ shape: [1, 3], data: [100, -100, 32000], dtype: "<i2" });
    expect(parseNPY(i2, 128).data[0]).toEqual([100, -100, 32000]);

    const i4 = buildNPY({ shape: [1, 3], data: [100000, -100000, 0], dtype: "<i4" });
    expect(parseNPY(i4, 128).data[0]).toEqual([100000, -100000, 0]);
  });

  it("supports float64 dtype", () => {
    const buf = buildNPY({ shape: [1, 2], data: [1.23456789, -9.87654321], dtype: "<f8" });
    const signal = parseNPY(buf, 128);
    expect(signal.data[0][0]).toBeCloseTo(1.23456789, 8);
    expect(signal.data[0][1]).toBeCloseTo(-9.87654321, 8);
  });

  it("records format/dtype/shape in meta", () => {
    const buf = buildNPY({ shape: [2, 3], data: [1, 2, 3, 4, 5, 6] });
    const signal = parseNPY(buf, 128);
    expect(signal.meta).toMatchObject({ format: "npy", dtype: "<f4", shape: [2, 3] });
  });
});
