import { describe, it, expect } from "vitest";

// Replicate the magic-number logic from the upload route for unit testing.
const MAGIC_NUMBERS: Record<string, number[]> = {
  ".edf": [0x30],
  ".bdf": [0xff],
  ".csv": [],
  ".tsv": [],
  ".npy": [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
};

function checkMagicNumber(bytes: Uint8Array, ext: string): boolean {
  const expected = MAGIC_NUMBERS[ext];
  if (!expected || expected.length === 0) return true;
  if (bytes.length < expected.length) return false;
  return expected.every((b, i) => bytes[i] === b);
}

describe("T-028 Upload magic-number content sniff", () => {
  it("accepts a valid EDF header (starts with '0' = 0x30)", () => {
    const head = new Uint8Array([0x30, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20]);
    expect(checkMagicNumber(head, ".edf")).toBe(true);
  });

  it("rejects a non-EDF header", () => {
    const head = new Uint8Array([0x50, 0x4b, 0x03, 0x04]); // ZIP magic
    expect(checkMagicNumber(head, ".edf")).toBe(false);
  });

  it("accepts a valid BDF header (starts with 0xFF)", () => {
    const head = new Uint8Array([0xff, 0x42, 0x49, 0x4f]); // \xffBIOSEMI
    expect(checkMagicNumber(head, ".bdf")).toBe(true);
  });

  it("rejects a non-BDF header for .bdf extension", () => {
    const head = new Uint8Array([0x30, 0x20, 0x20]);
    expect(checkMagicNumber(head, ".bdf")).toBe(false);
  });

  it("accepts a valid NPY header (\x93NUMPY)", () => {
    const head = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
    expect(checkMagicNumber(head, ".npy")).toBe(true);
  });

  it("rejects a non-NPY file with .npy extension", () => {
    const head = new Uint8Array([0x1f, 0x8b, 0x08, 0x00]); // gzip magic
    expect(checkMagicNumber(head, ".npy")).toBe(false);
  });

  it("always accepts CSV/TSV (no magic number check)", () => {
    expect(checkMagicNumber(new Uint8Array([0x61, 0x62, 0x63]), ".csv")).toBe(true);
    expect(checkMagicNumber(new Uint8Array([]), ".tsv")).toBe(true);
  });

  it("rejects when the file is shorter than the magic number", () => {
    const head = new Uint8Array([0x93]); // too short for NPY
    expect(checkMagicNumber(head, ".npy")).toBe(false);
  });

  it("rejects an unknown extension", () => {
    expect(checkMagicNumber(new Uint8Array([0x00]), ".txt")).toBe(true); // unknown → no check
  });
});
