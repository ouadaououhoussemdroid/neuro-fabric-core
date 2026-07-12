import { describe, it, expect } from "vitest";
import { createFileSource, type FileFormat } from "../acquisition";
import { parseEDF } from "../parsers";

/** Minimal spec-compliant EDF builder (mirrors the edf.test.ts helper). */
function buildEDF(opts: {
  channels: Array<{
    label: string;
    physMin: number;
    physMax: number;
    digMin: number;
    digMax: number;
    samplesPerRecord: number;
    rawSamples: number[];
  }>;
  numDataRecords: number;
  recordDuration: number;
}): ArrayBuffer {
  const channels = opts.channels;
  const ns = channels.length;
  const perSignalFieldsBytes = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80 + 8 + 32;
  const headerBytes = 256 + ns * perSignalFieldsBytes;
  const recordSize = channels.reduce((sum, c) => sum + c.samplesPerRecord, 0) * 2;
  const dataBytes = recordSize * opts.numDataRecords;

  const buf = new ArrayBuffer(headerBytes + dataBytes);
  const bytes = new Uint8Array(buf);
  bytes.fill(0x20);
  const enc = new TextEncoder();
  const writeField = (offset: number, value: string, width: number) => {
    const padded = value.slice(0, width).padEnd(width, " ");
    bytes.set(enc.encode(padded), offset);
  };

  writeField(0, "0", 8);
  writeField(8, "test patient", 80);
  writeField(88, "test recording", 80);
  writeField(168, "01.01.26", 8);
  writeField(176, "00.00.00", 8);
  writeField(184, String(headerBytes), 8);
  writeField(236, String(opts.numDataRecords), 8);
  writeField(244, String(opts.recordDuration), 8);
  writeField(252, String(ns), 4);

  let off = 256;
  channels.forEach((c, i) => writeField(off + i * 16, c.label, 16));
  off += 16 * ns;
  channels.forEach((_, i) => writeField(off + i * 80, "transducer", 80));
  off += 80 * ns;
  channels.forEach((_, i) => writeField(off + i * 8, "uV", 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.physMin), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.physMax), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.digMin), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.digMax), 8));
  off += 8 * ns;
  channels.forEach((_, i) => writeField(off + i * 80, "prefilter", 80));
  off += 80 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.samplesPerRecord), 8));

  const view = new DataView(buf);
  for (let r = 0; r < opts.numDataRecords; r++) {
    let cursor = headerBytes + r * recordSize;
    for (const c of channels) {
      for (let s = 0; s < c.samplesPerRecord; s++) {
        view.setInt16(cursor, c.rawSamples[r * c.samplesPerRecord + s], true);
        cursor += 2;
      }
    }
  }
  return buf;
}

const SAMPLE_EDF = buildEDF({
  channels: [
    {
      label: "Cz",
      physMin: -100,
      physMax: 100,
      digMin: -2048,
      digMax: 2047,
      samplesPerRecord: 5,
      rawSamples: [0, 1000, -1000, 2047, -2048],
    },
  ],
  numDataRecords: 1,
  recordDuration: 1,
});

describe("createFileSource", () => {
  it("emits exactly one chunk matching parseEDF output", async () => {
    const expected = parseEDF(SAMPLE_EDF);
    const source = createFileSource({ data: SAMPLE_EDF, format: "edf" });

    const chunks: (typeof expected)[] = [];
    for await (const chunk of source.stream()) chunks.push(chunk);

    expect(chunks).toHaveLength(1);
    expect(chunks[0].channels).toEqual(expected.channels);
    expect(chunks[0].sampleRate).toBe(expected.sampleRate);
    expect(chunks[0].data).toEqual(expected.data);
    expect(chunks[0].meta?.source).toBe("file");
  });

  it("exposes metadata without consuming the stream", () => {
    const source = createFileSource({ data: SAMPLE_EDF, format: "edf" });
    expect(source.metadata.channels).toEqual(["Cz"]);
    expect(source.metadata.sampleRate).toBe(5);
    expect(source.type).toBe("file");
  });

  it("parses CSV with an explicit sample rate", async () => {
    const csv = "Cz,Fz\n1,2\n3,4\n5,6\n";
    const source = createFileSource({ data: csv, format: "csv", sampleRate: 100 });

    const chunks = [];
    for await (const c of source.stream()) chunks.push(c);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].channels).toEqual(["Cz", "Fz"]);
    expect(chunks[0].sampleRate).toBe(100);
    expect(chunks[0].data).toEqual([
      [1, 3, 5],
      [2, 4, 6],
    ]);
  });

  it("throws when CSV is missing the sample rate", async () => {
    const source = createFileSource({ data: "a,b\n1,2\n", format: "csv" });
    await expect(async () => {
      for await (const _ of source.stream()) {
        void _;
      }
    }).rejects.toThrow(/sampleRate required/);
  });

  it("rejects an unsupported format", async () => {
    const source = createFileSource({ data: SAMPLE_EDF, format: "gdf" as FileFormat });
    await expect(async () => {
      for await (const _ of source.stream()) {
        void _;
      }
    }).rejects.toThrow(/Unsupported file format/);
  });

  it("propagates parser errors for a corrupt buffer", async () => {
    const bad = new ArrayBuffer(256);
    new Uint8Array(bad).fill(0x20);
    const source = createFileSource({ data: bad, format: "edf" });
    await expect(async () => {
      for await (const _ of source.stream()) {
        void _;
      }
    }).rejects.toThrow();
  });

  it("caches parsing so repeated stream() calls yield the same data", async () => {
    const source = createFileSource({ data: SAMPLE_EDF, format: "edf", filename: "rec.edf" });
    const first = [];
    for await (const c of source.stream()) first.push(c);
    const second = [];
    for await (const c of source.stream()) second.push(c);
    expect(second[0].data).toEqual(first[0].data);
    expect(second[0].meta?.filename).toBe("rec.edf");
  });
});
