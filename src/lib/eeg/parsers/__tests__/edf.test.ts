import { describe, it, expect } from "vitest";
import { parseEDF } from "../edf";

interface ChannelSpec {
  label: string;
  physMin: number;
  physMax: number;
  digMin: number;
  digMax: number;
  samplesPerRecord: number;
  /** Raw digital (int16) samples, length = samplesPerRecord * numDataRecords */
  rawSamples: number[];
}

/** Builds a byte-exact, spec-compliant minimal EDF buffer for testing. */
function buildEDF(opts: {
  channels: ChannelSpec[];
  numDataRecords: number;
  recordDuration: number;
}): ArrayBuffer {
  const { channels, numDataRecords, recordDuration } = opts;
  const ns = channels.length;
  const perSignalFieldsBytes = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80 + 8 + 32;
  const headerBytes = 256 + ns * perSignalFieldsBytes;
  const recordSize = channels.reduce((sum, c) => sum + c.samplesPerRecord, 0) * 2;
  const dataBytes = recordSize * numDataRecords;

  const buf = new ArrayBuffer(headerBytes + dataBytes);
  const bytes = new Uint8Array(buf);
  bytes.fill(0x20); // space-pad the whole header by default
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
  writeField(236, String(numDataRecords), 8);
  writeField(244, String(recordDuration), 8);
  writeField(252, String(ns), 4);

  // Write each fixed-width field block in its own pass, in the exact
  // order parseEDF reads them.
  let off = 256;
  channels.forEach((c, i) => writeField(off + i * 16, c.label, 16));
  off += 16 * ns;
  channels.forEach((c, i) => writeField(off + i * 80, "transducer", 80));
  off += 80 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, "uV", 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.physMin), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.physMax), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.digMin), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.digMax), 8));
  off += 8 * ns;
  channels.forEach((c, i) => writeField(off + i * 80, "prefilter", 80));
  off += 80 * ns;
  channels.forEach((c, i) => writeField(off + i * 8, String(c.samplesPerRecord), 8));

  const view = new DataView(buf);
  for (let r = 0; r < numDataRecords; r++) {
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

describe("parseEDF", () => {
  it("decodes digital samples to physical units via the physMin/physMax/digMin/digMax scale", () => {
    // scale = (physMax - physMin) / (digMax - digMin) = 200 / 4095
    // offset = physMin - scale * digMin = -100 - scale * -2048
    const physMin = -100,
      physMax = 100,
      digMin = -2048,
      digMax = 2047;
    const scale = (physMax - physMin) / (digMax - digMin);
    const offset = physMin - scale * digMin;
    const rawSamples = [0, 1000, -1000, digMax, digMin];

    const buf = buildEDF({
      channels: [
        {
          label: "Cz",
          physMin,
          physMax,
          digMin,
          digMax,
          samplesPerRecord: rawSamples.length,
          rawSamples,
        },
      ],
      numDataRecords: 1,
      recordDuration: 1,
    });

    const signal = parseEDF(buf);
    expect(signal.channels).toEqual(["Cz"]);
    expect(signal.sampleRate).toBe(rawSamples.length); // samplesPerRecord / recordDuration(1)
    for (let i = 0; i < rawSamples.length; i++) {
      expect(signal.data[0][i]).toBeCloseTo(rawSamples[i] * scale + offset, 6);
    }
  });

  it("concatenates multiple data records in order", () => {
    const spec: ChannelSpec = {
      label: "Fz",
      physMin: -1,
      physMax: 1,
      digMin: -100,
      digMax: 100,
      samplesPerRecord: 2,
      rawSamples: [10, 20, 30, 40, 50, 60], // 3 records x 2 samples
    };
    const buf = buildEDF({ channels: [spec], numDataRecords: 3, recordDuration: 1 });
    const signal = parseEDF(buf);
    expect(signal.data[0]).toHaveLength(6);
    // monotonically increasing raw samples -> monotonically increasing physical values
    for (let i = 1; i < signal.data[0].length; i++) {
      expect(signal.data[0][i]).toBeGreaterThan(signal.data[0][i - 1]);
    }
  });

  it("reads multiple channels independently with their own scale", () => {
    const chA: ChannelSpec = {
      label: "A",
      physMin: 0,
      physMax: 10,
      digMin: 0,
      digMax: 100,
      samplesPerRecord: 2,
      rawSamples: [50, 50],
    };
    const chB: ChannelSpec = {
      label: "B",
      physMin: -5,
      physMax: 5,
      digMin: -50,
      digMax: 50,
      samplesPerRecord: 2,
      rawSamples: [25, 25],
    };
    const buf = buildEDF({ channels: [chA, chB], numDataRecords: 1, recordDuration: 1 });
    const signal = parseEDF(buf);
    expect(signal.channels).toEqual(["A", "B"]);
    expect(signal.data[0][0]).toBeCloseTo(5, 6); // midpoint of A's range
    expect(signal.data[1][0]).toBeCloseTo(2.5, 6); // 25/50 * 5
  });

  it("skips a channel whose sample rate differs from the first channel and records a warning", () => {
    const fast: ChannelSpec = {
      label: "fast",
      physMin: -1,
      physMax: 1,
      digMin: -1,
      digMax: 1,
      samplesPerRecord: 4,
      rawSamples: [0, 0, 0, 0],
    };
    const slow: ChannelSpec = {
      label: "slow",
      physMin: -1,
      physMax: 1,
      digMin: -1,
      digMax: 1,
      samplesPerRecord: 2, // different fs -> should be skipped
      rawSamples: [0, 0],
    };
    const buf = buildEDF({ channels: [fast, slow], numDataRecords: 1, recordDuration: 1 });
    const signal = parseEDF(buf);
    expect(signal.channels).toEqual(["fast"]);
    expect(signal.meta?.warnings).toEqual(
      expect.arrayContaining([expect.stringContaining("slow")]),
    );
  });

  it("throws on an invalid header (ns <= 0)", () => {
    const buf = new ArrayBuffer(256);
    const bytes = new Uint8Array(buf);
    bytes.fill(0x20);
    // ns field (bytes 252-255) left as spaces -> parseInt -> NaN
    expect(() => parseEDF(buf)).toThrow("EDF: invalid header");
  });
});
