/**
 * M38 — Sleep-EDF loader unit tests.
 *
 * Tests cover:
 * - Dataset metadata correctness (matches M31 §9 / benchmark_archive manifest)
 * - Sleep stage label parsing (5-stage: W, N1, N2, N3, REM)
 * - Channel expansion 7→62 (nearest-neighbour interpolation)
 * - Annotation epoch parsing (TAL → 30s epochs)
 * - Integration: preprocessSleepEDF end-to-end on a synthetic EDF buffer
 */
import { describe, it, expect } from "vitest";
import {
  sleepEDFDataset,
  SLEEP_STAGES,
  SLEEP_STAGE_LABEL_TO_ID,
  SLEEP_STAGE_ID_TO_LABEL,
  SLEEP_STAGE_LABELS,
  parseSleepAnnotations,
  expandSleepToEEGPT,
  preprocessSleepEDF,
  JOINT_2312_SAMPLE_RATE_HZ,
  JOINT_WINDOW_SEC,
  SLEEP_EPOCH_SEC,
} from "../sleep-edf";
import { EEGPT_CHANNELS_62 } from "../../eeg/channels";
import type { EEGAnnotation as EDFAnnotationType } from "../../eeg/parsers/edf";
import type { EEGSignal } from "../../eeg/types";

// ─── EDF test buffer builder (mirrors src/lib/eeg/parsers/__tests__/edf.test.ts) ───

interface ChannelSpec {
  label: string;
  physMin: number;
  physMax: number;
  digMin: number;
  digMax: number;
  samplesPerRecord: number;
  rawSamples: number[];
}

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

  writeField(0, "0", 8); // version
  writeField(8, "test patient", 80);
  writeField(88, "test recording", 80);
  writeField(168, "01.01.26", 8); // start date
  writeField(176, "00.00.00", 8); // start time
  writeField(184, String(headerBytes), 8);
  writeField(236, String(numDataRecords), 8);
  writeField(244, String(recordDuration), 8);
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

// ─── Signal helpers ───────────────────────────────────────────────────────────

interface EDFAnnotation {
  onset: number;
  duration: number | null;
  text: string;
}

function makeAnn(onset: number, text: string, duration: number | null = 30): EDFAnnotation {
  return { onset, duration, text };
}

/** Build a synthetic 7-channel EEGSignal for testing. */
function make7ChannelSignal(samples: number, withChannels: Record<string, number[]> = {}): EEGSignal {
  const baseChannels = ["Fpz-Cz", "Pz-Oz", "EOG"];
  const data: number[][] = [];
  for (let i = 0; i < 3; i++) {
    const ch = baseChannels[i];
    data.push(
      withChannels[ch] ?? Array.from({ length: samples }, (_, j) => Math.sin(j * 0.01) * 100),
    );
  }
  return {
    channels: baseChannels,
    data,
    sampleRate: 100,
    meta: { format: "edf", numDataRecords: 1, recordDuration: samples / 100 },
  };
}

/** Build a minimal EDF buffer with 3 Sleep-EDF channels at 100 Hz. */
function makeTestEDF(samples: number): ArrayBuffer {
  const nRecords = Math.ceil(samples / 100);
  const perRecord = 100;
  return buildEDF({
    channels: [
      {
        label: "Fpz-Cz",
        physMin: -100,
        physMax: 100,
        digMin: -2048,
        digMax: 2047,
        samplesPerRecord: perRecord,
        rawSamples: Array.from({ length: perRecord * nRecords }, (_, i) =>
          Math.round(Math.sin((i / 100) * 0.5) * 1000),
        ),
      },
      {
        label: "Pz-Oz",
        physMin: -100,
        physMax: 100,
        digMin: -2048,
        digMax: 2047,
        samplesPerRecord: perRecord,
        rawSamples: Array.from({ length: perRecord * nRecords }, (_, i) =>
          Math.round(Math.cos((i / 100) * 0.5) * 1000),
        ),
      },
      {
        label: "EOG",
        physMin: -100,
        physMax: 100,
        digMin: -2048,
        digMax: 2047,
        samplesPerRecord: perRecord,
        rawSamples: Array.from({ length: perRecord * nRecords }, (_, i) =>
          Math.round(Math.sin((i / 100) * 2.0) * 800),
        ),
      },
    ],
    numDataRecords: nRecords,
    recordDuration: 1, // 1 second per record → 100 Hz
  });
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe("sleepEDFDataset", () => {
  it("has correct metadata matching M31 §9", () => {
    expect(sleepEDFDataset.name).toBe("Sleep-EDF");
    expect(sleepEDFDataset.license).toBe("BSD-3-Clause");
    expect(sleepEDFDataset.nSubjects).toBe(99);
    expect(sleepEDFDataset.nChannels).toBe(7);
    expect(sleepEDFDataset.sampleRate).toBe(100);
    expect(sleepEDFDataset.nClasses).toBe(5);
  });

  it("has the correct 5-stage sleep metadata", () => {
    expect(sleepEDFDataset.metadata.stages).toEqual(["W", "N1", "N2", "N3", "REM"]);
    expect(sleepEDFDataset.metadata.paradigm).toBe("sleep_staging");
    expect(sleepEDFDataset.metadata.sessions).toBe(2);
  });
});

describe("SLEEP_STAGES", () => {
  it("maps numeric stage codes (Sleep-EDF standard format)", () => {
    expect(SLEEP_STAGES["0"]).toBe("W");
    expect(SLEEP_STAGES["1"]).toBe("N1");
    expect(SLEEP_STAGES["2"]).toBe("N2");
    expect(SLEEP_STAGES["3"]).toBe("N3");
    expect(SLEEP_STAGES["4"]).toBe("N3");
    expect(SLEEP_STAGES["5"]).toBe("REM");
  });

  it("maps letter codes (W, R)", () => {
    expect(SLEEP_STAGES["W"]).toBe("W");
    expect(SLEEP_STAGES["R"]).toBe("REM");
  });

  it("maps named forms (Wake, N1-N3, REM)", () => {
    expect(SLEEP_STAGES["Wake"]).toBe("W");
    expect(SLEEP_STAGES["N1"]).toBe("N1");
    expect(SLEEP_STAGES["N2"]).toBe("N2");
    expect(SLEEP_STAGES["N3"]).toBe("N3");
    expect(SLEEP_STAGES["REM"]).toBe("REM");
  });

  it("maps SWS to N3", () => {
    expect(SLEEP_STAGES["SWS"]).toBe("N3");
  });
});

describe("SLEEP_STAGE_LABELS / ID mappings", () => {
  it("has all 5 standard stages", () => {
    expect(SLEEP_STAGE_LABELS).toEqual(["W", "N1", "N2", "N3", "REM"]);
  });

  it("round-trips all 5 stages through ID mapping", () => {
    const stages: ("W" | "N1" | "N2" | "N3" | "REM")[] = ["W", "N1", "N2", "N3", "REM"];
    for (const s of stages) {
      const id = SLEEP_STAGE_LABEL_TO_ID[s];
      expect(id).toBeGreaterThanOrEqual(0);
      expect(SLEEP_STAGE_ID_TO_LABEL[id]).toBe(s);
    }
  });

  it("encodes UNKNOWN as -1", () => {
    expect(SLEEP_STAGE_LABEL_TO_ID.UNKNOWN).toBe(-1);
  });
});

describe("parseSleepAnnotations", () => {
  it("parses all 5 standard Sleep-EDF stage annotations", () => {
    const anns = [
      makeAnn(0, "Sleep stage W"),
      makeAnn(30, "Sleep stage 1"),
      makeAnn(60, "Sleep stage 2"),
      makeAnn(90, "Sleep stage 3"),
      makeAnn(120, "Sleep stage R"),
    ];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs).toHaveLength(5);
    expect(epochs.map((e) => e.stage)).toEqual(["W", "N1", "N2", "N3", "REM"]);
  });

  it("parses named stage forms (N1, N2, N3, REM)", () => {
    const anns = [
      makeAnn(0, "Sleep stage N2"),
      makeAnn(30, "Sleep stage N3"),
      makeAnn(60, "Sleep stage REM"),
    ];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs).toHaveLength(3);
    expect(epochs.map((e) => e.stage)).toEqual(["N2", "N3", "REM"]);
  });

  it("skips non-sleep annotations", () => {
    const anns = [
      makeAnn(0, "Sleep stage 2"),
      makeAnn(30, "Some other event"),
      makeAnn(60, "Sleep stage R"),
    ];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs).toHaveLength(2);
    expect(epochs[0].stage).toBe("N2");
    expect(epochs[1].stage).toBe("REM");
  });

  it("uses default 30s epoch duration when absent", () => {
    const anns = [makeAnn(0, "Sleep stage W", null)];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs[0].duration).toBe(30);
    expect(epochs[0].endTime).toBe(30);
  });

  it("uses explicit duration when present", () => {
    const anns = [makeAnn(60, "Sleep stage 3", 60)];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs[0].duration).toBe(60);
    expect(epochs[0].startTime).toBe(60);
    expect(epochs[0].endTime).toBe(120);
  });

  it("assigns sequential indices", () => {
    const anns = [
      makeAnn(0, "Sleep stage W"),
      makeAnn(30, "Sleep stage N1"),
      makeAnn(60, "Sleep stage N2"),
    ];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs.map((e) => e.index)).toEqual([0, 1, 2]);
  });

  it("handles case-insensitive 'Sleep Stage' prefix", () => {
    const anns = [makeAnn(0, "sleep stage 2"), makeAnn(30, "SLEEP STAGE R")];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs).toHaveLength(2);
    expect(epochs[0].stage).toBe("N2");
    expect(epochs[1].stage).toBe("REM");
  });

  it("returns empty array for no sleep annotations", () => {
    const epochs = parseSleepAnnotations([makeAnn(0, "Event marker")]);
    expect(epochs).toEqual([]);
  });

  it("maps stage 4 to N3 (AASM merge)", () => {
    const anns = [makeAnn(0, "Sleep stage 4")];
    const epochs = parseSleepAnnotations(anns);
    expect(epochs[0].stage).toBe("N3");
  });
});

describe("expandSleepToEEGPT", () => {
  it("expands 7 channels to exactly 62 EEGPT channels", () => {
    const signal = make7ChannelSignal(1000);
    const expanded = expandSleepToEEGPT(signal);
    expect(expanded.channels).toHaveLength(62);
    expect(expanded.data).toHaveLength(62);
  });

  it("produces channels in EEGPT_CHANNELS_62 order", () => {
    const signal = make7ChannelSignal(1000);
    const expanded = expandSleepToEEGPT(signal);
    expect(expanded.channels).toEqual(EEGPT_CHANNELS_62);
  });

  it("copies Fpz-Cz data into frontal/central channels", () => {
    const ramp = Array.from({ length: 1000 }, (_, i) => i);
    const signal = make7ChannelSignal(1000, { "Fpz-Cz": ramp });
    const expanded = expandSleepToEEGPT(signal);
    // FP1 should be a copy of Fpz-Cz
    const fp1Idx = expanded.channels.indexOf("FP1");
    const fpzCzIdx = signal.channels.indexOf("Fpz-Cz");
    expect(expanded.data[fp1Idx]).toEqual(signal.data[fpzCzIdx]);
  });

  it("copies Pz-Oz data into parietal/occipital channels", () => {
    const ramp = Array.from({ length: 1000 }, (_, i) => i + 100);
    const signal = make7ChannelSignal(1000, { "Pz-Oz": ramp });
    const expanded = expandSleepToEEGPT(signal);
    // PZ should be a copy of Pz-Oz
    const pzIdx = expanded.channels.indexOf("PZ");
    const pzOzIdx = signal.channels.indexOf("Pz-Oz");
    expect(expanded.data[pzIdx]).toEqual(signal.data[pzOzIdx]);
  });

  it("preserves sample count across expansion", () => {
    const signal = make7ChannelSignal(2000);
    const expanded = expandSleepToEEGPT(signal);
    for (const ch of expanded.data) {
      expect(ch.length).toBe(2000);
    }
  });

  it("preserves sample rate in output", () => {
    const signal = make7ChannelSignal(1000);
    const expanded = expandSleepToEEGPT(signal);
    expect(expanded.sampleRate).toBe(signal.sampleRate);
  });

  it("records expansion metadata", () => {
    const signal = make7ChannelSignal(1000);
    const expanded = expandSleepToEEGPT(signal);
    expect(expanded.meta?.expansion).toBe("7→62 (nearest-neighbour spatial interpolation)");
    expect(expanded.meta?.source_channel_count).toBe(signal.channels.length);
    expect(expanded.meta?.source_channels).toEqual(signal.channels);
  });

  it("can expand from 1 channel (all default to Fpz-Cz)", () => {
    const ramp = Array.from({ length: 100 }, (_, i) => i);
    const signal: EEGSignal = {
      channels: ["Fpz-Cz"],
      data: [ramp],
      sampleRate: 100,
      meta: {},
    };
    const expanded = expandSleepToEEGPT(signal);
    expect(expanded.channels).toHaveLength(62);
    const fpzCzData = signal.data[0];
    for (const ch of expanded.data) {
      expect(ch).toEqual(fpzCzData);
    }
  });

  it("falls back to Fpz-Cz for channels with no explicit mapping", () => {
    const signal = make7ChannelSignal(100, { "Fpz-Cz": Array(100).fill(5) });
    const expanded = expandSleepToEEGPT(signal);
    // FCZ maps to Fpz-Cz in the neighbour table
    const fczIdx = expanded.channels.indexOf("FCZ");
    expect(expanded.data[fczIdx]).toEqual(signal.data[0]);
  });
});

describe("preprocessSleepEDF", () => {
  it("returns 62-channel expanded signal at 250 Hz", () => {
    // 50 records × 100 samples = 5000 samples at 100 Hz → 12500 samples at 250 Hz
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    expect(result.signal.channels).toHaveLength(62);
    expect(result.signal.sampleRate).toBe(JOINT_2312_SAMPLE_RATE_HZ);
  });

  it("produces aligned windows across all blocks", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    expect(result.report.cbramodWindows.length).toBe(result.report.eegptWindows.length);
    expect(result.report.v2Windows.length).toBe(result.report.eegptWindows.length);
    expect(result.report.cbramodWindows.length).toBeGreaterThan(0);
  });

  it("segments windows at 4-second duration (1000 samples at 250 Hz)", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    for (const w of result.windows) {
      expect(w.data[0].length).toBe(JOINT_WINDOW_SEC * JOINT_2312_SAMPLE_RATE_HZ);
    }
  });

  it("windows have correct sampleRate property", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    for (const w of result.windows) {
      expect(w.sampleRate).toBe(JOINT_2312_SAMPLE_RATE_HZ);
    }
  });

  it("CBraMod windows have 19 channels", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    for (const w of result.report.cbramodWindows) {
      expect(w.data.length).toBe(19);
    }
  });

  it("V2 windows have 22 channels", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    for (const w of result.report.v2Windows) {
      expect(w.data.length).toBe(22);
    }
  });

  it("EEGPT windows have 62 channels", () => {
    const edf = makeTestEDF(5000);
    const result = preprocessSleepEDF(edf);
    for (const w of result.report.eegptWindows) {
      expect(w.data.length).toBe(62);
    }
  });

  it("returns empty epochs array when no sleep annotations in EDF", () => {
    const edf = makeTestEDF(1000);
    const result = preprocessSleepEDF(edf);
    expect(result.epochs).toEqual([]);
  });

  it("preprocess report records all steps", () => {
    const edf = makeTestEDF(2000);
    const result = preprocessSleepEDF(edf);
    const stepNames = result.report.preprocessReport.steps.map((s) => s.name);
    expect(stepNames).toContain("bandpass");
    expect(stepNames).toContain("zscore");
    expect(stepNames).toContain("channel-expansion");
  });
});
