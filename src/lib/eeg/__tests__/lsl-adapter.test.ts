import { describe, it, expect } from "vitest";
import { decodeFrame, createLSLSource } from "../lsl-adapter";
import type { EEGStreamFrame } from "../stream-gateway";

function frame(overrides: Partial<EEGStreamFrame> = {}): EEGStreamFrame {
  return {
    seq: 0,
    model_id: "eegconformer-v1",
    source: "lsl:open",
    channels: ["Cz", "Fz"],
    sampleRate: 250,
    data: [
      [1, 2],
      [3, 4],
    ],
    ts: "2026-07-11T00:00:00.000Z",
    ...overrides,
  };
}

describe("decodeFrame", () => {
  it("decodes a first frame with channels into an EEGSignal", () => {
    const f = frame();
    const sig = decodeFrame(f);
    expect(sig.channels).toEqual(["Cz", "Fz"]);
    expect(sig.data).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(sig.sampleRate).toBe(250);
    expect(sig.meta?.source).toBe("lsl:open");
    expect(sig.meta?.seq).toBe(0);
  });

  it("falls back to knownChannels when the frame omits them", () => {
    const f = frame({
      channels: [],
      seq: 1,
      data: [
        [5, 6],
        [7, 8],
      ],
    });
    const sig = decodeFrame(f, ["Cz", "Fz"]);
    expect(sig.channels).toEqual(["Cz", "Fz"]);
    expect(sig.data).toEqual([
      [5, 6],
      [7, 8],
    ]);
  });
});

describe("createLSLSource", () => {
  it("streams decoded chunks from an async iterable of JSON frames", async () => {
    const frames = [
      frame({ seq: 0 }),
      frame({
        seq: 1,
        channels: [],
        data: [
          [9, 10],
          [11, 12],
        ],
      }),
    ];
    async function* gen(): AsyncIterable<string> {
      for (const f of frames) yield JSON.stringify(f);
    }
    const source = createLSLSource(gen(), 250, ["Cz", "Fz"]);
    const out = [];
    for await (const chunk of source.stream()) out.push(chunk);

    expect(out).toHaveLength(2);
    expect(out[0].channels).toEqual(["Cz", "Fz"]);
    expect(out[0].data).toEqual([
      [1, 2],
      [3, 4],
    ]);
    // Second frame omitted channels; source threads them forward.
    expect(out[1].channels).toEqual(["Cz", "Fz"]);
    expect(out[1].data).toEqual([
      [9, 10],
      [11, 12],
    ]);
  });

  it("exposes metadata matching the source", () => {
    async function* empty(): AsyncIterable<string> {}
    const source = createLSLSource(empty(), 200, ["Pz"]);
    expect(source.type).toBe("lsl");
    expect(source.metadata.sampleRate).toBe(200);
    expect(source.metadata.channels).toEqual(["Pz"]);
  });
});
