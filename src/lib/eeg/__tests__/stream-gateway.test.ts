import { describe, it, expect } from "vitest";
import {
  StreamGateway,
  parseSourceId,
  parseClientMessage,
  GatewayError,
  type EEGStreamFrame,
} from "../stream-gateway";
import type { AcquisitionSource } from "../acquisition";
import type { EEGSignal } from "../types";

/** Build a mock AcquisitionSource that emits a fixed list of chunks. */
function mockSource(
  chunks: EEGSignal[],
  type: AcquisitionSource["type"] = "file",
): AcquisitionSource {
  return {
    type,
    metadata: {
      sampleRate: chunks[0]?.sampleRate ?? 250,
      channels: chunks[0]?.channels ?? [],
    },
    async *stream() {
      for (const c of chunks) yield c;
    },
  };
}

function chunk(channels: string[], data: number[][], sampleRate = 250): EEGSignal {
  return { channels, data, sampleRate };
}

describe("StreamGateway", () => {
  it("pumps chunks as sequenced frames with model_id and source", async () => {
    const src = mockSource([chunk(["Cz"], [[1, 2, 3]]), chunk(["Cz"], [[4, 5, 6]])]);
    const gw = new StreamGateway({ defaultModelId: "test-model-v1" });
    gw.register("file:rec-1", src);

    const frames: EEGStreamFrame[] = [];
    await gw.pump("file:rec-1", "peer-A", (f) => frames.push(f));

    expect(frames).toHaveLength(2);
    expect(frames[0].seq).toBe(0);
    expect(frames[1].seq).toBe(1);
    expect(frames[0].model_id).toBe("test-model-v1");
    expect(frames[0].source).toBe("file:rec-1");
    expect(frames[0].channels).toEqual(["Cz"]); // first frame includes channels
    expect(frames[1].channels).toEqual([]); // subsequent frames omit channels
    expect(frames[0].data).toEqual([[1, 2, 3]]);
    expect(frames[1].data).toEqual([[4, 5, 6]]);
  });

  it("maintains independent sequence counters per peer", async () => {
    const src = mockSource([chunk(["Cz"], [[1]]), chunk(["Cz"], [[2]])]);
    const gw = new StreamGateway();
    gw.register("s1", src);

    await gw.pump("s1", "peer-A", () => {});
    await gw.pump("s1", "peer-B", () => {});

    expect(gw.peekSeq("peer-A")).toBe(2);
    expect(gw.peekSeq("peer-B")).toBe(2);
  });

  it("throws a 404 GatewayError for an unknown source", async () => {
    const gw = new StreamGateway();
    await expect(gw.pump("nope", "peer-A", () => {})).rejects.toThrow(/unknown source/);
    await expect(gw.pump("nope", "peer-A", () => {})).rejects.toBeInstanceOf(GatewayError);
  });

  it("stops pumping when shouldStop returns true", async () => {
    const src = mockSource([chunk(["Cz"], [[1]]), chunk(["Cz"], [[2]]), chunk(["Cz"], [[3]])]);
    const gw = new StreamGateway();
    gw.register("s1", src);

    const frames: EEGStreamFrame[] = [];
    await gw.pump(
      "s1",
      "peer-A",
      (f) => frames.push(f),
      () => frames.length >= 1,
    );
    expect(frames).toHaveLength(1);
  });

  it("resetPeer clears the peer's sequence counter", async () => {
    const src = mockSource([chunk(["Cz"], [[1]])]);
    const gw = new StreamGateway();
    gw.register("s1", src);
    await gw.pump("s1", "peer-A", () => {});
    expect(gw.peekSeq("peer-A")).toBe(1);
    gw.resetPeer("peer-A");
    expect(gw.peekSeq("peer-A")).toBe(0);
  });

  it("unregister removes a source and closes it", async () => {
    let closed = false;
    const src: AcquisitionSource = {
      type: "file",
      metadata: { sampleRate: 250, channels: ["Cz"] },
      async *stream() {
        yield chunk(["Cz"], [[1]]);
      },
      close: () => {
        closed = true;
      },
    };
    const gw = new StreamGateway();
    gw.register("s1", src);
    expect(gw.has("s1")).toBe(true);
    gw.unregister("s1");
    expect(gw.has("s1")).toBe(false);
    expect(closed).toBe(true);
  });
});

describe("parseSourceId", () => {
  it("returns the trimmed string", () => {
    expect(parseSourceId("file:rec-1")).toBe("file:rec-1");
  });
  it("throws for empty input", () => {
    expect(() => parseSourceId("")).toThrow(GatewayError);
    expect(() => parseSourceId(undefined)).toThrow(GatewayError);
  });
});

describe("parseClientMessage", () => {
  it("parses a subscribe message", () => {
    expect(parseClientMessage(JSON.stringify({ type: "subscribe", source: "s1" }))).toEqual({
      type: "subscribe",
      source: "s1",
    });
  });
  it("parses an unsubscribe message", () => {
    expect(parseClientMessage(JSON.stringify({ type: "unsubscribe" }))).toEqual({
      type: "unsubscribe",
      source: undefined,
    });
  });
  it("returns null for invalid JSON", () => {
    expect(parseClientMessage("not json")).toBeNull();
  });
  it("returns null for unknown message type", () => {
    expect(parseClientMessage(JSON.stringify({ type: "bogus" }))).toBeNull();
  });
});
