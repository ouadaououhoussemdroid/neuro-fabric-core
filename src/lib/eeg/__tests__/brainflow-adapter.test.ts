import { describe, it, expect } from "vitest";
import { createBrainFlowSource, type BrainFlowConfig } from "../brainflow-adapter";

const config: BrainFlowConfig = {
  boardId: "synthetic",
  sampleRate: 250,
  nChannels: 8,
  chunkSeconds: 0.1, // 25 samples per chunk for fast tests
};

describe("createBrainFlowSource (synthetic fallback)", () => {
  it("yields EEGSignal chunks with correct dimensions", async () => {
    const source = createBrainFlowSource(config);
    const iter = source.stream()[Symbol.asyncIterator]();
    const { value: chunk } = await iter.next();
    source.close?.();

    expect(chunk.channels).toHaveLength(8);
    expect(chunk.channels[0]).toBe("ch0");
    expect(chunk.data).toHaveLength(8);
    expect(chunk.data[0]).toHaveLength(25); // 250 * 0.1
    expect(chunk.sampleRate).toBe(250);
    expect(chunk.meta?.source).toBe("brainflow-synthetic");
  });

  it("produces deterministic sine-based data within a reasonable uV range", async () => {
    const source = createBrainFlowSource(config);
    const iter = source.stream()[Symbol.asyncIterator]();
    const { value: chunk } = await iter.next();
    source.close?.();

    for (const row of chunk.data) {
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(-50.1);
        expect(v).toBeLessThanOrEqual(50.1);
      }
    }
  });

  it("stops streaming after close() is called", async () => {
    const source = createBrainFlowSource({ ...config, chunkSeconds: 0.05 });
    const iter = source.stream()[Symbol.asyncIterator]();
    await iter.next();
    source.close?.();
    // Next pull should terminate the generator.
    const { done } = await iter.next();
    expect(done).toBe(true);
  });

  it("respects custom channel labels", async () => {
    const source = createBrainFlowSource({
      ...config,
      channels: ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
    });
    const iter = source.stream()[Symbol.asyncIterator]();
    const { value: chunk } = await iter.next();
    source.close?.();

    expect(chunk.channels).toEqual(["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]);
  });

  it("metadata matches config", () => {
    const source = createBrainFlowSource(config);
    expect(source.type).toBe("brainflow");
    expect(source.metadata.sampleRate).toBe(250);
    expect(source.metadata.channels).toHaveLength(8);
  });
});
