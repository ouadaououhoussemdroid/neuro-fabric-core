import { describe, it, expect } from "vitest";
import {
  BraindecodeAdapter,
  isBraindecodeAvailable,
  setBraindecodeBridge,
  type BraindecodeBridge,
} from "../braindecode-adapter";
import { embed } from "../../embeddings";
import { registerModel, unregisterModel } from "../../models/registry";
import type { ModelInput } from "../../types";

function makeFakeBridge(opts: { available?: boolean; failForward?: boolean } = {}): BraindecodeBridge {
  return {
    async isAvailable() {
      return opts.available ?? true;
    },
    async load() {
      /* noop */
    },
    async forward(window) {
      if (opts.failForward) throw new Error("forward failed");
      // Deterministic embedding = per-channel mean
      const emb = window.map((ch) => ch.reduce((a, b) => a + b, 0) / ch.length);
      return { embedding: emb, logits: emb.map((v) => v * 2) };
    },
    async unload() {},
  };
}

function makeWindowInput(channels: number, samples: number): ModelInput {
  const data: number[][] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Array<number>(samples);
    for (let t = 0; t < samples; t++) ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / samples);
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: 128, start: 0, end: samples }] };
}

describe("BraindecodeAdapter", () => {
  it("loads and embeds via a fake bridge", async () => {
    const adapter = new BraindecodeAdapter({
      id: "bd-eegnetv4-test",
      architecture: "EEGNetv4",
      channels: 4,
      sampleRate: 128,
      windowSamples: 256,
      bridge: () => makeFakeBridge(),
    });
    await adapter.load();
    expect(adapter.isLoaded()).toBe(true);
    const out = await adapter.embed(makeWindowInput(4, 256));
    expect(out.vector).toHaveLength(4);
    expect(out.modelId).toBe("bd-eegnetv4-test");
    await adapter.unload();
    expect(adapter.isLoaded()).toBe(false);
  });

  it("rejects wrong channel/sample counts before forwarding", async () => {
    const adapter = new BraindecodeAdapter({
      id: "bd-shape-test",
      architecture: "EEGNetv4",
      channels: 4,
      sampleRate: 128,
      windowSamples: 256,
      bridge: () => makeFakeBridge(),
    });
    await adapter.load();
    await expect(adapter.embed(makeWindowInput(3, 256))).rejects.toThrow(/expected 4 channels/);
    await expect(adapter.embed(makeWindowInput(4, 128))).rejects.toThrow(/expected 256 samples/);
  });

  it("throws on load() when no bridge is available", async () => {
    const adapter = new BraindecodeAdapter({
      id: "bd-no-bridge",
      architecture: "EEGNetv4",
      channels: 4,
      sampleRate: 128,
      windowSamples: 256,
      bridge: () => makeFakeBridge({ available: false }),
    });
    await expect(adapter.load()).rejects.toThrow(/runtime not available/);
  });

  it("isBraindecodeAvailable reflects the registered bridge", async () => {
    setBraindecodeBridge(null);
    expect(await isBraindecodeAvailable()).toBe(false);
    setBraindecodeBridge(() => makeFakeBridge({ available: true }));
    expect(await isBraindecodeAvailable()).toBe(true);
    setBraindecodeBridge(null);
  });

  it("cascades Braindecode → PCA when bridge forward fails", async () => {
    registerModel(
      () =>
        new BraindecodeAdapter({
          id: "bd-failing",
          architecture: "EEGNetv4",
          channels: 2,
          sampleRate: 128,
          windowSamples: 256,
          bridge: () => makeFakeBridge({ failForward: true }),
        }),
    );
    try {
      const out = await embed(makeWindowInput(2, 256), {
        modelId: "bd-failing",
        fallbackChain: ["pca-legacy-v1"],
      });
      expect(out.fellBack).toBe(true);
      expect(out.modelId).toBe("pca-legacy-v1");
      expect(out.fallbackReason).toMatch(/forward failed/);
    } finally {
      unregisterModel("bd-failing");
    }
  });
});