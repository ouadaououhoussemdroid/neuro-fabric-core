import { describe, it, expect, afterEach } from "vitest";
import {
  registerBraindecodeEEGConformer,
  hasModel,
  unregisterModel,
  getDescriptor,
} from "../registry";
import { embedEEG } from "../../inference/embed-eeg";
import { setRolloutStage } from "../../rollout";
import type { ModelInput } from "../../types";

function makeInput(C: number, T: number, sr: number): ModelInput {
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / T);
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: T }] };
}

describe("EEGConformer production registration", () => {
  const id = "braindecode-eegconformer-prod";
  afterEach(() => {
    unregisterModel(id);
  });

  it("registers with the documented defaults (22ch, 4 s @ 250 Hz, 32-D)", () => {
    registerBraindecodeEEGConformer({ artifact: new Uint8Array([0]) });
    expect(hasModel(id)).toBe(true);
    const d = getDescriptor(id)!;
    expect(d.capabilities.channels).toBe(22);
    expect(d.capabilities.sampleRate).toBe(250);
    expect(d.capabilities.windowSamples).toBe(1000);
  });

  it("falls back to PCA when the underlying ONNX runtime is unavailable", async () => {
    setRolloutStage("ga");
    registerBraindecodeEEGConformer({
      artifact: new Uint8Array([0]),
      // Force unavailable runtime to exercise the production fallback chain.
      runtime: async () => {
        throw new Error("no runtime in test env");
      },
    });
    const res = await embedEEG(makeInput(22, 1000, 250), { preferredModelId: id });
    expect(res.fellBack).toBe(true);
    expect(res.modelId).toBe("pca-legacy-v1");
    expect(res.normalized).toBe(true);
  });
});
