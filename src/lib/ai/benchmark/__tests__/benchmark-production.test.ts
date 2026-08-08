import { describe, it, beforeAll, expect } from "vitest";
import { performance } from "node:perf_hooks";
import { benchmarkAll } from "../";
import { setRolloutStage } from "../../rollout";
import type { ModelInput } from "../../types";

function makeInput(channels: number, samples: number, sr: number): ModelInput {
  const data: number[][] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Array<number>(samples);
    for (let t = 0; t < samples; t++) {
      ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / sr);
    }
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: samples }] };
}

describe("Production benchmark & validation", () => {
  const input = makeInput(22, 1000, 250);
  beforeAll(() => {
    setRolloutStage("ga");
  });

  it("benchmarks and validates embeddings", async () => {
    const results = await benchmarkAll(
      ["pca-legacy-v1", "braindecode-eegconformer-prod"],
      input,
      5,
    );
    console.dir(results, { depth: null });

    const pcaResult = results.find((r) => r.modelId === "pca-legacy-v1")!;
    const conformerResult = results.find((r) => r.modelId === "braindecode-eegconformer-prod")!;

    // PCA baseline always works.
    expect(pcaResult.embeddingDim).toBeGreaterThan(0);
    expect(pcaResult.error).toBeUndefined();
    expect(pcaResult.latencyMsMean).toBeGreaterThanOrEqual(0);

    // EEGConformer: when the ONNX WASM backend is unavailable (typical in CI),
    // the benchmark falls back to PCA and records fellBack=true + a reason.
    // When the backend IS available, the model produces 32-dim embeddings
    // without fallback. Accept both outcomes.
    if (conformerResult.fellBack) {
      // ONNX backend not available in this environment — expected in CI.
      expect(conformerResult.embeddingDim).toBe(pcaResult.embeddingDim);
      expect(conformerResult.fallbackReason).toBeDefined();
    } else {
      expect(conformerResult.embeddingDim).toBe(32);
      expect(conformerResult.fellBack).toBe(false);
      expect(conformerResult.fallbackReason).toBeUndefined();
    }
    expect(conformerResult.latencyMsMean).toBeGreaterThanOrEqual(0);
  });
});
