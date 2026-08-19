/**
 * T-016 — Canary observability metrics tests.
 *
 * Verifies that embedEEG() correctly records:
 *   - cohortChecksTotal{result="hit"} when the user is in the EEGConformer
 *     cohort (stage != off and isEEGConformerEnabledForUser returns true).
 *   - cohortChecksTotal{result="miss"} when the user is NOT in the cohort
 *     (stage == off, or user outside the hash threshold).
 *   - modelSelectedTotal{model="..."} after every embedEEG call, with the
 *     fell_back flag reflecting whether PCA was used as a fallback.
 *
 * Constraint note: these tests use setRolloutStage() to simulate canary /
 * ga states in-memory only — they never touch the AI_EEGCONFORMER_ENABLED
 * env var and always reset to "off" in afterEach. No rollout percentage
 * values are changed.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { embedEEG } from "../embed-eeg";
import { metrics, resetMetrics } from "../../../metrics";
import { setRolloutStage } from "../../rollout";
import {
  hasModel,
  unregisterModel,
  registerModel,
  registerBraindecodeEEGConformer,
} from "../../models/registry";
import { __resetManifestCache } from "../../artefacts/runtime-verifier";
import { ONNXAdapter } from "../../adapters/onnx-adapter";

const EEGCONFORMER_ID = "braindecode-eegconformer-prod-v2";

function makeWindowInput() {
  const C = 2;
  const T = 256;
  const sampleRate = 128;
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * (8 + c) * t) / sampleRate);
    data.push(ch);
  }
  return {
    kind: "windows" as const,
    windows: [{ data, sampleRate, start: 0, end: T }],
  };
}

describe("Canary observability metrics (T-016)", () => {
  beforeEach(() => {
    resetMetrics();
    __resetManifestCache();
    // Stub fetch so verification attempts don't make real network calls.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 404 })),
    );
  });

  afterEach(() => {
    setRolloutStage("off");
    vi.unstubAllGlobals();
    // Restore EEGConformer v2 registration so other test files see default state.
    if (!hasModel(EEGCONFORMER_ID)) {
      registerBraindecodeEEGConformer({
        id: "braindecode-eegconformer-prod-v2",
        artifact: "/models/eegconformer_finetuned.onnx",
        enableVerification: true,
      });
    }
  });

  it("records cohort miss when EEGConformer is off (default stage)", async () => {
    // Stage defaults to "off" → isEEGConformerEnabledForUser returns false.
    const result = await embedEEG(makeWindowInput());

    expect(metrics.cohortChecksTotal.value({ result: "miss" })).toBe(1);
    expect(metrics.cohortChecksTotal.value({ result: "hit" })).toBe(0);
    // Fell back to PCA since EEGConformer is not enabled.
    expect(result.modelId).toBe("pca-legacy-v1");
  });

  it("records cohort hit when EEGConformer is enabled (stage ga)", async () => {
    setRolloutStage("ga");
    // Ensure the production EEGConformer v2 model is NOT registered so we
    // don't accidentally load onnxruntime-web or make real network calls.
    // With stage ga, enabled=true, but hasModel=false → startId=chain[0]=PCA.
    const wasRegistered = hasModel(EEGCONFORMER_ID);
    if (wasRegistered) unregisterModel(EEGCONFORMER_ID);

    const result = await embedEEG(makeWindowInput(), { userId: "test-user-123" });

    expect(metrics.cohortChecksTotal.value({ result: "hit" })).toBe(1);
    expect(metrics.cohortChecksTotal.value({ result: "miss" })).toBe(0);
    expect(result.modelId).toBe("pca-legacy-v1");
  });

  it("records modelSelectedTotal after embedEEG completes", async () => {
    const result = await embedEEG(makeWindowInput());

    // The model that produced the embedding should be recorded with fell_back flag.
    const expected = result.modelId;
    const fellBack = String(result.fellBack);
    expect(metrics.modelSelectedTotal.value({ model: expected, fell_back: fellBack })).toBe(1);
  });

  it("records modelSelectedTotal with fell_back=true when preferred model fails", async () => {
    // Register a broken ONNX model that will fail to load → embedEEG falls back to PCA.
    registerModel(
      () =>
        new ONNXAdapter({
          id: "broken-for-metrics-test",
          name: "Broken",
          version: "0",
          description: "",
          artifact: "/models/broken.onnx",
          task: "embedding",
          inputShape: { kind: "raw", channels: 2, samples: 256 },
          runtime: async () => {
            throw new Error("runtime unavailable");
          },
        }),
    );

    try {
      // Use embedEEG with preferredModelId set to the broken model so the
      // model-selected metric is recorded by the embedEEG facade.
      // isEEGConformer is false (not DEFAULT_PREFERRED), so no cohort check fires.
      const result = await embedEEG(makeWindowInput(), {
        preferredModelId: "broken-for-metrics-test",
        normalize: false,
      });

      expect(result.fellBack).toBe(true);
      expect(result.modelId).toBe("pca-legacy-v1");
      expect(metrics.modelSelectedTotal.value({ model: "pca-legacy-v1", fell_back: "true" })).toBe(
        1,
      );
    } finally {
      unregisterModel("broken-for-metrics-test");
    }
  });
});
