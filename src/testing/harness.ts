/**
 * Browser WASM Smoke Test Harness — production code bridge.
 *
 * This file is the single source of truth for what the browser test harness
 * exposes to Playwright. It imports the REAL production code from the AI layer
 * — the exact same `embedEEG()` function that `src/routes/api/eeg/upload.ts`
 * calls in production — and bridges it to `window.__neuroTest` so Playwright
 * tests can exercise the full production inference path inside a real browser.
 *
 * The harness does NOT duplicate or simplify any production logic. It only
 * re-exports functions and types that already exist in the production codebase,
 * plus helpers for test orchestration (reset, input generation, diagnostics).
 *
 * Production code path exercised:
 *   embedEEG() → embed() → createAdapter() → BraindecodeAdapter →
 *   ONNXAdapter → defaultRuntime() (onnxruntime-web, wasmPaths="/ort/") →
 *   verifyRemoteArtifact() (crypto.subtle.digest, fetch, manifest) →
 *   InferenceSession.create() → session.run() → applyOutputPooling() →
 *   validateEmbedding() + l2Normalize()
 *
 * Loaded by: smoke-harness.html (served at /smoke-harness.html by Vite dev server)
 */
import { embedEEG, type EmbedEEGOptions } from "@/lib/ai/inference/embed-eeg";
import { setRolloutStage } from "@/lib/ai/rollout";
import { resetMetrics, metrics } from "@/lib/metrics";
import {
  __resetManifestCache,
  verifyRemoteArtifact,
  resolveVerification,
} from "@/lib/ai/artefacts/runtime-verifier";
import { hasModel, registerBraindecodeEEGConformer, unregisterModel } from "@/lib/ai/models/registry";
import { inferenceEngine } from "@/lib/ai/inference/engine";

/** Synthetic EEG input generator — deterministic mathematical signal. */
export function makeSyntheticInput(
  channels: number,
  samples: number,
  sampleRate: number,
): { kind: "windows"; windows: { data: number[][]; sampleRate: number; start: number; end: number }[] } {
  const data = Array.from({ length: channels }, (_, c) =>
    Array.from({ length: samples }, (_, t) => Math.sin((2 * Math.PI * (10 + c) * t) / sampleRate) * 0.5),
  );
  return {
    kind: "windows",
    windows: [{ data, sampleRate, start: 0, end: samples }],
  };
}

/**
 * Read a named counter from the metrics singleton.
 * Returns 0 if the label set has not been observed yet.
 */
export function metricValue(
  counter: { value: (labels?: Record<string, string>) => number },
  labels: Record<string, string> = {},
): number {
  return counter.value(labels);
}

/**
 * Read Performance API resource entries for ORT WASM files.
 * Proves the browser actually fetched wasm from /ort/ — not just that
 * InferenceSession.create() returned.
 */
export function wasmResourceEntries(): Array<{
  name: string;
  responseStatus: number;
  duration: number;
}> {
  return performance
    .getEntriesByType("resource")
    .filter((e: any) => e.name.includes("ort-wasm") && e.name.endsWith(".wasm"))
    .map((e: any) => ({
      name: e.name,
      responseStatus: e.responseStatus ?? (e.responseEnd > 0 ? 200 : 0),
      duration: e.responseEnd - e.startTime,
    }));
}

// ---------------------------------------------------------------------------
// Expose everything on window for Playwright page.evaluate() access.
//
// This is the ONLY bridge between the browser and the test runner. All
// functions called here are the REAL production implementations — no stubs,
// no duplicates, no simplified logic.
// ---------------------------------------------------------------------------
declare global {
  interface Window {
    __neuroTest: {
      /** Production embedEEG entry point (exact same function upload.ts uses). */
      embedEEG: typeof embedEEG;
      /** In-memory rollout stage selector (does NOT touch AI_EEGCONFORMER_ENABLED env). */
      setRolloutStage: typeof setRolloutStage;
      /** Reset all in-process metrics counters (test isolation). */
      resetMetrics: typeof resetMetrics;
      /** Reset the manifest cache (forces re-fetch on next verify). */
      __resetManifestCache: typeof __resetManifestCache;
      /** Direct verification call (for isolated SHA-256 tests). */
      verifyRemoteArtifact: typeof verifyRemoteArtifact;
      /** Resolve verification metadata from the manifest. */
      resolveVerification: typeof resolveVerification;
      /** Read a counter value from the metrics singleton. */
      metricValue: typeof metricValue;
      /** Read the full metrics singleton (for assertions). */
      metrics: typeof metrics;
      /** Check if a model is registered. */
      hasModel: typeof hasModel;
      /** Register EEGConformer with custom opts (test hook). */
      registerEEGConformer: typeof registerBraindecodeEEGConformer;
      /** Unregister a model (test cleanup). */
      unregisterModel: typeof unregisterModel;
      /** Generate synthetic EEG input matching a model's descriptor contract. */
      makeSyntheticInput: typeof makeSyntheticInput;
      /** Read Performance API entries for ORT WASM resource loads. */
      wasmResourceEntries: typeof wasmResourceEntries;
      /** Cached InferenceEngine (production singleton) for test teardown. */
      inferenceEngine: typeof inferenceEngine;
    };
  }
}

window.__neuroTest = {
  embedEEG,
  setRolloutStage,
  resetMetrics,
  __resetManifestCache,
  verifyRemoteArtifact,
  resolveVerification,
  metricValue,
  metrics,
  hasModel,
  registerEEGConformer: registerBraindecodeEEGConformer,
  unregisterModel,
  makeSyntheticInput,
  wasmResourceEntries,
  inferenceEngine,
};
