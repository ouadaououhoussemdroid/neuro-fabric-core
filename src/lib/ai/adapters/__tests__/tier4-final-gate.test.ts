/**
 * T-016 Final Gate — VERIFICATION 3: Real EEGConformer factory chain.
 *
 * Proves the full production routing path end-to-end:
 *   embedEEG() → InferenceEngine.embed() → createAdapter("braindecode-eegconformer-prod-v2")
 *     → BraindecodeAdapter → createONNXBraindecodeBridge → ONNXAdapter
 *     → onnxruntime-web (Node CPU/WASM EP) → real EEGConformer ONNX
 *     → 32-dim embedding → pgvector-compatible vector(32)
 *
 * The gate is set to "ga" so isEEGConformerEnabledForUser returns true and
 * embedEEG routes to braindecode-eegconformer-prod-v2 (not PCA fallback).
 *
 * The adapter is registered with the real filesystem artifact path (not the
 * production URL /models/eegconformer_finetuned.onnx) and a clean nodeRuntime
 * that clears any wasmPaths pollution from prior defaultRuntime() calls.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { embedEEG } from "../../inference/embed-eeg";
import { registerBraindecodeEEGConformer, unregisterModel, hasModel } from "../../models/registry";
import { setRolloutStage } from "../../rollout";
import { type OrtRuntime } from "../onnx-adapter";
import type { ModelInput } from "../../types";
import { getExecutionProviders } from "../webgpu-flag";

const MODELS_DIR = join(process.cwd(), "public", "models");
const EEGCONFORMER_ARTIFACT = join(MODELS_DIR, "eegconformer_finetuned.onnx");
const MANIFEST = JSON.parse(readFileSync(join(MODELS_DIR, "manifest.json"), "utf-8"));

/** Node-compatible runtime: clears wasmPaths pollution so WASM init is clean. */
async function nodeRuntime(): Promise<OrtRuntime> {
  const mod = await import("onnxruntime-web");
  // Clear any wasmPaths set by a prior defaultRuntime() call (e.g. "/ort/").
  if (mod?.env?.wasm && mod.env.wasm.wasmPaths != null) {
    mod.env.wasm.wasmPaths = undefined;
  }
  return mod as unknown as OrtRuntime;
}

/** 22-ch × 1000 @ 250 Hz input matching the EEGConformer descriptor contract. */
function makeEEGConformerInput(): ModelInput {
  const channels = 22;
  const samples = 1000;
  const sr = 250;
  const data = Array.from({ length: channels }, (_, c) =>
    Array.from({ length: samples }, (_, t) => Math.sin((2 * Math.PI * (10 + c) * t) / sr) * 0.5),
  );
  return {
    kind: "windows",
    windows: [{ data, sampleRate: sr, start: 0, end: samples }],
  };
}

const EEGCONFORMER_ID = "braindecode-eegconformer-prod-v2";

describe("T-016 Final Gate: Real EEGConformer factory chain", () => {
  beforeEach(() => {
    // Enable the full EEGConformer path: GA → 100% of users get EEGConformer.
    setRolloutStage("ga");
  });

  afterEach(() => {
    // Restore to the safe default so other tests in this file aren't affected.
    setRolloutStage("off");
    // Remove our re-registration so the registry is back to its module-load state.
    // (Each test file runs in its own module registry, so this is just hygiene.)
    if (hasModel(EEGCONFORMER_ID)) {
      // Re-register with the production URL artifact (default) instead of the
      // filesystem path used in the test, so the descriptor matches production.
      registerBraindecodeEEGConformer({
        id: EEGCONFORMER_ID,
        artifact: "/models/eegconformer_finetuned.onnx",
      });
    }
  });

  it("VERIFICATION 3: real EEGConformer factory chain through embedEEG", async () => {
    // Register with the REAL filesystem artifact path + clean Node WASM runtime.
    // This exercises the full chain: registerBraindecodeEEGConformer →
    //   BraindecodeAdapter → createONNXBraindecodeBridge → ONNXAdapter
    //   → onnxruntime-web (WASM) → real inference.
    expect(existsSync(EEGCONFORMER_ARTIFACT)).toBe(true);

    const manifestEntry = MANIFEST.models["eegconformer_finetuned"];
    expect(manifestEntry).toBeDefined();
    // Verify SHA-256 integrity of the artifact we're about to load.
    const buf = readFileSync(EEGCONFORMER_ARTIFACT);
    const hash = createHash("sha256").update(buf).digest("hex");
    expect(manifestEntry.sha256).toBe(hash);

    // Register the production EEGConformer with real artifact + nodeRuntime.
    registerBraindecodeEEGConformer({
      id: EEGCONFORMER_ID,
      artifact: EEGCONFORMER_ARTIFACT,
      runtime: nodeRuntime,
      executionProviders: ["wasm"],
    });
    expect(hasModel(EEGCONFORMER_ID)).toBe(true);

    // Feed real EEG-shaped input (22-ch × 1000 @ 250 Hz).
    const input = makeEEGConformerInput();

    // embedEEG routes through the gate → EEGConformer (not PCA fallback).
    const res = await embedEEG(input, {
      preferredModelId: EEGCONFORMER_ID,
      normalize: false,
    });

    // No fallback — the real EEGConformer model was used end-to-end.
    expect(res.fellBack).toBe(false);
    expect(res.modelId).toBe(EEGCONFORMER_ID);

    // Exact 32-D contract: producer dim == DB vector(32) dim.
    expect(res.vector).toHaveLength(32);
    expect(res.dim).toBe(32);

    // Non-degenerate output: proves real inference, not a zero/NaN stub.
    const sum = res.vector.reduce((a, b) => a + Math.abs(b), 0);
    expect(sum).toBeGreaterThan(0);
    for (const v of res.vector) {
      expect(Number.isFinite(v)).toBe(true);
    }
  }, 60000);

  it("falls back to PCA (32-dim) when rollout stage is 'off' (gate-preserving)", async () => {
    // With stage "off", embedEEG must NOT use EEGConformer even though it's
    // registered — it routes to PCA fallback, preserving the gate semantics.
    setRolloutStage("off");

    registerBraindecodeEEGConformer({
      id: EEGCONFORMER_ID,
      artifact: EEGCONFORMER_ARTIFACT,
      runtime: nodeRuntime,
      executionProviders: ["wasm"],
    });

    const input = makeEEGConformerInput();
    const res = await embedEEG(input, {
      preferredModelId: EEGCONFORMER_ID,
      normalize: false,
    });

    // When the gate is "off", embedEEG routes to PCA as the primary model
    // (not as a fallback — PCA IS the default when EEGConformer is gated off).
    // The key assertion: EEGConformer was NOT used despite being registered.
    expect(res.modelId).toBe("pca-legacy-v1");
    expect(res.vector).toHaveLength(32);
    expect(res.dim).toBe(32);
  }, 30000);
});
