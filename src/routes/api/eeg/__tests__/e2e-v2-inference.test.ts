/**
 * T-031 Final Gate — End-to-end verification:
 *   RAW EEG (NPY fixture) → upload.ts preprocessing path → EEGConformer v2 ONNX → 32-D embedding
 *
 * Uses a real fixture file (22 channels × 2500 samples @ 250 Hz), feeds it through
 * the actual preprocess() function used by upload.ts with the T-031 segment config,
 * then runs the result through the EEGConformer ONNX adapter.
 *
 * Verifies:
 * 1. Preprocessing produces exactly 1000-sample windows
 * 2. The ONNX adapter receives [1, 22, 1000] tensor input
 * 3. The model produces a valid 32-D embedding
 * 4. Output is deterministic for the same input
 * 5. PCA fallback works when the v2 model is unavailable
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { preprocess } from "@/lib/eeg/preprocessing";
import { parseNPY } from "@/lib/eeg/parsers/npy";
import { ONNXAdapter, type OrtRuntime } from "@/lib/ai/adapters/onnx-adapter";
import { embed, type EmbedResult } from "@/lib/ai/embeddings";
import { embedEEG } from "@/lib/ai/inference/embed-eeg";
import { getDescriptor, registerModel, unregisterModel, hasModel } from "@/lib/ai/models/registry";
import type { EEGSignal } from "@/lib/eeg/types";
import type { ModelInput } from "@/lib/ai/types";

const FIXTURE_PATH = join(process.cwd(), "test-fixtures", "eeg-raw-sample.npy");
const MODELS_DIR = join(process.cwd(), "public", "models");
const V2_ONNX = join(MODELS_DIR, "eegconformer_finetuned.onnx");
const V2_DATA = join(MODELS_DIR, "eegconformer_finetuned.onnx.data");
const MANIFEST = JSON.parse(readFileSync(join(MODELS_DIR, "manifest.json"), "utf-8"));

/** Node-compatible runtime wrapper — uses onnxruntime-web CPU backend (no wasmPaths). */
async function nodeRuntime(): Promise<OrtRuntime> {
  const mod = await import("onnxruntime-web");
  return mod as unknown as OrtRuntime;
}

/** Register a temporary V2 adapter pointing at the deployed v2 artifact. */
function registerV2Adapter(tempId: string): ONNXAdapter {
  const d = getDescriptor("braindecode-eegconformer-prod-v2")!;
  const adapter = new ONNXAdapter({
    id: tempId,
    name: d.name,
    version: "2.0.0",
    description: d.description + " (v2 fine-tuned on PhysioNet EEGMMIDB, 20 subjects)",
    artifact: join(process.cwd(), "public", "models", "eegconformer_finetuned.onnx"),
    task: "embedding",
    inputShape: { kind: "raw", channels: 22, samples: 1000 },
    channels: 22,
    sampleRate: 250,
    windowSamples: 1000,
    embeddingDim: 32,
    runtime: nodeRuntime,
  });
  registerModel(() => adapter);
  return adapter;
}

/** Convert preprocess() windows to ModelInput — exact production path shape. */
function windowsToInput(windows: ReturnType<typeof preprocess>["windows"]): ModelInput {
  return {
    kind: "windows",
    windows: windows.map((w) => ({
      data: w.data,
      sampleRate: w.sampleRate,
      start: w.start,
      end: w.end,
    })),
  };
}

/** Load the fixture and return an EEGSignal, preprocessing with T-031 config. */
function loadAndPreprocess(): {
  signal: EEGSignal;
  pre: ReturnType<typeof preprocess>;
  input: ModelInput;
} {
  const buf = readFileSync(FIXTURE_PATH);
  // NPY parser expects ArrayBuffer — convert Buffer's underlying ArrayBuffer
  const signal = parseNPY(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), 250);
  const pre = preprocess(signal, {
    segment: { windowSec: 4, overlap: 0.5 },
    bandpass: { low: 4, high: 38 },
  });
  const input = windowsToInput(pre.windows);
  return { signal, pre, input };
}

describe("T-031 Final Gate: RAW EEG → preprocessing → EEGConformer v2 → 32-D embedding", () => {
  // ── Test 1: Real fixture file loads correctly ──────────────────────
  it("fixture file exists and parses to 22×2500 @ 250 Hz", () => {
    expect(existsSync(FIXTURE_PATH)).toBe(true);
    const buf = readFileSync(FIXTURE_PATH);
    // NPY parser expects ArrayBuffer — convert Buffer's underlying ArrayBuffer
    const signal = parseNPY(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), 250);
    expect(signal.channels).toHaveLength(22);
    expect(signal.sampleRate).toBe(250);
    expect(signal.data[0]).toHaveLength(2500);
    // Verify non-zero data
    const sum = signal.data[0].reduce((a, b) => a + Math.abs(b), 0);
    expect(sum).toBeGreaterThan(0);
  });

  // ── Test 2: Production preprocessing produces 1000-sample windows ──
  it("preprocess() with T-031 config produces exactly 1000-sample windows", () => {
    const { pre } = loadAndPreprocess();

    expect(pre.windows.length).toBeGreaterThan(0);
    for (const w of pre.windows) {
      expect(w.data[0].length).toBe(1000); // 250 Hz × 4 s = 1000 samples
      expect(w.data.length).toBe(22); // 22 channels
    }
  });

  // ── Test 3: V2 ONNX adapter receives [1, 22, 1000] tensor ─────────
  it("V2 ONNX adapter builds tensor with shape [1, 22, 1000]", async () => {
    expect(existsSync(V2_ONNX)).toBe(true);

    const { input } = loadAndPreprocess();

    const tempId = "test-v2-tensor-shape";
    const adapter = registerV2Adapter(tempId);
    try {
      await adapter.load();

      // Access private buildTensor to verify shape
      const buildResult = (
        adapter as unknown as { buildTensor: (input: ModelInput) => { dims: number[] } }
      ).buildTensor(input);
      expect(buildResult.dims).toEqual([1, 22, 1000]);
    } finally {
      await adapter.unload();
      if (hasModel(tempId)) unregisterModel(tempId);
    }
  }, 60000);

  // ── Test 4: V2 ONNX produces valid 32-D embedding ──────────────────
  it("V2 ONNX produces valid non-zero 32-D embedding via embed() facade", async () => {
    const { input } = loadAndPreprocess();
    const tempId = "test-v2-valid-embed";
    const adapter = registerV2Adapter(tempId);
    try {
      const result = await embed(input, {
        modelId: tempId,
        fallbackToPCA: false, // Must NOT fall back — should succeed with ONNX
        normalize: false,
      });

      expect(result.fellBack).toBe(false);
      expect(result.modelId).toBe(tempId);
      expect(result.vector.length).toBe(32);
      expect(result.dim).toBe(32);

      // Non-zero
      const sum = result.vector.reduce((a, b) => a + Math.abs(b), 0);
      expect(sum).toBeGreaterThan(0);

      // No NaN/Inf
      for (const v of result.vector) {
        expect(Number.isFinite(v)).toBe(true);
      }
    } finally {
      if (hasModel(tempId)) unregisterModel(tempId);
      await adapter.unload();
    }
  }, 60000);

  // ── Test 5: Deterministic output for same input ───────────────────
  it("V2 ONNX produces deterministic embeddings for identical input", async () => {
    const { input } = loadAndPreprocess();
    const tempId = "test-v2-deterministic";
    const adapter = registerV2Adapter(tempId);
    try {
      // Run twice
      const result1 = await embed(input, {
        modelId: tempId,
        fallbackToPCA: false,
        normalize: false,
      });
      const result2 = await embed(input, {
        modelId: tempId,
        fallbackToPCA: false,
        normalize: false,
      });

      expect(result1.fellBack).toBe(false);
      expect(result2.fellBack).toBe(false);
      expect(result1.vector.length).toBe(32);
      expect(result2.vector.length).toBe(32);

      // Should be identical (deterministic)
      for (let i = 0; i < 32; i++) {
        expect(result1.vector[i]).toBeCloseTo(result2.vector[i], 5);
      }
    } finally {
      if (hasModel(tempId)) unregisterModel(tempId);
      await adapter.unload();
    }
  }, 120000);

  // ── Test 6: PCA fallback when V2 is unavailable ───────────────────
  it("PCA fallback when V2 adapter.load() fails (missing artifact)", async () => {
    // Use a deterministic sine window input (same pattern as existing fallback tests)
    const sr = 250;
    const ws = 1000;
    const data: number[][] = [];
    for (let c = 0; c < 22; c++) {
      const ch = new Array<number>(ws);
      for (let t = 0; t < ws; t++) ch[t] = Math.sin((2 * Math.PI * (c + 1) * t) / ws);
      data.push(ch);
    }
    const input: ModelInput = {
      kind: "windows",
      windows: [{ data, sampleRate: sr, start: 0, end: ws }],
    };

    const tempId = "test-v2-broken-fallback";
    const d = getDescriptor("braindecode-eegconformer-prod")!;
    const brokenAdapter = new ONNXAdapter({
      id: tempId,
      name: d.name,
      version: d.version,
      description: d.description,
      artifact: "/nonexistent/v2-model.onnx", // This will fail at load()
      task: "embedding",
      inputShape: { kind: "raw", channels: 22, samples: 1000 },
      channels: 22,
      sampleRate: 250,
      windowSamples: 1000,
      embeddingDim: 32,
      runtime: nodeRuntime,
    });
    registerModel(() => brokenAdapter);

    try {
      const result = await embedEEG(input, { preferredModelId: tempId, normalize: true });

      expect(result.fellBack).toBe(true);
      expect(result.modelId).toBe("pca-legacy-v1");
      expect(result.vector.length).toBe(32);

      // PCA vector is valid and non-zero
      const sum = result.vector.reduce((a, b) => a + Math.abs(b), 0);
      expect(sum).toBeGreaterThan(0);

      // L2 normalized
      const norm = Math.sqrt(result.vector.reduce((a, b) => a + b * b, 0));
      expect(norm).toBeCloseTo(1.0, 2);
    } finally {
      if (hasModel(tempId)) unregisterModel(tempId);
      await brokenAdapter.unload();
    }
  }, 30000);

  // ── Test 7: V2 artifact SHA-256 matches manifest ──────────────────
  it("V2 artifact SHA-256 matches manifest.json entry", () => {
    const v2OnnxPath = join(MODELS_DIR, "eegconformer_finetuned.onnx");
    expect(existsSync(v2OnnxPath)).toBe(true);

    const onnxBuf = readFileSync(v2OnnxPath);
    const onnxHash = createHash("sha256").update(onnxBuf).digest("hex");

    const entry = MANIFEST.models["eegconformer_finetuned"];
    expect(entry, "eegconformer_finetuned entry missing from manifest").toBeDefined();
    expect(entry.sha256).toBe(onnxHash);
    expect(entry.wasmCompatible).toBe(true);
    expect(entry.size).toBe(onnxBuf.length);
    // V2 model is now self-contained (external data merged into ONNX)
    expect(entry.externalData).toBeUndefined();
    expect(entry.sha256ExternalData).toBeUndefined();
  });

  // ── Test 8: V2 ONNX ops are WASM-compatible ───────────────────────
  it("V2 ONNX uses only WASM-compatible ops (verified via nodeRuntime CPU)", async () => {
    const { input } = loadAndPreprocess();
    const tempId = "test-v2-wasm-compat";
    const adapter = registerV2Adapter(tempId);
    try {
      await adapter.load();

      // If we can load and run the model, the ops are compatible
      const out = await adapter.embed(input);
      expect(out.modelId).toBe(tempId);
      expect(out.vector.length).toBe(32);
      expect(out.dim).toBe(32);
      const sum = out.vector.reduce((a, b) => a + Math.abs(b), 0);
      expect(sum).toBeGreaterThan(0);
    } finally {
      if (hasModel(tempId)) unregisterModel(tempId);
      await adapter.unload();
    }
  }, 60000);
});
