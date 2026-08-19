/**
 * T-036 / M25 — Tier-2 E2E smoke: drives the REAL CBraMod + V2 pipeline end-to-end.
 *
 * Unlike `joint-fusion.test.ts` (which tests the pure fusion function with
 * synthetic vectors), this loads the actual ONNX artifacts:
 *   - public/models/cbramod-encoder.onnx  (SHA c128ccfd…, 22 MB)
 *   - public/models/eegconformer_finetuned.onnx (SHA 18644de1…, 3.4 MB)
 *
 * Verifies their SHAs against the manifest, creates real onnxruntime-node
 * InferenceSessions, runs real [1,19,1000] and [1,22,1000] forward passes,
 * fuses with PCA-32, and checks the 264-D result through the strict dim gate
 * (validateEmbedding) enforced by `finalize`.
 *
 * Guarded: skipped when `onnxruntime-node` cannot be imported (e.g. CI on a
 * host without a prebuilt native binary). The optional dependency is declared in
 * package.json precisely so absence degrades to SKIP here rather than a hard
 * build/install failure.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { EEGWindow } from "@/lib/eeg/types";

let ortAvailable = false;
try {
  await import("onnxruntime-node");
  ortAvailable = true;
} catch {
  ortAvailable = false;
}

const maybe = ortAvailable ? describe : describe.skip;

const {
  embedJointWindows,
  embedJoint2312Windows,
  JOINT_MODEL_ID,
  JOINT_EMBEDDING_DIM,
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
  resetJointAdapter,
} =
  await import("../joint.server");

// Reused synthetic window generators — deterministic sine-wave signals
// (mirrors the harness.ts makeSyntheticInput pattern but in JS). The `idx`
// parameter shifts the base frequency so that each window has distinct
// spectral content. This is essential for the PCA path: with identical
// windows standardization produces zero variance and PCA projections
// collapse to all-zero (a real EEG recording always yields varied windows
// from the 4 s / 50 % overlap segmenter).
function makeCBRaModWindow(idx = 0): EEGWindow {
  const channels = 19;
  const samples = 1000;
  const sr = 250;
  return {
    data: Array.from({ length: channels }, (_, c) =>
      Array.from(
        { length: samples },
        (_, t) => Math.sin((2 * Math.PI * (10 + c + idx * 0.5) * t) / sr) * 0.5,
      ),
    ),
    sampleRate: sr,
    start: 0,
    end: samples,
  };
}

function makeV2Window(idx = 0): EEGWindow {
  const channels = 22;
  const samples = 1000;
  const sr = 250;
  return {
    data: Array.from({ length: channels }, (_, c) =>
      Array.from(
        { length: samples },
        (_, t) => Math.sin((2 * Math.PI * (10 + c + idx * 0.5) * t) / sr) * 0.5,
      ),
    ),
    sampleRate: sr,
    start: 0,
    end: samples,
  };
}

function makeEEGPTWindow(idx = 0): EEGWindow {
  const channels = 62;
  const samples = 1000;
  const sr = 250;
  return {
    data: Array.from({ length: channels }, (_, c) =>
      Array.from(
        { length: samples },
        (_, t) => Math.sin((2 * Math.PI * (10 + c + idx * 0.5) * t) / sr) * 0.5,
      ),
    ),
    sampleRate: sr,
    start: 0,
    end: samples,
  };
}

// Need ≥2 windows with distinct spectral content for PCA to have non-zero
// variance after standardisation. A single identical window → zero variance
// → zero PCA projection. Real EEG recordings always yield varied windows
// from the 4 s / 50 % overlap segmenter, so this mirrors production usage.
const N_WINDOWS = 5;

maybe("joint.server (Tier-2 E2E: real CBraMod + V2 artifacts + onnxruntime-node)", () => {
  beforeEach(() => resetJointAdapter());
  afterEach(() => resetJointAdapter());

  it("embeds aligned 19-ch and 22-ch windows into a 264-D L2-normalised vector", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    const [r] = await embedJointWindows(windows19, windows22);

    expect(r.modelId).toBe(JOINT_MODEL_ID);
    // No PCA / V2 / CBraMod fallback: fellBack is strictly false on the Tier-2 path.
    expect(r.fellBack).toBe(false);
    // Strict dim gate (validateEmbedding({expectedDim:264})) — must be exactly 264.
    expect(r.dim).toBe(JOINT_EMBEDDING_DIM);
    expect(r.vector).toHaveLength(264);
    // L2 normalisation enforced by finalize().
    const norm = Math.sqrt(r.vector.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 4);
    // No NaN / Inf.
    for (const v of r.vector) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("produces non-degenerate embeddings (not all-zero)", () => {
    // Non-degenerate: proves real ONNX inference, not a zero/NaN stub.
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    return embedJointWindows(windows19, windows22).then(([r]) => {
      const sum = r.vector.reduce((s, v) => s + Math.abs(v), 0);
      expect(sum).toBeGreaterThan(0);
    });
  });

  it("is deterministic — cos(runA, runB) ≈ 1.0", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));

    const [r1] = await embedJointWindows(windows19, windows22);
    const [r2] = await embedJointWindows(windows19, windows22);

    const cos = r1.vector.reduce((s, v, i) => s + v * r2.vector[i], 0);
    expect(cos).toBeCloseTo(1, 4); // ≈ 0.9999999999999998 per M18
  });

  it("embeds multiple windows independently (one result per window)", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));

    const results = await embedJointWindows(windows19, windows22);

    expect(results).toHaveLength(N_WINDOWS);
    for (const r of results) {
      expect(r.dim).toBe(264);
      expect(r.vector).toHaveLength(264);
      expect(r.fellBack).toBe(false);
    }
  });

  it("throws on window count mismatch", async () => {
    await expect(
      embedJointWindows(
        [makeCBRaModWindow(0), makeCBRaModWindow(1)],
        [makeV2Window(0), makeV2Window(1), makeV2Window(2)],
      ),
    ).rejects.toThrow(/window count mismatch/);
  });

  it("throws on empty windows", async () => {
    await expect(embedJointWindows([], [])).rejects.toThrow(/no windows/);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// M28 — Joint-2312 4-block E2E (real CBraMod + V2 + EEGPT ONNX artifacts)
// ──────────────────────────────────────────────────────────────────────────

// M28 E2E calls three ONNX embedders (CBraMod + V2 + EEGPT-2048); the 2048-D
// model adds ~10s per full suite, so the default 5s timeout is insufficient.
maybe("joint.server (M28 Tier-2 E2E: real CBraMod + V2 + EEGPT + onnxruntime-node)", () => {
  beforeEach(() => resetJointAdapter());
  afterEach(() => resetJointAdapter());

  // 5s default is too tight for loading/running 3 ONNX models per test.
  vi.setConfig({ testTimeout: 60000 });

  const N_WINDOWS = 5;

  it("embeds aligned 19/22/62-channel windows into a 2312-D L2-normalised vector", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    const windows62 = Array.from({ length: N_WINDOWS }, (_, i) => makeEEGPTWindow(i));
    const [r] = await embedJoint2312Windows(windows19, windows22, windows62);

    expect(r.modelId).toBe(JOINT_2312_MODEL_ID);
    expect(r.fellBack).toBe(false);
    expect(r.dim).toBe(JOINT_2312_EMBEDDING_DIM);
    expect(r.vector).toHaveLength(2312);
    const norm = Math.sqrt(r.vector.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 4);
    for (const v of r.vector) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("produces non-degenerate embeddings (not all-zero)", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    const windows62 = Array.from({ length: N_WINDOWS }, (_, i) => makeEEGPTWindow(i));
    const [r] = await embedJoint2312Windows(windows19, windows22, windows62);
    const sum = r.vector.reduce((s, v) => s + Math.abs(v), 0);
    expect(sum).toBeGreaterThan(0);
  });

  it("is deterministic — cos(runA, runB) ≈ 1.0", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    const windows62 = Array.from({ length: N_WINDOWS }, (_, i) => makeEEGPTWindow(i));

    const [r1] = await embedJoint2312Windows(windows19, windows22, windows62);
    const [r2] = await embedJoint2312Windows(windows19, windows22, windows62);

    const cos = r1.vector.reduce((s, v, i) => s + v * r2.vector[i], 0);
    expect(cos).toBeCloseTo(1, 4);
  });

  it("embeds multiple windows independently (one result per window)", async () => {
    const windows19 = Array.from({ length: N_WINDOWS }, (_, i) => makeCBRaModWindow(i));
    const windows22 = Array.from({ length: N_WINDOWS }, (_, i) => makeV2Window(i));
    const windows62 = Array.from({ length: N_WINDOWS }, (_, i) => makeEEGPTWindow(i));

    const results = await embedJoint2312Windows(windows19, windows22, windows62);

    expect(results).toHaveLength(N_WINDOWS);
    for (const r of results) {
      expect(r.dim).toBe(2312);
      expect(r.vector).toHaveLength(2312);
      expect(r.fellBack).toBe(false);
    }
  });

  it("throws on 3-way window count mismatch", async () => {
    await expect(
      embedJoint2312Windows(
        [makeCBRaModWindow(0), makeCBRaModWindow(1)],
        [makeV2Window(0), makeV2Window(1)],
        [makeEEGPTWindow(0), makeEEGPTWindow(1), makeEEGPTWindow(2)],
      ),
    ).rejects.toThrow(/window count mismatch/);
  });

  it("throws on empty windows", async () => {
    await expect(embedJoint2312Windows([], [], [])).rejects.toThrow(/no windows/);
  });
});
