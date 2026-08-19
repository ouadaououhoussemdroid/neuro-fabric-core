/**
 * T-036 — Tier-2 E2E smoke: drives the REAL CBraMod pipeline end-to-end.
 *
 * Unlike `foundation.server.test.ts` (which mocks the ort runtime), this loads
 * the actual 22 MB `cbramod-encoder.onnx` artifact, SHA-verifies it against the
 * manifest (c128ccfd…), creates a real onnxruntime-node InferenceSession, runs a
 * real [1,19,1000] forward, and checks the mean-tokens-pooled → 200-D → L2
 * result satisfies the strict dim gate (validateEmbedding) enforced by
 * `finalize`.
 *
 * Guarded: skipped when `onnxruntime-node` cannot be imported (e.g. CI on a
 * host without a prebuilt native binary). The optional dependency is declared in
 * package.json precisely so absence degrades to SKIP here rather than a hard
 * build/install failure.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
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
  embedFoundationWindows,
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
  resetFoundationAdapter,
} = await import("../foundation.server");

maybe("foundation.server (Tier-2 E2E: real 22MB artifact + onnxruntime-node)", () => {
  beforeEach(() => resetFoundationAdapter());
  afterEach(() => resetFoundationAdapter());

  it("embeds a real [19,1000] window into a 200-D L2 vector and passes the dim gate", async () => {
    // Synthetic but valid CBraMod-shaped input: 19 channels × 1000 samples @ 250 Hz.
    const data = Array.from(
      { length: 19 },
      () => Array.from({ length: 1000 }, () => Math.random() * 2 - 1) as number[],
    );
    const window: EEGWindow = { data, sampleRate: 250, start: 0, end: 1000 };

    const [r] = await embedFoundationWindows([window]);

    expect(r.modelId).toBe(FOUNDATION_MODEL_ID);
    // No PCA / V2 fallback: fellBack is strictly false on the Tier-2 path.
    expect(r.fellBack).toBe(false);
    // Strict dim gate (validateEmbedding{expectedDim:200}) — must be exactly 200.
    expect(r.dim).toBe(FOUNDATION_EMBEDDING_DIM);
    expect(r.vector).toHaveLength(200);
    // Forward latency observed (sanity: the native session actually ran).
    expect(r.durationMs).toBeGreaterThan(0);
    // L2 normalization enforced by finalize().
    const norm = Math.sqrt(r.vector.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1, 4);
    // Not a degenerate zero vector (the strict validateEmbedding rejects zeros too).
    expect(r.vector.some((v) => v !== 0)).toBe(true);
  }, 30_000); // generous: ~1-2s model load + forward on the 22 MB native graph
});
