import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { decodeCognitiveState, bandStats } from "../index";
import type { EEGSignal } from "../../eeg/types";

const FS = 250;
const N = 2000;

function sineSignal(freqs: number[], fs = FS, n = N): EEGSignal {
  return {
    channels: freqs.map((_, i) => `ch${i}`),
    data: freqs.map((f) =>
      Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / fs)),
    ),
    sampleRate: fs,
  };
}

// squash(x) = 1/(1+e^-ln(x)) = x/(x+1) for x>0, else 0. Verified algebraically
// against src/lib/decoder/index.ts's private squash() — used here as an
// independent closed-form re-implementation, not a copy of the source.
function squash(x: number): number {
  return Number.isFinite(x) && x > 0 ? x / (x + 1) : 0;
}

// Mock createONNXDecoder so tests don't require the ONNX model file.
// By default it rejects (simulating no model in test env) → fallback to heuristic.
const mockCreateONNXDecoder = vi.hoisted(() => vi.fn());
vi.mock("../trained-decoder", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../trained-decoder")>();
  return {
    ...actual,
    createONNXDecoder: mockCreateONNXDecoder,
  };
});

describe("decodeCognitiveState", () => {
  beforeEach(() => {
    // Default: no ONNX model available → fallback to heuristic
    mockCreateONNXDecoder.mockRejectedValue(new Error("no model in test env"));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("falls back to baseline-spectral-v1 when ONNX model is unavailable", async () => {
    const result = await decodeCognitiveState(sineSignal([10]));
    expect(result.decoder).toBe("baseline-spectral-v1");
    expect(result.trained).toBe(false);
  });

  it("uses trained-logistic-v0 when ONNX model is available", async () => {
    mockCreateONNXDecoder.mockResolvedValue(async () => [0.85, 0.42, 0.61]);
    const result = await decodeCognitiveState(sineSignal([10]));
    expect(result.decoder).toBe("trained-logistic-v0");
    expect(result.trained).toBe(true);
    expect(result.attention).toBe(0.85);
    expect(result.workload).toBe(0.42);
    expect(result.arousal).toBe(0.61);
    expect(result.confidence).toBeCloseTo(0.85, 2);
    expect(result.confidenceIntervals).toBeDefined();
  });

  it("falls back to baseline when the ONNX decoder throws at inference time", async () => {
    mockCreateONNXDecoder.mockResolvedValue(async () => {
      throw new Error("ONNX inference failed");
    });
    const result = await decodeCognitiveState(sineSignal([10]));
    expect(result.decoder).toBe("baseline-spectral-v1");
    expect(result.trained).toBe(false);
  });

  it("implements attention = squash(beta / (alpha + theta)) against independently-recomputed bandStats", async () => {
    const signal = sineSignal([10, 20, 6]); // mix of alpha, beta, theta
    const stats = bandStats(signal);
    const expectedAttention = squash(stats.beta / Math.max(1e-9, stats.alpha + stats.theta));
    const result = await decodeCognitiveState(signal);
    expect(result.attention).toBeCloseTo(expectedAttention, 10);
  });

  it("implements workload = squash(theta / alpha) against independently-recomputed bandStats", async () => {
    const signal = sineSignal([10, 6]); // alpha + theta
    const stats = bandStats(signal);
    const expectedWorkload = squash(stats.theta / Math.max(1e-9, stats.alpha));
    const result = await decodeCognitiveState(signal);
    expect(result.workload).toBeCloseTo(expectedWorkload, 10);
  });

  it("implements arousal = clamp(beta + gamma, 0, 1) against independently-recomputed bandStats", async () => {
    const signal = sineSignal([20, 35]); // beta + gamma
    const stats = bandStats(signal);
    const expectedArousal = Math.min(1, Math.max(0, stats.beta + stats.gamma));
    const result = await decodeCognitiveState(signal);
    expect(result.arousal).toBeCloseTo(expectedArousal, 10);
  });

  it("all three outputs stay within [0, 1] regardless of input", async () => {
    for (const freqs of [[2], [10], [20], [35], [2, 10, 20, 35]]) {
      const result = await decodeCognitiveState(sineSignal(freqs));
      expect(result.attention).toBeGreaterThanOrEqual(0);
      expect(result.attention).toBeLessThanOrEqual(1);
      expect(result.workload).toBeGreaterThanOrEqual(0);
      expect(result.workload).toBeLessThanOrEqual(1);
      expect(result.arousal).toBeGreaterThanOrEqual(0);
      expect(result.arousal).toBeLessThanOrEqual(1);
    }
  });

  it("returns squashed 0 outputs (not NaN/Infinity) for a signal too short to produce any bandStats", async () => {
    const signal = sineSignal([10], FS, 10); // shorter than the default window
    const result = await decodeCognitiveState(signal);
    expect(result.attention).toBe(0);
    expect(result.workload).toBe(0);
    expect(Number.isFinite(result.arousal)).toBe(true);
  });

  it("includes the underlying bandStats and a non-negative durationMs in the report", async () => {
    const result = await decodeCognitiveState(sineSignal([10]));
    expect(result.bandStats).toBeDefined();
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });
});
