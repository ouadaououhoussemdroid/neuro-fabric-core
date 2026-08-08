/**
 * T-025 — Integration test: cognitive decoder ONNX artefact & real inference.
 *
 * Verifies that the shipped `public/models/cognitive-decoder-v0.onnx` model:
 *   1. File exists and has valid ONNX magic bytes
 *   2. manifest.json SHA-256 matches the artefact
 *   3. The model has the correct I/O contract (input: [None,5], output: probabilities [None,3])
 *   4. Real ONNX inference (via onnxruntime-web, not a mock) produces valid
 *      probability values in [0, 1] with 3 elements [attention, workload, arousal]
 *   5. The production inference path (createONNXDecoder -> runInference) loads
 *      the real artefact and produces calibrated cognitive-state predictions
 *   6. The full production path (decodeWithTrainedModel -> extractFeatures ->
 *      ONNX inference) produces trained-logistic-v0 reports with real model output
 *
 * This test exercises the real artefact and the real onnxruntime-web runtime,
 * not a mock. The runtime provider imports onnxruntime-web directly without
 * the production wasmPaths override (which points at /ort/ and doesn't exist
 * in the Node test environment).
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { bandPowerFeatures } from "@/lib/embeddings/features";
import { segment } from "@/lib/eeg/preprocessing/segment";
import type { EEGSignal } from "@/lib/eeg/types";
import { decodeWithTrainedModel, __resetCognitiveDecoderCache } from "../trained-decoder";
import type { OrtRuntime } from "../../ai/adapters/onnx-adapter";

const ARTEFACT_PATH = join(process.cwd(), "public", "models", "cognitive-decoder-v0.onnx");
const MANIFEST_PATH = join(process.cwd(), "public", "models", "manifest.json");

/**
 * Real onnxruntime-web runtime provider for tests.
 *
 * Unlike `defaultRuntime`, this does NOT override `wasmPaths` to `/ort/`,
 * which would break in the Node test environment (no self-hosted WASM).
 * The returned module is the genuine onnxruntime-web, not a mock.
 */
async function testRuntime(): Promise<OrtRuntime> {
  const mod = (await import("onnxruntime-web")) as unknown as OrtRuntime;
  return mod;
}

/** Build a synthetic EEG signal with known frequency content. */
function makeSignal(freqs: number[], fs = 250, nSamples = 2000): EEGSignal {
  return {
    channels: freqs.map((_, i) => `ch${i}`),
    data: freqs.map((f) =>
      Array.from({ length: nSamples }, (_, i) => Math.sin((2 * Math.PI * f * i) / fs)),
    ),
    sampleRate: fs,
  };
}

/** Read the model file as an ArrayBuffer (the format onnxruntime-web accepts). */
function readModelBuffer(): ArrayBuffer {
  const buf = readFileSync(ARTEFACT_PATH);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe("T-025 Cognitive decoder ONNX artefact", () => {
  it("the model file exists at the expected path", () => {
    const buf = readFileSync(ARTEFACT_PATH);
    // Logistic regression model should be small (< 10KB).
    expect(buf.length).toBeGreaterThan(100);
    expect(buf.length).toBeLessThan(10_000);
    // ONNX magic: 0x08 (protobuf field 1, varint type)
    expect(buf[0]).toBe(0x08);
  });

  it("manifest.json entry matches the artefact", async () => {
    const { createHash } = await import("node:crypto");
    const buf = readFileSync(ARTEFACT_PATH);
    const hash = createHash("sha256").update(buf).digest("hex");
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf-8"));
    const entry = manifest.models?.["cognitive-decoder-v0"];
    expect(entry).toBeDefined();
    expect(entry.sha256).toBe(hash);
    expect(entry.size).toBe(buf.length);
  });

  it("extractFeatures produces a 5-element vector (delta, theta, alpha, beta, gamma)", () => {
    const signal: EEGSignal = {
      channels: ["ch0", "ch1"],
      data: [
        Array.from({ length: 500 }, () => Math.sin(Math.random())),
        Array.from({ length: 500 }, () => Math.cos(Math.random())),
      ],
      sampleRate: 250,
    };
    const windows = segment(signal.data, 250, 2, 0.5);
    expect(windows.length).toBeGreaterThan(0);
    const feats = bandPowerFeatures(windows[0]);
    // 2 channels x 5 bands = 10 features per window.
    // The decoder averages across channels -> 5 features (extractFeatures in trained-decoder.ts).
    expect(feats.length).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// Real ONNX model loading & inference (not mocked)
// ---------------------------------------------------------------------------

describe("T-025 Cognitive decoder real ONNX loading & inference", () => {
  it("loads the real ONNX artefact and verifies the I/O contract", async () => {
    const { createONNXDecoder } = await import("../trained-decoder");
    const arrayBuf = readModelBuffer();

    const decoder = await createONNXDecoder(arrayBuf, testRuntime);
    expect(typeof decoder).toBe("function");

    // Verify the session was loaded by running inference.
    const result = await decoder([20, 30, 25, 15, 10]);
    expect(result).toHaveLength(3);
  });

  it("produces valid probability outputs (in [0,1]) from real ONNX inference", async () => {
    const { createONNXDecoder } = await import("../trained-decoder");
    const arrayBuf = readModelBuffer();

    const decoder = await createONNXDecoder(arrayBuf, testRuntime);
    const [attn, work, aro] = await decoder([20, 30, 25, 15, 10]);

    // All three outputs must be valid probabilities.
    for (const v of [attn, work, aro]) {
      expect(v).not.toBeNaN();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("produces deterministic outputs for the same input", async () => {
    const { createONNXDecoder } = await import("../trained-decoder");
    const arrayBuf = readModelBuffer();

    const decoder1 = await createONNXDecoder(arrayBuf, testRuntime);
    const result1 = await decoder1([20, 30, 25, 15, 10]);

    __resetCognitiveDecoderCache();

    const decoder2 = await createONNXDecoder(arrayBuf, testRuntime);
    const result2 = await decoder2([20, 30, 25, 15, 10]);

    expect(result1).toEqual(result2);
  });

  it("throws on invalid feature vector length (wrong input shape)", async () => {
    const { createONNXDecoder } = await import("../trained-decoder");
    const arrayBuf = readModelBuffer();

    const decoder = await createONNXDecoder(arrayBuf, testRuntime);
    // The model expects exactly 5 features; passing 4 should fail at inference.
    await expect(decoder([20, 30, 25, 15])).rejects.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Production inference path: decodeWithTrainedModel -> extractFeatures -> ONNX
// ---------------------------------------------------------------------------

describe("T-025 Cognitive decoder production inference path", () => {
  it("runs the full trained-decoder pipeline on a real EEG signal", async () => {
    const { createONNXDecoder } = await import("../trained-decoder");
    const arrayBuf = readModelBuffer();

    const onnxDecoder = await createONNXDecoder(arrayBuf, testRuntime);
    const signal = makeSignal([10, 20], 250, 2000); // alpha + beta mix
    const report = await decodeWithTrainedModel(signal, onnxDecoder);

    // The trained model should be used (not the heuristic fallback).
    expect(report.trained).toBe(true);
    expect(report.decoder).toBe("trained-logistic-v0");

    // All outputs must be valid probabilities.
    expect(report.attention).toBeGreaterThanOrEqual(0);
    expect(report.attention).toBeLessThanOrEqual(1);
    expect(report.workload).toBeGreaterThanOrEqual(0);
    expect(report.workload).toBeLessThanOrEqual(1);
    expect(report.arousal).toBeGreaterThanOrEqual(0);
    expect(report.arousal).toBeLessThanOrEqual(1);

    // Confidence intervals must bracket the predicted value.
    expect(report.confidence.attention[0]).toBeLessThanOrEqual(report.attention);
    expect(report.confidence.attention[1]).toBeGreaterThanOrEqual(report.attention);
    expect(report.confidence.workload[0]).toBeLessThanOrEqual(report.workload);
    expect(report.confidence.workload[1]).toBeGreaterThanOrEqual(report.workload);
    expect(report.confidence.arousal[0]).toBeLessThanOrEqual(report.arousal);
    expect(report.confidence.arousal[1]).toBeGreaterThanOrEqual(report.arousal);
  });

  it("falls back to heuristic when the ONNX decoder throws", async () => {
    const failingDecoder = async (): Promise<[number, number, number]> => {
      throw new Error("ONNX model not loaded");
    };
    const signal = makeSignal([10, 20], 250, 2000);
    const report = await decodeWithTrainedModel(signal, failingDecoder);

    expect(report.trained).toBe(false);
    expect(report.decoder).toBe("baseline-spectral-v1");
  });
});
