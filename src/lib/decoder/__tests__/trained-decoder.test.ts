import { describe, it, expect } from "vitest";
import { decodeWithTrainedModel } from "../trained-decoder";
import type { EEGSignal } from "../../eeg/types";

function makeSignal(channels: number, samples: number, sampleRate = 250): EEGSignal {
  const data: number[][] = [];
  for (let c = 0; c < channels; c++) {
    const row = new Array<number>(samples);
    for (let i = 0; i < samples; i++) {
      // Mix of 10 Hz (alpha) and 20 Hz (beta) tones.
      row[i] =
        Math.sin((2 * Math.PI * 10 * i) / sampleRate) * 50 +
        Math.sin((2 * Math.PI * 20 * i) / sampleRate) * 30;
    }
    data.push(row);
  }
  return { channels: data.map((_, i) => `ch${i}`), data, sampleRate };
}

describe("decodeWithTrainedModel (heuristic fallback)", () => {
  it("returns baseline decoder when no ONNX decoder is provided", async () => {
    const signal = makeSignal(4, 1000);
    const report = await decodeWithTrainedModel(signal);
    expect(report.decoder).toBe("baseline-spectral-v1");
    expect(report.trained).toBe(false);
    expect(report.attention).toBeGreaterThanOrEqual(0);
    expect(report.attention).toBeLessThanOrEqual(1);
    expect(report.workload).toBeGreaterThanOrEqual(0);
    expect(report.arousal).toBeGreaterThanOrEqual(0);
  });

  it("returns confidence intervals for each metric", async () => {
    const signal = makeSignal(4, 1000);
    const report = await decodeWithTrainedModel(signal);
    expect(report.confidence.attention).toHaveLength(2);
    expect(report.confidence.workload).toHaveLength(2);
    expect(report.confidence.arousal).toHaveLength(2);
    // Lower ≤ upper
    expect(report.confidence.attention[0]).toBeLessThanOrEqual(report.confidence.attention[1]);
  });
});

describe("decodeWithTrainedModel (ONNX-backed)", () => {
  it("uses the ONNX decoder when provided and returns trained-logistic-v0", async () => {
    const signal = makeSignal(4, 1000);
    const mockDecoder = async (features: number[]): Promise<[number, number, number]> => {
      expect(features).toHaveLength(5);
      return [0.85, 0.42, 0.61];
    };
    const report = await decodeWithTrainedModel(signal, mockDecoder);
    expect(report.decoder).toBe("trained-logistic-v0");
    expect(report.trained).toBe(true);
    expect(report.attention).toBe(0.85);
    expect(report.workload).toBe(0.42);
    expect(report.arousal).toBe(0.61);
  });

  it("falls back to heuristic when the ONNX decoder throws", async () => {
    const signal = makeSignal(4, 1000);
    const failingDecoder = async (): Promise<[number, number, number]> => {
      throw new Error("ONNX model not loaded");
    };
    const report = await decodeWithTrainedModel(signal, failingDecoder);
    expect(report.decoder).toBe("baseline-spectral-v1");
    expect(report.trained).toBe(false);
  });

  it("confidence intervals contain the predicted value", async () => {
    const signal = makeSignal(4, 1000);
    const mockDecoder = async (): Promise<[number, number, number]> => [0.7, 0.3, 0.5];
    const report = await decodeWithTrainedModel(signal, mockDecoder);
    expect(report.confidence.attention[0]).toBeLessThanOrEqual(0.7);
    expect(report.confidence.attention[1]).toBeGreaterThanOrEqual(0.7);
  });
});
