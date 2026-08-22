/**
 * T-025 — Tests for server-side GPU preprocessing module.
 *
 * Tests GPU capability detection, CPU fallback, band power computation,
 * and metrics integration. onnxruntime-node is fully mocked so tests run
 * deterministically in any CI environment without a GPU.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock onnxruntime-node BEFORE importing the module under test.
// The module dynamically imports "onnxruntime-node" — vitest's vi.mock
// intercepts this so no native addon is loaded.
vi.mock("onnxruntime-node", () => {
  const mockSession = {
    run: vi.fn().mockResolvedValue({}),
    inputNames: ["input"],
    outputNames: ["output"],
    release: vi.fn().mockResolvedValue(undefined),
  };
  const mockTensor = vi.fn();
  return {
    InferenceSession: {
      create: vi.fn().mockResolvedValue(mockSession),
    },
    Tensor: mockTensor,
    env: {
      availableProviders: ["cpu"], // CPU-only in test env
    },
  };
});

import {
  detectGPUCapabilities,
  isGPUPreprocessingAvailable,
  getGPUExecutionProviders,
  bandpassGPU,
  bandpowerGPU,
  preprocessGPU,
  checkGPUHealth,
  resetGPUCache,
} from "../gpu-preprocess.server";
import { bandpass as cpuBandpass } from "@/lib/eeg/preprocessing/filters";

/** Generate synthetic multi-channel EEG signal. */
function makeSyntheticEEG(channels: number, samples: number, fs: number): number[][] {
  const data: number[][] = [];
  for (let ch = 0; ch < channels; ch++) {
    const row = new Array<number>(samples);
    for (let i = 0; i < samples; i++) {
      const t = i / fs;
      row[i] =
        Math.sin(2 * Math.PI * 10 * t + ch * 0.5) * 50 +
        Math.sin(2 * Math.PI * 50 * t) * 5 +
        (Math.random() * 2 - 1);
    }
    data.push(row);
  }
  return data;
}

describe("gpu-preprocess-server", () => {
  beforeEach(() => {
    resetGPUCache();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  describe("GPU capability detection", () => {
    it("detects capabilities without crashing", async () => {
      const caps = await detectGPUCapabilities();
      expect(caps).toHaveProperty("ep");
      expect(caps).toHaveProperty("device");
      expect(caps).toHaveProperty("available");
      expect(["cuda", "dml", "cpu"]).toContain(caps.ep);
    });

    it("falls back to CPU in test environment (no GPU)", async () => {
      const caps = await detectGPUCapabilities();
      expect(caps.ep).toBe("cpu");
      expect(caps.available).toBe(false);
      expect(caps.device).toBe("none");
    });

    it("isGPUPreprocessingAvailable returns false in test env", async () => {
      const available = await isGPUPreprocessingAvailable();
      expect(available).toBe(false);
    });

    it("getGPUExecutionProviders returns CPU config in test env", async () => {
      const providers = await getGPUExecutionProviders();
      expect(providers).toHaveProperty("executionProviders");
      expect(providers.executionProviders).toEqual(["cpu"]);
    });
  });

  describe("GPU bandpass filtering (CPU fallback path)", () => {
    it("falls back to CPU when GPU unavailable", async () => {
      const data = makeSyntheticEEG(4, 500, 250);
      const result = await bandpassGPU(data, 250, 1, 40, { allowCPUFallback: true });

      expect(result).toBeDefined();
      expect(result.length).toBe(4);
      expect(result[0].length).toBe(500);
    });

    it("CPU fallback produces same result as CPU bandpass", async () => {
      const data = makeSyntheticEEG(4, 200, 250);

      // Force CPU fallback (which is what happens when no GPU is available)
      const gpuResult = await bandpassGPU(data, 250, 1, 40, {
        allowCPUFallback: true,
        executionProvider: "cpu",
      });
      const cpuResult = cpuBandpass(data, 250, 1, 40);

      // Results should be identical (both using CPU)
      for (let ch = 0; ch < 4; ch++) {
        for (let i = 0; i < Math.min(200, gpuResult[ch].length); i++) {
          expect(gpuResult[ch][i]).toBeCloseTo(cpuResult[ch][i], 4);
        }
      }
    });

    it("throws when allowCPUFallback is false and GPU unavailable", async () => {
      const data = makeSyntheticEEG(2, 100, 250);
      await expect(
        bandpassGPU(data, 250, 1, 40, { allowCPUFallback: false }),
      ).rejects.toThrow(/No GPU execution provider/);
    });
  });

  describe("bandpower computation (CPU fallback path)", () => {
    it("computes band powers for all channels", async () => {
      const data = makeSyntheticEEG(22, 1000, 250);
      const result = await bandpowerGPU(data, 250, { allowCPUFallback: true });

      expect(result).toHaveLength(22);
      for (const band of result) {
        expect(band).toHaveProperty("delta");
        expect(band).toHaveProperty("theta");
        expect(band).toHaveProperty("alpha");
        expect(band).toHaveProperty("beta");
        expect(band).toHaveProperty("gamma");
        // All powers should be non-negative
        expect(band.delta).toBeGreaterThanOrEqual(0);
        expect(band.theta).toBeGreaterThanOrEqual(0);
        expect(band.alpha).toBeGreaterThanOrEqual(0);
        expect(band.beta).toBeGreaterThanOrEqual(0);
        expect(band.gamma).toBeGreaterThanOrEqual(0);
      }
    });

    it("dominant frequency appears in correct band", async () => {
      // Signal with strong 10Hz component (alpha band: 8-13 Hz)
      const fs = 250;
      const samples = 2500; // 10 seconds
      const data = [Array.from({ length: samples }, (_, i) => Math.sin(2 * Math.PI * 10 * i / fs) * 75)];

      const result = await bandpowerGPU(data, fs, { allowCPUFallback: true });
      expect(result[0].alpha).toBeGreaterThan(result[0].delta);
      expect(result[0].alpha).toBeGreaterThan(result[0].beta);
      expect(result[0].alpha).toBeGreaterThan(result[0].theta);
    });

    it("handles single-channel input", async () => {
      const data = makeSyntheticEEG(1, 500, 250);
      const result = await bandpowerGPU(data, 250, { allowCPUFallback: true });

      expect(result).toHaveLength(1);
      expect(result[0]).toHaveProperty("delta");
    });

    it("handles 22-channel input (standard EEG montage)", async () => {
      const data = makeSyntheticEEG(22, 1000, 250);
      const result = await bandpowerGPU(data, 250, { allowCPUFallback: true });

      expect(result).toHaveLength(22);
      for (const band of result) {
        expect(band.delta + band.theta + band.alpha + band.beta + band.gamma).toBeGreaterThanOrEqual(0);
      }
    });

    it("produces consistent results across runs", async () => {
      const data = makeSyntheticEEG(2, 250, 250);
      const result1 = await bandpowerGPU(data, 250, { allowCPUFallback: true });
      const result2 = await bandpowerGPU(data, 250, { allowCPUFallback: true });

      expect(result1[0].delta).toBeCloseTo(result2[0].delta, 6);
      expect(result1[0].alpha).toBeCloseTo(result2[0].alpha, 6);
    });
  });

  describe("full preprocessing pipeline (CPU fallback)", () => {
    it("runs CPU fallback mode correctly", async () => {
      const signal = {
        channels: ["ch0", "ch1", "ch2", "ch3"],
        data: makeSyntheticEEG(4, 1000, 250),
        sampleRate: 250,
      };

      const result = await preprocessGPU(signal, {
        bandpassHz: [1, 40],
        artifactRejection: true,
        gpuFFT: false,
      });

      expect(result.signal.data).toBeDefined();
      expect(result.signal.data.length).toBe(4);
      expect(result.durationMs).toBeGreaterThan(0);
      expect(result.mode).toBe("cpu"); // No GPU in test env
    });

    it("computes artifact scores", async () => {
      const signal = {
        channels: ["ch0"],
        data: makeSyntheticEEG(1, 500, 250),
        sampleRate: 250,
      };

      const result = await preprocessGPU(signal, {
        bandpassHz: [1, 40],
        artifactRejection: true,
      });

      expect(result.artifactScores).toBeDefined();
      expect(result.artifactScores!.length).toBe(1);
      for (const score of result.artifactScores!) {
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      }
    });

    it("returns usedGPU=false in CPU-only environment", async () => {
      const signal = {
        channels: ["ch0"],
        data: makeSyntheticEEG(1, 250, 250),
        sampleRate: 250,
      };

      const result = await preprocessGPU(signal, {
        bandpassHz: [1, 40],
      });

      expect(result.usedGPU).toBe(false);
      expect(result.mode).toBe("cpu");
    });

    it("does not compute bandPowers when gpuFFT is false", async () => {
      const signal = {
        channels: ["ch0"],
        data: makeSyntheticEEG(1, 250, 250),
        sampleRate: 250,
      };

      const result = await preprocessGPU(signal, {
        bandpassHz: [1, 40],
        gpuFFT: false,
      });

      expect(result.bandPowers).toBeUndefined();
    });
  });

  describe("GPU health check", () => {
    it("returns health status without crashing", async () => {
      const health = await checkGPUHealth();
      expect(health).toHaveProperty("healthy");
      expect(health).toHaveProperty("provider");
      expect(health).toHaveProperty("latencyMs");
      expect(typeof health.healthy).toBe("boolean");
      expect(typeof health.provider).toBe("string");
      expect(typeof health.latencyMs).toBe("number");
    });

    it("returns unhealthy when GPU unavailable", async () => {
      const health = await checkGPUHealth();
      expect(health.healthy).toBe(false);
      expect(health.provider).toBe("cpu");
    });
  });
});
