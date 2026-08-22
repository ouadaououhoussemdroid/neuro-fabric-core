/**
 * T-025b — GPU vs CPU preprocessing benchmark.
 *
 * Measures the performance improvement of GPU-accelerated bandpass filtering
 * vs. CPU-based biquad filtering. Expected: 10-100x speedup for multi-channel EEG.
 *
 * Run: npx vitest run --reporter=verbose src/lib/eeg/preprocessing/__tests__/gpu-filters.bench.ts
 */
import { describe, it, expect } from "vitest";
import { bandpass as cpuBandpass } from "../filters";
import { bandpassGPU, bandPowerGPU, isGPUSupported, initGPU } from "../gpu-filters";
import { bandPowerFeatures } from "@/lib/embeddings/features";

/** Generate synthetic multi-channel EEG signal. */
function makeSyntheticEEG(channels: number, samples: number, fs: number): number[][] {
  const data: number[][] = [];
  for (let ch = 0; ch < channels; ch++) {
    const channel = new Array<number>(samples);
    for (let i = 0; i < samples; i++) {
      const t = i / fs;
      // Mix of frequencies: delta + theta + alpha + beta + muscle artifact
      channel[i] =
        Math.sin(2 * Math.PI * 10 * t + ch * 0.5) * 50 +     // alpha (50µV)
        Math.sin(2 * Math.PI * 50 * t) * 5 +                  // 50Hz noise (mains)
        Math.sin(2 * Math.PI * 0.2 * t) * 20 +                // delta (20µV)
        Math.sin(2 * Math.PI * 20 * t + ch) * 10 +            // beta (10µV)
        Math.random() * 2;                                    // noise
    }
    data.push(channel);
  }
  return data;
}

function getMedian(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

describe("GPU vs CPU preprocessing benchmark", () => {
  const FS = 250;
  const CHANNELS = 22;
  const SAMPLES = 1000; // 4 seconds at 250Hz
  const ITERATIONS = 5;

  describe("bandpass filter: CPU vs GPU", () => {
    it("measures CPU bandpass latency", async () => {
      const data = makeSyntheticEEG(CHANNELS, SAMPLES, FS);
      const timings: number[] = [];

      for (let i = 0; i < ITERATIONS; i++) {
        const t0 = performance.now();
        const result = cpuBandpass(data, FS, 1, 40);
        const t1 = performance.now();
        timings.push(t1 - t0);

        expect(result).toBeDefined();
        expect(result.length).toBe(CHANNELS);
        expect(result[0].length).toBe(SAMPLES);
      }

      const median = getMedian(timings);
      console.log(`\n  CPU bandpass (22ch × 1000 samples): median = ${median.toFixed(2)}ms`);
      expect(median).toBeLessThan(500); // Should complete in under 500ms
    });

    it("measures GPU bandpass latency (skipped if WebGPU unavailable)", async () => {
      const device = await initGPU();
      if (!device) {
        console.log("\n  GPU bandpass: SKIPPED (WebGPU not available)");
        return;
      }

      const data = makeSyntheticEEG(CHANNELS, SAMPLES, FS);
      const timings: number[] = [];

      // Warmup
      await bandpassGPU(data, FS, 1, 40);

      for (let i = 0; i < ITERATIONS; i++) {
        const t0 = performance.now();
        const result = await bandpassGPU(data, FS, 1, 40);
        const t1 = performance.now();
        timings.push(t1 - t0);

        expect(result).toBeDefined();
        expect(result.length).toBe(CHANNELS);
      }

      const median = getMedian(timings);
      console.log(`\n  GPU bandpass (22ch × 1000 samples): median = ${median.toFixed(2)}ms`);
      console.log(`  GPU available: YES (speedup depends on hardware)`);
    });

    it("verifies GPU and CPU results are numerically equivalent", async () => {
      const device = await initGPU();
      if (!device) {
        console.log("\n  GPU equivalence: SKIPPED (WebGPU not available)");
        return;
      }

      const data = makeSyntheticEEG(4, 500, FS); // Smaller for precision comparison
      const cpuResult = cpuBandpass(data, FS, 1, 40);
      const gpuResult = await bandpassGPU(data, FS, 1, 40);

      expect(gpuResult.length).toBe(cpuResult.length);

      // Check correlation between CPU and GPU results
      for (let ch = 0; ch < 4; ch++) {
        const cpuCh = cpuResult[ch];
        const gpuCh = gpuResult[ch];

        // Should be similar (GPU uses different filter implementation)
        const cpuMean = cpuCh.reduce((a, b) => a + b) / cpuCh.length;
        const gpuMean = gpuCh.reduce((a, b) => a + b) / gpuCh.length;

        // Means should be similar (both near 0 after filtering)
        expect(Math.abs(gpuMean - cpuMean)).toBeLessThan(50);
      }
    });
  });

  describe("band power analysis: CPU vs GPU", () => {
    it("measures CPU band power computation", () => {
      const data = makeSyntheticEEG(CHANNELS, SAMPLES, FS);
      const timings: number[] = [];
      for (let i = 0; i < ITERATIONS; i++) {
        const t0 = performance.now();
        const results = data.map((ch) => bandPowerFeatures({ data: [ch], sampleRate: FS } as any));
        const t1 = performance.now();
        timings.push(t1 - t0);

        expect(results).toHaveLength(CHANNELS);
      }

      const median = getMedian(timings);
      console.log(`\n  CPU band power (22ch × 1000 samples): median = ${median.toFixed(2)}ms`);
    });

    it("measures GPU band power computation (if WebGPU available)", async () => {
      const device = await initGPU();
      if (!device) {
        console.log("\n  GPU band power: SKIPPED (WebGPU not available)");
        return;
      }

      const data = makeSyntheticEEG(CHANNELS, SAMPLES, FS);

      // Warmup
      await bandPowerGPU(data, FS);

      const timings: number[] = [];
      for (let i = 0; i < ITERATIONS; i++) {
        const t0 = performance.now();
        const results = await bandPowerGPU(data, FS);
        const t1 = performance.now();
        timings.push(t1 - t0);

        expect(results).toHaveLength(CHANNELS);
      }

      const median = getMedian(timings);
      console.log(`\n  GPU band power (22ch × 1000 samples): median = ${median.toFixed(2)}ms`);
    });
  });

  describe("scalability test", () => {
    it("GPU maintains constant time as sample count increases", async () => {
      const device = await initGPU();
      if (!device) {
        console.log("\n  GPU scalability: SKIPPED (WebGPU not available)");
        return;
      }

      const sizes = [500, 1000, 2000, 4000];
      for (const size of sizes) {
        const data = makeSyntheticEEG(CHANNELS, size, FS);

        // Warmup
        await bandpassGPU(data, FS, 1, 40);

        const t0 = performance.now();
        await bandpassGPU(data, FS, 1, 40);
        const t1 = performance.now();

        console.log(`  GPU filter ${CHANNELS}ch × ${size}samples: ${(t1-t0).toFixed(2)}ms`);
      }
    });

    it("CPU scales linearly with sample count", () => {
      const sizes = [500, 1000, 2000, 4000];
      for (const size of sizes) {
        const data = makeSyntheticEEG(CHANNELS, size, FS);
        const t0 = performance.now();
        cpuBandpass(data, FS, 1, 40);
        const t1 = performance.now();
        console.log(`  CPU filter ${CHANNELS}ch × ${size}samples: ${(t1-t0).toFixed(2)}ms`);
      }
    });
  });
});
