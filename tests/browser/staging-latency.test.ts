/**
 * Mission 5 — Staging validation harness for EEGConformer v2 GA promotion.
 *
 * Real browser WASM latency + fallback-rate measurement. Uses Playwright to
 * run in actual Chromium and Firefox (NOT Node.js CPU EP).
 *
 * Does NOT change AI_EEGCONFORMER_ENABLED env var (uses in-memory setRolloutStage).
 * Does NOT modify production fallback behavior or the production default model.
 */
import { test, expect, type Page } from "@playwright/test";

const MODELS = [
  { name: "EEGConformer v1", id: "braindecode-eegconformer-prod", ch: 22, sr: 250, samples: 1000, dim: 32 },
  { name: "EEGConformer v2", id: "braindecode-eegconformer-prod-v2", ch: 22, sr: 250, samples: 1000, dim: 32 },
] as const;

const ITERATIONS = 20;
const WARMUP = 3;

async function loadHarness(page: Page): Promise<void> {
  await page.goto("/staging-harness.html", { waitUntil: "networkidle" });
  await page.waitForFunction(() => (window as any).__stagingTest !== undefined, undefined, {
    timeout: 30_000,
  });
}

async function resetState(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__stagingTest.resetMetrics();
    (window as any).__stagingTest.__resetManifestCache();
    (window as any).__stagingTest.setRolloutStage("ga");
  });
}

async function runBenchmark(page: Page, modelId: string, ch: number, sr: number, samples: number, iterations: number): Promise<any> {
  return await page.evaluate(
    (opts: { modelId: string; channels: number; sr: number; samples: number; iterations: number }) =>
      (window as any).__stagingTest
        .runLatencyBenchmark(
          (window as any).__stagingTest.makeSyntheticInput(opts.channels, opts.samples, opts.sr),
          { preferredModelId: opts.modelId, normalize: false },
          opts.iterations,
        )
        .then((s: any[]) => (window as any).__stagingTest.latencyPercentiles(s)),
    { modelId, channels: ch, sr, samples, iterations },
  );
}

test.describe("Mission 5: Staging latency + fallback-rate validation (real browser WASM)", () => {
  test.describe("Group 1: P95/P50 latency gate (< 600ms / < 400ms) for each model", () => {
    for (const model of MODELS) {
      test(`[${model.name}] P95 latency < 600ms, P50 < 400ms in browser WASM (warm)`, async ({ page }) => {
        await loadHarness(page);
        await resetState(page);

        const registered = await page.evaluate(
          (id: string) => (window as any).__stagingTest.hasModel(id),
          model.id,
        );
        expect(registered).toBe(true);

        const stats = await runBenchmark(page, model.id, model.ch, model.sr, model.samples, ITERATIONS);

        // Log raw measurements for the report
        console.log(`[${model.name}] ${JSON.stringify(stats)}`);

        // Warm-start latency gates (excludes first WARMUP iterations):
        // GA exit criterion: P95 < 600ms
        expect(stats.p95).toBeLessThan(600);
        // GA exit criterion: P50 < 400ms
        expect(stats.p50).toBeLessThan(400);
        expect(stats.n).toBe(ITERATIONS);
        expect(stats.max).toBeGreaterThan(0);
      });
    }
  });

  test.describe("Group 2: Fallback-rate gate (< 0.5%)", () => {
    for (const model of MODELS) {
      test(`[${model.name}] fallback rate < 0.5% over ${ITERATIONS} inferences`, async ({ page }) => {
        await loadHarness(page);
        await resetState(page);

        const stats = await runBenchmark(page, model.id, model.ch, model.sr, model.samples, ITERATIONS);

        // Canary/Beta/GA exit criterion: < 0.5% fallback rate
        expect(stats.fallbackRate).toBeLessThan(0.005);
        expect(stats.fallbackCount).toBe(0);
      });
    }
  });

  test.describe("Group 3: Metrics snapshot for staging monitoring", () => {
    test("collectMetricsSnapshot() returns structured metrics with expected fields", async ({ page }) => {
      await loadHarness(page);
      await resetState(page);

      await page.evaluate(() =>
        (window as any).__stagingTest.embedEEG(
          (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
          { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
        ),
      );

      const snapshot = await page.evaluate(() =>
        (window as any).__stagingTest.collectMetricsSnapshot(),
      );

      expect(snapshot).toBeDefined();
      expect(snapshot.timestamp).toBeTruthy();
      expect(snapshot.metrics).toBeDefined();

      // Prometheus uses neuro_fabric_ prefixed names
      expect(snapshot.metrics.neuro_fabric_model_selected_total).toBeDefined();
    });

    test("metrics snapshot includes artifactVerificationTotal pass counter", async ({ page }) => {
      await loadHarness(page);
      await resetState(page);

      await page.evaluate(() =>
        (window as any).__stagingTest.embedEEG(
          (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
          { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
        ),
      );

      const snapshot = await page.evaluate(() =>
        (window as any).__stagingTest.collectMetricsSnapshot(),
      );

      const verifyMetrics = snapshot.metrics.neuro_fabric_artifact_verification_total;
      expect(verifyMetrics).toBeDefined();
      const hasPass = verifyMetrics.some((m: any) =>
        m.labels && typeof m.labels === "string" && m.labels.includes('result="pass"')
      );
      expect(hasPass).toBe(true);
    });
  });

  test.describe("Group 4: Determinism check", () => {
    test("same input produces same embedding (cosine = 1.0) across runs", async ({ page }) => {
      await loadHarness(page);
      await resetState(page);

      const results = await page.evaluate(() => {
        const input = (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250);
        return Promise.all([
          (window as any).__stagingTest.embedEEG(input, {
            preferredModelId: "braindecode-eegconformer-prod-v2",
            normalize: false,
          }),
          (window as any).__stagingTest.embedEEG(input, {
            preferredModelId: "braindecode-eegconformer-prod-v2",
            normalize: false,
          }),
        ]);
      });

      const v1 = results[0].vector;
      const v2 = results[1].vector;
      expect(v1.length).toBe(v2.length);

      let dot = 0;
      let n1 = 0;
      let n2 = 0;
      for (let i = 0; i < v1.length; i++) {
        dot += v1[i] * v2[i];
        n1 += v1[i] * v1[i];
        n2 += v2[i] * v2[i];
      }
      const cosine = dot / (Math.sqrt(n1) * Math.sqrt(n2));
      expect(cosine).toBeCloseTo(1.0, 5);
    });
  });

  test.describe("Group 5: Artifact verification in browser", () => {
    test("SHA-256 verification passes for v2 artifact in browser", async ({ page }) => {
      await loadHarness(page);
      await resetState(page);

      // Run embedEEG — the adapter.load() path calls verifyRemoteArtifact()
      // which fetches the ONNX from /models/ and verifies SHA-256 via Web Crypto.
      const result = await page.evaluate(() =>
        (window as any).__stagingTest.embedEEG(
          (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
          { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
        ),
      );

      // Real WASM inference succeeded — verification passed
      expect(result.fellBack).toBe(false);
      expect(result.modelId).toBe("braindecode-eegconformer-prod-v2");
      expect(result.vector.length).toBe(32);
    });
  });
});
