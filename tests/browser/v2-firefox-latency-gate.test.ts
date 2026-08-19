/**
 * Mission P1 — V2 Firefox WASM latency gate (COOP/COEP regression guard).
 *
 * Context (T-035):
 *   Chromium V2 P95 442 ms (PASS) vs Firefox V2 P95 1576 ms (FAIL) for the same
 *   threaded `ort-wasm-simd-threaded.wasm` (13.5 MB) bundle. Root cause: the dev
 *   server emitted no COOP/COEP headers → `window.crossOriginIsolated` was false
 *   in Firefox → `SharedArrayBuffer` unavailable → ORT's threaded WASM build ran
 *   single-threaded (~3.5x slower). Chromium enabled SAB heuristically, hiding the
 *   regression.
 *
 * Fix verified here:
 *   COOP/COEP/CORP headers are now applied permanently (vite.config.ts dev via
 *   coopCoepHeadersPlugin; src/server.ts production). This makes
 *   `crossOriginIsolated === true` + `SharedArrayBuffer` available in Firefox,
 *   unlocking threaded WASM → Firefox V2 P95 drops to the latency gate.
 *
 * What this test exercises — the REAL production path:
 *   /staging-harness.html → src/testing/staging-harness.ts → window.__stagingTest
 *   → embedEEG() → InferenceEngine (cached, reused session) → ONNXAdapter →
 *   defaultRuntime() (onnxruntime-web, wasmPaths="/ort/") → verifyRemoteArtifact()
 *   (crypto.subtle digest SHA-256 + manifest) → InferenceSession.create() [run
 *   once per model, reused across calls] → session.run() → applyOutputPooling()
 *   → validateEmbedding() + l2Normalize(). Nothing is stubbed.
 *   preferredModelId="braindecode-eegconformer-prod-v2" bypasses the rollout
 *   cohort so we gate the model itself.
 *
 * Gate (GA exit criterion = T-035 / staging-latency.test.ts Group 1):
 *   - crossOriginIsolated === true   (P1 COOP/COEP fix — Firefox SAB unlocked)
 *   - typeof SharedArrayBuffer === "function"  (threaded ORT-WASM can run)
 *   - V2: 0 fallbacks, modelId===v2, dim===32, SHA-256 verified (fellBack===false)
 *   - P95 < 600 ms, P50 < 400 ms (3 warmup discarded, 20 measured)
 *
 * Post-P3: the < 600 ms Firefox gate is CLEARED by routing embedEEG() through the
 * persistent InferenceEngine (session reuse) — NOT by INT8 (which remains
 * experimental in /models/_bench/). Default numThreads=1.
 *
 * Does NOT change DEFAULT_PREFERRED, rollout stage, PCA fallback, registry
 * semantics, the V2 FP32 artifact, or unrelated models.
 */
import { test, expect, type Page } from "@playwright/test";

const MODEL_ID = "braindecode-eegconformer-prod-v2";
const ITERATIONS = 20;

async function loadHarness(page: Page): Promise<void> {
  await page.goto("/staging-harness.html", { waitUntil: "networkidle" });
  await page.waitForFunction(() => (window as any).__stagingTest !== undefined, undefined, {
    timeout: 60_000,
  });
}

async function resetState(page: Page): Promise<void> {
  await page.evaluate(() => {
    (window as any).__stagingTest.resetMetrics();
    (window as any).__stagingTest.__resetManifestCache();
    (window as any).__stagingTest.setRolloutStage("ga"); // in-memory only
  });
}

test.describe("V2 Firefox/Chromium WASM latency gate (COOP/COEP)", () => {
  // ---------------------------------------------------------------------------
  // 1. The permanent fix: cross-origin isolation MUST be active.
  //    Without it, Firefox cannot use SharedArrayBuffer and the threaded ORT
  //    WASM build degrades to single-thread (T-035 regression). This is the
  //    primary regression guard — it fails loudly if COOP/COEP is ever removed.
  // ---------------------------------------------------------------------------
  test("COOP/COEP headers active: crossOriginIsolated===true + SharedArrayBuffer available", async ({
    page,
  }) => {
    await loadHarness(page);
    const caps = await page.evaluate(() => ({
      crossOriginIsolated: (window as any).crossOriginIsolated,
      sab: typeof (window as any).SharedArrayBuffer,
    }));
    expect(caps.crossOriginIsolated).toBe(true);
    expect(caps.sab).toBe("function");
  });

  // ---------------------------------------------------------------------------
  // 2. Correctness: real V2 inference (not PCA fallback), SHA-256 verified.
  // ---------------------------------------------------------------------------
  test(`V2 real-browser inference: no fallback, dim=32, SHA-256 verified`, async ({
    page,
  }) => {
    await loadHarness(page);
    await resetState(page);

    const result = await page.evaluate(() =>
      (window as any).__stagingTest.embedEEG(
        (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
        { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
      ),
    );

    // fellBack===false proves verifyRemoteArtifact() + InferenceSession.create()
    // succeeded in-browser — i.e. the SHA-256 gate passed and real WASM ran.
    expect(result.fellBack).toBe(false);
    expect(result.modelId).toBe(MODEL_ID);
    expect(result.dim).toBe(32);
    expect(result.vector.length).toBe(32);
    const sum = result.vector.reduce((a: number, b: number) => a + Math.abs(b), 0);
    expect(sum).toBeGreaterThan(0);
  });

  // ---------------------------------------------------------------------------
  // 3. Latency gate: P95 < 600 ms, P50 < 400 ms (warm; 3 warmup discarded).
  //    This is the GA exit criterion — the regression that was FAILING on Firefox
  //    before the COOP/COEP fix (T-035 Firefox P95 1576 ms).
  // ---------------------------------------------------------------------------
  test(`${MODEL_ID} real-browser latency: P95<600ms & P50<400ms (warm, ${ITERATIONS} measured)`, async ({
    page,
    browserName,
  }) => {
    await loadHarness(page);
    await resetState(page);

    const registered = await page.evaluate(
      (id: string) => (window as any).__stagingTest.hasModel(id),
      MODEL_ID,
    );
    expect(registered).toBe(true);

    const stats = await page.evaluate((opts: { modelId: string; iterations: number }) =>
      (window as any).__stagingTest
        .runLatencyBenchmark(
          (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
          { preferredModelId: opts.modelId, normalize: false },
          opts.iterations,
        )
        .then((samples: any[]) =>
          (window as any).__stagingTest.latencyPercentiles(samples),
        ),
      { modelId: MODEL_ID, iterations: ITERATIONS },
    );

    // eslint-disable-next-line no-console
    console.log(`V2 latency [${MODEL_ID}]`, JSON.stringify(stats));

    expect(stats.n).toBe(ITERATIONS);
    expect(stats.fallbackCount).toBe(0);
    expect(stats.fallbackRate).toBe(0);
    expect(stats.max).toBeGreaterThan(0);

    // Latency gate (GA exit criterion = T-035 / staging-latency.test.ts Group 1):
    // P95 < 600 ms, P50 < 400 ms, warmup excluded (3 cold-start iterations discarded).
    //
    // Post-P3, production embedEEG() routes V2 through the persistent
    // InferenceEngine (one InferenceSession per model, reused across requests)
    // instead of createAdapter→load→unload per call. Combined with the P1
    // COOP/COEP fix (SharedArrayBuffer available → threaded ORT-WASM), Firefox V2
    // P95 drops from ~1576 ms (per-call, single effective thread) to ~131 ms
    // (persistent session, numThreads=1) — clearing the < 600 ms gate.
    //
    // numThreads is NOT forced here: per-call thread-pool spin-up was measured
    // strictly NEGATIVE (P95 1742 ms → 2658 ms with numThreads=12 in P1); reuse is
    // what wins, and numThreads=1 (ORT default) is optimal for this 3.3 MB model.
    expect(stats.p95).toBeLessThan(600);
    expect(stats.p50).toBeLessThan(400);
  });
});
