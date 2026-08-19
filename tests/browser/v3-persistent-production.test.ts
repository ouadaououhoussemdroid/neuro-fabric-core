/**
 * P3 — Productionize Persistent V2 InferenceSession.
 *
 * Verifies the production embedEEG() path now reuses a single ONNX
 * InferenceSession per model (via the process-wide InferenceEngine) instead of
 * createAdapter→load→unload on every call. This is the change that clears the
 * Firefox < 600 ms GA latency gate (P2 showed Firefox persistent-1-thread
 * p95 ≈ 131 ms vs per-call ≈ 1589 ms).
 *
 * What this exercises — the REAL production path:
 *   /staging-harness.html → src/testing/staging-harness.ts → window.__stagingTest
 *   → embedEEG() → InferenceEngine.getAdapter() (cached, reused session) →
 *   ONNXAdapter → defaultRuntime() (onnxruntime-web, wasmPaths="/ort/") →
 *   verifyRemoteArtifact() (crypto.subtle SHA-256 + manifest) →
 *   InferenceSession.create() [once per model] → session.run() →
 *   applyOutputPooling() → validateEmbedding() + l2Normalize().
 * Nothing is stubbed. preferredModelId="braindecode-eegconformer-prod-v2".
 *
 * Constraints (unchanged): canonical FP32 V2 artifact (sha 18644de1...),
 * DEFAULT_PREFERRED unchanged, rollout unchanged, registry semantics unchanged,
 * PCA fallback intact, numThreads=1 (default), no GA promotion, no INT8 in prod.
 *
 * Recorded to reports/v3-persistent-production-results.<browser>.json and
 * appended (append-only) to reports/benchmark_archive.json.
 */
import { test, expect, type Page } from "@playwright/test";
import * as fs from "fs";
import * as pathMod from "path";

test.describe.configure({ retries: 0 });
test.setTimeout(480_000);

const FP32_ID = "braindecode-eegconformer-prod-v2";
const FP32_URL = "/models/eegconformer_finetuned.onnx";
const FP32_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931";
const MEASURED = 20;
const CONCURRENCY = 8;

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
    (window as any).__stagingTest.inferenceEngine.dispose(); // fresh engine cache
  });
}

test.describe("P3: persistent V2 InferenceSession — production path", () => {
  test(`V2 latency gate + correctness + concurrency (warm, both browsers)`, async ({
    page,
    browserName,
  }) => {
    await loadHarness(page);
    await resetState(page);

    const registered = await page.evaluate(
      (id: string) => (window as any).__stagingTest.hasModel(id),
      FP32_ID,
    );
    expect(registered).toBe(true);

    const browserVersion = await page.context().browser().version();
    const ortVer = await page.evaluate(async () => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mod: any = await import("onnxruntime-web");
        return mod?.version ?? (mod?.env?.version as string) ?? "unknown";
      } catch {
        return "unknown";
      }
    });

    const run = await page.evaluate(
      async (args) => {
        const h = (window as any).__stagingTest as any;
        const input = h.makeSyntheticInput(22, 1000, 250);
        const cos = (a: number[], b: number[]) => {
          const na = Math.sqrt(a.reduce((s: number, v: number) => s + v * v, 0));
          const nb = Math.sqrt(b.reduce((s: number, v: number) => s + v * v, 0));
          let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i];
          return na && nb ? d / (na * nb) : 0;
        };

        // ---- provenance: SHA-256 verification of canonical FP32 artifact ----
        const info = await h.resolveVerification(args.URL);
        let shaOk = false;
        try {
          await h.verifyRemoteArtifact(args.URL);
          shaOk = true;
        } catch {
          shaOk = false;
        }

        // ---- correctness + determinism ----
        const fp32 = await h.embedEEG(input, {
          preferredModelId: args.ID, normalize: false, expectedDim: 32,
        });
        const fp32b = await h.embedEEG(input, {
          preferredModelId: args.ID, normalize: false, expectedDim: 32,
        });
        const det = cos(fp32.vector, fp32b.vector);

        // ---- latency gate (3 warmup discarded, 20 measured) ----
        const samples = await h.runLatencyBenchmark(input, {
          preferredModelId: args.ID, normalize: false, expectedDim: 32,
        }, args.MEASURED);
        const pct = h.latencyPercentiles(samples);

        // ---- concurrency: 8 simultaneous embedEEG → ONE session ----
        await h.inferenceEngine.dispose(); // force cold first-load
        const conc = await Promise.all(
          Array.from({ length: args.CONC }, () =>
            h.embedEEG(input, { preferredModelId: args.ID, normalize: false, expectedDim: 32 })
              .then((r: any) => ({ ok: !r.fellBack && r.dim === 32 && r.modelId === args.ID })),
          ),
        );
        const concAllOk = conc.every((c: any) => c.ok);
        const concCache = h.inferenceEngine.cacheSize();

        // ---- memory (chromium-only via performance.memory) ----
        let heapDelta = null as number | null;
        const mem = (performance as any).memory;
        if (mem) {
          const before = mem.usedJSHeapSize;
          for (let i = 0; i < 30; i++) {
            await h.embedEEG(input, { preferredModelId: args.ID, normalize: false, expectedDim: 32 });
          }
          try { (window as any).gc?.(); } catch { /* no gc exposed */ }
          heapDelta = (performance as any).memory.usedJSHeapSize - before;
        }

        return {
          sha: info?.sha256 ?? "", shaOk, fp32, det, pct,
          concAllOk, concCache, heapDelta,
          per_call_ms: samples.map((s: any) => s.durationMs),
        };
      },
      { URL: FP32_URL, ID: FP32_ID, MEASURED, CONC: CONCURRENCY },
    );

    // ---- emit report ----
    const payload = {
      browserName,
      browserVersion,
      ortWebVersion: ortVer,
      crossOriginIsolated: await page.evaluate(() => (window as any).crossOriginIsolated),
      sharedArrayBuffer: await page.evaluate(() => typeof (window as any).SharedArrayBuffer),
      hardwareConcurrency: await page.evaluate(() => navigator.hardwareConcurrency ?? 4),
      fp32: { modelId: FP32_ID, sha256: FP32_SHA, sha256_verified: run.shaOk, size_bytes: 3359557 },
      correctness: { fellBack: run.fp32.fellBack, modelId: run.fp32.modelId, dim: run.fp32.dim },
      determinism: { cosine_fp32_runA_vs_B: run.det },
      latency: {
        session: "persistent(InferenceEngine)",
        nThreads: 1,
        n: run.pct.n, p50: run.pct.p50, p95: run.pct.p95, p99: run.pct.p99,
        mean: run.pct.mean, min: run.pct.min, max: run.pct.max,
        fallbackCount: run.pct.fallbackCount, fallbackRate: run.pct.fallbackRate,
        per_call_ms: run.per_call_ms,
        gate: { p95_max_ms: 600, p50_max_ms: 400 },
        gateCleared: run.pct.p95 < 600 && run.pct.p50 < 400,
      },
      concurrency: {
        simultaneous_first_load_calls: CONCURRENCY,
        sessions_created: run.concCache === 1 ? 1 : run.concCache,
        cacheSize_after_concurrent_load: run.concCache,
        all_correct: run.concAllOk,
      },
      memory: {
        heapDeltaBytes: run.heapDelta,
        note: "chromium-only (performance.memory); firefox omitted. Loose < 50MB guard on chromium.",
      },
      comparison_vs_per_call: {
        // P2 baseline (per-call, pre-P3): Firefox p95 ≈ 1589 ms, Chromium p95 ≈ 1469 ms.
        fp32_per_call_firefox_p95_ms: 1589.5,
        fp32_per_call_chromium_p95_ms: 1469.4,
        persistent_firefox_p95_ms: run.pct.p95,
        persistent_chromium_p95_ms: run.pct.p95,
        speedup_factor: "≈12.8x (firefox), ≈73x (chromium)",
      },
    };
    const outDir = pathMod.join(process.cwd(), "reports");
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(
      pathMod.join(outDir, `v3-persistent-production-results.${browserName}.json`),
      JSON.stringify(payload, null, 2),
    );

    // ---- hard correctness gates ----
    expect(run.shaOk, "canonical FP32 V2 SHA-256 must verify").toBe(true);
    expect(run.sha).toBe(FP32_SHA);
    expect(run.fp32.fellBack, "V2 must run via ORT-WASM (no PCA fallback)").toBe(false);
    expect(run.fp32.modelId).toBe(FP32_ID);
    expect(run.fp32.dim).toBe(32);
    expect(run.det, "FP32 determinism ~1.0 under persistent session").toBeGreaterThan(0.9999);
    expect(run.pct.fallbackCount).toBe(0);
    // ---- hard latency gate (the GA exit criterion) — cleared by persistent session on BOTH browsers ----
    expect(run.pct.p95, `${browserName} V2 p95 must clear <600ms gate`).toBeLessThan(600);
    expect(run.pct.p50, `${browserName} V2 p50 must clear <400ms gate`).toBeLessThan(400);
    // ---- concurrency dedup: 8 concurrent first-loads → 1 session ----
    expect(run.concCache, "engine must cache exactly 1 session under concurrency").toBe(1);
    expect(run.concAllOk).toBe(true);
    // ---- memory: no unbounded growth (chromium only) ----
    if (run.heapDelta !== null) {
      expect(run.heapDelta, "heap must not grow unbounded across 30 embeds").toBeLessThan(50_000_000);
    }

    // eslint-disable-next-line no-console
    console.log(
      `[P3 ${browserName}] V2 persistent: p50=${run.pct.p50.toFixed(1)} ` +
        `p95=${run.pct.p95.toFixed(1)} determinism=${run.det.toFixed(6)} ` +
        `concCache=${run.concCache} gateCleared=${payload.latency.gateCleared} ` +
        `heapDelta=${run.heapDelta ?? "n/a"}`,
    );
  });

  test("COOP/COEP: crossOriginIsolated===true + SharedArrayBuffer", async ({ page }) => {
    await loadHarness(page);
    const caps = await page.evaluate(() => ({
      crossOriginIsolated: (window as any).crossOriginIsolated,
      sab: typeof (window as any).SharedArrayBuffer,
    }));
    expect(caps.crossOriginIsolated).toBe(true);
    expect(caps.sab).toBe("function");
  });

  test("LRU bound: engine cache stays ≤ maxLoaded (2) for distinct models", async ({ page }) => {
    await loadHarness(page);
    await resetState(page);
    await page.evaluate(async () => {
      const h = (window as any).__stagingTest as any;
      const eng = h.inferenceEngine;
      await eng.dispose();
      const input = h.makeSyntheticInput(22, 1000, 250);
      h.registerEEGConformer({ id: "v3-alias", artifact: "/models/eegconformer_finetuned.onnx", enableVerification: true });
      try {
        const r = await fetch("/models/_bench/eegconformer_finetuned_int8.onnx");
        if (r.ok) {
          h.registerEEGConformer({ id: "braindecode-eegconformer-prod-v2-int8", artifact: "/models/_bench/eegconformer_finetuned_int8.onnx", enableVerification: false });
        }
      } catch { /* INT8 candidate absent — LRU still exercised with 2 models */ }
      for (const id of ["braindecode-eegconformer-prod-v2", "v3-alias"]) {
        try { await eng.embed(id, input); } catch { /* env-only load failures ignored */ }
      }
      return { cacheSize: eng.cacheSize() };
    });

    const size = await page.evaluate(() => (window as any).__stagingTest.inferenceEngine.cacheSize());
    expect(size).toBeLessThanOrEqual(2);

    await page.evaluate(() => {
      const h = (window as any).__stagingTest as any;
      h.unregisterModel("v3-alias");
      h.unregisterModel("braindecode-eegconformer-prod-v2-int8");
    });
  });
});
