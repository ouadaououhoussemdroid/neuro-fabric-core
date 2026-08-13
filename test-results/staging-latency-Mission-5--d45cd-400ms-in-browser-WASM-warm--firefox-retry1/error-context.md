# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: staging-latency.test.ts >> Mission 5: Staging latency + fallback-rate validation (real browser WASM) >> Group 1: P95/P50 latency gate (< 600ms / < 400ms) for each model >> [EEGConformer v1] P95 latency < 600ms, P50 < 400ms in browser WASM (warm)
- Location: tests\browser\staging-latency.test.ts:52:7

# Error details

```
Error: expect(received).toBeLessThan(expected)

Expected: < 600
Received:   968
```

# Test source

```ts
  1   | /**
  2   |  * Mission 5 — Staging validation harness for EEGConformer v2 GA promotion.
  3   |  *
  4   |  * Real browser WASM latency + fallback-rate measurement. Uses Playwright to
  5   |  * run in actual Chromium and Firefox (NOT Node.js CPU EP).
  6   |  *
  7   |  * Does NOT change AI_EEGCONFORMER_ENABLED env var (uses in-memory setRolloutStage).
  8   |  * Does NOT modify production fallback behavior or the production default model.
  9   |  */
  10  | import { test, expect, type Page } from "@playwright/test";
  11  | 
  12  | const MODELS = [
  13  |   { name: "EEGConformer v1", id: "braindecode-eegconformer-prod", ch: 22, sr: 250, samples: 1000, dim: 32 },
  14  |   { name: "EEGConformer v2", id: "braindecode-eegconformer-prod-v2", ch: 22, sr: 250, samples: 1000, dim: 32 },
  15  | ] as const;
  16  | 
  17  | const ITERATIONS = 20;
  18  | const WARMUP = 3;
  19  | 
  20  | async function loadHarness(page: Page): Promise<void> {
  21  |   await page.goto("/staging-harness.html", { waitUntil: "networkidle" });
  22  |   await page.waitForFunction(() => (window as any).__stagingTest !== undefined, undefined, {
  23  |     timeout: 30_000,
  24  |   });
  25  | }
  26  | 
  27  | async function resetState(page: Page): Promise<void> {
  28  |   await page.evaluate(() => {
  29  |     (window as any).__stagingTest.resetMetrics();
  30  |     (window as any).__stagingTest.__resetManifestCache();
  31  |     (window as any).__stagingTest.setRolloutStage("ga");
  32  |   });
  33  | }
  34  | 
  35  | async function runBenchmark(page: Page, modelId: string, ch: number, sr: number, samples: number, iterations: number): Promise<any> {
  36  |   return await page.evaluate(
  37  |     (opts: { modelId: string; channels: number; sr: number; samples: number; iterations: number }) =>
  38  |       (window as any).__stagingTest
  39  |         .runLatencyBenchmark(
  40  |           (window as any).__stagingTest.makeSyntheticInput(opts.channels, opts.samples, opts.sr),
  41  |           { preferredModelId: opts.modelId, normalize: false },
  42  |           opts.iterations,
  43  |         )
  44  |         .then((s: any[]) => (window as any).__stagingTest.latencyPercentiles(s)),
  45  |     { modelId, channels: ch, sr, samples, iterations },
  46  |   );
  47  | }
  48  | 
  49  | test.describe("Mission 5: Staging latency + fallback-rate validation (real browser WASM)", () => {
  50  |   test.describe("Group 1: P95/P50 latency gate (< 600ms / < 400ms) for each model", () => {
  51  |     for (const model of MODELS) {
  52  |       test(`[${model.name}] P95 latency < 600ms, P50 < 400ms in browser WASM (warm)`, async ({ page }) => {
  53  |         await loadHarness(page);
  54  |         await resetState(page);
  55  | 
  56  |         const registered = await page.evaluate(
  57  |           (id: string) => (window as any).__stagingTest.hasModel(id),
  58  |           model.id,
  59  |         );
  60  |         expect(registered).toBe(true);
  61  | 
  62  |         const stats = await runBenchmark(page, model.id, model.ch, model.sr, model.samples, ITERATIONS);
  63  | 
  64  |         // Log raw measurements for the report
  65  |         console.log(`[${model.name}] ${JSON.stringify(stats)}`);
  66  | 
  67  |         // Warm-start latency gates (excludes first WARMUP iterations):
  68  |         // GA exit criterion: P95 < 600ms
> 69  |         expect(stats.p95).toBeLessThan(600);
      |                           ^ Error: expect(received).toBeLessThan(expected)
  70  |         // GA exit criterion: P50 < 400ms
  71  |         expect(stats.p50).toBeLessThan(400);
  72  |         expect(stats.n).toBe(ITERATIONS);
  73  |         expect(stats.max).toBeGreaterThan(0);
  74  |       });
  75  |     }
  76  |   });
  77  | 
  78  |   test.describe("Group 2: Fallback-rate gate (< 0.5%)", () => {
  79  |     for (const model of MODELS) {
  80  |       test(`[${model.name}] fallback rate < 0.5% over ${ITERATIONS} inferences`, async ({ page }) => {
  81  |         await loadHarness(page);
  82  |         await resetState(page);
  83  | 
  84  |         const stats = await runBenchmark(page, model.id, model.ch, model.sr, model.samples, ITERATIONS);
  85  | 
  86  |         // Canary/Beta/GA exit criterion: < 0.5% fallback rate
  87  |         expect(stats.fallbackRate).toBeLessThan(0.005);
  88  |         expect(stats.fallbackCount).toBe(0);
  89  |       });
  90  |     }
  91  |   });
  92  | 
  93  |   test.describe("Group 3: Metrics snapshot for staging monitoring", () => {
  94  |     test("collectMetricsSnapshot() returns structured metrics with expected fields", async ({ page }) => {
  95  |       await loadHarness(page);
  96  |       await resetState(page);
  97  | 
  98  |       await page.evaluate(() =>
  99  |         (window as any).__stagingTest.embedEEG(
  100 |           (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
  101 |           { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
  102 |         ),
  103 |       );
  104 | 
  105 |       const snapshot = await page.evaluate(() =>
  106 |         (window as any).__stagingTest.collectMetricsSnapshot(),
  107 |       );
  108 | 
  109 |       expect(snapshot).toBeDefined();
  110 |       expect(snapshot.timestamp).toBeTruthy();
  111 |       expect(snapshot.metrics).toBeDefined();
  112 | 
  113 |       // Prometheus uses neuro_fabric_ prefixed names
  114 |       expect(snapshot.metrics.neuro_fabric_model_selected_total).toBeDefined();
  115 |     });
  116 | 
  117 |     test("metrics snapshot includes artifactVerificationTotal pass counter", async ({ page }) => {
  118 |       await loadHarness(page);
  119 |       await resetState(page);
  120 | 
  121 |       await page.evaluate(() =>
  122 |         (window as any).__stagingTest.embedEEG(
  123 |           (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250),
  124 |           { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
  125 |         ),
  126 |       );
  127 | 
  128 |       const snapshot = await page.evaluate(() =>
  129 |         (window as any).__stagingTest.collectMetricsSnapshot(),
  130 |       );
  131 | 
  132 |       const verifyMetrics = snapshot.metrics.neuro_fabric_artifact_verification_total;
  133 |       expect(verifyMetrics).toBeDefined();
  134 |       const hasPass = verifyMetrics.some((m: any) =>
  135 |         m.labels && typeof m.labels === "string" && m.labels.includes('result="pass"')
  136 |       );
  137 |       expect(hasPass).toBe(true);
  138 |     });
  139 |   });
  140 | 
  141 |   test.describe("Group 4: Determinism check", () => {
  142 |     test("same input produces same embedding (cosine = 1.0) across runs", async ({ page }) => {
  143 |       await loadHarness(page);
  144 |       await resetState(page);
  145 | 
  146 |       const results = await page.evaluate(() => {
  147 |         const input = (window as any).__stagingTest.makeSyntheticInput(22, 1000, 250);
  148 |         return Promise.all([
  149 |           (window as any).__stagingTest.embedEEG(input, {
  150 |             preferredModelId: "braindecode-eegconformer-prod-v2",
  151 |             normalize: false,
  152 |           }),
  153 |           (window as any).__stagingTest.embedEEG(input, {
  154 |             preferredModelId: "braindecode-eegconformer-prod-v2",
  155 |             normalize: false,
  156 |           }),
  157 |         ]);
  158 |       });
  159 | 
  160 |       const v1 = results[0].vector;
  161 |       const v2 = results[1].vector;
  162 |       expect(v1.length).toBe(v2.length);
  163 | 
  164 |       let dot = 0;
  165 |       let n1 = 0;
  166 |       let n2 = 0;
  167 |       for (let i = 0; i < v1.length; i++) {
  168 |         dot += v1[i] * v2[i];
  169 |         n1 += v1[i] * v1[i];
```