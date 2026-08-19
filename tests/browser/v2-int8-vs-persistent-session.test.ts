/**
 * P2 — V2 Firefox latency optimization: INT8-QDQ vs persistent-session ablation.
 *
 * Evidence-gathering ONLY. Does NOT change production behaviour:
 *  - The INT8 candidate is served from the non-prod path /models/_bench/ (FP32
 *    canonical artifact at /models/eegconformer_finetuned.onnx, sha 18644de1...,
 *    is never modified). The INT8 model is registered through the staging
 *    harness's registerEEGConformer (NOT the app registry) with
 *    enableVerification:false so no production manifest is needed.
 *  - Persistent sessions are exercised via the staging-exposed `inferenceEngine`
 *    (Test B instrumentation in src/testing/staging-harness.ts). Production
 *    embedEEG() is untouched BY P2. NOTE (P3): production embedEEG() was later
 *    routed through the persistent `inferenceEngine` in P3; consequently the
 *    per-call baseline below is measured via the `embed()` facade directly
 *    (window.__stagingTest.embedFacade), NOT via embedEEG(), so the fresh-session
 *    cost is still captured for comparison.
 *  - DEFAULT_PREFERRED, rollout stages, registry semantics, PCA fallback, and
 *    all other models (EEGPT/CBraMod/LaBraM/FEMBA) are unchanged.
 *
 * Recorded to reports/v2-int8-vs-persistent-results.<browser>.json (archive-style).
 */
import { test, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

// benchmark: no auto-retry; a gate miss should surface once, not double-run
test.describe.configure({ retries: 0 });
test.setTimeout(420_000);

const FP32_URL = "/models/eegconformer_finetuned.onnx";
const FP32_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931";
const INT8_URL = "/models/_bench/eegconformer_finetuned_int8.onnx";
const INT8_SHA = "59e9555a18536a716e7c1bdf9bba46bca5b0ad3b753529e9b871d272ae45e880";
const FP32_ID = "braindecode-eegconformer-prod-v2";
const INT8_ID = "braindecode-eegconformer-prod-v2-int8";

const WARMUP = 3;
const MEASURED = 12;

interface LatencyStats {
  n: number; p50: number; p95: number; p99: number; mean: number; min: number; max: number;
}
interface ConfigResult {
  name: string;
  session: "fresh-per-call" | "persistent(InferenceEngine)";
  model: string;
  model_sha: string;
  nThreads: number | "default";
  n: number;
  stats: LatencyStats;
  per_call_ms: number[];
}
interface BrowserResult {
  browserName: string;
  browserVersion: string;
  ortWebVersion: string;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: string;
  hardwareConcurrency: number;
  fp32: { sha256: string; sha256_verified: boolean; size_bytes: number };
  int8: { available: boolean; sha256: string; size_bytes: number };
  parity: {
    embedding_cosine_fp32_vs_int8: number | null;
    dim: number;
    fp32_fellBack: boolean;
    int8_fellBack: boolean;
    fp32_modelId: string;
    int8_modelId: string;
  };
  determinism: { cosine_fp32_runA_vs_B: number };
  configs: ConfigResult[];
}

test("P2: V2 latency ablation — INT8 vs persistent session (real app, both browsers)", async ({
  page, browserName, browser }, testInfo) => {
  const browserVersion = await browser.version();
  await page.goto("/staging-harness.html", { waitUntil: "networkidle" });
  await page.waitForFunction(() => (window as any).__stagingTest !== undefined, null, {
    timeout: 30_000,
  });

  const result: BrowserResult = await page.evaluate(
    async (args): Promise<Omit<BrowserResult, "browserName" | "browserVersion">> => {
      const h = (window as any).__stagingTest as any;
      const HW = navigator.hardwareConcurrency ?? 4;
      // ort-web version (best-effort)
      let ortVer = "unknown";
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mod: any = await import("onnxruntime-web");
        ortVer = mod?.version ?? (mod?.env?.version as string) ?? "unknown";
      } catch {
        /* ignore */
      }

      const isCoIso = (self as any).crossOriginIsolated === true;
      const sab = typeof (window as any).SharedArrayBuffer;

      const input = h.makeSyntheticInput(22, 1000, 250); // [1,22,1000] @ 250Hz

      // ---- provenance: SHA-verify the canonical FP32 artifact (must remain intact) ----
      let fp32Sha = "";
      let fp32Ok = false;
      try {
        const info = await h.resolveVerification(args.FP32_URL);
        fp32Sha = info?.sha256 ?? "";
      } catch {
        /* resolve may return null if manifest entry absent */
      }
      try {
        await h.verifyRemoteArtifact(args.FP32_URL);
        fp32Ok = true;
      } catch {
        fp32Ok = false;
      }

      // ---- INT8 candidate fetchability ----
      let int8Ok = false;
      let int8Size = 0;
      try {
        const r = await fetch(args.INT8_URL);
        int8Ok = r.ok;
        int8Size = Number(r.headers.get("content-length") ?? 0);
      } catch {
        int8Ok = false;
      }
      if (int8Ok) {
        h.registerEEGConformer({
          id: args.INT8_ID,
          artifact: args.INT8_URL,
          enableVerification: false, // no manifest entry — staging-only candidate
        });
      }

      // ---- in-browser parity + correctness (FP32 vs INT8) ----
      let parityCos: number | null = null;
      let dim = 0;
      let fp32Fb = false;
      let int8Fb = false;
      let fp32Id = "";
      let int8Id = "";
      try {
        const fp32 = await h.embedEEG(input, {
          preferredModelId: args.FP32_ID, normalize: false, expectedDim: 32,
        });
        dim = fp32.dim; fp32Fb = !!fp32.fellBack; fp32Id = fp32.modelId;
        if (int8Ok) {
          const int8r = await h.embedEEG(input, {
            preferredModelId: args.INT8_ID, normalize: false, expectedDim: 32,
          });
          int8Fb = !!int8r.fellBack; int8Id = int8r.modelId;
          const a = fp32.vector as number[];
          const b = int8r.vector as number[];
          const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
          const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
          let dot = 0;
          for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
          parityCos = na && nb ? dot / (na * nb) : 0;
        }
      } catch (e) {
        /* parity unavailable — leave null */
      }

      // ---- determinism (FP32: two runs on identical input must match) ----
      let det = 0;
      try {
        const r1 = await h.embedEEG(input, {
          preferredModelId: args.FP32_ID, normalize: false, expectedDim: 32,
        });
        const r2 = await h.embedEEG(input, {
          preferredModelId: args.FP32_ID, normalize: false, expectedDim: 32,
        });
        const a = r1.vector as number[];
        const b = r2.vector as number[];
        const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
        const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
        let dot = 0;
        for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
        det = na && nb ? dot / (na * nb) : 0;
      } catch {
        det = 0;
      }

      const stats = (durations: number[]): LatencyStats => {
        const s = [...durations].sort((a, b) => a - b);
        const n = s.length;
        const p = (q: number) => s[Math.min(n - 1, Math.max(0, Math.floor((q / 100) * (n - 1))))];
        const mean = durations.reduce((a, b) => a + b, 0) / n;
        return { n, p50: p(50), p95: p(95), p99: p(99), mean, min: s[0], max: s[n - 1] };
      };

      const configs: ConfigResult[] = [];

      // (B0) Baseline: PRE-P3 per-call cost via the embed() facade (fresh
      // InferenceSession.create + verify + unload each call). Production
      // embedEEG() now routes the preferred model through the persistent
      // InferenceEngine (P3), so we call the facade directly to capture the
      // per-call session-create cost for comparison against the persistent run.
      {
        const times: number[] = [];
        for (let i = 0; i < args.WARMUP; i++) {
          await h.embedFacade(input, {
            modelId: args.FP32_ID, fallbackToPCA: true, normalize: false, expectedDim: 32,
          });
        }
        for (let i = 0; i < args.MEASURED; i++) {
          const t = performance.now();
          await h.embedFacade(input, {
            modelId: args.FP32_ID, fallbackToPCA: true, normalize: false, expectedDim: 32,
          });
          times.push(performance.now() - t);
        }
        configs.push({
          name: "FP32_per_call", session: "fresh-per-call", model: args.FP32_ID,
          model_sha: args.FP32_SHA, nThreads: "default", n: times.length,
          stats: stats(times), per_call_ms: times,
        });
      }

      // (A) INT8-QDQ per-call (Track A) — measured via the per-call facade
      // (fresh session each call) for the same reason as (B0).
      if (int8Ok) {
        const times: number[] = [];
        for (let i = 0; i < args.WARMUP; i++) {
          await h.embedFacade(input, {
            modelId: args.INT8_ID, fallbackToPCA: true, normalize: false, expectedDim: 32,
          });
        }
        for (let i = 0; i < args.MEASURED; i++) {
          const t = performance.now();
          await h.embedFacade(input, {
            modelId: args.INT8_ID, fallbackToPCA: true, normalize: false, expectedDim: 32,
          });
          times.push(performance.now() - t);
        }
        configs.push({
          name: "INT8_per_call", session: "fresh-per-call", model: args.INT8_ID,
          model_sha: args.INT8_SHA, nThreads: "default", n: times.length,
          stats: stats(times), per_call_ms: times,
        });
      }

      // (B1) FP32 persistent session, numThreads=1
      {
        await h.setOrtWasmThreads(1);
        const eng = h.inferenceEngine;
        for (let i = 0; i < args.WARMUP; i++) await eng.embed(args.FP32_ID, input); // 1st triggers session.create
        const times: number[] = [];
        for (let i = 0; i < args.MEASURED; i++) {
          const t = performance.now();
          await eng.embed(args.FP32_ID, input);
          times.push(performance.now() - t);
        }
        eng.dispose();
        configs.push({
          name: "FP32_persistent_1thread", session: "persistent(InferenceEngine)",
          model: args.FP32_ID, model_sha: args.FP32_SHA, nThreads: 1, n: times.length,
          stats: stats(times), per_call_ms: times,
        });
      }

      // (B2) FP32 persistent session, numThreads=hardwareConcurrency (SAB threads now unlocked)
      {
        await h.setOrtWasmThreads(HW);
        const eng = h.inferenceEngine;
        for (let i = 0; i < args.WARMUP; i++) await eng.embed(args.FP32_ID, input);
        const times: number[] = [];
        for (let i = 0; i < args.MEASURED; i++) {
          const t = performance.now();
          await eng.embed(args.FP32_ID, input);
          times.push(performance.now() - t);
        }
        eng.dispose();
        configs.push({
          name: "FP32_persistent_hw", session: "persistent(InferenceEngine)",
          model: args.FP32_ID, model_sha: args.FP32_SHA, nThreads: HW, n: times.length,
          stats: stats(times), per_call_ms: times,
        });
      }

      // (A+B) INT8-QDQ persistent session, numThreads=HW (best-case combination)
      if (int8Ok) {
        await h.setOrtWasmThreads(HW);
        const eng = h.inferenceEngine;
        for (let i = 0; i < args.WARMUP; i++) await eng.embed(args.INT8_ID, input);
        const times: number[] = [];
        for (let i = 0; i < args.MEASURED; i++) {
          const t = performance.now();
          await eng.embed(args.INT8_ID, input);
          times.push(performance.now() - t);
        }
        eng.dispose();
        configs.push({
          name: "INT8_persistent_hw", session: "persistent(InferenceEngine)",
          model: args.INT8_ID, model_sha: args.INT8_SHA, nThreads: HW, n: times.length,
          stats: stats(times), per_call_ms: times,
        });
      }

      return {
        sharedArrayBuffer: sab,
        crossOriginIsolated: isCoIso,
        hardwareConcurrency: HW,
        ortWebVersion: ortVer,
        fp32: { sha256: fp32Sha, sha256_verified: fp32Ok, size_bytes: 3359557 },
        int8: { available: int8Ok, sha256: args.INT8_SHA, size_bytes: int8Size },
        parity: {
          embedding_cosine_fp32_vs_int8: parityCos,
          dim,
          fp32_fellBack: fp32Fb,
          int8_fellBack: int8Fb,
          fp32_modelId: fp32Id,
          int8_modelId: int8Id,
        },
        determinism: { cosine_fp32_runA_vs_B: det },
        configs,
      };
    },
    { FP32_URL, FP32_SHA, INT8_URL, INT8_SHA, FP32_ID, INT8_ID, WARMUP, MEASURED },
  );

  result.browserName = browserName;
  result.browserVersion = browserVersion;

  // ---- hard correctness gates (latency is data, not a hard-assert here) ----
  expect(result.crossOriginIsolated, "cross-origin isolation (COOP/COEP) must be on").toBe(true);
  expect(result.sharedArrayBuffer, "SharedArrayBuffer must be available").toBe("function");
  expect(result.fp32.sha256_verified, "canonical FP32 artifact SHA-256 must verify").toBe(true);
  expect(result.fp32.sha256, "FP32 sha").toBe(FP32_SHA);
  expect(result.parity.dim, "embedding dim = 32").toBe(32);
  expect(result.parity.fp32_fellBack, "FP32 must not fall back").toBe(false);
  expect(result.determinism.cosine_fp32_runA_vs_B, "FP32 determinism ~1.0").toBeGreaterThan(0.9999);
  if (result.int8.available) {
    expect(result.parity.int8_fellBack, "INT8 must run via ORT-WASM (no PCA fallback)").toBe(false);
    expect(result.parity.embedding_cosine_fp32_vs_int8 ?? 0, "INT8 parity cosine > 0.99").toBeGreaterThan(0.99);
  }

  // ---- emit report ----
  const outDir = path.join(process.cwd(), "reports");
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `v2-int8-vs-persistent-results.${browserName}.json`);
  fs.writeFileSync(outFile, JSON.stringify(result, null, 2));

  // eslint-disable-next-line no-console
  console.log(
    `[P2 ${browserName}] ` +
      result.configs
        .map(
          (c) =>
            `${c.name}: numThreads=${c.nThreads} p50=${c.stats.p50.toFixed(1)} ` +
            `p95=${c.stats.p95.toFixed(1)} (n=${c.stats.n})`,
        )
        .join(" | "),
  );

  // ---- gate verdict (reported, not hard-failed so a miss isn't a test-fatal) ----
  const find = (n: string) => result.configs.find((c) => c.name === n);
  const b0 = find("FP32_per_call");
  const a = find("INT8_per_call");
  const b1 = find("FP32_persistent_1thread");
  const b2 = find("FP32_persistent_hw");
  const ab = find("INT8_persistent_hw");
  const gate = (c: ConfigResult | undefined) =>
    c ? `p95=${c.stats.p95.toFixed(0)}ms p50=${c.stats.p50.toFixed(0)}ms` : "n/a";
  // eslint-disable-next-line no-console
  console.log(
    `[P2 ${browserName}] gate verdict -> ` +
      `baseline=${gate(b0)} | INT8-percall=${gate(a)} | ` +
      `FP32-persist-1t=${gate(b1)} | FP32-persist-HW=${gate(b2)} | ` +
      `INT8-persist-HW=${gate(ab)}`,
  );
});
