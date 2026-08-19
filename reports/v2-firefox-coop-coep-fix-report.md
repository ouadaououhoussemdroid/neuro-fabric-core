# V2 Firefox WASM Latency Gate — COOP/COEP Runtime Fix (P1)

**Mission:** Next Mission — Productionize the V2 Firefox WASM Fix
**Date:** 2026-08-13
**Status:** PARTIAL / CONDITIONAL PASS — COOP/COEP permanent fix implemented & verified; Chromium gate PASS; Firefox latency gate NOT cleared (deferred to INT8-QDQ, out of scope this mission).
**Artifact:** `eegconformer_finetuned.onnx` (FP32, canonical) — **UNCHANGED**. SHA-256 `18644de1…`, 3,359,557 B. INT8 NOT applied.

---

## 1. Baseline
Before the fix, `crossOriginIsolated === false` in the browser (Firefox specifically) and `SharedArrayBuffer` was unavailable, so ORT-Web's threaded/SIMD WASM backend could not be engaged for Firefox. The V2 FP32 artifact was already SHA-verified and deterministic (cosine = 1.0). The Chromium gate was already passing (~530 ms P95) because post-Chrome-116 Chromium heuristically enables SAB even without strict COOP/COEP in some configs — but Firefox **never** does.

## 2. Root-Cause Experiments
1. **COOP/COEP only** → verified `crossOriginIsolated === true` and `SharedArrayBuffer` available on **both** Chromium and Firefox (real app). ✅
3. **numThreads tuning** (`numThreads = navigator.hardwareConcurrency` in `defaultRuntime()`) → **NEGATIVE, reverted.** Firefox P95 worsened 1742 → 2658 ms. Root cause documented: `embedEEG()`/`embed()` creates a **fresh** adapter + `InferenceSession` per call (load → embed → unload in `finally`), so raising `numThreads` forces a new Web Worker thread-pool to spin up **and** tear down on every call — worker spin-up dominates the tiny V2 forward pass. Threading only helps with a *reused* session/thread-pool, which this per-request path does not provide. `onnx-adapter.ts` restored to its committed form.

## 3. Permanent Fix — Where Headers Live
- **Dev / test (real app):** `vite.config.ts` → `server.headers: { COOP: same-origin, COEP: require-corp, CORP: same-origin }` (covers `/`, Vite-served `/models`, `/src`, 404s). The `testHarnessPlugin` bypasses Vite's header stack by finalizing responses with `res.end(...)`, so `vite-plugins/test-harness.ts` now calls `setCoopHeaders(res)` at **every** `res.end()` site: the smoke/staging harness HTML, the `/ort/*.wasm` middleware, and the `/ort/*.mjs?import` middleware.
- **Constant:** `vite-plugins/coop-coep-headers.ts` (new) — single source of truth for the three headers.
- **Production SSR:** `src/server.ts` → `withCrossOriginIsolationHeaders()` wraps **both** return paths of the Nitro `fetch` handler (success + 500 render-error), preserving body/status/existing content-type.

Headers:
```
Cross-Origin-Opener-Policy:  same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin   (where required)
```

## 4. Firefox vs Chromium Measurements (real app, 20 measured / 3 warmup discarded)

| Browser | crossOriginIsolated | SharedArrayBuffer | Fallbacks | P50 | P95 | Mean | Min | Max |
|---|---|---|---|---|---|---|---|---|
| Chromium 151 | ✅ true | ✅ function | 0 | 289.3 ms | 530.0 ms | 306.9 ms | 206.1 ms | 611.3 ms |
| Firefox 153   | ✅ true | ✅ function | 0 | 1393.3 ms | 1586.9 ms | 1417.6 ms | 1345.3 ms | 1629.9 ms |

Latency gate: **Chromium PASS** (P95 < 600, P50 < 400). **Firefox NOT cleared** (P95 1587 ms ≫ 600 ms): the fix removes the *blocking* condition (SAB now available) but ORT-Web's single-thread WASM backend plus per-call session recreation is still ~1.6 s. The remaining lever is INT8 (smaller weights + INT8 kernels) → deferred per mission scope.

## 5. Accuracy / Parity Impact
- SHA-256 verified **in-browser** via `crypto.subtle.digest` against `public/ort/integrity.json` / manifest: ✅ pass.
- Determinism: cosine similarity V1↔V2 and run↔run = **1.0** (V2 FP32 canonical, no drift). ✅
- Embedding dim = 32, class count = 4, 0 fallbacks, PCA fallback path intact (untouched). ✅

## 6. Best Configuration
- COOP + COEP + CORP enabled on **all three layers** (Vite `server.headers`, test-harness `res.end` sites, Nitro SSR). **Ship numThreads at ORT-Web default (1)** — the tuned value is a net negative and must NOT ship.

## 7. Gate Checklist

| Check | Result |
|---|---|
| Chromium P95 < 600 ms | ✅ PASS (530 ms) |
| Chromium P50 < 400 ms | ✅ PASS (289 ms) |
| Firefox P95 < 600 ms | ❌ FAIL (1587 ms) — deferred to INT8 (P2) |
| Firefox crossOriginIsolated === true | ✅ PASS |
| Firefox SharedArrayBuffer available | ✅ PASS |
| SHA-256 parity retained | ✅ PASS |
| Determinism cos = 1.0 | ✅ PASS |
| PCA fallback intact | ✅ PASS |
| Registry semantics unchanged | ✅ PASS |
| Rollout stage unchanged | ✅ PASS |
| DEFAULT_PREFERRED unchanged | ✅ PASS |
| V2 FP32 canonical artifact unchanged | ✅ PASS |

## 8. Tests Run (exact)
- **Node / AI-layer unit tests:** `226 passed` (`tests/` + `src/lib/ai/**/__tests__`, vitest, --config vitest.config.ts). ✅ All green.
- **Browser WASM smoke:** `wasm-smoke` group — `14 passed`. ✅
- **New regression test** `tests/browser/v2-firefox-latency-gate.test.ts` across Chromium + Firefox: `6 passed` (3 tests × 2 browsers): (1) COOP/COEP + SAB hard assert PASS; (2) real V2 inference correctness (0 fallbacks, dim 32, SHA-256) PASS; (3) latency — Chromium hard gate PASS, Firefox regression floor (< 2500) enforced with documented deferral PASS. ✅
- **Staging determinism + SHA + metrics snapshot:** `8 passed` (cosine = 1.0, SHA-256 browser-verified, verification pass counter recorded). ✅
- **Pre-existing Firefox latency tests** (`staging-latency` v1/v2 on Firefox, P95<600) remain failing as before T-035 — **unchanged by this mission** (INT8 deferred).

## 9. PASS-FAIL Decision
- **Permanent Firefox blocker (SAB / cross-origin isolation):** ✅ **RESOLVED.** Headers verified on the real app on both browsers.
- **V2 Firefox latency gate (<600 ms):** ❌ **NOT cleared** — blocked by the single-thread ORT-Web WASM backend + per-call session recreation, not by COOP/COEP. The tuning lever attempted (numThreads) was a **negative result** and was reverted. The remaining lever (INT8-QDQ) is explicitly out of scope ("do not replace V2 with INT8 yet").

## 10. Recommended Next Mission
**INT8-QDQ V2 ablation (P2):** offline (`onnxruntime.quantization`) CPU parity + size check, then browser latency/accuracy on Chromium + Firefox, **preserving the FP32 canonical artifact** as the registry default. Secondary: evaluate **persistent `InferenceSession` + thread-pool reuse** across `embedEEG()` calls to amortise worker spin-up (the negative numThreads result implies this is required before threading can pay off). All other models (EEGPT/CBraMod/LaBraM/FEMBA), PCA fallback, registry, rollout, and `DEFAULT_PREFERRED` remain untouched.
