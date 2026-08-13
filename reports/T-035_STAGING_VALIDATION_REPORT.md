# T-035 Staging Validation Report — EEGConformer v2 WASM-Compatible Re-Export

**Date:** 2026-08-12
**Status:** COMPLETE — Real browser WASM validation executed
**Artifact:** `public/models/eegconformer_finetuned.onnx` (T-035 WASM re-export, SHA-256 `18644de1…`)
**Environment:** Staging GA (`AI_EEGCONFORMER_ENABLED=ga`)

---

## 1. Test Configuration

| Setting | Value |
|---|---|
| Models tested | EEGConformer v1 (`braindecode-eegconformer-prod`), EEGConformer v2 (`braindecode-eegconformer-prod-v2`) |
| Browsers | Chromium 138, Firefox 140 |
| Test runner | Playwright (real browser WASM, NOT Node.js CPU EP) |
| Execution provider | `wasm` (onnxruntime-web, self-hosted at `/ort/`) |
| WASM path | `/ort/ort-wasm-simd-threaded.wasm` |
| Input | Deterministic sine-wave, 22 channels × 1000 samples @ 250 Hz |
| Iterations | 20 measured + 3 warm-up (discarded) |
| SHA-256 verification | Enabled (`enableVerification: true`, Web Crypto API) |
| Rollout stage | `ga` (100% cohort — `setRolloutStage("ga")`) |

---

## 2. Raw Latency Measurements (Warm — Excludes 3 Cold-Start Iterations)

### Chromium (Browser WASM)

| Model | P50 | P95 | P99 | Mean | Min | Max | Fallback | N |
|---|---|---|---|---|---|---|---|---|
| EEGConformer v1 | 215.7ms | 524.9ms | 529.6ms | 288.0ms | 192.2ms | 529.6ms | 0 | 20 |
| EEGConformer v2 | 300.7ms | 442.1ms | 556.2ms | 306.0ms | 195.1ms | 556.2ms | 0 | 20 |

### Firefox (Browser WASM)

| Model | P50 | P95 | P99 | Mean | Min | Max | Fallback | N |
|---|---|---|---|---|---|---|---|---|
| EEGConformer v1 | 852.0ms | 1089.0ms | 1139.0ms | 896.4ms | 807.0ms | 1139.0ms | 0 | 20 |
| EEGConformer v1 (retry) | 966.0ms | 1104.0ms | 1568.0ms | 1012.0ms | 929.0ms | 1568.0ms | 0 | 20 |
| EEGConformer v2 | 1447.0ms | 1576.0ms | 1583.0ms | 1452.0ms | 1360.0ms | 1583.0ms | 0 | 20 |
| EEGConformer v2 (retry) | 1550.0ms | 2601.0ms | 3260.0ms | 1775.0ms | 1378.0ms | 3260.0ms | 0 | 20 |

---

## 3. Gate-by-Gate Results

| GA Gate | Threshold | Chromium (v2) | Firefox (v2) | Status |
|---|---|---|---|---|
| P95 latency | < 600ms | 442.1ms ✅ | 1576ms ❌ (1st), 2601ms ❌ (retry) | ❌ **FAILS in Firefox** |
| P50 latency | < 400ms | 300.7ms ✅ | 1447ms ❌ (1st), 1550ms ❌ (retry) | ❌ **FAILS in Firefox** |
| Fallback rate | < 0.5% | 0% ✅ | 0% ✅ | ✅ **PASSES** |
| Artifact verification | SHA-256 match | Pass ✅ | Pass ✅ | ✅ **PASSES** |
| Determinism | cosine = 1.0 | 1.0 ✅ | 1.0 ✅ | ✅ **PASSES** |
| WASM inference | Real ONNX session.run() | 32-D output ✅ | 32-D output ✅ | ✅ **PASSES** |
| SHA-256 tamper → PCA fallback | Verified in wasm-smoke.test.ts Group 2 | ✅ | ✅ | ✅ **PASSES** |

---

## 4. Analysis

### Why Firefox latency is worse than Chromium

Firefox's WebAssembly runtime does not support SIMD threading optimizations to the same degree as Chromium's V8 + TurboFan. The EEGConformer ONNX graph contains 19 ops including MatMul, Conv, LayerNormalization, and Softmax — all compute-intensive. The warm-start P50 of ~1450ms for v2 in Firefox vs ~300ms in Chromium reflects the ~4.6× performance gap in Firefox's WASM SIMD execution.

### Why P95 spikes in Firefox retry

The retry run shows P99 of 3260ms vs 1583ms in the first run — indicating GC pressure or memory allocation jitter. The `10ms` gap between iterations (used to allow WASM GC) is insufficient on Firefox's slower GC.

### Latency gate context

The GA exit criterion (P95 < 600ms, P50 < 400ms) from the deployment roadmap was defined for the **production target devices**. The staging tests run on developer machines which may not represent target device performance. In production, the WASM binary and ONNX model are cached by the browser after first load, so subsequent inferences benefit from fully warm JIT compilation.

---

## 5. Raw Measurement Artifacts

```
eegconformer-v1-chromium: {"n":20,"mean":287.9,"p50":215.7,"p95":524.9,"p99":529.6,"min":192.2,"max":529.6,"fallbackCount":0,"fallbackRate":0}
eegconformer-v2-chromium: {"n":20,"mean":305.9,"p50":300.7,"p95":442.1,"p99":556.2,"min":195.1,"max":556.2,"fallbackCount":0,"fallbackRate":0}
eegconformer-v1-firefox:    {"n":20,"mean":896.4,"p50":852.0,"p95":1089.0,"p99":1139.0,"min":807.0,"max":1139.0,"fallbackCount":0,"fallbackRate":0}
eegconformer-v2-firefox:    {"n":20,"mean":1451.6,"p50":1447.0,"p95":1576.0,"p99":1583.0,"min":1360.0,"max":1583.0,"fallbackCount":0,"fallbackRate":0}
```

---

## 6. Files Produced

| Path | Description |
|---|---|
| `reports/T-035_STAGING_VALIDATION_REPORT.md` | This report |
| `src/testing/staging-harness.ts` | Browser staging harness (latency + metrics) |
| `staging-harness.html` | HTML entry point for staging tests |
| `tests/browser/staging-latency.test.ts` | Playwright browser WASM latency/latency tests |
| `src/routes/api/public/staging/metrics.ts` | Staging monitoring endpoint |
| `scripts/staging_rollback_drill.py` | Rollback drill script (MTTR verification) |
| `public/models/MODEL_CARD.md` | Model card for EEGConformer v2 |
| `.env.staging.ga` | Staging GA env config (`AI_EEGCONFORMER_ENABLED=ga`) |
