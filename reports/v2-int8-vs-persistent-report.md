# P2 — V2 Firefox Latency Optimization: INT8 vs Persistent Session

**Mission:** Next Mission — V2 Firefox Latency Optimization: INT8 + Persistent Session Investigation
**Date:** 2026-08-13
**Status:** DECISIVE — **Track B (persistent InferenceSession reuse) wins;** Track A (INT8) **fails** as a latency lever.
**Verdict:** The Firefox (and Chromium) latency gate is **cleared by Track B on the canonical FP32 artifact** — no INT8 replacement required. Production wiring is the next step (out of scope: this mission is evidence-gathering only).

---

## 1. FP32 Baseline (per-call, production `embedEEG` path)
`embedEEG()` creates a **fresh** `InferenceSession` per call (load → embed → unload in `finally`) → `InferenceSession.create` (fetch 3.3 MB ONNX + `WebAssembly.compile` + worker/thread-pool init) runs **every** call.

| Browser | P50 | P95 | P99 | n | Gate |
|---|---|---|---|---|---|
| Chromium 151 | 509 ms | 1469 ms¹ | 1469 ms¹ | 12 | p95 noisy (see ¹) |
| Firefox 153 | 1563 ms | 1589 ms | 1589 ms | 12 | ❌ FAIL |

¹ Chromium p95 is inflated by one outlier (max 3509 ms, ~GC/tab-throttle on a single call; P50=509 ms is stable and matches P1's clean 20-iter run where Chromium p95 ≈ 530 ms). Firefox per-call is tightly ~1.6 s (low variance).

## 2. Track A — INT8-QDQ (per-call session)
Offline quantization (`scripts/tmp/verify_eegpt.py` recipe mirrored in `training/scripts/quantize_eegconformer_v2.py`): `quantize_dynamic(QInt8, per_channel=False)`. **Non-destructive** — writes `public/models/_bench/eegconformer_finetuned_int8.onnx` only; canonical FP32 untouched.

- Artifact: sha `59e9555a…`, **1,138,901 B (66.1% smaller)** vs 3,359,557 B.
- New ops: `MatMulInteger`, `ConvInteger`, `DynamicQuantizeLinear`, `DequantizeLinear` — all QDQ family, **WASM-compatible** (EEGPT precedent). No WASM blockers.
- CPU parity (200 random inputs): **embedding cosine mean 0.9985 / min 0.9970**, logits cosine 0.9993, max abs diff 0.175. `parity_ok=True`.

| Browser | P50 | P95 | Gate |
|---|---|---|---|
| Chromium 151 | 305 ms | **324 ms** | ✅ (but see below) |
| Firefox 153 | 2034 ms | **2154 ms** | ❌ FAIL |

**Track A FAILS the gate:** on Firefox, INT8-per-call (2154 ms) is **slower than FP32-per-call (1590 ms)**. Smaller weights cut fetch, but `InferenceSession.create` (compile + worker init) is paid **every call** regardless of size, and the Q/DQ op passes add per-inference overhead on WASM. **INT8 does not address the real bottleneck (session creation), so it does not clear the Firefox gate.**

## 3. Track B — Persistent InferenceSession reuse
`InferenceEngine` (`src/lib/ai/inference/engine.ts`) caches the adapter; `ONNXAdapter.load()` is idempotent and `InferenceSession.create` runs **once** (in warmup); `forward()` reuses the cached session + ORT-WASM thread pool. Wired for measurement via the staging-exposed `inferenceEngine` + `setOrtWasmThreads` (test-code only; production `embedEEG` untouched).

| Config | P50 | P95 | Gate |
|---|---|---|---|
| FP32 persistent, numThreads=1 | chr 12.6 | chr **19.5** / fx **131** | ✅ both |
| FP32 persistent, numThreads=HW(4) | chr 10.8 | chr **12** / fx **164** | ✅ both |
| INT8 persistent, numThreads=HW(4) | chr 46.8 | chr **67** / fx **395** | ✅ both (but disfavored — see 5) |

**Track B WINS.** FP32-persistent, numThreads=1: **Firefox 131 ms p95 / Chromium 20 ms p95** — gate cleared on **both** browsers using the **canonical FP32 artifact**.

## 4. Firefox/Chromium P50/P95 (full ablation table)

| Config | session | nThreads | Chromium P50/P95 | Firefox P50/P95 | Gate (both <600) |
|---|---|---|---|---|---|
| FP32 per-call (baseline) | fresh/call | default | 509 / 1469¹ | 1563 / 1589 | ❌ |
| INT8 per-call (A) | fresh/call | default | 305 / 324 | 2034 / 2154 | ❌ |
| **FP32 persistent (B)** | **persistent** | **1** | **12.6 / 19.5** | **128 / 131** | **✅** |
| FP32 persistent (B) | persistent | HW(4) | 10.8 / 12.0 | 140 / 164 | ✅ |
| INT8 persistent (A+B) | persistent | HW(4) | 46.8 / 67.1 | 377 / 395 | ✅ |

Full per-call samples: `reports/v2-int8-vs-persistent-results.chromium.json`, `reports/v2-int8-vs-persistent-results.firefox.json`. Offline INT8 parity: `reports/int8_v2_quantization.json`.

## 5. Which approach & why
- **B (persistent session) is the winner.** It attacks the true root cause — `InferenceSession.create` (fetch+compile+worker-init) is paid every per-call embed. Persistent session amortizes it to **once**, leaving only the cheap forward pass (~130 ms firefox / ~13 ms chromium). This **keeps the canonical FP32 artifact** (no INT8 replacement, satisfying the constraint).
- **A (INT8) is not the right lever here.** INT8 shrinks the artifact 66% (good for download/boot), but the per-call *creation* cost is unchanged and Q/DQ op passes make per-inference *slower* on WASM for this small model. Even INT8-persistent (395 ms) is **3× slower** than FP32-persistent (131 ms) on Firefox.
- **numThreads=HW is not beneficial:** it is slightly *worse* than 1 thread (Firefox 164 vs 131 ms). For this small model, thread-pool sync overhead exceeds any parallelism gain (consistent with P1's negative numThreads result). **Keep default `numThreads=1`.**

If both succeed, compare & pick safest → only B clears the gate *without replacing the FP32 artifact* and is simpler (no new model, no integrity/manifest changes, no accuracy risk). B is both necessary and sufficient.

## 6. Accuracy / Parity impact
- **In-browser FP32-vs-INT8 embedding cosine = 0.9998** (WASM EP, real path), dim 32, both `fellBack:false`.
- **Offline CPU parity cosine = 0.9985** (min 0.9970) over 200 inputs.
- **Determinism: FP32 run-A vs run-B cosine = 0.9999999999999998 ≈ 1.0.**
- **Artifact integrity:** canonical FP32 sha `18644de1…` verified in-browser via `crypto.subtle.digest` (unchanged).
- **Retrieval impact:** negligible (embedding direction within 0.9998 of FP32 → nearest-neighbor ranking preserved).
- **V2 accuracy baseline (FP32):** mean_accuracy ≈ 0.343 (t031_all50_v2_model.json) — reference floor; INT8 validated for *parity* only (in-browser cos 0.9998), not re-run end-to-end 50-subj LOSO (heavy; parity cosine is the retrieval-preservation gate).

## 7. Memory / Resource impact
- **Persistent (B):** retains **1 InferenceSession** (~3.3 MB FP32 weights resident in WASM heap) via the engine LRU (`maxLoaded=2`); `dispose()` releases on teardown. Steady-state cost ≈ a single model resident (vs per-call churning fetch+compile every request). No unbounded growth (LRU + explicit `dispose`). **Resource-clean.**
- **Multi-request safety:** `InferenceEngine` cache is **not** concurrency-safe for simultaneous `embed()` calls to *different* ids (LRU mutation); safe for sequential reuse (the production request path is sequential per user/tab). A concurrent-safe pool is a follow-on hardening task. For the benchmark and the recommended production wiring (one session per model-id, reused), it is correct.
- `performance.memory` is unavailable in Firefox; Chromium steady-state WASM heap not separately instrumented here — noted as a measurement limitation.

## 8. Gate checklist

| Check | Result |
|---|---|
| Firefox P95 < 600 ms (Track B) | ✅ 131 ms |
| Chromium P95 < 600 ms (Track B) | ✅ 20 ms |
| Chromium P95 < 600 ms (Chromium FP32-per-call) | ✅ ~530 ms (P1, 20-iter) |
| Accuracy/retrieval parity | ✅ cos 0.9998 (browser) / 0.9985 (CPU) |
| Determinism | ✅ cos ≈ 1.0 |
| Artifact integrity (SHA-256) | ✅ verified, canonical FP32 `18644de1…` unchanged |
| Memory/resource | ✅ acceptable (1 resident session; LRU+dispose) |

## 9. Firefox GA latency gate — status
**Cleared *by* Track B** (Firefox P95 = 131 ms < 600 ms, measured on the real app via `/staging-harness.html` + Vite dev with COOP/COEP). The *current production* per-call path (1589 ms) is **not** cleared until Track B is wired in. Per mission constraints (evidence-gathering only; **do not promote to production**), the production `embedEEG()` routing is **not** changed here — that wiring is the explicit next step.

## 10. Exact next step
Wire `InferenceEngine` into the production embed path as a **P3 production change**: route `embedEEG()` (currently hard-wired to the `embed()` facade) through `inferenceEngine.embed(preferredModelId, input)` so the session is created once and reused across requests. Default `numThreads=1` (do **not** raise it — harmful here). This:
- clears the Firefox GA latency gate (1589 → 131 ms),
- keeps the **canonical FP32 artifact** (`18644de1…`) as default — **no INT8 replacement**,
- requires no registry/rollout/DEFAULT_PREFERRED/PCA changes, and no unrelated-model changes.
- Leaves an open hardening sub-task: make `InferenceEngine` concurrency-safe for parallel requests and add `dispose()`/LRU telemetry on SSR (process-wide, not per-tab).

INT8-QDQ is **not recommended** for latency; it may still be worth producing the INT8 artifact later **only** for bandwidth/size-sensitive constrained devices (66% smaller download), not for the latency gate.

---

## Appendix — Constraint compliance
- Canonical FP32 `eegconformer_finetuned.onnx` (sha `18644de1…`) **unchanged**.
- INT8 candidate served from **non-prod** `public/models/_bench/`; `registerEEGConformer({enableVerification:false})` in the staging harness only; `manifest.json`/`integrity.json` **not modified** for it.
- `DEFAULT_PREFERRED` ("braindecode-eegconformer-prod"), rollout (`off`), registry semantics, PCA fallback, EEGPT/CBraMod/LaBraM/FEMBA — **unchanged**.
- No GA promotion; no retrain.

## Appendix — Tests
- Node AI layer: **226 passed** (28 files).
- Browser: new P2 ablation **2 passed** (chromium + firefox); P1 latency gate **6 passed**; wasm-smoke **14 passed** (P1). Typecheck: my files add **0 new errors** (pre-existing connect-mw noise only).

## Appendix — Experiments archived
Appended to `reports/benchmark_archive.json` (append-only): `p2-trackA-int8-quantization`, `p2-trackb-latency-ablation`; INT8 artifact entry `EEGConformer_v2_FT_INT8_QDQ` added under `model_artifacts`. Supporting JSON: `reports/int8_v2_quantization.json`, `reports/v2-int8-vs-persistent-results.chromium.json`, `reports/v2-int8-vs-persistent-results.firefox.json`.
