# Mission 22 — Master Execution Roadmap
## Roadmap Reconciliation & Next-Generation TODO

**Mission Date:** 2026-08-17  
**Mission Director:** e725bb9d (Pooled)  
**Status:** ✅ COMPLETE — Planning / Reconciliation Only  
**Constraint:** Do NOT implement code; modify production behavior; change models; change rollout; promote anything; delete old roadmap items; rewrite benchmark history; start new experiments. This document is a planning artifact synthesizing evidence from Missions 4, 10–21 and Tiers 2–4.

---

## 0. Executive Summary

This document reconciles the Neuro-Fabric Core execution roadmap (from the
original 2026-06-19 Open-Source Execution Blueprint and the
2026-06-17 EEGConformer Deployment Roadmap) against **all new evidence**
produced through Missions 4, 10–21 and Tiers 2–4. It produces a single
**Priority-ranked master roadmap** (P0/P1/P2/P3) grounded in measured
outcomes, a record of what is **materially better**, a **DO NOT REPEAT**
list, the **next 6 missions**, and a **classification** of every original
roadmap item.

### 0.1 What Changed Since the Last Roadmap

| Area | Before (June 2026) | After (August 2026) | Delta |
|---|---|---|---|
| **V2 browser latency** | Firefox P95 ≈ 1589 ms (FAIL, gate < 600 ms) | Firefox P95 = **161.9 ms** (PASS, ~9.8×) | ✅ Gate cleared |
| **Inference architecture** | Per-call `InferenceSession.create` (fetch+compile+worker every embed) | Persistent cached session + LRU + async mutex | ✅ Root cause fixed |
| **INT8 quantization** | Not attempted | 66.1% smaller, cos 0.9985, but **slower** (2154 ms vs 1590 ms per-call) | ❌ Not the lever |
| **Model zoo** | v1 = GA; CBraMod/EEGPT/LaBraM/FEMBA = stubs/research | V2 registered & validated; CBraMod server-ready (M15); EEGPT/LaBraM/FEMBA = evaluated & mostly dropped | ✅ Portfolio rationalized |
| **CBraMod role** | Unclear (MI accuracy inconclusive) | Server-side subject-identity specialist (R@5 +0.312, p=1.66e-59); MI remap negative (p=0.353) | ✅ Role clarified |
| **Cognitive decoder** | 1,333-byte stub | Real 1.5 KB skl2onnx model (TIER-2 bug fixed) | ✅ Functional |
| **CI/CD** | None | GitHub Actions (typecheck, lint, test, build, nightly SLO) | ✅ Live |
| **Rate limiting** | None | PostgreSQL-backed atomic UPSERT, per-user | ✅ Live |
| **WASM delivery** | jsDelivr CDN (SPOF) | Self-hosted `/ort/` (13.5 MB threaded SIMD) + SHA-384 + COOP/COEP | ✅ No CDN SPOF |
| **Feature extraction** | O(M²) DFT | O(M log M) FFT (TD-005) | ✅ 2–5× faster |
| **Vector store** | In-memory | pgvector `vector(32)` + `foundation_embeddings` `vector(200)` | ✅ Persistent |
| **Statistical infra** | Minimal | Full stats module (t-test, Cohen's d, CIs, p-values), LOSO, ground-truth, benchmarks | ✅ TIER-3 GO |
| **Scientific validation** | Inconclusive (synthetic probes) | Tier-4 validated (83/83), 50-subject LOSO, T-030/T-031 corrections applied | ✅ Validated |
| **Technical maturity** | 63/100 (Research Platform) | 63/100 (score stable; gates cleared but GA promotion pending) | ⚠️ Score unchanged — promotion decision is P0 |

### 0.2 What Has NOT Changed

- **`DEFAULT_PREFERRED`** remains `braindecode-eegconformer-prod` (v1).
- **V2 rollout** remains `off` (opt-in only).
- **Canonical V2 ONNX artifact** SHA `18644de1…` unchanged (byte-for-byte).
- **No retraining** of any model since Mission 4.
- **Benchmark archive** records preserved byte-for-byte (append-only).
- **No INT8 in production** (remains in `public/models/_bench/`).

### 0.3 Constraint Compliance This Mission

| Constraint (Mission 22) | Compliance | Verification |
|---|---|---|
| No code implementation | ✅ | No source files modified; this is a planning document |
| No production behavior change | ✅ | No routing, rollout, or config changes |
| No model changes | ✅ | No weights, artifacts, or registry changes |
| No promotion | ✅ | V2 still opt-in; DEFAULT_PREFERRED still v1 |
| No old roadmap deletion | ✅ | Original roadmaps preserved under `docs/archived/roadmaps/` |
| No benchmark history rewrite | ✅ | `benchmark_archive.json` append-only; prior records intact |
| No new experiments | ✅ | All evidence sourced from completed missions |

---

## Phase 1 — Recovered Roadmap

The original roadmap consists of two primary planning documents, plus the
technical debt register and the audit. All are preserved under
`docs/archived/roadmaps/` and `reports/`.

### 1.1 Open-Source Execution Blueprint (2026-06-19)
**File:** `docs/archived/roadmaps/2026-06-19_open_source_execution_blueprint.md`

- **30 tasks** (T-001 through T-028) across 13 categories (A–Q).
- **Top-25 ranked** by Scientific × Engineering × Investor × Builder scores.
- **Three horizons:** 30-day (Days 1–30), 90-day (Days 31–90), 180-day (Days 91–180).
- **Shortest realistic path:** 6 steps, ~9 engineer-weeks.

### 1.2 EEGConformer Deployment Roadmap (2026-06-17)
**File:** `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md`

- **6 phases:** Phase 0 (Artifact) → Phase 1 (Hosting) → Phase 2 (Wiring) →
  Phase 3 (Validation) → Phase 4 (Rollout) → Phase 5 (Post-Launch).

### 1.3 Next-Phase Roadmap (Post-Blueprint)
**File:** `reports/ROADMAP_NEXT_PHASE.md`

- **4 phases:** Phase 1 (Operational Foundation, Weeks 1–4) →
  Phase 2 (Core Capability, Weeks 5–8) → Phase 3 (Platform Maturation,
  Months 3–4) → Phase 4 (Pilot, Months 4–6).
- Each phase has explicit **gate criteria**.

### 1.4 Technical Debt Register
**File:** `reports/TECHNICAL_DEBT.md`

- **15 items** (TD-001 through TD-015), classified P1–P3.
- **P1:** 4 items (CI/CD, rate limiting, WASM SPOF, artifact bundling).
- **P2:** 6 items (FFT, pgvector, cognitive decoder, metrics, dataset loaders, input validation).
- **P3:** 5 items (acquisition coupling, vector scalability, doc cleanup, test coverage, contributing guide).

### 1.5 Top-25 Task Catalog (from Blueprint)

| Rank | Task | S | E | I | A | Σ | Status |
|---|---|---|---|---|---|---|---|
| 1 | T-010 EEGConformer empirical validation | 5 | 4 | 5 | 4 | 18 | ✅ Done (T-030/T-031/50-subj) |
| 2 | T-011 pgvector migration | 4 | 5 | 5 | 4 | 18 | ✅ Done (TD-006) |
| 3 | T-025 Trained cognitive decoder v0 | 5 | 4 | 5 | 4 | 18 | ✅ Done (TIER-2) |
| 4 | T-012 Recall@10 SLO harness | 5 | 5 | 3 | 4 | 17 | ⚠️ Partial (CI SLO referenced) |
| 5 | T-017 MOBBM-driven evaluation harness | 5 | 4 | 4 | 4 | 17 | ❌ Not started |
| 6 | T-008 Self-hosted ORT WASM | 3 | 5 | 4 | 5 | 17 | ✅ Done (TD-003) |
| 7 | T-009 Content-hashed ONNX artefact | 3 | 5 | 4 | 5 | 17 | ❌ Not started |
| 8 | T-015 Braindecode model zoo | 5 | 4 | 4 | 3 | 16 | ⚠️ Partial (stubs remain) |
| 9 | T-018 Captum saliency | 5 | 3 | 5 | 3 | 16 | ❌ Not started |
| 10 | T-027 CI/CD release gates | 3 | 5 | 4 | 4 | 16 | ✅ Done (TD-001) |
| 11 | T-005 BrainFlow integration | 4 | 4 | 5 | 3 | 16 | ❌ Not started |
| 12 | T-003 WebSocket EEG gateway | 4 | 5 | 4 | 3 | 16 | ❌ Not started |
| 13 | T-021 MLflow tracking | 4 | 5 | 3 | 4 | 16 | ❌ Not started |
| 14 | T-013 Concept graph schema | 5 | 4 | 4 | 3 | 16 | ❌ Not started |
| 15 | T-020 Reproducible training container | 4 | 5 | 3 | 4 | 16 | ⚠️ Partial (Dockerfile exists) |
| 16 | T-022 Registry↔MLflow↔Storage sync | 3 | 5 | 4 | 3 | 15 | ❌ Not started |
| 17 | T-006 Pyodide-MNE parity harness | 5 | 4 | 2 | 4 | 15 | ❌ Not started |
| 18 | T-002 EDF+/BDF/GDF parser hardening | 4 | 4 | 3 | 4 | 15 | ❌ Not started |
| 19 | T-007 FFT replacement | 3 | 5 | 3 | 4 | 15 | ✅ Done (TD-005) |
| 20 | T-024 WebGPU EP flag | 3 | 5 | 4 | 3 | 15 | ❌ Not started |
| 21 | T-028 Upload hardening | 2 | 5 | 4 | 4 | 15 | ✅ Done (TD-002) |
| 22 | T-014 Subject-level aggregation | 5 | 3 | 3 | 3 | 14 | ❌ Not started |
| 23 | T-019 Dataset manifest | 4 | 4 | 3 | 3 | 14 | ❌ Not started |
| 24 | T-004 LSL bridge | 4 | 3 | 3 | 4 | 14 | ❌ Not started |
| 25 | T-026 Notebook portal | 3 | 3 | 4 | 4 | 14 | ❌ Not started |
| — | T-001 Hardware-agnostic acquisition | — | — | — | — | — | ❌ Not started |
| — | T-016 EEGPT honesty | — | — | — | — | — | ❌ Not started |
| — | T-023 ONNX optimizer | — | — | — | — | — | ❌ Not started |

---

## Phase 2 — New Evidence Incorporated

### 2.1 Mission 4 — Fine-Tuned EEGConformer (2026-08-10)

- **Fine-tuned V2** (PhysioNet EEGMMIDB, 20 subjects, 505 training trials):
  **0.3342** accuracy vs Original **0.2825**, p=0.013, Cohen's d=0.70 (medium).
- **T-031 label-mapping bug** found and fixed (was inflating feet-class scores).
- **50-subject validation** (T-031-50SUBJ): V2 FT = **0.343** vs Original 0.283,
  p=0.0002, d=0.701 — significant. V2 vs PCA: 0.343 vs 0.313, p≈0.07 — **not
  significant**.
- **V3 (30-subject FT):** 0.310 vs Original 0.272, p=0.097 — **not significant**.
  Not deployed.
- **ONNX parity:** PyTorch↔ORT cos = 1.0 for V2; opset 17, 17 WASM-compatible ops.
- **Evidence source:** `reports/T-031_FINAL_REPORT.md`, `reports/MISSION_4_COMPLETION_REPORT.md`.

### 2.2 Missions 10–15 — CBraMod Server-Path Operationalization

- **Mission 10** (CBraMod product role decision): CBraMod MI accuracy
  (0.3043 @ 10 subj) **loses** to V2 but highest point estimate. Role =
  server-only representation specialist.
- **Mission 11** (cross-session validation): CBraMod 200-D **R@5 = 0.5273**
  vs V2 32-D **0.2158**, Δ=+0.3116, p=1.66e-59 (Bonferroni). CBraMod
  **wins subject-identity retrieval** decisively. MI accuracy is a safety
  floor only (0.275, near chance).
- **Mission 12** (Tier-2 architecture): Defined server-side native-200-D
  store + `/api/eeg/embed/foundation` route pattern.
- **Mission 13** (Tier-2 utility validation): Confirmed native-200-D store
  provides value V2-32-D cannot.
- **Mission 14** (Phase-1 GA readiness): All 7 mandatory gates PASS; 4
  production-operational gates INCONCLUSIVE (browser/WASM, signed-artifact,
  live-Supabase rate-limit, real-JWT API).
- **Mission 15** (Phase-2 production validation): **READY_FOR_OPT_IN.** All 4
  previously-INCONCLUSIVE gates now **PASS** against real local Supabase stack:
  - Real JWT auth (8/8), rate limiting (5/5), concurrency ramp (7/7),
    artifact SHA serving-path (5/5), API contract (16/16).
  - 720 tests passed, 6 pre-existing failures (Playwright version conflict,
    engine-lifecycle API mismatch), 6 skipped.
- **Evidence sources:** `reports/MISSION10_CBRAMOD_PRODUCT_ROLE_DECISION.md`,
  `reports/MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md`,
  `reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.md`,
  `reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.md`,
  `reports/MISSION14_PHASE1_GA_READINESS.md`,
  `reports/MISSION15_COMPLETION_REPORT.md`.

### 2.3 Missions 16–20 — Joint Embedding & Robustness

- **Mission 16** (linear probe): CBraMod-200 does NOT significantly outperform
  PCA or V2 on linear-probe MI. MI-accuracy is the wrong gate for a
  representation model.
- **Mission 17** (learned metric): LDA improves CBraMod retrieval but does
  **not** beat PCA. LDA not adopted.
- **Mission 18** (learned joint embedding): Block-weighted 264-D concat
  `[CBraMod_200 × 0.62 ⊕ V2_32 × 0.16 ⊕ PCA_32 × 0.22]` is **BEST**
  embedding: **R@5 = 0.7856**, p = 4.5e-9. Joint embedding wins.
- **Mission 19** (dimension-wise weighting): Not significant vs block weighting
  (p=0.157). Block weighting retained.
- **Mission 20** (robustness): M18 independently reproduced — **PASS**.
- **Evidence sources:** `reports/MISSION16_LINEAR_PROBE_REPORT.md`,
  `reports/MISSION17_LEARNED_METRIC_REPORT.md`,
  `reports/MISSION18_LEARNED_JOINT_EMBEDDING_REPORT.md`,
  `reports/MISSION19_DIMENSIONWISE_EMBEDDING_REPORT.md`,
  `reports/MISSION20_EMBEDDING_ROBUSTNESS_REPORT.md`.

### 2.4 Mission 21 — V2 Firefox WASM Latency (2026-08-17)

- **Root cause:** Per-call `InferenceSession.create` (fetch 3.3 MB ONNX +
  `WebAssembly.compile` of 13.5 MB threaded WASM + worker init) on every
  `embedEEG()` call = ~1589 ms P95 Firefox.
- **Track A — INT8-QDQ quantization:** `quantize_dynamic(QInt8,
  per_channel=False)` → 1.14 MB (66.1% smaller), cos 0.9985. **FAILS** as
  latency lever: Firefox per-call INT8 P95 = **2154 ms** (slower than FP32
  per-call 1590 ms). Q/DQ op overhead on WASM exceeds size savings.
- **Track B — Persistent session reuse:** Route `embedEEG()` through
  `InferenceEngine` (LRU cache maxLoaded=2, pending-promise dedup, per-model
  async mutex `withLock`). **CLEARS** the gate:
  - Firefox FP32 persistent: **P50 = 108.6 ms, P95 = 161.9 ms** (was 1589 ms).
  - Chromium FP32 persistent: **P50 = 19.7 ms, P95 = 35.8 ms** (was 1469 ms).
  - Determinism: cos(runA, runB) = **0.9999999999999998**.
  - Concurrency: 8 concurrent → `sessions_created: 1`, `concCache = 1`.
- **numThreads tuning:** numThreads > 1 is **negative** (Firefox 164 ms vs
  131 ms at 1 thread). Thread-pool spin-up outweighs parallelism for 3.3 MB
  model. Default 1 retained.
- **Production wiring applied to engine.ts + embed-eeg.ts** (P3 production
  wiring). Canonical FP33 artifact untouched.
- **Evidence source:** `reports/MISSION21_V2_FIREFOX_WASM_LATENCY_REPORT.md`,
  `reports/v3-persistent-production-report.md`,
  `reports/v2-int8-vs-persistent-report.md`.

### 2.5 Tier Completion Reports

| Tier | Report | Status | Key Numbers |
|---|---|---|---|
| TIER-2 | `reports/TIER_2_COMPLETION_REPORT.md` | ✅ COMPLETE | Cognitive decoder bug fixed (non-tensor output → silent PCA fallback); real-inference integration tests added |
| TIER-3 | `reports/TIER_3_COMPLETION_REPORT.md` | ✅ GO | 504 tests pass; stats module (t-test, Cohen's d, CIs); LOSO; ground-truth; benchmark comparison |
| TIER-4 | `reports/TIER_4_FINAL_SCIENTIFIC_VALIDATION.md` | ✅ VALIDATED | 83/83 tests pass across 5 models; SHA-256 integrity; real ONNX inference; no fallback; CBraMod WASM-block confirmed |

### 2.6 Benchmark Archive

- **File:** `reports/benchmark_archive.json` + `reports/BENCHMARK_ARCHIVE.md`
- **16 experiments** (tier4-original through t031-50subj-all) + M21 append + M15 append.
- **7 fine-tuning experiments**, **9 bugs/corrections**, **11 model artifacts**,
  **17 preserved artifacts**.
- All records append-only; prior records byte-for-byte preserved.

### 2.7 Technical Debt Completion

| TD | Status | Report |
|---|---|---|
| TD-001 (CI/CD) | ✅ COMPLETED | `reports/P1_COMPLETION_REPORT.md` |
| TD-002 (Rate limiting & upload) | ✅ COMPLETED | `reports/P1_COMPLETION_REPORT.md` |
| TD-003 (WASM self-hosting) | ✅ COMPLETED | `reports/P1_COMPLETION_REPORT.md` |
| TD-004 (Artifact bundling) | ❌ DEFERRED | — |
| TD-005 (FFT) | ✅ COMPLETED | `reports/P2_TD-005_COMPLETION_REPORT.md` |
| TD-006 (pgvector migration) | ✅ COMPLETED | `reports/P2_TD-006_COMPLETION_REPORT.md` |
| TD-007 (Cognitive decoder) | ✅ COMPLETED | `reports/TIER_2_COMPLETION_REPORT.md` |
| TD-008 (Metrics & alerting) | ⚠️ PARTIAL | Foundation metrics registered; full observability stack TBD |
| TD-009 (Dataset loaders) | ❌ NOT STARTED | Only synthetic + sleep-edf loader |
| TD-010 (Input validation) | ✅ COMPLETED | Folded into TD-002 upload hardening |
| TD-011 (Acquisition coupling) | ❌ NOT STARTED | — |
| TD-012 (Vector scalability/ANN) | ⚠️ PARTIAL | pgvector ivfflat exists; HNSW/Qdrant TBD |
| TD-013 (Doc cleanup) | ❌ NOT STARTED | — |
| TD-014 (Test coverage) | ⚠️ PARTIAL | Coverage improved; 720 tests pass |
| TD-015 (Contributing guide) | ❌ NOT STARTED | — |

### 2.8 Audit & Maturity

- **AUDIT_2026.md** (2026-08-01): **63/100**, Research Platform.
  - AI 65%, Backend 75%, Frontend 80%, Infrastructure 50%, Security 70%,
    Testing 40%, Documentation 85%, Research 60%.
- Score stable since June (63→63). Improvements in CI/CD, rate limiting,
  WASM self-hosting, pgvector offset weaknesses in Infrastructure (50%) and
  Testing (40%). Promotion decision is the binding constraint for score
  movement.

---

## Phase 3 — Current Technical State Snapshot

### 3.1 Model Portfolio

| Model | ID | Dim | Channels | Sample Rate | WASM? | SHA (prefix) | Size | Role | Status |
|---|---|---|---|---|---|---|---|---|---|
| **EEGConformer V2** | `braindecode-eegconformer-prod-v2` | 32 | 22 | 250 Hz | ✅ self-contained | `18644de1…` | 3.36 MB | Browser/interactive backbone (opt-in) | ✅ Validated, gate cleared |
| **EEGConformer V1** | `braindecode-eegconformer-prod` | 32 | 22 | 250 Hz | ⚠️ external-data | `31cd3651…` (+ext `892b5a77…`) | 3.0+3.2 MB | `DEFAULT_PREFERRED` (GA); rollback only | ✅ Live, v1 |
| **CBraMod-200** | `onnx-cbramod` | 200 (mean-tokens) | 19 | 250 Hz | ❌ DFT, ReduceL2 | `c128ccfd…` | 22.0 MB | Server-side subject-identity specialist (opt-in) | ✅ M15 READY_FOR_OPT_IN |
| **EEGPT** | `onnx-eegpt` | 2048 (mean-tokens) | 62 | 250 Hz | ✅ INT8 | `a92daf44…` | 24.9 MB | Experimental; server 2048-D backbone (blocked on 62→22 remap) | ⚠️ Blocked |
| **LaBraM** | `onnx-labram` | 200 | 16 | 250 Hz | ✅ | `61f28d12…` | 22.2 MB | Below PCA (0.2533); drop | ❌ Dropped |
| **FEMBA-tiny** | `onnx-femba-tiny` | 30800 (raw) / 385 (pooled) | 22 | 200 Hz | ✅ | `e0242279…` | 30.7 MB | Contract bug + below PCA; drop | ❌ Dropped |
| **Cognitive decoder** | `cognitive-decoder-v0` | 3 | bandpower 110 | — | ✅ | `ea4f216c…` | 1.5 KB | 3-class (att/work/ars); real model (TIER-2 fixed) | ✅ Functional |
| **PCA** | `pca-legacy-v1` | 32 | 22 | 250 Hz | ✅ (pure JS) | (code) | ~0 | Terminal fallback | ✅ Production |
| EEGNet/Shallow/Deep4 | stubs | — | — | — | — | — | — | No artifacts; ablation only | ⚠️ Stubs |
| PyTorch export | `pytorch-export-placeholder` | — | — | — | — | — | — | `implemented:false` | ❌ Stub |

**Key production state:**
- `DEFAULT_PREFERRED` = `braindecode-eegconformer-prod` (v1).
- V2 registered as `braindecode-eegconformer-prod-v2` at rollout `off` (opt-in).
- CBraMod registered but `wasmCompatible:false` — server-side only, wired to
  `foundation_embeddings` (`vector(200)`) table via `/api/eeg/embed/foundation`.
- PCA is the JS fallback (runs everywhere, ~0 ms).
- Joint embedding: `[CBraMod×0.62 ⊕ V2×0.16 ⊕ PCA×0.22]` = 264-D, R@5=0.7856.

### 3.2 Inference Architecture

```
embedEEG()  (embed-eeg.ts)
   │
   ├── if enabled && hasModel(preferred) && preferred ≠ DEFAULT_EMBEDDER_ID:
   │     → inferenceEngine.embed(preferred, input)   // locked, cached, reused
   │     → finalize(out, ...)                         // validate + L2-normalize
   │     → on failure: disposeModel(preferred) → per-call facade → SHA re-verify → fallbackChain → PCA
   │
   └── InferenceEngine (engine.ts):
        ├── LRU cache (maxLoaded=2) of loaded adapters
        ├── pending promise map (concurrent first-load dedup)
        ├── per-model async mutex (withLock) for non-reentrant ORT-Web WASM session.run()
        └── SHA-256 verifyRemoteArtifact (once per session bootstrap)
```

**Performance (P3 persistent session, numThreads=1):**

| Browser | P50 | P95 | Determinism | concCache | Gate |
|---|---|---|---|---|---|
| Chromium 151 | 19.7 ms | 35.8 ms | 1.0000000 | 1 | ✅ < 600 ms |
| Firefox 153 | 108.6 ms | 161.9 ms | 1.0000000 | 1 | ✅ < 600 ms |

**INT8-QDQ (experimental, NOT adopted):**
- 66.1% smaller (3.36 MB → 1.14 MB), cos 0.9985.
- Per-call Firefox P95 = 2154 ms (slower than FP32 1590 ms).
- Persistent INT8 Firefox P95 = 395 ms (3× slower than FP32 persistent 131 ms).
- **Verdict:** NOT the right lever. Remains in `public/models/_bench/`.

### 3.3 Data & Training

- **Fine-tuning corpus:** PhysioNet EEGMMIDB S001–S050 (4-class MI, runs 5/6).
- **V2 FT (v2):** 40 train → 10 test (strictly held-out), 0.327 vs 0.280 (p=0.143,
  n.s. at 10-subj). At 50-subj all-LOSO: 0.343 vs 0.283 (p=0.0002, d=0.701).
- **V3 FT (30-subj):** 0.310 vs 0.272 (p=0.097, n.s.). Not deployed.
- **T-034 contrastive FT:** `t034_contrastive` 0.274, `t034_aug_contrastive` 0.259 —
  both lost to V2 (0.325). Not re-run.
- **T-033:** 32-D dimension is not the accuracy bottleneck; gains are data/objective-limited.
- **CBraMod remap (Mission 6):** 0.304 vs V2 0.325, p=0.353 — negative for MI Recall@K.
- PCA baseline: 110 bandpower features (5 bands × 22 channels), per-fold fit,
  train-only, seed=42, 32 components.

### 3.4 Testing & CI

| Suite | Command | Result |
|---|---|---|
| Node AI suite | `npx vitest run src/lib/ai` | **226 passed** (28 files) |
| Full suite | (vitest + Playwright) | **720 passed**, 6 pre-existing failures, 6 skipped |
| Wasm-smoke (Chromium + Firefox) | `tests/browser/wasm-smoke.test.ts` | **14 passed** |
| V2 Firefox latency gate | `tests/browser/v2-firefox-latency-gate.test.ts` | **6 passed** |
| V3 persistent production | `tests/browser/v3-persistent-production.test.ts` | **6 passed** |
| V2 INT8-vs-persistent | `tests/browser/v2-int8-vs-persistent-session.test.ts` | **passed** |
| TIER-4 validation | — | **83/83 passed** |
| TIER-3 validation | `bunx vitest run` | **504 passed** (62 new), 0 failures |

**CI (`.github/workflows/ci.yml`):** Typecheck, ESLint, `test:coverage`,
production build, nightly SLO, security scanning (dependency audit + secret
scanning), Supabase migration validation, Recall@10 SLO gate.

### 3.5 Infrastructure

| Layer | Status | Notes |
|---|---|---|
| **CI/CD** | ✅ Done | GitHub Actions on every PR + nightly |
| **Rate limiting** | ✅ Done | PostgreSQL-backed `check_rate_limit`, atomic UPSERT, per-user isolation |
| **Upload validation** | ✅ Done | 50 MB cap, magic-number check, MIME validation, file-extension allowlist |
| **WASM delivery** | ✅ Done | Self-hosted `/ort/` (13.5 MB threaded SIMD), SHA-384 integrity, COOP/COEP for SharedArrayBuffer |
| **Vector store** | ✅ Done | pgvector `vector(32)` (interactive) + `foundation_embeddings` `vector(200)` (CBraMod server) |
| **Auth** | ✅ Done | Supabase Auth (GoTrue) + RLS policies; real JWT on M15 |
| **Feature extraction** | ✅ Done | FFT O(M log M) replacing DFT (TD-005) |
| **Observability** | ⚠️ Partial | `foundationRequestsTotal/ErrorsTotal/BytesTotal/EmbedMs` registered; Prometheus endpoint + tracing + SLO alerting deferred (TD-008) |
| **Dataset loaders** | ❌ Not started | Only synthetic + sleep-edf loader exist (TD-009) |

### 3.6 Metrics Snapshot

| Metric | Value | Source |
|---|---|---|
| Technical maturity | 63/100 (Research Platform) | `AUDIT_2026.md`, `TECHNICAL_MATURITY_REPORT.md` |
| AI layer score | 65/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Backend score | 75/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Frontend score | 80/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Infrastructure score | 50/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Security score | 70/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Testing score | 40/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Documentation score | 85/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Research score | 60/100 | `TECHNICAL_MATURITY_REPORT.md` |
| Test pass rate | 720 passed, 6 failures | Full suite |
| V2 Firefox latency gate | 161.9 ms P95 (< 600 ms) | M21/P3 |
| V2 determinism | 0.9999999999999998 | M21/P3 |
| CBraMod subject R@5 | +0.3116 vs V2 (p=1.66e-59) | Mission 11 |
| Joint embedding R@5 | 0.7856 (p=4.5e-9) | Mission 18 |

---

## Phase 4 — Reconciled Roadmap (P0/P1/P2/P3)

> **Priority definitions:**
> - **P0:** Blocking GA / production promotion / user-facing release. Must resolve
>   before V2-to-GA flip.
> - **P1:** Required for MVP readiness or opens a server-side capability V2
>   cannot provide. Must complete within 60 days.
> - **P2:** Important for platform maturity or model-portfolio expansion. Complete
>   within 6 months.
> - **P3:** Nice-to-have, quality-of-life, or low-ROI. Complete opportunistically.

### 4.1 Priority Definitions

| Priority | Meaning | Examples |
|---|---|---|
| **P0** | Blocks GA promotion or user-facing release | V2 rollout decision, production hardening blockers |
| **P1** | Required for MVP or opens server-side capability V2 cannot provide | CBraMod native 200-D store, EEGPT remap, metrics/observability |
| **P2** | Improves platform maturity or model diversity | Cross-domain validation, ablation suite, documentation |
| **P3** | Nice-to-have / low-ROI / quality-of-life | Acquisition refactors, notebook portal, contributing guide |

### 4.2 Reconciled Task Table

| ID | Task | Priority | Status | Why Now | Dependencies | Evidence | Outcome |
|---|---|---|---|---|---|---|---|
| **R-01** | **V2 GA promotion decision** (flip `rollout`→`active`, set `DEFAULT_PREFERRED`→V2) | **P0** | ⏳ Ready but not promoted | Firefox gate cleared (161.9 ms); all infrastructure done; M15 READY_FOR_OPT_IN. Promotion is a product/business decision, not engineering. | All P1 latency + infra complete | M21, P3 report, M15, TIER-4 | ✅ Gate cleared; pending promotion decision |
| **R-02** | **CBraMod native 200-D server representation validation** (M9/P0 experiment) | **P1** | ⏳ Operational ready (M15); representation experiment pending | M11 proves subject-identity wins (R@5 +0.312); M18 proves joint embedding wins (R@5=0.7856). Need native-200-D Fisher/silhouette subject-Recall@K vs V2 to formally earn server-specialist role. | M15 infrastructure done; PhysioNet S001–S050 available | MISSION11, MISSION18, MODEL_ROLE_EXPERIMENT_PLAN §7.1 | ⚠️ Infra ready; experiment not yet run |
| **R-03** | **CBraMod 264-D joint embedding productionization** | **P1** | ⏳ Validated (M18/M20); productionization pending | Joint `[CBraMod×0.62 ⊕ V2×0.16 ⊕ PCA×0.22]` is BEST embedding (R@5=0.7856). Needs `vector(264)` native store + `/api/eeg/embed/foundation?model=joint` route. | R-02 (CBraMod store) | MISSION18, MISSION20 | ✅ Reproduced; store/route not built |
| **R-04** | **EEGPT 62→22 remap viability study** (P1 from Mission 8) | **P1** | ⏳ Not started | EEGPT's only plausible role is server 2048-D backbone; blocked on whether dropping 40/62 channels destroys the signal. cos(62→22) ≥ 0.90 + 22-ch acc ≥ V2 required. | None (eval-only, no training) | MODEL_ROLE_EXPERIMENT_PLAN §7.2, MODEL_STRATEGY §3 | ❌ Blocked — remap feasibility unknown |
| **R-05** | **Full observability stack** (Prometheus endpoint + tracing + SLO alerting) | **P1** | ⚠️ Partial | Foundation metrics registered (TD-008), but no Prometheus endpoint, distributed tracing, or SLO alerting for Recall@10 / latency. Testing score is 40/100 — observability is the gap. | None | AUDIT_2026.md (Testing 40/100), TIER-3 | Foundation metrics registered; full stack pending |
| **R-06** | **V2 cross-domain / cross-dataset validation** (BCI-IV-2a + non-MI task) | **P2** | ⏳ Not started | V2 validated on PhysioNet MI only. Model is MI-specific; accuracy ceiling likely data/object-limited (T-033), not dimension-limited. Must prove generalization before claiming robustness. | R-01 (if promotion proceeds) | MODEL_STRATEGY §1, T-033 reasoning | ❌ Not validated |
| **R-07** | **EEGNet/Shallow/Deep4 ablation suite** (ONNX export + 50-subj eval) | **P2** | ⚠️ Partial (stubs registered, no artifacts) | Architecture selection needs comparison. EEGNetv4 could become a "fast small backbone" if competitive. Cheap eval, high information. | T-015 export script exists | MODEL_STRATEGY §6, BENCHMARK_ARCHIVE.md | Stub registered; not exported/evaluated |
| **R-08** | **Dataset manifest + loader maturity** (Sleep-EDF, CHB-MIT, TUH) | **P2** | ⚠️ Partial (sleep-edf loader only) | TD-009; reproducibility requires dataset metadata table (name, license, sha256, source). Without real loaders, validation is limited to PhysioNet. | None | TECHNICAL_DEBT TD-009 | Sleep-EDF loader only |
| **R-09** | **Content-hashed ONNX artifact in object storage** (TD-004) | **P2** | ⚠️ Partial (self-hosted WASM done; artifacts still in `public/models/`) | TD-004. `public/models/` bundling increases cold load; no content-hash URLs for cache busting / integrity. | TD-003 (done) | TECHNICAL_DEBT TD-004 | WASM self-hosted; artifact storage pending |
| **R-10** | **Concept graph schema** (T-013) | **P2** | ⏳ Not started | "Neuro-Fabric" provenance (subject→session→window→embedding→label) enables audit queries + longitudinal studies. pgvector migration done; graph schema is next. | TD-006 (done) | Blueprint §4.2, TECHNICAL_MATURITY_REPORT | Not started |
| **R-11** | **Subject-level embedding aggregation** (T-014) | **P2** | ⚠️ Partial (per-window only) | Per-subject signatures with stability metrics are needed for cohort matching (M11 subject identity). Currently only per-window 32-D vectors exist. | R-02 | MODEL_ROLE_EXPERIMENT_PLAN §6.1 | Per-window only |
| **R-12** | **Trained cognitive decoder v1** (decouple from backbone) | **P2** | ✅ Done (v0 via TIER-2) | TIER-2 fixed the v0 bug (non-tensor output → silent PCA fallback). Real 1.5 KB model now functional. v1 = cross-validate + individual calibration. | TIER-2 complete | TIER_2_COMPLETION_REPORT, MODEL_STRATEGY §7 | ✅ v0 functional; v1 validation pending |
| **R-13** | **WebGPU execution provider feature flag** (T-024) | **P2** | ⚠️ Partial (no flag) | 5–20× browser speedup on supported browsers. Latency gate cleared via session reuse; WebGPU is an optimization, not a blocker. | T-008 (done) | MODEL_ROLE_EXPERIMENT_PLAN §7.3b, Blueprint | Not started |
| **R-14** | **ONNX simplifier + optimizer in export pipeline** (T-023) | **P3** | ⚠️ Partial (V2 already stripped of Einsum) | Smaller graphs + faster cold start. V2 re-export already done (T-035). Future exports should auto-simplify. | T-015 | MODEL_STRATEGY §1 | V2 done; pipeline not automated |
| **R-15** | **MOABB-driven evaluation harness** (T-017) | **P3** | ❌ Not started | Current benchmarking is in Python (`benchmark_tier4.py`); a MOABB-driven harness would standardize evaluation across BCI-IV-2a, BCI-IV-2b, PhysioNetMI. | R-07 | Blueprint §4.2 | Not started |
| **R-16** | **Captum saliency on /embeddings route** (T-018) | **P3** | ❌ Not started | Investor + scientific story; turns embedding from black box. No reconstruction product defined yet. | R-10 | Blueprint §4.3 | Not started |
| **R-17** | **Acquisition source refactoring** (T-001, T-005, T-003, T-004) | **P3** | ❌ Not started | BrainFlow + LSL + WebSocket gateway for live EEG. Live decoding requires live sources. | None | TECHNICAL_DEBT TD-011 | Not started |
| **R-18** | **EEGNet/Deep4 parser hardening** (T-002) | **P3** | ❌ Not started | EDF+/BDF/GDF support without Python round-trip. Gates BCI-IV-2a evaluation. | None | Blueprint §4.1 | Not started |
| **R-19** | **Pyodide-MNE preprocessing parity harness** (T-006) | **P3** | ❌ Not started | Golden-file tests comparing TS filters vs MNE. Closes long-standing parity question. | T-002 | Blueprint §4.2 | Not started |
| **R-20** | **Training container + MLflow tracking** (T-020, T-021) | **P3** | ⚠️ Partial (Dockerfile exists) | Reproducible training + experiment tracking. Required for any future FT. | R-08 | Blueprint §4.2 | Dockerfile exists; MLflow not wired |
| **R-21** | **Registry↔MLflow↔Storage three-way sync** (T-022) | **P3** | ❌ Not started | TS registry should pull from MLflow + verify Storage hash. Closes training↔serving loop. | R-20 | Blueprint §4.2 | Not started |
| **R-22** | **Subject recall@10 SLO harness** (T-012) | P3 → **P2 (post-GA)** | ⚠️ Partial (CI references SLO) | Nightly recall@10 sampling → alert on regression. Currently a CI gate, not a nightly SLO harness. | R-09, R-11 | Blueprint §4.1 | CI gate exists; nightly harness pending |
| **R-23** | **Documentation cleanup / deprecation** (TD-013) | **P3** | ❌ Not started | Historical docs cause confusion; deprecation banners needed. | None | TECHNICAL_DEBT TD-013 | Not started |
| **R-24** | **Test coverage expansion** (TD-014) | **P3** | ⚠️ Partial | EEG processing + acquisition modules minimally tested. Coverage improved to 720 passing but gaps remain. | None | TECHNICAL_DEBT TD-014 | Improved; gaps remain |
| **R-25** | **Contributing guidelines** (TD-015) | **P3** | ❌ Not started | No `CONTRIBUTING.md`; slows external contributions. | None | TECHNICAL_DEBT TD-015 | Not started |

### 4.3 Priority Summary

| Priority | Count | Key Items |
|---|---|---|
| **P0** | 1 | R-01: V2 GA promotion decision |
| **P1** | 5 | R-02: CBraMod native 200-D validation; R-03: Joint embedding productionization; R-04: EEGPT remap; R-05: Observability stack |
| **P2** | 9 | R-06–R-13: Cross-domain validation, ablation suite, dataset loaders, content-hashed storage, concept graph, subject aggregation, cognitive decoder v1, WebGPU, ONNX optimizer |
| **P3** | 12 | R-14–R-25: MOABB harness, Captum, acquisition refactoring, parser hardening, MNE parity, training container, registry sync, SLO harness, docs, test coverage, contributing guide |

---

## Phase 5 — What Is Materially Better + DO NOT REPEAT

### 5.1 What Is Materially Better (This Cycle)

| # | Improvement | Before | After | Evidence |
|---|---|---|---|---|
| 1 | **Browser V2 latency** | Firefox P95 ≈ 1589 ms (FAIL) | Firefox P95 = **161.9 ms** (✅ PASS, ~9.8×) | M21, v3-persistent |
| 2 | **Browser V2 latency (Chromium)** | Chromium P95 ≈ 1469 ms | Chromium P95 = **35.8 ms** (~41×) | M21, v3-persistent |
| 3 | **Inference session management** | Per-call `createAdapter→load→unload` | Persistent LRU cache + async mutex | engine.ts, embed-eeg.ts |
| 4 | **CBraMod cross-session identity** | Subject identity unproven | R@5 +0.312 (p=1.66e-59) vs V2; +0.322 vs PCA | MISSION11 |
| 5 | **Joint 264-D embedding** | Single-model 32-D | Block-weighted concat R@5 = **0.7856** (p=4.5e-9) | MISSION18, MISSION20 |
| 6 | **CBraMod operational readiness** | Tier-2 gates INCONCLUSIVE (M14) | **READY_FOR_OPT_IN** (M15; all 4 gates PASS) | MISSION15 |
| 7 | **WASM self-hosting** | jsDelivr CDN (SPOF) | Self-hosted `/ort/` (13.5 MB threaded SIMD) + SHA-384 + COOP/COEP | TD-003, P1 completion |
| 8 | **Feature extraction** | O(M²) DFT | O(M log M) FFT (2–5× faster) | TD-005 |
| 9 | **Vector persistence** | In-memory (lost on reload) | pgvector `vector(32)` + `vector(200)` | TD-006, M12 |
| 10 | **CI/CD** | None | GitHub Actions (typecheck, lint, test, build, nightly SLO) | TD-001, P1 completion |
| 11 | **Rate limiting** | None | PostgreSQL atomic UPSERT, per-user | TD-002, P1 completion |
| 12 | **Upload validation** | None | 50 MB cap, magic-number, MIME validation | TD-002, TD-010 |
| 13 | **Cognitive decoder** | 1,333-byte stub | Real 1.5 KB skl2onnx model (bug fixed) | TIER-2 |
| 14 | **Statistical infrastructure** | None | Stats module (t-test, Cohen's d, CIs), LOSO, ground-truth, benchmarks | TIER-3 (504 tests, GO) |
| 15 | **Scientific validation** | Inconclusive (synthetic probes) | Tier-4 validated (83/83 pass) | TIER-4 |
| 16 | **Model portfolio rationalization** | 6 live models, unclear roles | V2 KEPT (browser), PCA KEPT (fallback), CBraMod SERVER SPECIALIST, EEGPT BLOCKED, LaBraM/FEMBA DROPPED | MODEL_STRATEGY, MODEL_ROLE_EXPERIMENT_PLAN |
| 17 | **SHA-256 artifact verification** | None | `verifyRemoteArtifact` runs once per session bootstrap (not per call) | engine.ts |
| 18 | **Concurrency safety** | Shared session → silent throws → PCA fallback | Per-model async mutex (`withLock`); 8 concurrent → `concCache=1` | v3-persistent |

### 5.2 DO NOT REPEAT (Learned Negative Results)

| # | Finding | Why It Fails | Evidence | Re-run? |
|---|---|---|---|---|
| **DN-01** | **INT8-QDQ quantization as latency lever** | Q/DQ op overhead on WASM dominates size savings for 3.3 MB model; per-call INT8 (2154 ms) > per-call FP32 (1590 ms); persistent INT8 (395 ms) > persistent FP32 (131 ms) | M21, v2-int8-vs-persistent | ❌ No |
| **DN-02** | **numThreads > 1 for V2** | Thread-pool spin-up exceeds parallelism gain for small model; Firefox 164 ms (4 threads) vs 131 ms (1 thread) | M21, v3-persistent §7 | ❌ No |
| **DN-03** | **Contrastive / augmentation FT of V2** | T-034: `t034_contrastive` 0.274, `t034_aug_contrastive` 0.259 — both < V2 0.325. v2 32-D is not the bottleneck; gains are data/objective-limited. | BENCHMARK_ARCHIVE.md §5 | ❌ No |
| **DN-04** | **CBraMod as 32-D MI Recall@K replacer** | M6: 0.304 vs V2 0.325, p=0.353 (negative). CBraMod's value is 200-D representation geometry, not 32-D MI accuracy. | CBRAMOD_REMAP_50SUBJ_REPORT | ❌ No |
| **DN-05** | **Per-call InferenceSession.create** | Root cause of Firefox latency failure (~1589 ms/call). Session reuse is the fix, not model shrinking. | M21 root cause | ❌ No |
| **DN-06** | **Re-running CBraMod 19→22 channel remap for MI accuracy** | Model is server-only (`wasmCompatible:false`); MI accuracy is the wrong gate. The remap question is already answered by M6 (negative for MI). Do not re-invest in MI comparison. | MODEL_ROLE_EXPERIMENT_PLAN §5 | ❌ No |
| **DN-07** | **Dimension-wise embedding weighting** | M19: not significant vs block weighting (p=0.157). Block weighting retained. | MISSION19 | ❌ No |
| **DN-08** | **LDA as a standalone metric** | M17: improves CBraMod but doesn't beat PCA. Not adopted as a production lever. | MISSION17 | ❌ No |
| **DN-09** | **Per-call SHA re-verification** | Pre-P3, `embed()` re-ran `verifyRemoteArtifact` on every call. Post-P3, SHA verified once per session bootstrap. | v3-persistent §4 | ❌ No |
| **DN-10** | **`const embed = adapter.embed` detached `this`** | First locked run: forwards threw silently → PCA fallback → Firefox P95 ≈ 3458 ms. Fixed by `adapter.embed!(input)`. | v3-persistent §8 | ❌ No (fixed) |

---

## Phase 6 — Next 6 Missions

> These are **proposed** missions for the next planning cycle. No work has begun.
> Each includes: objective, why now, inputs, deliverables, success gate,
> constraint compliance.

| # | Mission | Priority | Objective |
|---|---|---|---|
| **M23** | V2 GA Promotion | **P0** | Flip `rollout` from `off` → `active` and set `DEFAULT_PREFERRED` from v1 → v2. Clear the final gate to production GA. |
| **M24** | CBraMod Native 200-D Server Representation | **P1** | Run the M9/P0 experiment: native 200-D Fisher/silhouette + subject-Recall@K vs V2-32-D on 50-subject LOSO. Earn (or fail) the server-specialist role. |
| **M25** | CBraMod Joint 264-D Embedding Productionization | **P1** | Build the `vector(264)` native store + `/api/eeg/embed/foundation?model=joint` route. Productize M18's best embedding. |
| **M26** | EEGPT 62→22 Remap Viability | **P1** | Test whether EEGPT's 2048-D ViT survives dropping 40/62 channels. Open (or close) the EEGPT server backbone path. |
| **M27** | Observability Stack Completion | **P1** | Prometheus endpoint + distributed tracing + SLO alerting for Recall@10 + latency + fallback rate. Close Testing 40/100 gap. |
| **M28** | V2 Cross-Domain Generalization | **P2** | Validate V2 on BCI-IV-2a held-out + a non-MI task. Prove the MI accuracy ceiling is data-limited, not architecture-limited. |

### Mission 23 — V2 GA Promotion (P0)

| Field | Detail |
|---|---|
| **Why now** | Firefox gate cleared (161.9 ms < 600 ms); all infrastructure done (CI, rate-limit, pgvector, WASM self-hosting); M15 READY_FOR_OPT_IN; 720 tests pass. The engineering is complete; only the product decision remains. |
| **Inputs** | `DEFAULT_PREFERRED` in registry; `rollout` stage in rollout config; `manifest.json` SHA verification; existing v1 rollout `active`/`DEFAULT_PREFERRED` values |
| **Deliverables** | (1) Flip `rollout: off → active` for `braindecode-eegconformer-prod-v2`; (2) set `DEFAULT_PREFERRED` from `braindecode-eegconformer-prod` (v1) → `braindecode-eegconformer-prod-v2`; (3) update model card marking v1 as rollback-only; (4) CI gate: V2 smoke test on both browsers |
| **Success** | V2 becomes the default `embedEEG()` path; Firefox P95 < 600 ms in production traffic; fallback rate < 0.5%; SHA-256 verification still passes; no regression in 720-test suite |
| **Constraint compliance** | No model retraining; no artifact modification (V2 SHA `18644de1…` unchanged); no INT8 promotion |
| **Risk** | Representation collapse (intra≈inter cos 0.907/0.904) means V2's MI accuracy ceiling is low — this is a **known limitation** to communicate to users, not a blocker |

### Mission 24 — CBraMod Native 200-D Server Representation (P1)

| Field | Detail |
|---|---|
| **Why now** | M11 proved CBraMod wins subject-identity retrieval (R@5 +0.312, p=1.66e-59); M18 proved joint embedding wins (R@5=0.7856). The M15 infrastructure is ready (real JWT, pgvector, ONNX). The representation experiment (M9/P0) has not been run. |
| **Inputs** | PhysioNet EEGMMIDB S001–S050; CBraMod ONNX (SHA `c128ccfd…`); V2 ONNX (SHA `18644de1…`); PCA bandpower; `foundation_embeddings` `vector(200)` table; `scripts/t032-embedding-quality.py` (LOSO 50-fold harness) |
| **Deliverables** | (1) Run CBraMod 200-D native Fisher/silhouette + subject-Recall@K vs V2-32-D on 50-subject LOSO; (2) append to `benchmark_archive.json`; (3) write `reports/CBRAMOD_SERVER_REP_50SUBJ_REPORT.md`; (4) decision: if CBraMod acc ≥ PCA AND ≥ V2 (Bonferroni p<0.05) → authorise `/api/eeg/embed/foundation?model=cbramod`; else → DROP |
| **Success** | CBraMod 200-D separation/Recall@K ≥ V2-32-D (Δ≥0.05, Bonferroni p<0.05) AND MI nearest-centroid acc ≥ V2 (safety floor) → promotes server-specialist role; OR documented negative → CBraMod remains research-only |
| **Constraint compliance** | No V2 changes; CBraMod already registered; no retraining; server-only route (no WASM) |
| **Note** | M6 already showed CBraMod loses MI Recall@K (0.304 vs 0.325, p=0.353) — this is expected and does NOT disqualify the server role. The question is native-200-D geometry, not 32-D MI accuracy |

### Mission 25 — CBraMod Joint 264-D Embedding Productionization (P1)

| Field | Detail |
|---|---|
| **Why now** | M18/M20 proved the block-weighted joint embedding `[CBraMod×0.62 ⊕ V2×0.16 ⊕ PCA×0.22]` is the BEST embedding (R@5=0.7856, p=4.5e-9, deterministically reproduced). It must be productized. |
| **Inputs** | M18 learned weights (0.62, 0.16, 0.22); `foundation_embeddings` pattern; `eeg_analyses.embedding FLOAT8[]`; existing `embedEEG()` + `InferenceEngine` |
| **Deliverables** | (1) `vector(264)` native store (new table, partitioned by model_id); (2) `/api/eeg/embed/foundation?model=joint-264` server route; (3) SHA-256 verification for all 3 constituent models; (4) integration test: determinism cos ≈ 1.0 |
| **Success** | Joint 264-D embedding retrievable from `vector(264)` index; R@5 reproducing 0.7856; `vector(32)` interactive index untouched; no V2/GA/DEFAULT_PREFERRED change |
| **Constraint compliance** | No V2 change; no GA promotion; no rollout change; joint route is server-only, additive |

### Mission 26 — EEGPT 62→22 Remap Viability (P1)

| Field | Detail |
|---|---|
| **Why now** | EEGPT is the only remaining model with a plausible **server 2048-D backbone** role (V2's 32-D ceiling is hit; CBraMod has 200-D; EEGPT has 2048-D headroom). But its 62-channel contract vs the 22-channel pipeline is a hard blocker. |
| **Inputs** | EEGPT ONNX (SHA `a92daf44…`); PhysioNet EEGMMIDB (62-channel subset where available); BCI-IV-2a (22-channel standard); `benchmark_tier4.py` harness |
| **Deliverables** | (1) cos-sim(62-ch output, 22-ch projected output) ≥ 0.90; (2) 22-ch MI acc ≥ V2; (3) report + archive append; (4) decision |
| **Success** | Both thresholds → proceed to EEGPT 2048-D server-representation experiment; Failure → DROP EEGPT, close the thread |
| **Constraint compliance** | Eval-only, no training; no V2 change; EEGPT already registered experimental |

### Mission 27 — Observability Stack Completion (P1)

| Field | Detail |
|---|---|
| **Why now** | Testing score is 40/100 (lowest dimension). Foundation metrics (`RequestsTotal/ErrorsTotal/BytesTotal/EmbedMs`) are registered but there is no Prometheus endpoint, no distributed tracing, no SLO alerting. Recall@10 regression is a CI gate but not a nightly observability SLO. |
| **Inputs** | `src/lib/metrics/index.ts`; AUDIT_2026.md (Testing 40/100); TIER-3 stats module |
| **Deliverables** | (1) `/api/metrics` Prometheus endpoint; (2) distributed tracing on `/api/eeg/upload` + `/api/eeg/embed/foundation`; (3) SLO alerting: Recall@10 > 0.7, embed P95 < 600 ms, fallback rate < 0.5%, 429 rate per-user; (4) dashboard |
| **Success** | SLOs enforced in CI + nightly; Testing score ≥ 60/100; no alert fatigue |
| **Constraint compliance** | Additive; no production behavior change |

### Mission 28 — V2 Cross-Domain Generalization (P2)

| Field | Detail |
|---|---|
| **Why now** | V2 is validated on PhysioNet MI only. The model has a 4-class MI logits head and a 32-D representation. T-033 established 32-D is not the bottleneck — but generalization to other datasets/tasks is unproven. This de-risks claiming GA. |
| **Inputs** | V2 ONNX (SHA `18644de1…`); BCI-IV-2a (runs 4–7); a non-MI task dataset (e.g., P300 or SSVEP from MOABB); `evaluateLOSO()` (TIER-3) |
| **Deliverables** | (1) LOSO accuracy on BCI-IV-2a 4-class MI; (2) accuracy on non-MI task; (3) comparison vs PhysioNet (0.343); (4) report + archive append |
| **Success** | V2 acc ≥ 0.9× best PCA on cross-domain; if it collapses, the accuracy ceiling is data/task-specific (not architecture) → informs joint-embedding (M25) priority |
| **Constraint compliance** | Eval-only; no training; no production change |

---

## Phase 7 — Original Roadmap Classification

Every original roadmap item from the Blueprint (T-001–T-028), the EEGConformer
deployment roadmap, ROADMAP_NEXT_PHASE.md, and the TD register is classified:

| Original ID | Title | Classification | Rationale |
|---|---|---|---|
| **Blueprint tasks** |
| T-001 | Hardware-agnostic acquisition adapter | **DEFERRED (P3/R-17)** | Live EEG streaming is post-MVP; file-upload path is sufficient for GA |
| T-002 | EDF+/BDF/GDF parser hardening | **DEFERRED (P3/R-19)** | EDF/BDF already parse; GDF is niche; gates BCI-IV-2a eval but not GA |
| T-003 | WebSocket EEG gateway | **DEFERRED (P3/R-17)** | Same as T-001 — live streaming post-MVP |
| T-004 | LSL bridge | **DEFERRED (P3/R-17)** | Academic hardware integration; not needed for GA |
| T-005 | BrainFlow integration | **DEFERRED (P3/R-17)** | Same category — device abstraction post-MVP |
| T-006 | Pyodide-MNE parity harness | **DEFERRED (P3/R-19)** | Validation parity; not blocking GA |
| T-007 | FFT replacement of DFT | **COMPLETED** | TD-005 complete (M21 evidence: O(M log M) achieved) |
| T-008 | Self-hosted ORT WASM | **COMPLETED** | TD-003 complete (self-hosted `/ort/` + SHA-384 + COOP/COEP) |
| T-009 | Content-hashed ONNX artefact in storage | **DEFERRED (P2/R-09)** | Still in `public/models/`; not blocking GA but affects cold-start |
| T-010 | EEGConformer empirical validation on BCI-IV-2a | **COMPLETED (partial)** | PhysioNet 50-subj validation done (M4, T-031). BCI-IV-2a specific validation deferred to M28 (P2) |
| T-011 | pgvector migration | **COMPLETED** | TD-006 complete; `vector(32)` + `foundation_embeddings` `vector(200)` live |
| T-012 | Recall@10 SLO harness | **PARTIAL** | CI gate exists; nightly SLO harness deferred to R-22 (P3→P2 post-GA) |
| T-013 | Concept graph schema | **DEFERRED (P2/R-10)** | Provenance queries; post-GA |
| T-014 | Subject-level embedding aggregation | **PARTIAL (P2/R-11)** | Per-window only; per-subject aggregation pending |
| T-015 | Braindecode model zoo registration | **PARTIAL** | EEGConformer exported; EEGNet/Shallow/Deep4 remain stubs (R-07) |
| T-016 | EEGPT honesty (stub or remove) | **COMPLETED** | EEGPT registered experimental; role evaluated (BLOCKED on remap → R-04/M26) |
| T-017 | MOABB-driven evaluation harness | **DEFERRED (P3/R-15)** | Custom Python harness works; MOABB standardization is polish |
| T-018 | Captum saliency | **DEFERRED (P3/R-16)** | No reconstruction product defined; investor story later |
| T-019 | Dataset manifest + DVC-lite | **DEFERRED (P2/R-08)** | Reproducibility; not blocking GA |
| T-020 | Reproducible training container | **PARTIAL** | Dockerfile exists (R-20); MLflow not wired |
| T-021 | MLflow tracking server | **DEFERRED (P3/R-20)** | No future FT pending; defer until M24/M25 |
| T-022 | Registry↔MLflow↔Storage sync | **DEFERRED (P3/R-21)** | Depends on M21; no active training cycle |
| T-023 | ONNX simplifier + optimizer | **PARTIAL (P3/R-14)** | V2 already stripped (T-035); pipeline not automated |
| T-024 | WebGPU execution provider flag | **DEFERRED (P2/R-13)** | Latency gate cleared via session reuse; WebGPU is optimization |
| T-025 | Trained cognitive decoder v0 | **COMPLETED** | TIER-2 fixed the bug; real 1.5 KB model functional |
| T-026 | Notebook portal | **DEFERRED (P3)** | Research surface; not blocking GA |
| T-027 | CI security & quality gates | **COMPLETED** | TD-001; GitHub Actions live |
| T-028 | Upload hardening | **COMPLETED** | TD-002/T0010; 50 MB cap + validation |
| **EEGConformer deployment roadmap** |
| Phase 0 (Artifact) | Re-export V2 WASM-compatible | **COMPLETED** | T-035 re-export done; Einsum→MatMul |
| Phase 1 (Hosting) | Self-hosted WASM + SHA verification | **COMPLETED** | TD-003; `/ort/` + SHA-384 |
| Phase 2 (Wiring) | embedEEG → InferenceEngine | **COMPLETED** | P3 production wiring done |
| Phase 3 (Validation) | Staging latency gate | **COMPLETED** | M21 Firefox gate cleared |
| Phase 4 (Rollout) | Promote V2 to GA | **DEFERRED (P0/R-01)** | Gate cleared; promotion decision pending |
| Phase 5 (Post-launch) | Monitor + iterate | **IN PROGRESS** | Monitoring infra built; full observability pending M27 |
| **ROADMAP_NEXT_PHASE.md** |
| Phase 1 (Operational Foundation) | CI/CD, rate limit, WASM, FFT, pgvector | **COMPLETED** | TD-001, TD-002, TD-003, TD-005, TD-006 all done |
| Phase 2 (Core Capability) | Cognitive decoder, metrics, dataset loaders | ⚠️ PARTIAL | Cognitive decoder done (TIER-2); metrics partial (TD-008); dataset loaders deferred (TD-009) |
| Phase 3 (Platform Maturation) | Acquisition, vector scalability, docs, tests, contributing | **DEFERRED** | All P3 items; post-MVP |
| Phase 4 (Pilot) | Acquisition+streaming, multi-model, explainability, reconstruction | **DEFERRED** | Post-GA; depends on P0/P1 completion |
| **Technical Debt** |
| TD-001 | No CI/CD Pipeline | **COMPLETED** | P1 completion report |
| TD-002 | Missing rate limiting & upload | **COMPLETED** | P1 completion report |
| TD-003 | WASM CDN SPOF | **COMPLETED** | P1 completion report |
| TD-004 | Model artifact bundling | **PARTIAL** | WASM self-hosted done; ONNX still in `public/models/` |
| TD-005 | DFT → FFT | **COMPLETED** | P2 completion report |
| TD-006 | pgvector migration | **COMPLETED** | P2 completion report |
| TD-007 | Heuristic cognitive decoder | **COMPLETED** | TIER-2 fixed + trained model |
| TD-008 | Lack of metrics & alerting | ⚠️ PARTIAL | Foundation metrics registered; full stack M27 (P1) |
| TD-009 | Missing real EEG loaders | **DEFERRED** | R-08 (P2) |
| TD-010 | Insufficient input validation | **COMPLETED** | Folded into TD-002 upload hardening |
| TD-011 | Acquisition source coupling | **DEFERRED** | P3 (R-17) |
| TD-012 | Linear vector search (scalability) | ⚠️ PARTIAL | pgvector ivfflat done; HNSW/Qdrant scale TBD |
| TD-013 | Outdated/historical docs | **DEFERRED** | P3 (R-23) |
| TD-014 | Inconsistent test coverage | ⚠️ PARTIAL | Improved (720 pass); gaps remain (R-24, P3) |
| TD-015 | Missing contributing guidelines | **DEFERRED** | P3 (R-25) |

### 7.1 Classification Legend

| Classification | Meaning |
|---|---|
| **COMPLETED** | Task delivered and verified; evidence in reports |
| **PARTIAL** | Core done; known residual gap documented with a follow-up ID |
| **DEFERRED** | Valid future task; deprioritized to P2/P3; tracked under reconciled ID |
| **COMPLETED (partial)** | Core objective met but with a documented residual (e.g., PhysioNet done, BCI-IV-2a pending) |
| **IN PROGRESS** | Active but not yet closed |
| **SUPERSEDED** | Replaced by a different approach that already succeeded |
| **DROPPED** | Model/task evaluated and deliberately retired (LaBraM, FEMBA, stubs, v1 as product) |

### 7.2 What Was Superseded

| Original | Superseded By | Why |
|---|---|---|
| T-007's DFT→FFT | Same task — COMPLETED | No supersession; completed as specified |
| INT8-QDQ as latency fix (implicit from Mission 8 §7.3b) | Persistent session reuse (P3) | INT8 proven slower (DN-01); session reuse is the actual fix |
| Contrastive FT (T-033 recommendation) | Nothing — DROPPED | T-034 already ran it and lost; re-proposal would be DN-03 |
| CBraMod as 32-D MI replacer | CBraMod as 200-D server specialist | M6 negative for MI; M11 positive for subject identity |

---

## Appendix A — Constraint Compliance Matrix

Mission 22 constraint: "Do NOT implement code; modify production behavior;
change models; change rollout; promote anything; delete old roadmap items;
rewrite benchmark history; start new experiments."

| Constraint | Compliance | Evidence |
|---|---|---|
| No code implementation | ✅ | No source files modified (this is a `.md` report) |
| No production behavior change | ✅ | `embedEEG`, `DEFAULT_PREFERRED`, routing, rollout all unchanged |
| No model changes | ✅ | No ONNX, SHA, or registry changes |
| No rollout changes | ✅ | V2 rollout remains `off`; CBraMod remains opt-in |
| No promotion | ✅ | No GA flip; no DEFAULT_PREFERRED change |
| No old roadmap deletion | ✅ | All original roadmaps preserved under `docs/archived/roadmaps/`; classification table is additive |
| No benchmark history rewrite | ✅ | `benchmark_archive.json` append-only; prior records byte-intact |
| No new experiments | ✅ | All evidence sourced from completed Missions 4–21, Tiers 2–4 |

---

## Appendix B — Evidence Index

| Artifact | File(s) |
|---|---|
| Original blueprint (T-001–T-028, Top-25) | `docs/archived/roadmaps/2026-06-19_open_source_execution_blueprint.md` |
| EEGConformer deployment roadmap | `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md` |
| Next-phase 4-phase roadmap | `reports/ROADMAP_NEXT_PHASE.md` |
| Technical debt register | `reports/TECHNICAL_DEBT.md` |
| P1 debt completion | `reports/P1_COMPLETION_REPORT.md` |
| P2 TD-005 (FFT) completion | `reports/P2_TD-005_COMPLETION_REPORT.md` |
| P2 TD-006 (pgvector) completion | `reports/P2_TD-006_COMPLETION_REPORT.md` |
| Audit 2026 (63/100) | `reports/AUDIT_2026.md` |
| Technical maturity | `reports/TECHNICAL_MATURITY_REPORT.md` |
| Model portfolio assessment | `reports/MODEL_STRATEGY_OTHER_MODELS.md` |
| Model role × product architecture | `reports/MODEL_ROLE_PRODUCT_ARCHITECTURE.md` |
| Experiment plan (M8) | `reports/MODEL_ROLE_EXPERIMENT_PLAN.md` |
| CBraMod product role decision | `reports/MISSION10_CBRAMOD_PRODUCT_ROLE_DECISION.md` |
| CBraMod cross-session validation | `reports/MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md` |
| CBraMod Tier-2 architecture | `reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.md` |
| CBraMod Tier-2 utility validation | `reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.md` |
| CBraMod GA readiness (M14) | `reports/MISSION14_PHASE1_GA_READINESS.md` |
| CBraMod operational validation (M15) | `reports/MISSION15_COMPLETION_REPORT.md` |
| Linear probe report (M16) | `reports/MISSION16_LINEAR_PROBE_REPORT.md` |
| Learned metric report (M17) | `reports/MISSION17_LEARNED_METRIC_REPORT.md` |
| Joint embedding report (M18) | `reports/MISSION18_LEARNED_JOINT_EMBEDDING_REPORT.md` |
| Dimension-wise embedding (M19) | `reports/MISSION19_DIMENSIONWISE_EMBEDDING_REPORT.md` |
| Embedding robustness (M20) | `reports/MISSION20_EMBEDDING_ROBUSTNESS_REPORT.md` |
| V2 Firefox latency (M21) | `reports/MISSION21_V2_FIREFOX_WASM_LATENCY_REPORT.md` |
| P3 persistent production | `reports/v3-persistent-production-report.md` |
| P2 INT8-vs-persistent ablation | `reports/v2-int8-vs-persistent-report.md` |
| CBraMod remap 50-subj (M6) | `reports/CBRAMOD_REMAP_50SUBJ_REPORT.md` |
| TIER-2 completion | `reports/TIER_2_COMPLETION_REPORT.md` |
| TIER-3 completion | `reports/TIER_3_COMPLETION_REPORT.md` |
| TIER-4 final validation | `reports/TIER_4_FINAL_SCIENTIFIC_VALIDATION.md` |
| Benchmark archive | `reports/benchmark_archive.json`, `reports/BENCHMARK_ARCHIVE.md` |
| V1/V2/FT reports | `reports/T-030_FINAL_REPORT.md`, `reports/T-031_FINAL_REPORT.md`, `reports/MISSION_4_COMPLETION_REPORT.md` |
| Production inference code | `src/lib/ai/inference/engine.ts`, `src/lib/ai/inference/embed-eeg.ts` |
| Browser test suites | `tests/browser/wasm-smoke.test.ts`, `tests/browser/v2-firefox-latency-gate.test.ts`, `tests/browser/v2-int8-vs-persistent-session.test.ts`, `tests/browser/v3-persistent-production.test.ts` |
| CI/CD | `.github/workflows/ci.yml` |

---

## Appendix C — Original Roadmap Preservation Statement

All original roadmap documents remain **byte-for-byte unmodified** under:
- `docs/archived/roadmaps/2026-06-19_open_source_execution_blueprint.md`
- `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md`
- `reports/ROADMAP_NEXT_PHASE.md`
- `reports/TECHNICAL_DEBT.md`
- `reports/AUDIT_2026.md`
- `reports/TECHNICAL_MATURITY_REPORT.md`

This `MISSION22_MASTER_EXECUTION_ROADMAP.md` is an **additive** reconciliation
layer. It does not delete, overwrite, or alter any prior document. Classification
references (e.g., "T-007 → COMPLETED") point to their source files for traceability.

---

*This document is planning-only. No code, models, artifacts, or production
configuration were modified, created, or promoted during Mission 22. The
next engineering step is a **product decision** (M23: V2 GA promotion), not an
engineering implementation.*