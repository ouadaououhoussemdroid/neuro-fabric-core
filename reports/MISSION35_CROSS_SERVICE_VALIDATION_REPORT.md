# M35 — Cross-Service Validation & Tier 1 Beta

**Mission:** Validate that all three Tier-1 downstream services (Subject Identity, Cognitive State, Anomaly Detection) are correctly wired for cross-service operation on the shared Joint-2312 embedding layer.

**Date:** 2026-08-19
**Status:** ✅ PASS — 68/68 code checks, 74/74 Tier-1 tests passing

---

## 1. Objective

M31 §21.1 defines the Tier-1 critical path:
1. M31 Shared Service Layer ✅ (completed as M32)
2. Subject Identity & Cohort Similarity ✅ (completed as M32)
3. Cognitive State + Anomaly Detection (parallel) ✅ (completed as M33/M34)
4. **Cross-Service Validation** ← M35
5. Tier 1 Beta
6. Tier 1 Production Candidates

M35 validates Phase 4: that the three services can share a single Joint-2312 embedding
("Embed Once → Reuse Many") with consistent provenance, latency budgets, and no
regressions.

---

## 2. Validation Results

### 2.1 Code Checks (68/68 passed)

| Check Set | Description | Status |
|-----------|-------------|--------|
| CHECK 1 | All Tier-1 service files exist (12 files) | ✅ 12/12 |
| CHECK 2 | Shared `embedding_id` / `query_embedding` interface | ✅ 6/6 |
| CHECK 3 | Shared provenance via `buildServiceProvenance` | ✅ 3/3 |
| CHECK 4 | Joint-2312 artifact SHA consistency (manifest.json) | ✅ 3/3 |
| CHECK 5 | Task head SHA consistency (registry) | ✅ 2/2 |
| CHECK 6 | Benchmark archive records (M33, M34) | ✅ 5/5 |
| CHECK 7 | ONNX artifact verification (SHA-256) | ✅ 2/2 |
| CHECK 8 | Tier-1 metrics registration (22 metrics) | ✅ 22/22 |
| CHECK 9 | Inference index barrel exports | ✅ 3/3 |
| CHECK 10 | Cross-service integration test content | ✅ 7/7 |
| CHECK 11 | Mission reports present | ✅ 2/2 |
| CHECK 12 | Tier-1 test suite (vitest) | ✅ 1/1 |

### 2.2 Test Suite Results

| Test File | Tests | Status |
|-----------|-------|--------|
| `src/lib/ai/inference/__tests__/cross-service.test.ts` | 7 | ✅ All pass |
| `src/lib/ai/inference/__tests__/cognitive-decode.test.ts` | 12 | ✅ All pass |
| `src/lib/ai/inference/__tests__/anomaly-detect.test.ts` | 14 | ✅ All pass |
| `src/lib/ai/decoders/__tests__/registry.cognitive.test.ts` | 10 | ✅ All pass |
| `src/lib/ai/decoders/__tests__/registry.anomaly.test.ts` | 10 | ✅ All pass |
| `src/routes/api/joint2312/cognitive/__tests__/-decode.test.ts` | 11 | ✅ All pass |
| `src/routes/api/joint2312/anomaly/__tests__/-decode.test.ts` | 12 | ✅ All pass |
| **Total** | **74** | **✅ All pass** |

### 2.3 Pre-existing Test Suite (No Regressions)

Full suite: 863 passed, 33 failed (all pre-existing browser/live infra tests, unrelated
to Tier-1 services). See §4 for details on the pre-existing failures.

---

## 3. Cross-Service Integration Test

Created `src/lib/ai/inference/__tests__/cross-service.test.ts` with 7 tests validating:

1. **Shared embedding_id reuse** — All 3 services accept and reuse the same `embedding_id`
   from `joint_embeddings_2312`, reporting `embedding_reused: true`

2. **Provenance SHA consistency** — All 3 services return provenance with the same 4
   artifact SHAs (CBRaMod `c128ccfd…`, V2 `18644de1…`, PCA label, EEGPT `a92daf44…`)

3. **Dimension validation** — All 3 services reject non-2312-D embeddings with
   `DIMENSION_MISMATCH` errors

4. **Latency budget** — All 3 services complete within 500ms ceiling for the reuse path
   (mock target: <100ms with no re-embedding overhead)

5. **Shared metrics** — All 3 services increment `tier1ServiceRequestsTotal` with correct
   service labels (`subject-identity`, `cognitive-intelligence`, `anomaly-detection`)

6. **Input validation** — All 3 services throw when neither `embedding_id` nor
   `query_embedding` is provided

7. **Uniform interface** — All 3 services accept 2312-D `query_embedding` arrays identically

---

## 4. Inference Index Wiring

The inference barrel export (`src/lib/ai/inference/index.ts`) now exports all three
Tier-1 service modules:

```ts
export * from "./engine";
export * from "./embed-eeg";
export * from "./cognitive.server";
export * from "./anomaly.server";
export * from "./subject-identity.server";
```

---

## 5. Tier Architecture Summary

Based on M31 §3:

| Tier | Components | Status |
|------|-----------|--------|
| **Tier 2** (Foundation) | CBRaMod-200, V2-32, PCA-32, EEGPT-2048 → Joint-2312 (2312-D) | ✅ Complete (M28 productionized) |
| **Tier 1** (Downstream) | Subject Identity (M32), Cognitive State (M33), Anomaly Detection (M34) | ✅ Complete — validated (M35) |
| **Tier 1 → Tier 2** | Browser fallback: V2-32 projections for each service | ✅ Complete (browser.* files) |

### Embed-Once → Reuse-Many Implementation

Every Tier-1 service follows the same pattern:

```
┌─────────────────────────────┐
│  /api/eeg/embed/foundation   │  ← Computes & stores Joint-2312 ONCE
│  ?model=joint-2312           │
│  → joint_embeddings_2312     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Tier-1 Services (all reuse via embedding_id)   │
│                                                 │
│  • Subject Identity: /api/joint2312/similarity/search  │
│  • Cognitive State: /api/joint2312/cognitive/decode      │
│  • Anomaly Detection: /api/joint2312/anomaly/detect    │
│                                                 │
│  Each also accepts raw query_embedding (2312-D) │
└─────────────────────────────────────────────────┘
```

All 3 services:
- Accept `embedding_id` (reuse from `joint_embeddings_2312`) or `query_embedding` (raw 2312-D)
- Use `buildServiceProvenance()` for consistent artifact SHA reporting
- Follow the same security pattern: CORS → auth → rate-limit (20 req/min) → security headers → timeout
- Increment shared `tier1Service*` metrics with service-specific counters

---

## 6. Benchmark Archive

The validation script confirms both M33 and M34 records are present in
`reports/benchmark_archive.json` with correct SHAs. The M35 validation record
(documented here) confirms cross-service consistency.

---

## 7. Pre-existing Test Failures (Not Regressions)

The full test suite shows 33 failures, all in pre-existing files unrelated to Tier-1:

- `tests/browser/*.test.ts` (7 files) — Browser WASM tests requiring a live browser/playwright
  environment
- `src/routes/api/eeg/embed/__tests__/-foundation-*.test.ts` (5 files) — Live route tests requiring
  real Supabase JWT auth and onnxruntime-node at runtime
- `src/lib/ai/inference/__tests__/foundation-serving-m13.test.ts` — Real EDF file processing test

**No Tier-1 tests fail.** All 74 Tier-1 unit + route tests pass.

---

## 8. Next Steps

Per M31 §21.1, the next phases are:

- **Tier 1 Beta** — Internal beta with 3-5 researchers using all 3 services
- **Tier 1 Production Candidates** — Scientific + operational gate review

For the development roadmap beyond Tier 1, refer to:
- M31 §22: Experiment roadmap includes T-036 (EEGConformer canary observability),
  T-037 (production canary), and Tier-2 services (sleep staging, attention decoding)
- The `docs/roadmaps/` directory for execution blueprints

---

## 9. Files Created/Modified in M35

| Action | File |
|--------|------|
| **Create** | `src/lib/ai/inference/__tests__/cross-service.test.ts` (7 tests) |
| **Create** | `scripts/tmp/m35_cross_service_validation.py` (68 code checks) |
| **Create** | `reports/MISSION35_CROSS_SERVICE_VALIDATION_REPORT.md` |
| **Modify** | `src/lib/ai/inference/index.ts` (added `cognitive.server` + `subject-identity.server` exports) |

---

*Report generated: 2026-08-19 · Neuro-Fabric Core M35 Validation*
