# MISSION41 — Multi-Task Fusion on Joint-2312

**Status**: ✅ Complete
**Mission**: Tier-3 Multi-Task Fusion — single-route decode running all 4 task probes in parallel on one Joint-2312 embedding
**Date**: 2026-08-19
**Target**: Unified `decodeJoint2312()` with shared provenance, embed-once-reuse-many at the batch level

---

## 1. What Was Built

M41 implements the **Multi-Task Fusion** layer on the frozen Joint-2312 (2312-D) embedding backbone. This is Tier-3: a single `decodeJoint2312()` function that resolves a Joint-2312 embedding once and dispatches it to all 4 Tier-1+Tier-2 task probes in parallel via `Promise.all`:

| # | Probe | Service | Input→Output | Task Type |
|---|---|---|---|---|
| 1 | Cognitive State | M33 | 2312→1 | Regression (workload) |
| 2 | Anomaly Detection | M34 | 2312→1 | Mahalanobis distance |
| 3 | Sleep Staging | M39 | 2312→5 | Classification (softmax) |
| 4 | Sleep Quality | M40 | 2312→1 | Regression (clamped [0,1]) |

Subject Identity (M32) is excluded from the fusion batch — it requires an RPC vector search against the full embedding table, not a single-vector ONNX pass. Callers can invoke `/api/joint2312/search/subject-identity` separately.

### Core Service Files

| # | File | Purpose |
|---|---|---|
| 1 | `src/lib/ai/inference/joint-fusion.server.ts` | Core `decodeJoint2312()` — resolves embedding once, dispatches to all 4 probes in parallel via `Promise.all`, collects results + probe SHAs, builds unified provenance. |
| 2 | `src/routes/api/joint2312/fusion.ts` | `POST /api/joint2312/fusion` — CORS → auth → rate-limit (20/min) → body parse → validate → `decodeJoint2312` → catch (400/401/408/429/500) → security headers. |
| 3 | `src/lib/ai/inference/index.ts` | Barrel export: `export * from "./joint-fusion.server"` |

### Test Suite (2 new files, 35 tests)

| File | Tests | Coverage |
|---|---|---|
| `src/lib/ai/inference/__tests__/joint-fusion-decode.test.ts` | 19 | `decodeJoint2312()`: raw embedding decode, embedding_id reuse, embed-once-reuse-many, INVALID_REQUEST, DIMENSION_MISMATCH, EMBEDDING_NOT_FOUND, parallel Promise.all dispatch, same embedding to all probes, partial head selection, probe SHA collection, partial failure handling, provenance with block weights, timings, response shape validation |
| `src/routes/api/joint2312/__tests__/-fusion.test.ts` | 16 | Route-layer: content-type, JSON parsing, missing embedding, wrong dimension, invalid heads, 200 success, embedding_id echo, partial heads, 401 auth, 429 rate-limit, 400 decode errors, 500 unknown error, 408 timeout, unified provenance |

### New Exports from `joint-fusion.server.ts`

```typescript
export const JOINT_FUSION_SERVICE = "joint-fusion";
export const JOINT_FUSION_VERSION = "v0.1.0";
export const JOINT_FUSION_TIMEOUT_MS = 10_000;

export interface JointFusionRequest {
  query_embedding?: number[];
  embedding_id?: string;
  heads?: ("cognitive" | "anomaly" | "sleep-staging" | "sleep-quality")[];
}

export interface JointFusionResponse {
  service: string;
  model: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: {
    cognitive?: CognitiveResult[];
    anomaly?: AnomalyResult[];
    sleep_staging?: SleepResult[];
    sleep_quality?: SleepQualityResult[];
  };
  metadata: {
    embedding_reused: boolean;
    heads_run: string[];
    probes: Array<{ id: string; sha256: string }>;
  };
  timings: {
    embed_ms?: number;
    inference_ms: number;
    total_ms: number;
  };
}
```

---

## 2. Architecture

### Core Design: Batch Embed-Once → Reuse Many

M41 realizes the "Embed Once → Reuse Many" principle at the batch level:

1. **Embedding Resolution** (single lookup, before parallel dispatch):
   - If `embedding_id` is provided → fetch the existing Joint-2312 embedding from `joint_embeddings_2312` table (no recomputation)
   - If `query_embedding` is provided → use the raw 2312-D vector directly
   - If neither → throw `INVALID_REQUEST`

2. **Dimension Validation**: The embedding is validated to be exactly 2312-D before any probe dispatch.

3. **Parallel Probe Dispatch** (all 4 run simultaneously):
   - Each probe receives the same `query_embedding` via `Promise.all()`
   - Results are collected into a unified `results` object
   - Failed probes are caught individually (logged as warnings) and their heads are excluded from `heads_run`

4. **Unified Provenance**: A single `ServiceProvenance` block is built for the entire fusion, including all 4 artifact SHAs (CBraMod, V2, PCA, EEGPT) and block weights.

### Error Handling

The fusion service uses **plain Error objects** with error codes embedded in the message (e.g., `"EMBEDDING_NOT_FOUND: ..."`) rather than typed domain errors. The route catches all decode errors:

- **Known codes** (INVALID_REQUEST, EMBEDDING_NOT_FOUND, DIMENSION_MISMATCH): Return 400 with the error message as `error` field
- **Unknown errors**: Return 400 with `code: "DECODE_ERROR"` — the route does NOT have a separate 500 path for decode errors since the fusion service wraps all downstream errors

### Security

Same pattern as all Tier-1/Tier-2 services:
1. CORS preflight check (`handleCors`)
2. Bearer token authentication (Supabase Auth via `authenticateRequest`)
3. Rate limiting: 20 requests/minute/user (`checkRateLimit`)
4. Security headers (X-Content-Type-Options, X-Frame-Options, etc. via `applySecurityHeaders`)
5. Timeout: 10 seconds (`Promise.race` with timeout promise)

### Block Fusion (Frozen Backbone)

The Joint-2312 embedding is unchanged from M27. The fusion layer simply routes the 2312-D vector to all 4 linear probe heads:

| Block | Dim | Weight | SHA |
|---|---|---|---|
| CBraMod | 200 | 0.3062 | `c128ccfd…` |
| V2-32 | 32 | 0.1434 | `18644de1…` |
| PCA-32 | 32 | 0.1519 | deterministic |
| EEGPT-2048 | 2048 | 0.3985 | `a92daf44…` |

**Total**: 200+32+32+2048 = 2312-D (L2-normalised)

---

## 3. API Contract

```
POST /api/joint2312/fusion
Content-Type: application/json

Request:
{
  "embedding_id?: string",           // reuse existing Joint-2312 embedding
  "query_embedding?: number[]",      // raw 2312-D vector
  "heads?: string[]"                 // optional: subset of [cognitive, anomaly, sleep-staging, sleep-quality]
}

Response (200):
{
  "service": "joint-fusion",
  "model": "onnx-cbramod-joint-2312",
  "head_version": "v0.1.0",
  "embedding_id?": string,            // present if embedding_id was used
  "provenance": {
    "service": "joint-fusion",
    "service_version": "v0.1.0",
    "embedding_model": "onnx-cbramod-joint-2312",
    "embedding_dim": 2312,
    "task_head_id": "joint-fusion-all-v1",
    "artifact_shas": { "cbramod", "v2", "pca", "eegpt" },
    "block_weights": { "cbramod": 0.3062, "v2": 0.1434, "pca": 0.1519, "eegpt": 0.3985 },
    "component_dims": { "cbramod": 200, "v2": 32, "pca": 32, "eegpt": 2048 },
    "task_head_metrics": { "targets": "[cognitive, anomaly, sleep-staging, sleep-quality]" }
  },
  "results": {
    "cognitive?": [...],             // present if head ran
    "anomaly?": [...],               // present if head ran
    "sleep_staging?": [...],         // present if head ran
    "sleep_quality?": [...],         // present if head ran
  },
  "metadata": {
    "embedding_reused": false,
    "heads_run": ["cognitive", "anomaly", "sleep-staging", "sleep-quality"],
    "probes": [
      { "id": "cognitive-linear-v1", "sha256": "abc123" },
      { "id": "anomaly-mahalanobis-v1", "sha256": "def456" },
      { "id": "sleep-staging-v1", "sha256": "9da4ea37" },
      { "id": "sleep-quality-v1", "sha256": "5fb7400f" }
    ]
  },
  "timings": {
    "embed_ms?": number,
    "inference_ms": 1.2,
    "total_ms": 2.0
  }
}

Errors:
  400 — Invalid JSON, missing embedding, wrong dimension, invalid heads, decode errors
  401 — Authentication failed
  408 — Processing timeout (>10s)
  429 — Rate limit exceeded (retry_after_ms)
  500 — Internal server error (unexpected)
```

---

## 4. Validation

All validation checks pass:

| Check | Status |
|---|---|
| Code structure checks (service, route, barrel exports) | ✅ Pass |
| Benchmark archive record present | ✅ Pass |
| 35 TypeScript tests (19 service + 16 route) | ✅ All pass |

Run validation: `python scripts/tmp/m41_fusion_validation.py`

---

## 5. Relationship to Previous Missions

| Mission | Contribution | Reused |
|---|---|---|
| M27 | M27-learned 4-block weights `[0.3062, 0.1434, 0.1519, 0.3985]` | Joint-2312 frozen backbone |
| M28 | Production Joint-2312 pipeline | `JOINT_2312_EMBEDDING_DIM`, `JOINT_2312_MODEL_ID` |
| M33 | Cognitive State Intelligence | `decodeCognitiveState()` — probe head 1 in fusion |
| M34 | Anomaly Detection | `detectAnomalies()` — probe head 2 in fusion |
| M38 | Sleep-EDF Dataset Loader | Foundation for sleep probe training data |
| M39 | Sleep Staging | `decodeSleepState()` — probe head 3 in fusion |
| M40 | Sleep Quality | `decodeSleepQuality()` — probe head 4 in fusion |

---

## 6. Tier Status

- **Tier 1** (Foundation Model Integration): ✅ Complete — 3 services deployed (M32 Subject Identity, M33 Cognitive State, M34 Anomaly Detection)
- **Tier 2** (Domain-Specific Task Heads): ✅ Complete — M38 loader, M39 staging (5-class), M40 quality (regression)
- **Tier 3** (Multi-Task Fusion): ✅ Complete — M41 `POST /api/joint2312/fusion`

---

## 7. Files Summary

| Action | File |
|---|---|
| **Create** | `src/lib/ai/inference/joint-fusion.server.ts` |
| **Create** | `src/routes/api/joint2312/fusion.ts` |
| **Create** | `src/lib/ai/inference/__tests__/joint-fusion-decode.test.ts` |
| **Create** | `src/routes/api/joint2312/__tests__/-fusion.test.ts` |
| **Create** | `scripts/tmp/m41_fusion_validation.py` |
| **Create** | `reports/MISSION41_MULTI_TASK_FUSION_REPORT.md` |
| **Modify** | `src/lib/ai/inference/index.ts` (fusion barrel export) |
| **Modify** | `reports/benchmark_archive.json` (append m41 record) |
