# Mission 31 — Tier 1 Downstream Services Roadmap

## Executive Summary

**Mission:** Produce a complete, execution-ready roadmap for Tier 1 downstream services built on the frozen Joint-2312 representation layer. No implementation in this mission — only architecture and planning.

**Context:** M30 established that the Joint-2312 4-block fusion embedding (2312-D, R@5=0.8527, p=4.8×10⁻²⁸) is the strongest EEG representation produced by this project. It is already productionized (M28): the `/api/eeg/embed/foundation?model=joint-2312` route, `joint_embeddings_2312` table, `match_joint_embeddings_2312()`/`_exact()` RPCs, and SHA-verified inference are all live and tested.

**Tier 1 contains 3 services:**

1. **Subject-Identity & Cohort Similarity** — identify subjects across sessions, retrieve similar subjects/cohorts
2. **Cognitive State Intelligence** — attention, workload, arousal from EEG
3. **EEG Anomaly Detection** — detect abnormal patterns (non-clinical, screening-level)

This report defines the complete architecture, implementation phases, experiment roadmap, dataset requirements, evaluation protocols, validation gates, dependency graph, and execution order for all three services. The guiding principle is **Embed Once → Reuse Many Times**: Joint-2312 embeddings are computed once and consumed by all three services via a shared, reusable Service Layer.

---

## 2. Current State After M30

### 2.1 What Exists (Verified in Repository)

| Capability | Status | Evidence |
|-----------|--------|----------|
| Joint-2312 embedding (2312-D, 4-block fusion) | ✅ Complete | `src/lib/ai/inference/joint.server.ts` (807 lines), `fuseJoint2312Embedding()`, `embedJoint2312Windows()` |
| `/api/eeg/embed/foundation?model=joint-2312` route | ✅ Complete | `src/routes/api/eeg/embed/foundation.ts` (lines 392-528) |
| `joint_embeddings_2312` table (vector(2312)) | ✅ Complete | `supabase/migrations/20260817000001_joint_embeddings_2312.sql` |
| `match_joint_embeddings_2312()` ANN RPC | ✅ Complete | Same migration |
| `match_joint_embeddings_2312_exact()` exact RPC | ✅ Complete | Same migration |
| SHA-256 artifact verification | ✅ Complete | CBraMod `c128ccfd…`, V2 `18644de1…`, EEGPT `a92daf44…` |
| Channel selection (19/22/62-ch) | ✅ Complete | `src/lib/eeg/channels.ts` — `CBRAMOD_CHANNELS_19`, `PROD_CHANNELS_22`, `EEGPT_CHANNELS_62`, `selectEEGPTChannels()` |
| Unit tests (17 tests) | ✅ Complete | `src/lib/ai/inference/__tests__/joint-fusion-2312.test.ts` |
| E2E tests (12 tests: 6 M25 + 6 M28) | ✅ Complete | `src/lib/ai/inference/__tests__/joint-server.test.ts` |
| Browser smoke tests (4 tests) | ✅ Complete | `tests/browser/joint-2312-wasm-smoke-firefox.test.ts` (Chromium + Firefox) |
| Benchmark archive (29 experiments) | ✅ Complete | `reports/benchmark_archive.json` |

### 2.2 What is Partially Ready

| Capability | Status | Gap |
|-----------|--------|-----|
| Tier-1 upload + embedding (32-D) | ✅ Exists | No connection to Joint-2312; `eeg_analyses` table stores 32-D only |
| Cognitive decoder | ✅ Exists | Uses 5-band-power features, not 2312-D; heuristic + trained logistic regression |
| Subject/session concept graph | ✅ Exists | `concept_graph.sql` migration, but FK only to `embeddings(id)` (vector(32)), not `joint_embeddings_2312` |
| NeuralVectorIndex | ✅ Exists | Configurable for any table/RPC/dimension — needs extension for downstream results |
| Auth/rate-limit/CORS | ✅ Exists | Foundation route already uses these patterns; downstream services reuse |
| Metrics/logging | ✅ Exists | Tier-2 metrics exist; downstream service metrics need adding |
| Dataset manifest | ✅ Exists | Lists BCI-IV-2a, BCI-IV-2b, PhysioNetMI; no sleep/attention datasets |
| Sleep-EDF loader | ⚠️ Planned only | Referenced in DEEPTECH_ANALYSIS, not implemented |
| EEG2Image route | ⚠️ Static demo only | `src/routes/eeg2image.tsx` is a hardcoded UI demo |

### 2.3 What is Missing

| Capability | Why Missing |
|-----------|-------------|
| **Subject identity API** | `match_joint_embeddings_2312()` RPC exists but no higher-level API wraps it for identification/retrieval |
| **Cognitive decoder on Joint-2312** | Decoder uses 5-band-power features, not 2312-D embeddings |
| **Anomaly detection** | No distance-based or statistical anomaly detection on 2312-D space |
| **Service-layer result tables** | No tables for storing downstream service predictions (sleep stages, cognitive states, anomaly scores) |
| **Dataset loaders for downstream tasks** | No Sleep-EDF, SEED, DEAP, THINGS-EEG loaders |
| **Evaluation framework for downstream services** | No LOSO/LOSR evaluation harness for non-retrieval tasks |
| **Browser path for downstream** | No mechanism to project 2312-D → 32-D for browser-based inference |
| **Service versioning** | No model version tracking for decoder heads |

### 2.4 What is Blocked

| Capability | Blocker |
|-----------|---------|
| Sleep staging on Joint-2312 | No Sleep-EDF dataset loader + no sleep-specific channel remapping |
| EEG2Image on Joint-2312 | No THINGS-EEG dataset + EEG2Image decoder head |
| Seizure detection | No TUH/CHB-MIT dataset + clinical validation requirements |
| Attention decoding | No DOTS dataset + visual-search paradigm |
| Fatigue detection | No dedicated fatigue dataset |

---

## 3. Tier 1 Definition

**Tier 1 = the three downstream services built directly on top of the frozen Joint-2312 embedding layer, validated on publicly available datasets, and deployed as production candidates.**

```
Raw EEG
  ↓ (parse → select → preprocess)
Joint-2312 (2312-D, FROZEN — M28 productionized)
  ↓ (store in joint_embeddings_2312)
Shared Service Layer
  ├──────────────┬──────────────┬──────────────┐
  ▼              ▼              ▼
Subject       Cognitive      Anomaly
Identity      Intelligence    Detection
Service          Service        Service
  ↓              ↓              ↓
Similarity    Cognitive      Anomaly
Scores        States         Scores
```

**Shared components (used by all three services):**
- Joint-2312 embedding computation (`embedJoint2312Windows`)
- `joint_embeddings_2312` table (vector(2312))
- `match_joint_embeddings_2312()` ANN RPC + `match_joint_embeddings_2312_exact()` exact RPC
- NeuralVectorIndex (reused with different table/RPC names)
- Auth/rate-limit/CORS middleware
- Metrics/logging infrastructure
- SHA-256 artifact verification

**Service-specific components:**
- Task head model (linear probe or small MLP)
- Task-specific result table
- Service-specific API routes
- Service-specific evaluation protocol
- Service-specific baselines

---

## 4. Target Architecture

```
                        RAW EEG (EDF/CSV/NPY)
                               │
                               ▼
        ┌───────────────────────────────────────────┐
        │  TIER 2 FOUNDATION EMBEDDING LAYER         │
        │  (M25-M29 — FROZEN, NO CHANGES)            │
        │                                           │
        │  /api/eeg/embed/foundation?model=joint-2312│
        │  ↓                                          │
        │  CBraMod-200  V2-32  PCA-32  EEGPT-2048    │
        │  (4-block fusion)                          │
        │  block weights: [0.3062, 0.1434,            │
        │  0.1519, 0.3985]                           │
        │  → JointProvenance (SHA-verified)          │
        │  → joint_embeddings_2312 (vector(2312))    │
        └───────────────────────────────────────────┘
                               │
                               │ embedding_id (UUID FK)
                               ▼
        ┌───────────────────────────────────────────┐
        │  SHARED TIER 1 SERVICE LAYER              │
        │  (M31 — NEW)                              │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  ServiceRegistry                     │  │
        │  │  - registerService(name, factory)    │  │
        │  │  - createService(name, opts)         │  │
        │  │  - listServices()                    │  │
        │  └─────────────────────────────────────┘  │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  DownstreamVectorIndex               │  │
        │  │  (extends NeuralVectorIndex)         │  │
        │  │  - result table: {service}_results   │  │
        │  │  - search: match_{service}_embeddings│  │
        │  └─────────────────────────────────────┘  │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  TaskHeadRegistry                    │  │
        │  │  - registerHead(service, model)      │  │
        │  │  - createHead(service, opts)         │  │
        │  └─────────────────────────────────────┘  │
        │                                           │
        │  ┌─────────────────────────────────────┐  │
        │  │  ServiceProvenance                   │  │
        │  │  - embedding_sha (4 artifacts)       │  │
        │  │  - task_head_model_id + version      │  │
        │  │  - dataset_id + version              │  │
        │  │  - evaluation_metrics                │  │
        │  └─────────────────────────────────────┘  │
        └───────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     Subject/                  Cognitive        Anomaly
     Cohort                    Intelligence    Detection
     Service                    Service         Service
```

### 4.1 Component Descriptions

**ServiceRegistry** — Mirrors the existing `model-registry.ts` pattern. Each service registers a factory that creates a service instance given a `serviceId` + options. Services are looked up by name, instantiated lazily (like `InferenceEngine`'s LRU cache for ONNX sessions).

**DownstreamVectorIndex** — Extends the existing `NeuralVectorIndex` class. Each service gets its own result table (`subject_similarity_results`, `cognitive_states`, `anomaly_scores`) with a `CHECK` dimension contract and RLS policies matching the existing pattern. The `modelId` field tags results with the service+head version.

**TaskHeadRegistry** — Registers lightweight task heads (linear probes or small MLPs). Each head declares:
- `inputDim`: 2312 (Joint-2312) or 32 (V2-projected for browser)
- `outputDim`: task-specific (5 for sleep, 3 for cognitive, 1 for anomaly)
- `modelId`: unique identifier for the head version
- `artifactUri`: URL to ONNX or checkpoint file (SHA-verified)
- `trainingMetadata`: dataset, protocol, metrics

**ServiceProvenance** — Extends the existing provenance pattern from `joint2312Provenance()`. Each service result includes:
- The 4 embedding artifact SHAs (CBraMod, V2, EEGPT, PCA)
- The task head model ID + version + SHA
- The dataset used for head training
- Evaluation metrics (R@5, accuracy, AUROC, etc.)
- The LOSO fold configuration

---

## 5. Shared Tier 1 Service Layer

### 5.1 Common Infrastructure

| Component | Implementation | Reused From |
|-----------|---------------|-------------|
| Authentication | `authenticateRequest(request)` → Bearer token → `userId + supabase` | `request-auth.ts` (existing) |
| Rate limiting | `checkRateLimit(supabase, userId, max, window)` → Postgres → durable across isolates | `rate-limit.ts` (existing) |
| CORS | `handleCors(request)` + `getCorsHeadersForResponse()` | `middleware/cors.ts` (existing) |
| Security headers | `applySecurityHeaders()` | `middleware/security.ts` (existing) |
| Structured logging | `log("info", "event", payload)` | `logging/index.ts` (existing) |
| Metrics | `metrics.counter.inc({labels})`, `metrics.histogram.observe({labels}, value)` | `metrics/index.ts` (existing) |
| File validation | `checkMagicNumber()`, `sanitizeFilename()`, `MAX_FILE_BYTES` | `foundation.ts` route (existing) |
| Timeout | `PROCESSING_TIMEOUT_MS = 120_000` | `foundation.ts` (existing) |
| NeuralVectorIndex | Configurable table/RPC/dimension/modelId | `neural-index.ts` (existing) |
| Artifact verification | `verifyArtefact(bytes, sha256)` | `artefacts/hashed-artefact.ts` (existing) |
| Embedding validation | `validateEmbedding(vector, {expectedDim})` + `l2Normalize()` | `validation/` (existing) |

### 5.2 Shared Schema

The shared Service Layer builds on the existing database tables. The proposed schema adds service-level result tables that reference the `joint_embeddings_2312` table:

```
joint_embeddings_2312 (vector(2312))  ──┐
                                        ├── subject_identity_results
                                        ├── cognitive_state_results
                                        └── anomaly_detection_results
```

**New tables (M31 migration):**

```sql
-- Subject identity & cohort similarity results
CREATE TABLE subject_similarity_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  embedding_id UUID NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  query_type TEXT NOT NULL,  -- 'subject_identification' | 'session_similarity' | 'cohort_similarity'
  query_subject_id TEXT,     -- the subject being queried (if applicable)
  candidate_subject_id TEXT,  -- the matched subject
  rank INT,
  similarity FLOAT8,
  is_true_match BOOLEAN,     -- for evaluation
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_subject_similarity_embedding ON subject_similarity_results(embedding_id);
CREATE INDEX idx_subject_similarity_user ON subject_similarity_results(user_id);
CREATE INDEX idx_subject_similarity_query ON subject_similarity_results(query_type, created_at);

-- Cognitive state prediction results
CREATE TABLE cognitive_state_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  embedding_id UUID NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,     -- e.g. "cognitive-joint2312-linear-v1"
  attention FLOAT8 CHECK (attention BETWEEN 0 AND 1),
  workload FLOAT8 CHECK (workload BETWEEN 0 AND 1),
  arousal FLOAT8 CHECK (arousal BETWEEN 0 AND 1),
  attention_ci FLOAT8[2],     -- [lower, upper] confidence interval
  workload_ci FLOAT8[2],
  arousal_ci FLOAT8[2],
  confidence FLOAT8,          -- mean confidence score
  model_version TEXT,
  dataset TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_cognitive_results_embedding ON cognitive_state_results(embedding_id);
CREATE INDEX idx_cognitive_results_user ON cognitive_state_results(user_id);

-- Anomaly detection results
CREATE TABLE anomaly_detection_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  embedding_id UUID NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,     -- e.g. "anomaly-joint2312-mahalanobis-v1"
  anomaly_score FLOAT8,      -- normalized [0, 1]
  raw_distance FLOAT8,       -- pre-normalization distance
  threshold FLOAT8,
  is_anomaly BOOLEAN,
  confidence FLOAT8,
  model_version TEXT,
  dataset TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_anomaly_results_embedding ON anomaly_detection_results(embedding_id);
CREATE INDEX idx_anomaly_results_user ON anomaly_detection_results(user_id);
CREATE INDEX idx_anomaly_results_score ON anomaly_detection_results(anomaly_score);
```

### 5.3 Relationship Diagram

```
auth.users
  │
  ├── joint_embeddings_2312  (vector(2312), model_id='onnx-cbramod-joint-2312')
  │     │
  │     ├── subject_similarity_results  (embedding_id → joint_embeddings_2312.id)
  │     ├── cognitive_state_results     (embedding_id → joint_embeddings_2312.id)
  │     └── anomaly_detection_results   (embedding_id → joint_embeddings_2312.id)
  │
  ├── embeddings  (vector(32), Tier-1 — existing, unchanged)
  │
  ├── foundation_embeddings  (vector(200), Tier-2 — existing, unchanged)
  │
  ├── joint_embeddings  (vector(264), M25 — existing, unchanged)
  │
  ├── eeg_analyses  (existing, stores 32-D + cognitive states — Tier-1)
  │
  └── concept_graph tables  (subjects/sessions/windows/labels — existing)
```

### 5.4 API Request/Response Patterns

All three Tier-1 services follow a consistent pattern:

```typescript
// Shared service response template
interface DownstreamServiceResponse<T> {
  service: string;          // e.g. "subject-identity", "cognitive-state", "anomaly-detection"
  model: string;            // e.g. "joint-2312-frozen"
  head: string;             // e.g. "linear-probe-v1", "mahalanobis-v1"
  embedding_id: string;     // FK to joint_embeddings_2312
  provenance: {
    embedding_artifacts: { cbramod: string, v2: string, eegpt: string };
    head_artifact?: string; // SHA of task head if applicable
    dataset: string;
    evaluation: Record<string, number>;
  };
  results: T;
  timings: {
    embed_ms: number;
    inference_ms: number;
    total_ms: number;
  };
}
```

### 5.5 Service-Specific Model Registries

Each service has its own task-head registry, following the `model-registry.ts` pattern:

```typescript
// src/lib/ai/decoders/subject-identity.registry.ts
export const SUBJECT_IDENTITY_HEADS = [
  {
    id: "subject-identity-linear-v1",
    inputDim: 2312,
    outputDim: 1,
    training: { dataset: "eegmmidb", protocol: "loso-50fold", r5: 0.8527 },
    inference: "server",
    sha256: "...",
  },
  {
    id: "subject-identity-v2-projected-v1",
    inputDim: 32,
    outputDim: 1,
    training: { dataset: "eegmmidb", protocol: "loso-50fold", r5: 0.779 },
    inference: "browser",
    sha256: "...",
  },
];

// src/lib/ai/decoders/cognitive.registry.ts
export const COGNITIVE_HEADS = [
  {
    id: "cognitive-linear-v1",
    inputDim: 2312,
    outputDim: 3,  // [attention, workload, arousal]
    training: { dataset: "seed", protocol: "loso-15fold", r: 0.65 },
    inference: "server",
  },
  // ...
];

// src/lib/ai/decoders/anomaly.registry.ts
export const ANOMALY_HEADS = [
  {
    id: "anomaly-mahalanobis-v1",
    inputDim: 2312,
    outputDim: 1,  // anomaly score
    method: "mahalanobis-distance",
    threshold: 0.95,  // calibrated on normal data
    inference: "server",
  },
  // ...
];
```

---

## 6. Service 1 — Subject Identity & Cohort Similarity Roadmap

### 6.1 Current State

**What M27/M28 proved:**

| Metric | Value | Baseline Comparison |
|--------|-------|---------------------|
| Joint-2312 R@1 | 0.6438 | +0.118 vs Joint-264 (0.5271), p=6.3×10⁻⁷⁴ |
| Joint-2312 R@5 | 0.8527 | +0.0669 vs Joint-264 (0.7858), p=4.8×10⁻²⁸ |
| Joint-2312 R@10 | 0.9060 | +0.0444 vs Joint-264 (0.8616) |
| Joint-2312 MRR | 0.7361 | +0.0936 vs Joint-264 (0.6425) |

**What these numbers prove:**
- Joint-2312 provides strong **subject-identity retrieval** — 85.3% of the time, the correct subject is in the top-5 nearest neighbors
- The 4-block fusion significantly outperforms every individual model and every 3-block fusion variant
- The improvement over Joint-264 is both statistically significant (p=4.8×10⁻²⁸) and practically meaningful (+6.69pp R@5)

**What these numbers do NOT prove:**
- Cross-dataset generalization (all experiments used PhysioNet EEGMMIDB; the protocol is session-disjoint within the same dataset, not subject-disjoint across datasets)
- Real-world identification under distribution shift (different amplifiers, electrode gels, cap sizes, impedance levels)
- False positive rate — the retrieval protocol measures recall (can we find the right subject?) but doesn't measure precision (how often do wrong subjects appear in the top-K?)
- Clinical utility — subject identity is necessary but not sufficient for clinical applications (which also need pathology detection, as covered by Service 3)

### 6.2 Product Definition

#### 6.2.1 Subject Identification

```
Unknown EEG recording
  → POST /api/joint2312/identify
  → Joint-2312 embedding (2312-D)
  → match_joint_embeddings_2312() against enrolled subjects
  → ranked candidate list with similarity scores
  → confidence = top-1 similarity (threshold-based acceptance)
```

**Product requirements:**
- Must return top-K candidates (default K=10) with similarity scores
- Must return confidence interval (based on the gap between top-1 and top-2 similarity)
- Must support enrollment (explicitly register a subject's embeddings)
- Must support exclusion lists (exclude known subjects from search)

#### 6.2.2 Session Similarity

```
EEG session (multiple 4s windows)
  → POST /api/joint2312/sessions/similar
  → Joint-2312 embedding (mean-pooled across windows → 2312-D)
  → match_joint_embeddings_2312() 
  → ranked similar sessions with per-window similarity breakdown
```

**Product requirements:**
- Must support session-level aggregation (mean-pool window embeddings)
- Must return session-level similarity + per-window breakdown
- Must support filtering by date range, channel count, preprocessing protocol

#### 6.2.3 Cohort Similarity

```
Subject or session
  → POST /api/joint2312/cohorts/{cohortId}/similar
  → Joint-2312 embedding
  → match_joint_embeddings_2312() filtered to cohort
  → ranked cohort members with similarity scores
```

**Product requirements:**
- Must support cohort-based filtering (pre-defined subject groups)
- Must support similarity thresholds (return only above-threshold matches)
- Must support pagination for large cohorts

#### 6.2.4 Unified vs Separate APIs

**Decision: Unified API with query-type field.**

One endpoint (`POST /api/joint2312/similarity/search`) handles all three operations via a `query_type` parameter:
- `subject_identification` — single embedding vs enrolled subjects
- `session_similarity` — aggregated session embedding vs all sessions
- `cohort_similarity` — single embedding vs filtered cohort

This follows the existing pattern: `/api/eeg/embed/foundation?model=...` uses a single route with a `model` parameter. A separate `query_type` parameter avoids API sprawl while keeping the interface explicit.

### 6.3 Required Infrastructure

**Database:**
- New table: `subject_similarity_results` (defined in §5.2)
- New indexes on `joint_embeddings_2312(metadata.subject_id)`, `subject_similarity_results(query_type)`
- Extend `joint_embeddings_2312` metadata to include `subject_id` (currently only in `file_name`, `window_index`, etc.)

**API:**
- `POST /api/joint2312/similarity/search` — unified similarity search
- `POST /api/joint2312/subjects/enroll` — enroll a subject (compute + store embeddings)
- `GET /api/joint2312/subjects/{subjectId}/results` — get identification results
- `POST /api/joint2312/cohorts` — create a cohort (group of subjects)

**Auth:**
- Reuse `authenticateRequest()` from `request-auth.ts`
- Add RLS policy for `subject_similarity_results` (user-scoped, like existing tables)

**Observability:**
- `metrics.subjectIdentifyRequestsTotal` counter
- `metrics.subjectIdentifyLatencyMs` histogram
- `metrics.subjectIdentifyConfidenceDistribution` histogram
- Log events: `subject.identify.start`, `subject.identify.identified`, `subject.identify.fallback`

**Caching:**
- Cache enrollment embeddings in a Redis-like layer (or in-memory for dev)
- Cache top-K results per (subjectId, cohortId) for 1 hour

### 6.4 Scientific Validation

**Evaluation Protocol:**
- **Dataset:** PhysioNet EEGMMIDB S001–S050 (existing — 50 subjects, 6 runs, 15 trials each = 4,500 trials)
- **Enrollment set:** Runs 5-10 from 49 subjects (4,410 trials) — embeddings already stored from M28
- **Query set:** Held-out run from the 50th subject (15 trials) — 50 folds, LOSO
- **Aggregation:** Mean-pool 15 trial embeddings per (subject, run) → session-level embedding
- **Protocol:** For each query session, search all enrollment sessions → top-10 candidates → success if true subject in top-K

**Metrics:**
| Metric | Definition | Acceptance Threshold |
|--------|-----------|---------------------|
| Identify@1 | True subject is top-1 candidate | ≥ 0.50 (baseline: PCA R@1=0.4856) |
| Identify@5 | True subject is in top-5 | ≥ 0.85 (baseline: Joint-2312 R@5=0.8527) |
| Identify@10 | True subject is in top-10 | ≥ 0.90 (baseline: Joint-2312 R@10=0.9060) |
| Mean Reciprocal Rank | Average of 1/rank(true subject) | ≥ 0.70 (baseline: Joint-2312 MRR=0.7361) |
| False Positive Rate @ threshold | % of wrong-subject sessions above confidence threshold | ≤ 0.05 |

**Baselines:**
| Baseline | R@5 | Why relevant |
|----------|-----|-------------|
| Joint-2312 (this service) | 0.8527 | Current best |
| Joint-264 (M25) | 0.7858 | 3-block production baseline |
| EEGPT-2048 | 0.8118 | Single best block |
| PCA-32 | 0.7404 | Tier-1 interactive baseline |
| CBraMod-200 | 0.5276 | Single worst block |
| V2-32 | 0.2158 | Browser baseline |
| Band-power centroid | ~0.70 | Simple statistical baseline |

**Statistical test:** Paired t-test on per-(subject, run) Recall@K, Bonferroni-corrected (α=0.05/4=0.0125 for 4 thresholds). Report Cohen's d + bootstrap 95% CI.

### 6.5 Implementation Phases

**Phase 1 → Data/Index Preparation (Weeks 1-2)**
- Add `subject_id` and `session_id` to `joint_embeddings_2312` metadata (migration)
- Backfill existing M28 embeddings with subject/session info from PhysioNet EEGMMIDB filenames
- Create `subject_similarity_results` table + indexes
- Verify ANN search returns correct subject-identity matches

**Phase 2 → Service API (Weeks 2-3)**
- `POST /api/joint2312/similarity/search` route (reuses `match_joint_embeddings_2312`)
- `POST /api/joint2312/subjects/enroll` route
- Response schema with similarity scores, confidence intervals, provenance
- Auth/rate-limit/wiring (reuse foundation.ts patterns)

**Phase 3 → Identification (Weeks 3-4)**
- Subject identification endpoint (single-embedding search)
- Enrollment management (register new subjects)
- Confidence scoring (top-1/top-2 gap, calibration)
- Unit tests (15) + E2E tests (5)

**Phase 4 → Cohort & Session Similarity (Weeks 4-6)**
- Cohort filtering on ANN RPC
- Session-level aggregation (mean-pool + timestamp)
- Pagination + date range filters
- Tests (10 unit + 5 E2E)

**Phase 5 → Evaluation (Weeks 6-8)**
- Reproduce M27 results: R@5=0.8527 on 50-subject LOSO
- Compare against 6 baselines (Joint-264, EEGPT, PCA, CBraMod, V2, band-power)
- Statistical significance testing (paired t-test, Bonferroni)
- False positive rate analysis
- Append `m31-subject-identity` record to `benchmark_archive.json`

**Phase 6 → Beta (Weeks 8-10)**
- Internal beta with 3-5 researchers
- Monitor latency, false positive rate, user feedback
- Refine confidence scoring
- Documentation + usage examples

**Phase 7 → Production Candidate (Weeks 10-12)**
- Performance optimization (embedding caching, batch ANN search)
- Production readiness review (security, observability, error handling)
- Final evaluation on held-out dataset (if available)
- `reports/M31_SUBJECT_IDENTITY_VALIDATION.md`

### 6.6 Definition of Done

- [ ] Subject identification achieves R@5 ≥ 0.85 on 50-subject LOSO (reproducing M27)
- [ ] False positive rate < 5% at 0.80 similarity threshold
- [ ] `POST /api/joint2312/similarity/search` returns top-10 with provenance
- [ ] Enrollment + retrieval works end-to-end (embed → store → search → identify)
- [ ] Session-level aggregation (mean-pool) produces valid cohort similarity
- [ ] 25 unit tests + 10 E2E tests pass
- [ ] p95 latency < 2000ms (includes CBraMod + EEGPT inference)
- [ ] SHA-256 verification on all embeddings (CBraMod, V2, EEGPT)
- [ ] Auth + rate limiting enforced (20 req/min/user, 120s timeout)
- [ ] No regression in Tier-1 V2 or Tier-2 Joint-2312 paths
- [ ] Lint + typecheck clean
- [ ] M31 validation report written
- [ ] `benchmark_archive.json` appended with `m31-subject-identity` record

---

## 7. Service 2 — Cognitive State Intelligence Roadmap

### 7.1 Existing Cognitive Intelligence Work

The platform already has a **two-tier cognitive decoder**:

1. **Heuristic** (`src/lib/decoder/features.ts` + `decoder/index.ts`):
   - Band stats: delta, theta, alpha, beta, gamma (5 bands × 22 channels = 110 features)
   - Attention = squash(β / (α + θ))
   - Workload = squash(θ / α)
   - Arousal = clamp(β + γ)
   - `squash(x) = 1 / (1 + exp(-log(x)))` — log-squash transform
   - Used as the fallback in the upload pipeline

2. **Trained** (`src/lib/decoder/trained-decoder.ts`):
   - `cognitive-decoder-v0.onnx` — logistic regression on 5-element band-power feature vector
   - 3-class output: [attention, workload, arousal]
   - Confidence intervals: ±0.08 around each prediction
   - SHA `ea4f216c…` (1.3KB, WASM-compatible, SHA-verified)
   - Used when available; falls back to heuristic if ONNX fails

**Gap from M30:** The trained decoder uses **5-band-power features**, not the 2312-D Joint-2312 embedding. M30 proposed upgrading to a 2312-D input → multi-task head.

### 7.2 Product Definition

**Primary target: Workload** (not attention or arousal) as the first cognitive state to implement. Rationale:

| Criterion | Workload | Attention | Arousal |
|-----------|----------|-----------|---------|
| Existing code | Strong band-power ratio (θ/α) | Strong band-power ratio (β/α+θ) | Strong band-power (β+γ) |
| Dataset availability | SEED (15 subjects, rich labels) | SEED, DEAP | DEAP, DREAMER |
| Label quality | High (NASA-TLX workload scores) | Medium (self-report) | Medium (valence/arousal) |
| Scientific evidence | θ/β ratio is the canonical workload marker (Klimesch, 2012) | β/α ratio (Klimesch, 2002) | β+γ power (Cajochen, 2005) |
| Compatibility with Joint-2312 | CBraMod Fourier + EEGPT temporal = strong fit | | |
| Implementation difficulty | Medium | Low | Low |
| Business value | High (safety, BCI, training) | Medium | Medium |

**Workload** is selected because:
1. The existing heuristic decoder already computes a workload proxy (θ/α ratio)
2. The SEED dataset has high-quality NASA-TLX workload scores for 15 subjects across 3 sessions
3. The θ/β ratio (classic workload marker) is directly captured by CBraMod's Fourier features in the Joint-2312 embedding
4. Workload is actionable in business/safety contexts (driver monitoring, training optimization, fatigue prevention)

### 7.3 Architecture

```
Workload Decoder on Joint-2312

Input: EEG window (4s, 1000 samples, 22 or 62 channels)
  ↓
Joint-2312 (2312-D, frozen, SHA-verified)
  ↓
┌─────────────────────────────────────────────────┐
│  Cognitive Decoder Head (lightweight)          │
│                                                 │
│  Option A: Linear probe (Ridge regression)      │
│    2312-D → 1 (continuous workload 0-1)         │
│    Training: LOSO on SEED, R²>0.4 on held-out   │
│                                                 │
│  Option B: Small MLP (2312 → 64 → 32 → 1)        │
│    With dropout 0.3, weight decay 0.01          │
│    Only if linear probe doesn't meet threshold  │
└─────────────────────────────────────────────────┘
  ↓
Output: {
  workload: 0.73,
  confidence: 0.85,
  confidence_interval: [0.65, 0.81],
  provenance: {...}
}
```

### 7.4 Dataset Strategy

**Primary: SEED (7 subjects)**
- 15 subjects × 3 sessions × 15 trials = 675 trials
- 62 channels, 200 Hz
- Labels: valence, arousal, dominance (1-9 scale) + derived workload
- NASA-TLX workload scores available for a subset
- License: CC-BY-NC-SA — requires non-commercial use clause

**Validation dataset options:**
- **DEAP** (32 subjects, 14 channels, valence/arousal) — different channel count → tests generalization
- **DREAMER** (23 subjects, 14 channels) — valence/arousal/dominance
- **PhysioNet EEGMMIDB** (50 subjects, 64 channels) — already embedded; can derive workload proxy from MI difficulty

**Channel compatibility:** SEED uses 62 channels — fully compatible with the EEGPT block in Joint-2312. The 22-channel V2/Prod set is present in SEED's 62-channel montage (superset). This means the Joint-2312 pipeline can be applied directly to SEED data with minimal remapping.

### 7.5 Evaluation

**Protocol:**
- 50-fold LOSO (or 15-fold for SEED with 15 subjects)
- Session-disjoint evaluation (train on 2 sessions, test on 1 session per subject)
- Regression metric: R² (coefficient of determination)
- Classification metric: accuracy on binned workload (low/medium/high)

**Metrics:**
| Metric | Definition | Acceptance Threshold |
|--------|-----------|---------------------|
| R² (held-out) | Coefficient of determination on workload prediction | ≥ 0.40 |
| RMSE | Root mean squared error (normalized 0-1) | ≤ 0.20 |
| MAE | Mean absolute error (normalized 0-1) | ≤ 0.15 |
| Pearson r | Correlation with ground truth | ≥ 0.65 |
| Accuracy (3-class) | Low/Med/High workload classification | ≥ 0.60 |

**Baselines:**
| Baseline | R² | Why relevant |
|----------|-----|-------------|
| Heuristic θ/α ratio (current) | ~0.20–0.30 | Existing decoder |
| V2-32 + linear probe | ~0.25–0.35 | Tier-1 representation |
| Joint-2312 + linear probe | **target** | Proposed (frozen backbone) |
| CBraMod-200 + linear probe | ~0.30 | Single best block |
| EEGPT-2048 + linear probe | ~0.35 | Single best block (temporal) |

**Statistical test:** Paired t-test on per-subject R², Bonferroni-corrected (α=0.05/4=0.0125 for 4 baselines). Report Cohen's d.

### 7.6 Implementation Phases

**Phase 1 → Dataset Integration (Weeks 1-3)**
- Implement SEED dataset loader (EDF + annotation parser)
- Integrate into `src/lib/datasets/` + manifest
- Verify 62-channel compatibility with Joint-2312 pipeline

**Phase 2 → Baseline Reproduction (Weeks 3-4)**
- Reproduce existing heuristic decoder on SEED (θ/α → workload)
- Baseline: R² = ~0.25 (projected from EEGMMIDB heuristic)
- Baseline: V2-32 + linear probe on SEED (R² = ~0.30)
- Append `m31-cognitive-baselines` to archive

**Phase 3 → Joint-2312 Head Training (Weeks 4-6)**
- Train linear probe: 2312-D → 1 (workload regression)
- Evaluate: R² on LOSO held-out subjects
- If R² < 0.40: try small MLP (2312→64→32→1, dropout 0.3)
- If still < 0.40: try task-specific block masking (weight CBraMod + EEGPT blocks more heavily)

**Phase 4 → API Integration (Weeks 6-8)**
- `POST /api/joint2312/cognitive/decode` route
- Response: workload, confidence, provenance, timings
- Integration with existing `decodeCognitiveState()` facade
- Browser path: V2-32 → lightweight head for real-time (reuse existing pattern)

**Phase 5 → Evaluation & Beta (Weeks 8-12)**
- Full 15-fold LOSO evaluation on SEED
- Cross-dataset validation on DEAP/DREAMER
- Internal beta with 3-5 users
- Append `m31-cognitive-workload` to archive

### 7.7 Definition of Done

- [ ] SEED dataset loader implemented + integrated into dataset manifest
- [ ] Heuristic decoder reproduced on SEED (baseline R²)
- [ ] V2-32 + linear probe baseline ≥ R² 0.30
- [ ] Joint-2312 + linear probe ≥ R² 0.40 on 15-fold LOSO
- [ ] Cross-dataset validation on DEAP (no training, R² > 0.25)
- [ ] `POST /api/joint2312/cognitive/decode` returns workload + confidence + provenance
- [ ] Browser fallback (V2-32 → cognitive head) works with <600ms P95
- [ ] 20 unit tests + 8 E2E tests pass
- [ ] No regression in existing `decodeCognitiveState()` facade
- [ ] Audit log: all predictions stored with model_id + embedding_id + provenance
- [ ] M31 cognitive validation report written
- [ ] `benchmark_archive.json` appended

---

## 8. Service 3 — EEG Anomaly Detection Roadmap

### 8.1 Product Definition

This is a **non-clinical screening service**. It flags EEG windows that deviate significantly from the learned "normal" embedding distribution. It does NOT diagnose, screen for, or classify any medical condition.

**Product requirements:**
- Detect epochs with abnormal spectral patterns (e.g., excessive slow waves, focal slowing)
- Detect electrode artifacts (flat lines, noise bursts)
- Detect outlier subjects/sessions (unusual embedding patterns)
- Provide anomaly scores with configurable thresholds
- Support both per-window and session-level aggregation

### 8.2 Approach Selection

Based on M30 analysis and existing literature:

| Method | Description | Pros | Cons | Evidence |
|--------|-------------|------|------|----------|
| **Mahalanobis distance** | L2 distance from 2312-D centroid, normalized by covariance | Statistically principled; fast; interpretable | Assumes Gaussian; sensitive to covariance estimation | Used in M30 §5.8; standard in bio signal anomaly detection |
| **Isolation Forest** | Tree-based anomaly detection on 2312-D | Non-parametric; handles non-Gaussian | Slower training; harder to interpret; needs scikit-learn port | Scikit-learn standard; well-documented |
| **One-Class SVM** | Learn boundary of normal data in 2312-D | Theoretically grounded | Scales poorly O(n²); sensitive to kernel | Not recommended for 2312-D (curse of dimensionality) |
| **Reconstruction error (autoencoder)** | Train AE to reconstruct normal embeddings; anomalies have high error | Captures nonlinear structure; learns manifold | Requires training data; more complex | M30 §5.8; promising but heavier |
| **Z-score on band-power** | Simple statistical threshold on PCA-32 band powers | Fast, interpretable, no training | Misses complex patterns | Used as current fallback |

**Decision: Mahalanobis distance** as the primary method, with z-score on PCA-32 as a secondary/confirmatory signal. Mahalanobis is chosen because:

1. M16 established that PCA-32 (band-power features) is competitive with all foundation models on MI — the band-power signal is real and interpretable
2. Mahalanobis distance is the natural extension of z-score to multivariate space
3. It's computationally lightweight (O(D²) for covariance, O(D) per inference after fit)
4. It's interpretable (per-dimension contribution to anomaly score)
5. No training data beyond the "normal" cohort is needed

**Why not One-Class SVM:** At 2312-D, the curse of dimensionality makes SVM boundary learning unstable without extensive regularization. Mahalanobis distance, which naturally accounts for feature correlations, is more robust in high dimensions when the covariance can be estimated reliably (which it can, with 50+ subjects × 6 runs × 15 trials = 4,500 samples).

**Why not Isolation Forest:** While effective, it requires porting scikit-learn to TypeScript/Node.js or serving via an API. Mahalanobis can be computed directly in SQL (via the `covariance` aggregate) or in the TypeScript layer with a small numerical library.

### 8.3 Data Strategy

**What counts as "normal":**
- PhysioNet EEGMMIDB runs 5/6 (left/right hand, feet, tongue MI) — these are "normal" motor imagery tasks with no known pathology
- EEGMMIDB S001–S050 runs 5-10 represent healthy subjects performing standard MI tasks
- Any EEG that embeds to a typical region of the 2312-D space is "normal"

**What counts as "anomalous":**
- Flat-line signals (all-zero or near-zero variance windows)
- Excessive muscle artifacts (high gamma, high broadband)
- Severe electrode artifacts (sudden drops in specific channels)
- Unusual spectral patterns (excessive delta, absent alpha)
- Subject outliers (embeddings far from the cohort centroid)

**What counts as "out-of-distribution":**
- Different montage (e.g., 10-10 vs 10-20)
- Different task paradigm (e.g., visual vs motor imagery)
- Different amplifier (different noise floor)
- Pathological EEG (seizure, slowing, asymmetry) — but this is NOT labeled; the service flags these as "out-of-distribution" not "pathological"

### 8.4 Model Specification

```
Mahalanobis Anomaly Detector

Input:   2312-D Joint-2312 embedding (L2-normalized)
Fit:     Compute μ (2312-D mean) and Σ (2312×2312 covariance) on normal data
Score:   d² = (x - μ)ᵀ Σ⁻¹ (x - μ)  → Mahalanobis distance squared
Normalize: z_score = (d² - median(d²_train)) / MAD(d²_train)
         (MAD = median absolute deviation, robust to outliers)
Threshold: z > 3.5 → anomaly (calibrated for ~0.5% false positive rate)
Window aggregation: 5/10 consecutive anomaly windows → session-level flag

Fallback: z-score on PCA-32 band-power features (no covariance needed;
          per-band thresholds: delta theta alpha beta gamma)
```

### 8.5 Evaluation

**Protocol:**
- "Normal" data: EEGMMIDB S001–S050 runs 5-10 (4,500 trials, already embedded)
- "Anomalous" data: artificially injected artifacts:
  - Flat-line (zero variance)
  - Gaussian noise injection (SNR 0, 5, 10 dB)
  - Channel dropout (random 5–15 channels zeroed)
  - Spectral spike (add 50 Hz line noise × 10)
  - Slow-wave injection (add 0.5-2 Hz oscillation)
- 50-fold LOSO: train Mahalanobis on 49 subjects, test on held-out subject
- Metrics computed per-subject, then averaged

**Metrics:**
| Metric | Definition | Acceptance Threshold |
|--------|-----------|---------------------|
| AUROC | Area under ROC curve | ≥ 0.90 |
| AUPRC | Area under PR curve | ≥ 0.80 |
| Sensitivity @ 5% FPR | True positive rate at 5% false positive rate | ≥ 0.70 |
| Specificity @ 5% FPR | True negative rate at 5% FPR | ≥ 0.95 |
| False discovery rate | % anomaly flags that are false | ≤ 0.10 |

**Baselines:**
| Baseline | AUROC (expected) | Why relevant |
|----------|-----------------|-------------|
| Mahalanobis on Joint-2312 (proposed) | ~0.90+ | Full 2312-D multivariate distance |
| Z-score on PCA-32 band-power | ~0.70-0.80 | Current fallback approach |
| Z-score on Joint-2312 L2-norm | ~0.60-0.70 | Simple magnitude anomaly |
| Euclidean distance from centroid | ~0.65-0.75 | Univariate distance |
| Isolation Forest on PCA-32 | ~0.75-0.85 | Non-parametric alternative |

**Statistical test:** DeLong's test for paired AUROC comparison, Bonferroni-corrected (α=0.05/4=0.0125 for 4 baselines).

### 8.6 Safety Constraints

**Allowed claims:**
- "This service flags EEG windows with atypical spectral/temporal patterns"
- "Anomaly scores reflect deviation from the normal embedding distribution"
- "Results are screening-level only and require expert review"

**Prohibited claims:**
- ❌ "Detects seizures" / "Detects epilepsy"
- ❌ "Diagnoses" / "Identifies pathology"
- ❌ "Medical-grade" / "clinically validated"
- ❌ Any accuracy guarantees for clinical use

**Safety measures:**
- All anomalies require human review
- Service labeled "non-clinical, screening-level"
- Confidence intervals on all scores
- Audit log of all anomaly flags
- Opt-in for pathology datasets (TUH, CHB-MIT) — must be in separate evaluation, not training

### 8.7 Implementation Phases

**Phase 1 → Normal Data Characterization (Weeks 1-2)**
- Compute μ and Σ of 2312-D Joint-2312 embeddings from EEGMMIDB (4,500 trials)
- Analyze distribution: Is it Gaussian? (Shapiro-Wilk, per-dimension)
- Calibrate threshold: z > 3.5 → ~0.5% false positive rate
- Create `anomaly_detection_results` table

**Phase 2 → Artifact Injection (Weeks 2-3)**
- Create synthetic anomaly injection utilities
- Inject 5 anomaly types × 4 severity levels × 4,500 trials = 90,000 test cases
- Verify all injections produce detectable signal in 2312-D space

**Phase 3 → Mahalanobis Detector (Weeks 3-5)**
- Implement Mahalanobis distance computation (2312×2312 covariance inverse)
- Handle high-dimensional covariance: use pseudo-inverse with regularization
- Implement z-score normalization using MAD (robust)
- Calibrate threshold on normal data
- Unit tests (15) + E2E tests (5)

**Phase 4 → Evaluation (Weeks 5-7)**
- Run 50-fold LOSO evaluation
- Compute AUROC, AUPRC, sensitivity/specificity at 5% FPR
- Compare against 4 baselines
- Statistical significance testing
- Append `m31-anomaly-mahalanobis` to archive

**Phase 5 → API Integration (Weeks 7-9)**
- `POST /api/joint2312/anomaly/detect` route
- Batch + streaming modes
- Response: anomaly score, threshold, confidence, provenance
- Browser fallback: z-score on V2-32 (32-D)

**Phase 6 → Safety & Beta (Weeks 9-12)**
- Safety review (prohibited claims enforcement)
- Audit log implementation
- Internal beta with 3 researchers
- M31 anomaly report

### 8.8 Definition of Done

- [ ] Mahalanobis detector achieves AUROC ≥ 0.90 on 50-fold LOSO
- [ ] Sensitivity ≥ 0.70 at 5% FPR
- [ ] False discovery rate ≤ 0.10 on synthetic anomalies
- [ ] Statistical significance vs all 4 baselines (Bonferroni p < 0.0125)
- [ ] `POST /api/joint2312/anomaly/detect` returns anomaly score + provenance
- [ ] Safety constraints enforced (no clinical claims, all anomalies require review)
- [ ] Audit log of all anomaly flags with model_id + threshold + metadata
- [ ] 20 unit tests + 8 E2E tests pass
- [ ] Browser fallback (z-score on V2-32) produces reasonable scores
- [ ] M31 anomaly validation report written
- [ ] `benchmark_archive.json` appended

---

## 9. Dataset Roadmap

**P0 = immediately available/required | P1 = required soon | P2 = future validation | P3 = optional**

| # | Dataset | Subjects | Recordings | Channels | SR | Labels | Modality | License | Availability | Current Support | Purpose | Effort | Priority |
|---|---------|----------|------------|----------|----|--------|----------|---------|-------------|-----------------|--------|----------|
| 1 | **PhysioNet EEGMMIDB** | 50 (S001–S050) | 6 runs each (4,500 trials) | 64 | 160 | 4-class MI | Motor imagery | CC-BY-4.0 | ✅ Repo (benchmarked) | `src/lib/eeg/parsers/` | All 3 services (existing embeddings) | N/A | **P0** |
| 2 | **BCI-IV-2a** | 9 | 4 sessions | 22 | 250 | 4-class MI | Motor imagery | BSD-3 | ✅ Repo (training) | Training pipeline | Cross-dataset validation | Low | **P0** |
| 3 | **SEED** | 15 | 3 sessions (675 trials) | 62 | 200 | valence, arousal, dominance | Emotion/visual | CC-BY-NC-SA | ⚠️ Need loader | Datasets manifest only | Cognitive services | Medium | **P1** |
| 4 | **Sleep-EDF (expanded)** | 99 | 2 nights each | Fpz-Cz, Pz-Oz, EOG | 100/128 | 5-stage sleep | Sleep | BSD-3-Clause | ⚠️ Need loader | DEEPTECH_ANALYSIS only | Sleep analysis | Medium | **P1** |
| 5 | **DEAP** | 32 | 1 per subject | 14 | 128 | valence, arousal, dominance | Emotion | CC-BY-4.0 | ⚠️ Need loader | Datasets manifest only | Cognitive validation | Medium | **P1** |
| 6 | **DREAMER** | 23 | 1 per subject | 14 | 128 | valence, arousal, dominance | Emotion | CC-BY-4.0 | ⚠️ Need loader | Datasets manifest only | Cognitive validation | Low | **P2** |
| 7 | **TUH EEG Abnormal** | 10,000+ | variable | 256 (varies) | 100-256 | normal/abnormal | Pathology | CC-BY-4.0 | ⚠️ Need download | None | Anomaly validation | High | **P2** |
| 8 | **CHB-MIT** | 22 | long-term | 256 | 256 | seizure onset | Seizure | BSD-3 | ⚠️ Need download | None | Anomaly validation | Medium | **P2** |
| 9 | **THINGS-EEG** | 10 | 1 per subject | 62 | 200 | 1920 images | Vision | CC-BY-4.0 | ⚠️ Need loader | None | EEG2Image prep | Medium | **P2** |
| 10 | **DOTS** | 6 | 1 per subject | 64 | 1000 | 9 locations | Attention | CC-BY-4.0 | ⚠️ Need loader | None | Attention decoding | Low | **P3** |
| 11 | **DROZY** | 60 | 1 per subject | 64 | 128 | 5 fatigue stages | Fatigue | CC-BY-4.0 | ⚠️ Need loader | None | Fatigue detection | Medium | **P3** |
| 12 | **MASS** | 174 | 2-3 nights | 20 | 256 | 5-stage sleep | Sleep | CC-BY-4.0 | ⚠️ Need loader | None | Sleep validation | Medium | **P2** |
| 13 | **BIDS-VISUAL** | variable | variable | variable | variable | visual stimuli | Vision | CC-BY-4.0 | ⚠️ Need loader | None | Attention + EEG2Image prep | Medium | **P3** |

### 9.1 Channel Compatibility Analysis

| Dataset | Channels | CBraMod 19-ch | V2 22-ch | EEGPT 62-ch | Compatible? |
|---------|----------|---------------|----------|-------------|-------------|
| EEGMMIDB | 64 (10-10) | ✅ Superset | ✅ Superset | ✅ Superset (PO5/PO6 interp) | ✅ Full |
| BCI-IV-2a | 22 (10-20) | ✅ Superset (19 ⊆ 22) | ✅ Exact | ⚠️ Need 40 channels zero-filled | ⚠️ CBraMod+V2 only for EEGPT |
| SEED | 62 (10-10) | ✅ Superset | ✅ Superset | ✅ Exact | ✅ Full |
| Sleep-EDF | 7 (Fpz-Cz, Pz-Oz, EOG) | ❌ 4/7 match | ❌ 4/7 match | ❌ 0/7 match | ❌ Requires remapping |
| DEAP | 14 | ✅ Superset | ✅ Superset | ⚠️ Need 48 zero-filled | ⚠️ CBraMod+V2 only |
| TUH | 256 | ✅ Superset | ✅ Superset | ✅ Superset | ✅ Full |
| THINGS-EEG | 62 | ✅ Superset | ✅ Superset | ✅ Exact | ✅ Full |
| DOTS | 64 | ✅ Superset | ✅ Superset | ✅ Superset | ✅ Full |

**Key finding:** Sleep-EDF is the only major downstream dataset with channel incompatibility (7 channels vs 62 required). This requires either a sleep-specific Joint-2312 variant or a channel interpolation strategy. All other datasets are compatible.

### 9.2 Preprocessing Alignment

| Dataset | SR | Bandpass | Window | M30 Compatible? |
|---------|----|---------|--------|-----------------|
| EEGMMIDB | 160 | 4-38 Hz (CBraMod), 1-40 Hz (EEGPT) | 4s @ 250 Hz | ✅ (requires resample to 250 Hz) |
| BCI-IV-2a | 250 | 4-38 Hz | 4s | ✅ Direct |
| SEED | 200 | 1-40 Hz | 4s @ 200 Hz → resample to 250 | ✅ |
| Sleep-EDF | 100/128 | 0.5-40 Hz | 30 min epochs → 4s windows | ⚠️ Non-standard window size |
| DEAP | 128 | 1-40 Hz | Variable | ✅ |
| TUH | 100-256 | 1-40 Hz | Variable | ✅ |
| THINGS-EEG | 200 | 1-40 Hz | 4s | ✅ |

---

## 10. Experiment Roadmap

All experiments follow the established M18-M28 protocol: 50-fold (or n-fold for smaller datasets) Leave-One-Subject-Out (LOSO) cross-validation, session-disjoint evaluation, train-only weight learning, seed=42, Bonferroni correction.

### Experiment 1: Baseline Reproduction
- **ID:** `m31-subject-identity-baseline`
- **Objective:** Reproduce M27 Joint-2312 retrieval on PhysioNet EEGMMIDB (R@5=0.8527)
- **Dataset:** EEGMMIDB S001-S050, runs 5-10
- **Model:** Joint-2312 (frozen, no head)
- **Baseline:** Joint-264 (M25, R@5=0.7858), PCA-32 (R@5=0.7404)
- **Variables:** Block weights (fixed vs learned), window aggregation method
- **Evaluation:** 50-fold LOSO, session-disjoint, R@1/5/10/MRR
- **Metric:** R@5 (primary), R@1, R@10, MRR
- **Expected output:** R@5 ≥ 0.85
- **Success threshold:** p < 0.0125 vs Joint-264, Cohen's d > 0.5
- **Artifact:** `reports/.m27_augmented_joint_2312_results.json` (already exists)
- **Report:** Part of `reports/MISSION31_TIER1_ROADMAP.md`

### Experiment 2: Subject Identification on SEED
- **ID:** `m31-subject-identity-seed`
- **Objective:** Demonstrate subject identity works on a different dataset
- **Dataset:** SEED (15 subjects × 3 sessions)
- **Model:** Joint-2312 (frozen) → mean-pool → cosine similarity
- **Baseline:** V2-32 (Tier-1), PCA-32
- **Variables:** Session-disjoint split (train 2 sessions, test 1)
- **Evaluation:** 15-fold LOSO, session-disjoint
- **Metric:** R@1/5/10/MRR
- **Expected output:** R@5 ≥ 0.75
- **Success threshold:** p < 0.05 vs V2-32
- **Artifact:** `reports/m31-subject-identity-seed-results.json`
- **Report:** Appended to M31 report

### Experiment 3: Cognitive Workload Linear Probe
- **ID:** `m31-cognitive-workload-probe`
- **Objective:** Train linear probe on Joint-2312 for workload prediction
- **Dataset:** SEED (15 subjects, valence/arousal/dominance + derived workload)
- **Model:** Joint-2312 (frozen) → Ridge regression (2312 → 1)
- **Baseline:** Heuristic θ/α (existing), V2-32 + Ridge, CBraMod-200 + Ridge
- **Variables:** Regularization strength (C=0.001, 0.01, 0.1, 1.0, 10.0)
- **Evaluation:** 15-fold LOSO, session-disjoint
- **Metric:** R² (primary), RMSE, Pearson r
- **Expected output:** R² ≥ 0.40
- **Success threshold:** R² ≥ 0.40, p < 0.0125 vs all baselines
- **Artifact:** `reports/m31-cognitive-workload-probe-results.json`
- **Report:** `reports/M31_COGNITIVE_WORKLOAD_VALIDATION.md`

### Experiment 4: Cognitive Workload MLP
- **ID:** `m31-cognitive-workload-mlp` (conditional — only if Experiment 3 fails)
- **Objective:** Try small MLP if linear probe insufficient
- **Dataset:** SEED
- **Model:** Joint-2312 → 2312→64→32→1 MLP (dropout 0.3, weight decay 0.01)
- **Baseline:** Experiment 3 (linear probe)
- **Variables:** Hidden layer sizes, dropout rate
- **Evaluation:** 15-fold LOSO
- **Metric:** R²
- **Expected output:** R² ≥ 0.40
- **Success threshold:** R² ≥ 0.40 AND outperforms linear probe (p < 0.05)
- **Artifact:** `reports/m31-cognitive-workload-mlp-results.json`
- **Report:** Appended to cognitive validation report

### Experiment 5: Block Weighting for Workload
- **ID:** `m31-cognitive-block-weighting` (conditional — only if M3 linear probe meets threshold)
- **Objective:** Learn task-specific block weights (CBraMod, V2, PCA, EEGPT) for workload
- **Dataset:** SEED
- **Model:** Joint-2312 with learned block weights → Ridge regression
- **Baseline:** M31-cognitive-workload-probe (equal weights)
- **Variables:** Block weights (learn 4, simplex-constrained)
- **Evaluation:** 15-fold LOSO (train-only weights)
- **Metric:** R²
- **Expected output:** R² ≥ 0.40, improvement over equal weights
- **Success threshold:** ΔR² ≥ 0.05 vs equal weights, p < 0.05
- **Artifact:** `reports/m31-cognitive-block-weighting-results.json`
- **Report:** Appended to cognitive validation report

### Experiment 6: Anomaly Detection Calibration
- **ID:** `m31-anomaly-calibration`
- **Objective:** Calibrate Mahalanobis threshold on normal data
- **Dataset:** EEGMMIDB S001-S050 runs 5-10 (4,500 normal trials)
- **Model:** Joint-2312 → compute μ, Σ → Mahalanobis → z-score (MAD-normalized)
- **Baseline:** Z-score on PCA-32 band-power (existing heuristic)
- **Variables:** Threshold (z=3.0, 3.5, 4.0, 4.5)
- **Evaluation:** None (unsupervised calibration)
- **Metric:** False positive rate on normal data
- **Expected output:** FPR ≤ 0.5% at z=3.5
- **Success threshold:** FPR = 0.3–0.7% (matches expected normal distribution)
- **Artifact:** `reports/m31-anomaly-calibration-results.json`
- **Report:** Appended to M31 report

### Experiment 7: Anomaly Detection Evaluation
- **ID:** `m31-anomaly-evaluation`
- **Objective:** Evaluate anomaly detection on synthetic artifacts
- **Dataset:** EEGMMIDB (normal) + 5 injected artifact types × 4 severities
- **Model:** Joint-2312 → Mahalanobis → z-score → anomaly flag
- **Baseline:** Z-score on PCA-32, Euclidean distance from centroid, Isolation Forest on PCA-32
- **Variables:** Artifact type, severity level
- **Evaluation:** 50-fold LOSO (Mahalanobis fit on 49 subjects)
- **Metric:** AUROC (primary), AUPRC, sensitivity @ 5% FPR, FDR
- **Expected output:** AUROC ≥ 0.90
- **Success threshold:** AUROC ≥ 0.90, p < 0.0125 vs all baselines
- **Artifact:** `reports/m31-anomaly-evaluation-results.json`
- **Report:** `reports/M31_ANOMALY_VALIDATION.md`

### Experiment 8: Browser Projection Validation
- **ID:** `m31-browser-projection`
- **Objective:** Validate that a 2312→32 projection preserves task performance
- **Dataset:** EEGMMIDB (subject identity), SEED (workload), synthetic anomalies
- **Model:** Joint-2312 block weights → project to V2-32 → task head
- **Baseline:** Full 2312-D Joint-2312
- **Variables:** Projection method (block-weighted sum, PCA, LDA)
- **Evaluation:** Task-specific metrics (R@5, R², AUROC)
- **Metric:** ΔMetric vs 2312-D (should be within 5pp)
- **Expected output:** ΔR@5 ≤ 0.05 for subject identity
- **Success threshold:** Performance loss < 5pp for all tasks
- **Artifact:** `reports/m31-browser-projection-results.json`
- **Report:** Appended to M31 report

---

## 11. Baselines

### 11.1 Subject-Identity & Cohort Similarity

| Baseline | Type | R@5 | R@1 | MRR | Implementation |
|----------|------|------|----|----|----------------|
| **Joint-2312 (proposed)** | Frozen 4-block | 0.8527 | 0.6438 | 0.7361 | `/api/eeg/embed/foundation?model=joint-2312` |
| Joint-264 | Frozen 3-block | 0.7858 | 0.5271 | 0.6425 | M25 production route |
| EEGPT-2048 | Single block | 0.8118 | 0.5391 | 0.6584 | `embedEEGPTWindows` (existing) |
| PCA-32 | Band-power | 0.7404 | 0.4856 | 0.6016 | Tier-1 JS fallback |
| CBraMod-200 | Single block | 0.5276 | 0.2427 | 0.3775 | `embedFoundationWindows` |
| V2-32 | Single block | 0.2158 | 0.0687 | 0.1568 | Tier-1 production |
| Band-power centroid | Statistical | ~0.70 | ~0.45 | ~0.55 | No model needed |

### 11.2 Cognitive State Intelligence

| Baseline | Type | R² (expected) | Workload marker | Implementation |
|----------|------|---------------|-----------------|----------------|
| **Joint-2312 + Ridge (proposed)** | Frozen 4-block + linear | ≥0.40 | All 4 blocks | New head on frozen embedding |
| Heuristic θ/α ratio | Statistical | ~0.25 | Theta/Alpha | Existing `decoder/index.ts` |
| V2-32 + Ridge | 32-D linear | ~0.30 | Attention-weighted | `cognitive-decoder-v0.onnx` pattern |
| CBraMod-200 + Ridge | 200-D linear | ~0.30 | Fourier spectral | `embedFoundationWindows` + Ridge |
| EEGPT-2048 + Ridge | 2048-D linear | ~0.35 | Temporal attention | `embedEEGPTWindows` + Ridge |
| PCA-32 + Ridge | 32-D linear | ~0.30 | Band-power | Existing PCA adapter |

### 11.3 Anomaly Detection

| Baseline | Type | AUROC (expected) | Method | Implementation |
|----------|------|------------------|--------|----------------|
| **Mahalanobis on Joint-2312 (proposed)** | Multivariate distance | ≥0.90 | Covariance-normalized distance | New on 2312-D space |
| Z-score on PCA-32 | Univariate | ~0.70-0.80 | Per-band thresholding | Existing heuristic |
| Z-score on Joint-2312 norm | Magnitude | ~0.60-0.70 | L2 norm thresholding | Simple statistic |
| Euclidean from centroid | Distance | ~0.65-0.75 | Euclidean distance | Simple |
| Isolation Forest on PCA-32 | Non-parametric | ~0.75-0.85 | Random forest | scikit-learn style |

---

## 12. Anti-Leakage Strategy

### 12.1 Leakage Types and Prevention

| Leakage Type | Risk | Prevention Strategy |
|-------------|------|-------------------|
| **Subject leakage** | High — same subject in train and test inflates all metrics | 50-fold LOSO: each subject held out exactly once; never in both train and test |
| **Session leakage** | High — sessions from same recording share artifacts | Session-disjoint: query = 1 run; pool = all other runs × all subjects. The held-out run is excluded entirely. |
| **Window leakage** | Medium — overlapping windows share samples | Use non-overlapping windows for evaluation; 4s windows with 50% overlap → evaluate on even-indexed windows only (no overlap) |
| **Preprocessing leakage** | Medium — fit-on-all contaminates test statistics | StandardScaler, PCA fit on training folds only per LOSO fold; bandpass filters are zero-phase and do not leak |
| **Normalization leakage** | Medium — global L2 norm includes test data | L2-normalize per-fold on training data only; apply training normalization to test |
| **Hyperparameter leakage** | Medium — tuning on test set | All hyperparameters (Ridge C, threshold, block weights) learned from training subjects only within each fold; no cross-fold tuning |
| **Embedding caching leakage** | Low — precomputed embeddings may be contaminated | Verify SHA-256 of cached embeddings matches manifest; recompute if mismatch; cache key includes dataset + preprocessing hash |
| **Test-set contamination** | Low — results from prior experiments in test set | Use fresh evaluation scripts; assert all metrics differ from prior by >ε; append to archive, never overwrite |

### 12.2 LOSO Protocol Enforcement

```
For each fold f in {0, ..., 49}:
  Test subject = subject[f]
  Train subjects = all subjects except subject[f]

  Training phase (fold f):
    1. Load Joint-2312 embeddings for train subjects only
    2. Fit task head on train embeddings (Ridge/linear probe)
    3. Learn block weights from train Ridge coefficients (if applicable)
    4. Compute μ, Σ on train embeddings (for anomaly detection)
    5. Calibrate threshold on train data (MAD, percentile)

  Evaluation phase (fold f):
    1. Load Joint-2312 embeddings for test subject only
    2. Apply trained task head / Mahalanobis to test embeddings
    3. Compute metrics on test (R@5, R², AUROC, etc.)
    4. Record per-fold metrics

  Aggregation:
    - Mean + std across 50 folds
    - Bootstrap 95% CI (2000 resamples, seed=42)
    - Paired t-test: Joint-2312 vs each baseline
    - Bonferroni correction: α = 0.05 / (number of comparisons)
```

### 12.3 Experiment Isolation

Each M31 experiment:
- Has a unique ID in `benchmark_archive.json` (e.g., `m31-cognitive-workload-probe`)
- Produces a unique results JSON (never overwrites prior results)
- Appends exactly one record to the archive via an append script
- Uses a fresh Python script in `scripts/tmp/` with a unique name
- Produces a report in `reports/` with a unique name

---

## 13. API Roadmap

### 13.1 Tier-1 Service APIs

All APIs follow the existing pattern from `src/routes/api/eeg/embed/foundation.ts`:
- `createFileRoute` (TanStack Start)
- `.server.ts` suffix for server-only modules
- `authenticateRequest()` for auth
- `checkRateLimit()` for rate limiting (20 req/min/user, 120s timeout)
- `handleCors()` for CORS
- `applySecurityHeaders()` for security
- Response includes `provenance`, `timings`, `model` fields

#### 13.1.1 Subject Identity APIs

```typescript
// POST /api/joint2312/similarity/search
interface SimilaritySearchRequest {
  query_embedding?: number[];     // optional: skip re-embedding if already have one
  embedding_id?: string;           // optional: reference existing Joint-2312 embedding
  query_type: "subject_identification" | "session_similarity" | "cohort_similarity";
  match_count?: number;           // default 10
  filter_cohort_id?: string;      // cohort filter
  filter_subject_ids?: string[];  // exclude these subjects
  threshold?: number;             // minimum similarity (0.0-1.0)
}

interface SimilaritySearchResponse {
  service: "subject-identity";
  model: "onnx-cbramod-joint-2312";
  query_type: string;
  provenance: Joint2312Provenance;
  results: SimilarityResult[];
  metadata: {
    match_count: number;
    threshold: number;
    total_matches: number;
  };
  timings: {
    embed_ms?: number;
    search_ms: number;
    total_ms: number;
  };
}

interface SimilarityResult {
  rank: number;
  embedding_id: string;
  subject_id: string;
  session_id?: string;
  similarity: number;
  confidence: number;
  metadata: Record<string, unknown>;
}
```

#### 13.1.2 Cognitive State APIs

```typescript
// POST /api/joint2312/cognitive/decode
interface CognitiveDecodeRequest {
  embedding_id?: string;          // reference existing embedding
  target_state: "workload" | "attention" | "arousal";
  window_aggregation: "mean" | "max" | "per-window";  // default: "mean"
}

interface CognitiveDecodeResponse {
  service: "cognitive-intelligence";
  model: "cognitive-joint2312-linear-v1";
  provenance: {
    embedding_artifacts: Joint2312Provenance;
    head_model: string;
    head_sha256: string;
    dataset: string;
  };
  results: CognitiveResult[];
  timings: { embed_ms?: number; inference_ms: number; total_ms: number; };
}

interface CognitiveResult {
  window_index: number;
  [state: string]: number | [number, number] | string;
  // e.g. workload: 0.73, workload_ci: [0.65, 0.81]
}
```

#### 13.1.3 Anomaly Detection APIs

```typescript
// POST /api/joint2312/anomaly/detect
interface AnomalyDetectRequest {
  embedding_id?: string;
  threshold?: number;             // z-score threshold (default 3.5)
  aggregate_window?: number;      // consecutive anomalies to flag (default 5)
}

interface AnomalyDetectResponse {
  service: "anomaly-detection";
  model: "anomaly-joint2312-mahalanobis-v1";
  provenance: {
    embedding_artifacts: Joint2312Provenance;
    method: "mahalanobis-distance";
    threshold: number;
  };
  results: AnomalyResult[];
  aggregate_flags: AggregateAnomaly[];
  stats: {
    n_windows: number;
    n_anomalies: number;
    false_positive_rate: number;
    mean_z_score: number;
    std_z_score: number;
  };
  timings: { embed_ms?: number; inference_ms: number; total_ms: number; };
}

interface AnomalyResult {
  window_index: number;
  anomaly_score: number;         // z-score
  raw_distance: number;          // Mahalanobis distance
  is_anomaly: boolean;
  per_block_contribution: {
    cbramod: number;
    v2: number;
    pca: number;
    eegpt: number;
  };
}
```

### 13.2 API Design Principles

1. **Reference by ID:** All APIs accept an optional `embedding_id` to reuse already-computed Joint-2312 embeddings (EMBED ONCE → REUSE MANY TIMES)
2. **Provenance transparency:** Every response includes full provenance (artifact SHAs, head model, dataset, evaluation metrics)
3. **Graceful degradation:** If server-side Joint-2312 is unavailable, fall back to V2-32 (32-D) with a `fellBack` flag
4. **Consistent response schema:** All APIs return `service`, `model`, `provenance`, `results`, `timings`
5. **Rate limiting:** 20 req/min per user (matching foundation route)

---

## 14. Database Roadmap

### 14.1 Existing Tables (No Changes)

| Table | Dimension | Migration | Purpose |
|-------|-----------|-----------|---------|
| `embeddings` | vector(32) | `20260711060000` | Tier-1 (V2-32) |
| `foundation_embeddings` | vector(200) | `20260814000000` | Tier-2 (CBraMod-200) |
| `joint_embeddings` | vector(264) | `20260711060000` + `20260711060100` | M25 (3-block) |
| `joint_embeddings_2312` | vector(2312) | `20260817000001` | M28 (4-block) ✅ |
| `eeg_analyses` | FLOAT8[] | `20260607000000` | Tier-1 cognitive states |
| `graph_subjects` | ltree | `20260711070000` | Concept graph |
| `graph_sessions` | ltree | `20260711070000` | Concept graph |
| `graph_windows` | FK→embeddings | `20260711070000` | Concept graph |

### 14.2 New Tables (M31 Migration)

```
Migration: 20260820000000_tier1_service_layer.sql

1. subject_similarity_results
   - FK: embedding_id → joint_embeddings_2312.id
   - Columns: query_type, subject_id, similarity, rank, is_true_match
   - RLS: user-scoped (auth.uid() = user_id)

2. cognitive_state_results
   - FK: embedding_id → joint_embeddings_2312.id
   - Columns: model_id, attention, workload, arousal, confidence, CIs
   - RLS: user-scoped

3. anomaly_detection_results
   - FK: embedding_id → joint_embeddings_2312.id
   - Columns: model_id, anomaly_score, threshold, is_anomaly
   - RLS: user-scoped

4. service_provenance
   - Tracks all service runs: service_id, embedding_id, head_model, dataset, metrics
   - RLS: user-scoped

5. Indexes on joint_embeddings_2312.metadata->>'subject_id', ->>'session_id'
   - Enables efficient subject/session filtering in ANN RPC

6. New RPCs:
   - match_subjects_by_similarity (session-disjoint, excludes self)
   - match_sessions_for_cohort (filters by cohort metadata)
   - get_anomalies_for_user (filter by is_anomaly=true)
```

### 14.3 Schema Relationship Diagram

```
auth.users
  │
  ├── joint_embeddings_2312 (vector(2312))
  │     │
  │     ├── subject_similarity_results (embedding_id FK)
  │     ├── cognitive_state_results (embedding_id FK)
  │     └── anomaly_detection_results (embedding_id FK)
  │
  ├── embeddings (vector(32))
  ├── foundation_embeddings (vector(200))
  ├── joint_embeddings (vector(264))
  ├── eeg_analyses (FLOAT8[])
  ├── graph_subjects / graph_sessions / graph_windows
  └── datasets (manifest)
```

---

## 15. Browser vs Server Strategy

### 15.1 Tier-1 Service Classification

| Service | Primary | Browser? | Fallback | Latency Requirement |
|--------|---------|---------|----------|---------------------|
| **Subject Identity** | Server | ❌ (2312-D) | V2-32 projection | Batch (< 3s) |
| **Cognitive State** | Hybrid | ✅ (V2-32) + Server | Heuristic | RT: < 600ms P95 |
| **Anomaly Detection** | Server | ❌ (2312-D) | Z-score on V2-32 | Batch (< 3s) |

### 15.2 Browser Path

For cognitive state intelligence, the browser path mirrors the existing pattern:

```
Browser (V2-32, 32-D, WASM):
  EEG → embedEEG() → V2-32 (32-D, L2-normalized) → cognitive head (small ONNX)
  → attention/workload/arousal → real-time dashboard

Server (Joint-2312, 2312-D):
  EEG → /api/eeg/embed/foundation?model=joint-2312 → 2312-D
  → cognitive head (linear probe) → batch report with uncertainty
```

The browser and server paths use different embedding dimensions (32 vs 2312) but the same task head architecture. The V2-32 path is the "fast preview" — lower accuracy but real-time. The Joint-2312 path is the "authoritative report" — higher accuracy but batch.

### 15.3 2312→32 Projection for Browser

For subject identity and anomaly detection on the browser, a projection from 2312-D to 32-D is needed:

1. **Block-weighted projection:** Use M27 block weights [0.3062, 0.1334, 0.1519, 0.3985] to weight the V2-32 block relative to others, then project the full 2312-D to 32-D via learned linear projection (train-only, LOSO). This is essentially the "Linear 32-D" approach from M18 but with block weighting.

2. **Direct V2-32:** Use V2-32 directly (32-D, already in vector(32) store) — this is the existing Tier-1 path. It provides the 32-D embedding but without the CBraMod/EEGPT/PCA enrichment.

3. **Concatenated V2+PCA (64-D → 32-D):** Concatenate V2-32 + PCA-32 (64-D), then apply PCA or LDA to project to 32-D. This uses 2 of the 4 blocks but omits CBraMod and EEGPT (which are server-only).

**Recommended: Direct V2-32** as the browser fallback. It's already implemented, tested, and validated (R@5=0.779 on 50 subjects). The Joint-2312 path is strictly server-side for Tier-1 services.

### 15.4 When to Compute Joint-2312

**Embed Once, Reuse Many Times** principle:
- Joint-2312 is computed once when the user uploads EEG
- The embedding is stored in `joint_embeddings_2312`
- All three Tier-1 services read from the stored embedding
- If the user wants to run multiple services on the same EEG, only one embedding computation is needed

```
Upload → Compute Joint-2312 → Store (vector(2312)) → Cache embedding_id
  │
  ├── Subject Identity (read stored embedding, search ANN)
  ├── Cognitive State (read stored embedding, run head)
  └── Anomaly Detection (read stored embedding, compute Mahalanobis)
```

This pattern is already used by the `/api/eeg/embed/foundation` route — it stores embeddings and returns the embedding IDs. The Tier-1 services simply add a `embedding_id` parameter to their request schemas.

---

## 16. Security/Privacy

### 16.1 Authentication & Authorization

**Reuse existing:** `authenticateRequest()` from `src/integrations/supabase/request-auth.ts`
- Bearer token verification via Supabase Auth
- Returns `userId + supabase client`
- All Tier-1 service tables use RLS policies (user-scoped):
  ```sql
  CREATE POLICY "Users can view own {service}_results" 
    ON public.{service}_results FOR SELECT TO authenticated
    USING (auth.uid() = user_id);
  ```

### 16.2 Data Minimization

- **Embeddings, not raw EEG:** Once Joint-2312 embeddings are stored, the raw EEG can be discarded (the embedding is a lossy compression). This follows the existing pattern where `eeg_analyses` stores the 32-D embedding, not the raw signal.
- **Subject anonymization:** Subject IDs in metadata are user-defined labels, not PII. The platform never stores names, birth dates, or medical record numbers.
- **Retention policy:** Embeddings and results are retained for 30 days by default (configurable per user). Users can delete their data via the API.

### 16.3 Encryption

- **In transit:** HTTPS everywhere (existing TLS termination in Vite/Nitro)
- **At rest:** Supabase Postgres provides transparent encryption; embeddings in `vector(N)` columns are stored encrypted
- **API keys:** Stored in environment variables (`.env`), never in code

### 16.4 Access Control

- **Per-service scopes:** Each API route checks `auth.uid()` against RLS policies
- **Rate limiting:** 20 requests/min per user (matching foundation route)
- **File upload limits:** 50MB max (matching foundation route)
- **Query depth limiting:** ANN search limited to K=100 max to prevent resource exhaustion

### 16.5 Audit Trail

Every Tier-1 service operation creates an audit entry:

```typescript
interface AuditEntry {
  user_id: string;
  service: string;            // "subject-identity" | "cognitive-intelligence" | "anomaly-detection"
  action: string;             // "embed" | "search" | "decode" | "detect"
  resource_id: string;        // embedding_id or result_id
  model: string;              // e.g. "onnx-cbramod-joint-2312"
  timestamp: string;
  client_ip?: string;
  request_hash?: string;      // SHA-256 of request body for idempotency
}
```

Stored in a `service_audit_log` table with RLS policies.

### 16.6 Security Review Checklist

- [ ] All new tables have RLS policies (user-scoped)
- [ ] All RPCs use `SECURITY DEFINER` with `SET search_path = public`
- [ ] No raw SQL injection (all queries via Supabase RPC or parameterized)
- [ ] No PII in embeddings or metadata
- [ ] Rate limiting applied to all endpoints
- [ ] File upload limits enforced
- [ ] SHA-256 verification on all model artifacts
- [ ] Input validation on all numeric parameters (no NaN, no infinity, bounded ranges)
- [ ] Error messages sanitized (no internal paths, no stack traces)

---

## 17. Testing Strategy

### 17.1 Test Categories

| Test Type | Scope | Pattern | Tooling |
|-----------|-------|---------|---------|
| **Unit tests** | Pure functions (fusion, scoring, normalization) | `vitest` | `src/lib/.../__tests__/` |
| **Component tests** | Single service (API route + head) | `vitest` + mocked DB | `src/routes/api/.../__tests__/` |
| **Integration tests** | Full route → DB → search | `vitest` + real ONNX runtime | `__tests__/integration/` |
| **E2E tests** | Browser + real embeddings | Playwright | `tests/browser/` |
| **Regression tests** | Existing Tier-1/Tier-2 unchanged | `vitest` (existing) | Existing test files |
| **Performance tests** | Latency, throughput | `autocannon` / custom | `scripts/tmp/` |
| **Security tests** | Input validation, auth, RLS | `vitest` + OWASP ZAP | `tests/security/` |

### 17.2 Test Patterns (Reuse Existing)

The existing tests provide templates:

1. **Fusion tests** (`joint-fusion-2312.test.ts`): Test pure functions with synthetic inputs
2. **E2E tests** (`joint-server.test.ts`): Load real ONNX artifacts, verify SHAs, run forward passes
3. **Browser tests** (`joint-2312-wasm-smoke-firefox.test.ts`): Playwright, Chromium + Firefox

Following this pattern:

```typescript
// Subject Identity
describe("subject-identity", () => {
  // Unit: similarity search response formatting
  // E2E: embed → store → search → identify (real CBraMod + EEGPT)
  // Browser: V2-32 fallback path
});

// Cognitive State
describe("cognitive-state", () => {
  // Unit: linear probe coefficient application
  // E2E: embed → Ridge → workload prediction (real Joint-2312)
  // Browser: V2-32 → cognitive head
});

// Anomaly Detection
describe("anomaly-detection", () => {
  // Unit: Mahalanobis distance computation
  // E2E: fit on normal → detect synthetic artifacts
  // Regression: all injections produce detectable signal
});
```

### 17.3 Test Coverage Targets

| Component | Target Coverage | Rationale |
|-----------|----------------|-----------|
| Service Layer (shared) | 95% | All branches of auth/rate-limit/provenance |
| Subject Identity | 90% | Core business logic |
| Cognitive Decoder Head | 85% | Linear algebra, edge cases |
| Anomaly Detector | 85% | Statistical edge cases |
| API Routes | 80% | Error handling paths |
| Browser fallback | 75% | Limited by browser execution constraints |

---

## 18. Scientific Validation Gates

### 18.1 Subject Identity Validation

| Gate | Metric | Threshold | Test |
|------|--------|-----------|------|
| G1.1 | R@5 on 50-subject LOSO | ≥ 0.85 | Reproduces M27 |
| G1.2 | R@1 on 50-subject LOSO | ≥ 0.64 | Reproduces M27 |
| G1.3 | Statistical significance vs Joint-264 | p < 0.0125 | Paired t-test |
| G1.4 | False positive rate @ 0.80 threshold | ≤ 5% | Threshold calibration |
| G1.5 | Cross-dataset generalization (SEED) | R@5 ≥ 0.75 | 15-fold LOSO |
| G1.6 | No test-set leakage | 0 injected test data in train | Code audit + assertion |

### 18.2 Cognitive State Validation

| Gate | Metric | Threshold | Test |
|------|--------|-----------|------|
| G2.1 | Workload R² on held-out | ≥ 0.40 | 15-fold LOSO on SEED |
| G2.2 | Workload RMSE | ≤ 0.20 | Normalized to 0-1 scale |
| G2.3 | Pearson r with ground truth | ≥ 0.65 | Correlations across subjects |
| G2.4 | Statistical significance vs baselines | p < 0.0125 | 4 baselines, Bonferroni |
| G2.5 | Cross-dataset generalization | R² ≥ 0.25 | DEAP (unseen, no training) |
| G2.6 | Browser fallback performance | ΔR² ≤ 5pp | V2-32 vs Joint-2312 |

### 18.3 Anomaly Detection Validation

| Gate | Metric | Threshold | Test |
|------|--------|-----------|------|
| G3.1 | AUROC on synthetic artifacts | ≥ 0.90 | 5 artifact types, 4 severities |
| G3.2 | Sensitivity @ 5% FPR | ≥ 0.70 | Per-fold, then averaged |
| G3.3 | False discovery rate | ≤ 0.10 | On normal data (false alerts) |
| G3.4 | Statistical significance vs baselines | p < 0.0125 | 4 baselines, Bonferroni |
| G3.5 | Threshold stability | σ(thresholds) < 0.10 | Across 50 folds |

### 18.4 Cross-Service Validation

| Gate | Metric | Threshold | Test |
|------|--------|-----------|------|
| G4.1 | Embed once, reuse many | 0 re-embeddings | Embedding_id passed between services |
| G4.2 | No regression in Tier-1/2 | R@5 unchanged | Existing tests pass (122 tests) |
| G4.3 | Provenance integrity | SHA-256 match | All artifacts verified at every service call |
| G4.4 | Latency budget | P95 < 2s per service | End-to-end including CBraMod + EEGPT |

---

## 19. Production Readiness Gates

### 19.1 Technical Readiness

| Gate | Requirement | Validation Method |
|------|-------------|-------------------|
| TR-1 | All unit tests pass | `vitest run` — 100% pass |
| TR-2 | All E2E tests pass | `vitest run` (onnxruntime-node) — 100% pass |
| TR-3 | Browser smoke tests pass | Playwright on Chromium + Firefox |
| TR-4 | Lint + typecheck clean | `bunx lint` + `bunx typecheck` |
| TR-5 | SHA-256 verification | All 4 artifact hashes verified at load |
| TR-6 | No regression in Tier-1/2 | Existing 122 tests pass |
| TR-7 | Rate limiting works | 429 response at >20 req/min |
| TR-8 | RLS policies enforced | User cannot access other users' data |
| TR-9 | Error handling | 400/401/403/408/422/424/429/500 responses correct |
| TR-10 | Timeout handling | 120s timeout, 408 on timeout |

### 19.2 Operational Readiness

| Gate | Requirement | Validation Method |
|------|-------------|-------------------|
| OR-1 | Metrics instrumented | `metrics.ts` — new counters/histograms |
| OR-2 | Prometheus metrics endpoint | `/api/public/metrics` returns valid format |
| OR-3 | Structured logging | All events logged with `log("info", ...)` |
| OR-4 | Health check endpoint | `/api/public/staging/metrics` or similar |
| OR-5 | Error alerting | Errors logged with user_id + request context |
| OR-6 | Audit trail | `service_audit_log` table populated on every request |

### 19.3 Scientific Readiness

| Gate | Requirement | Validation Method |
|------|-------------|-------------------|
| SR-1 | Validation on held-out data | 50-fold LOSO (subject identity), 15-fold (cognitive) |
| SR-2 | Statistical significance | Bonferroni-corrected p-values, Cohen's d |
| SR-3 | Baseline comparison | All baselines from §11 |
| SR-4 | Report written | `reports/M31_{SERVICE}_VALIDATION.md` |
| SR-5 | Archive appended | `benchmark_archive.json` — new experiment record |
| SR-6 | Reproducibility | Seed=42, deterministic inference verified |

---

## 20. Dependency Graph

```
Joint-2312 Production (M25-M29)  [COMPLETE — FROZEN]
        │
        ├── joint_embeddings_2312 table (vector(2312))
        ├── match_joint_embeddings_2312() RPC
        ├── match_joint_embeddings_2312_exact() RPC
        ├── channel selection (19/22/62-ch)
        ├── SHA-256 verification
        └── foundation.ts route (+model=joint-2312)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  TIER 1 SHARED SERVICE LAYER (M31)                     │
│                                                       │
│  ┌──────────────┐   ┌───────────────┐   ┌─────────┐│
│  │ ServiceReg.  │   │ DownstreamIdx │   │ TaskHead││
│  │ Registry     │   │ (NeuralVecIdx  │   │ Registry││
│  └──────────────┘   │  extension)    │   └─────────┘│
│                     └───────────────┘              │
│                                                       │
│  ┌─────────────────────────────────────────────────┐│
│  │  Database: subject_*, cognitive_*, anomaly_*    ││
│  │  tables + service_audit_log                     ││
│  └─────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────┘
        │
        ├─────────────┬─────────────┬─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Subject ID  │ │ Cognitive   │ │ Anomaly     │
│ Service     │ │ State       │ │ Detection   │
│             │ │ Service     │ │ Service     │
└─────────────┘ └─────────────┘ └─────────────┘
        │
        ├── depends on: SEED dataset (M31-SV-001)
        ├── depends on: Sleep-EDF loader (M31-SV-002)
        ├── depends on: synthetic artifact injection (M31-SV-003)
        └── depends on: M31 experiment roadmap (§10)
```

**Dataset dependencies:**
- Subject Identity: P0 — EEGMMIDB (existing) + P1 — SEED (needs loader)
- Cognitive State: P1 — SEED (needs loader) → P2 — DEAP (needs loader)
- Anomaly Detection: P0 — EEGMMIDB (existing) + synthetic artifacts (no dataset needed)

**Infrastructure dependencies:**
- All services depend on Joint-2312 (complete)
- Cognitive and Anomaly services depend on M31 shared service layer
- Subject Identity has no upstream dependencies beyond M31 shared layer

---

## 21. Execution Order

### 21.1 Critical Path (Must Happen in Order)

```
1. M31 Shared Service Layer
   → ServiceRegistry, DownstreamVectorIndex, TaskHeadRegistry
   → Database migration
   → Auth/rate-limit wiring
   ↓
2. Subject Identity & Cohort Similarity
   → Uses existing Joint-2312 embeddings (no new datasets)
   → Fastest path to validated result
   ↓
3. Cognitive State (Workload) + Anomaly Detection (parallel)
   → Both depend on Subject Identity service patterns
   → Both need SEED dataset (Cognitive) / synthetic artifacts (Anomaly)
   ↓
4. Cross-Service Validation
   → Embed once, reuse many times
   → Latency budgeting
   → No regression verification
   ↓
5. Tier 1 Beta
   → All 3 services in internal beta
   ↓
6. Tier 1 Production Candidates
   → Scientific + operational gates pass
```

### 21.2 Parallel Work

| Team | Tasks | Duration |
|------|-------|----------|
| Team A (shared infra) | Service Layer, DB migration, API framework | Weeks 1-3 |
| Team B (subject identity) | API routes, evaluation, validation | Weeks 2-8 |
| Team C (cognitive state) | SEED loader, linear probe, evaluation | Weeks 3-10 |
| Team D (anomaly detection) | Mahalanobis, artifact injection, evaluation | Weeks 3-10 |
| Team E (browser) | V2-32 projection, browser fallback | Weeks 4-12 |

### 21.3 Blocking Relationships

| Blocker | Blocks | Mitigation |
|---------|--------|------------|
| SEED dataset loader not implemented | Cognitive State Service | Can use PhysioNet EEGMMIDB as proxy (derive workload from MI difficulty) |
| Sleep-EDF not available | Sleep Analysis (Tier 2, not Tier 1) | Not blocking Tier 1 |
| Joint-2312 latency (CBraMod 55ms + EEGPT 800ms) | All services | Mitigated by "embed once, reuse many" pattern; latency is per-upload, not per-query |
| No cross-dataset generalization data | Validation claims | Validate on EEGMMIDB first; cross-dataset is a stretch goal |

### 21.4 Execution Order Justification

**Subject Identity first** because:
1. It uses existing Joint-2312 embeddings (no new datasets needed)
2. It has the strongest empirical evidence (R@5=0.8527, already validated)
3. It establishes the serving pattern (API → search RPC → results)
4. It has the lowest risk (frozen embeddings, no head training)
5. It provides reusable infrastructure for Cognitive and Anomaly services

**Cognitive and Anomaly in parallel** because:
1. They both depend on the Subject Identity service layer
2. Cognitive needs SEED (P1 dataset), Anomaly needs synthetic artifacts (no dataset)
3. They use different head types (regression vs distance) — low coupling

---

## 22. Milestone Plan

| Milestone | Objective | Dependencies | Experiments | Code | Data | Tests | Deliverable | Acceptance Criteria |
|-----------|-----------|-------------|-------------|------|------|-------|-------------|---------------------|
| **M31.0** | Tier 1 Architecture | M30 report | None | None | None | None | `MISSION31_TIER1_ROADMAP.md` | This report complete |
| **M31.1** | Shared Service Layer | M25-M29 complete | E1 (baseline repro) | ServiceLayer, migration | None | 15 unit | Shared layer code + DB migration | ServiceRegistry + migration pass tests |
| **M31.2** | Subject Identity API | M31.1 | E1, E2 | `/api/joint2312/*` routes | EEGMMIDB (existing) | 25 unit + 10 E2E | Subject Identity Service MVP | R@5 ≥ 0.85, API returns results |
| **M31.3** | Subject Identity Validation | M31.2 | E2 | None | EEGMMIDB (existing) | 5 eval | `M31_SUBJECT_IDENTITY_VALIDATION.md` | All scientific gates pass |
| **M31.4** | Cognitive State (Workload) | M31.1, SEED loader | E3, E4 (conditional), E5 | `cognitive/*.ts`, decoder heads | SEED (new loader) | 20 unit + 8 E2E | Cognitive Workload Service MVP | R² ≥ 0.40 on SEED |
| **M31.5** | Anomaly Detection | M31.1, artifact injection | E6, E7 | `anomaly/*.ts` | Synthetic artifacts | 20 unit + 8 E2E | Anomaly Detection Service MVP | AUROC ≥ 0.90 |
| **M31.6** | Cross-Service Validation | M31.2, M31.4, M31.5 | E8 | None | All | 10 integration | Cross-Service Validation Report | Embed-once-reuse-many passes |
| **M31.7** | Browser Fallback | M31.2 | E8 (browser) | Browser projection | None | 5 browser | Browser Fallback Implementation | V2-32 fallback < 600ms P95 |
| **M31.8** | Tier 1 Beta | M31.2, M31.4, M31.5 | E2, E3, E7 | Beta configuration | All | 15 smoke | Tier 1 Beta deployment | 3 internal users, no regressions |
| **M31.9** | Tier 1 Production Candidates | M31.8 | All | Production hardening | All | 30 full | Tier 1 Production Readiness Reports | All gates in §18-19 pass |

---

## 23. Effort Estimates

### By Milestone

| Milestone | Effort | Classification |
|-----------|--------|----------------|
| M31.0 — Architecture | 3-5 days | Research-heavy |
| M31.1 — Shared Layer | 1-2 weeks | Medium |
| M31.2 — Subject Identity | 2-3 weeks | Medium |
| M31.3 — Subject Identity Validation | 1 week | Medium |
| M31.4 — Cognitive State | 2-3 weeks | Research-heavy |
| M31.5 — Anomaly Detection | 2-3 weeks | Medium |
| M31.6 — Cross-Service Validation | 1 week | Medium |
| M31.7 — Browser Fallback | 1-2 weeks | Medium |
| M31.8 — Tier 1 Beta | 1 week | Medium |
| M31.9 — Production Candidates | 1-2 weeks | Medium |

### By Team

| Team | Total Effort | Person-Weeks |
|------|-------------|-------------|
| Shared Infrastructure | 2-3 weeks | 2.5 pw |
| Subject Identity | 3-4 weeks | 3.5 pw |
| Cognitive State | 3-5 weeks | 4 pw (incl. SEED loader) |
| Anomaly Detection | 2-3 weeks | 3 pw (incl. artifact injection) |
| Browser Fallback | 1-2 weeks | 1.5 pw |
| QA / Validation / Release | 2-3 weeks | 2.5 pw |
| **Total** | **13-18 weeks** | **17 pw (single team)** or **12-14 weeks (parallel teams)** |

---

## 24. Risks

### 24.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Joint-2312 latency too high | Medium | High | Embed-once pattern amortizes; cache embeddings; batch ANN search |
| SEED channel mismatch | Medium | Medium | SEED 62-channel is superset of CBraMod 19 + V2 22; EEGPT 62 exact |
| 2312-D to 32-D projection loses quality | High | Medium | Use V2-32 directly as browser fallback (already validated at R@5=0.779) |
| High-dimensional covariance singular (Mahalanobis) | Medium | High | Add regularization (shrinkage); use pseudo-inverse; fallback to PCA-32 covariance |
| pgvector ivfflat recall at 2312-D | Low | Medium | Use exact RPC (`match_joint_embeddings_2312_exact`) for validation; tune lists parameter |
| CORS/Security bypass | Low | High | Reuse existing `handleCors()` + `applySecurityHeaders()` middleware |

### 24.2 Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| SEED workload labels unreliable | Medium | High | Cross-validate on DEAP; use correlation across multiple labeling schemes |
| Anomaly detection false positive rate too high | Medium | Medium | Calibrate threshold with extreme care; use MAD not std; validate on multiple artifact types |
| Cognitive workload generalizes poorly | High | Medium | Validate on DEAP (different paradigm); report per-subject variance |
| No clinical dataset for anomaly detection | High | High | Use synthetic artifacts (guaranteed signal); clearly label as non-clinical |
| Block weighting degrades downstream tasks | Low | Medium | Test equal weights vs learned weights; revert if no improvement |

### 24.3 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Dataset licensing incompatibility | Medium | High | Pre-vet all datasets (§9.1); BSD/CC-BY only; no NC-only datasets in production |
| Rate limiting too strict | Low | Medium | Match existing 20 req/min; add per-service rate limit if needed |
| No monitoring on Day 1 | Medium | High | Instrument all metrics in M31.1; set up alerts in M31.8 |
| Rollout confusion with Tier-1 | Low | High | Tier-1 services are server-only; use separate routes, no `DEFAULT_PREFERRED` involvement |

---

## 25. Blockers

### 25.1 Hard Blockers (Must Resolve Before Implementation)

| Blocker | Resolution |
|---------|-----------|
| **No SEED loader** — Cognitive State service requires SEED dataset | Implement SEED loader as part of M31.4 (estimated 1 week); can use PhysioNet EEGMMIDB as interim |
| **No synthetic artifact generator** — Anomaly Detection requires synthetic artifacts | Implement artifact injection utilities as part of M31.5 (3-5 days) |
| **Sleep-EDF not available** — Not blocking Tier 1 (sleep is Tier 2) | ✅ Not a blocker for Tier 1 |

### 25.2 Soft Blockers (Mitigated by Workarounds)

| Blocker | Workaround |
|---------|-----------|
| CBraMod + EEGPT latency (~1s) per upload | Embed-once pattern; batch multiple windows; async processing |
| No browser path for 2312-D | Use V2-32 (32-D) as browser fallback; clearly mark accuracy difference |
| SEED license is CC-BY-NC-SA (non-commercial) | Use for research/beta only; switch to DEAP (CC-BY-4.0) for commercial |
| Cross-dataset generalization unproven | Validate on EEGMMIDB first; cross-dataset is stretch goal |

---

## 26. Complete Definition of Done

### 26.1 Architecture

- [ ] Tier 1 architecture documented and approved
- [ ] Shared Service Layer designed and reviewed
- [ ] Database schema extension designed and reviewed
- [ ] API contracts defined and versioned
- [ ] Browser/server split strategy documented

### 26.2 Implementation

- [ ] All Tier-1 service code implemented (subject identity, cognitive, anomaly)
- [ ] Shared Service Layer operational
- [ ] Database migration applied and tested
- [ ] All API routes functional (auth, rate-limit, CORS, error handling)
- [ ] SHA-256 verification on all artifacts
- [ ] Embed-once, reuse-many pattern enforced
- [ ] Browser fallback path implemented (V2-32)

### 26.3 Testing

- [ ] 50+ unit tests (all services + shared layer)
- [ ] 15+ E2E tests (real ONNX inference)
- [ ] 5+ browser smoke tests (Chromium + Firefox)
- [ ] 10+ integration tests (cross-service)
- [ ] 0 regressions in existing Tier-1/Tier-2 tests (122 existing tests)
- [ ] Lint + typecheck clean
- [ ] 95%+ coverage on shared layer, 85%+ on service-specific code

### 26.4 Scientific Validation

- [ ] Subject Identity: R@5 ≥ 0.85 on 50-fold LOSO (reproduces M27)
- [ ] Subject Identity: statistical significance vs Joint-264 (p < 0.0125)
- [ ] Subject Identity: cross-dataset generalization on SEED (R@5 ≥ 0.75)
- [ ] Cognitive State: workload R² ≥ 0.40 on 15-fold LOSO (SEED)
- [ ] Cognitive State: statistical significance vs baselines (p < 0.0125)
- [ ] Cognitive State: cross-dataset validation (DEAP, no training)
- [ ] Anomaly Detection: AUROC ≥ 0.90 on synthetic artifacts
- [ ] Anomaly Detection: sensitivity ≥ 0.70 @ 5% FPR
- [ ] Anomaly Detection: statistical significance vs baselines (p < 0.0125)
- [ ] All evaluations use train-only weight learning, session-disjoint, no leakage

### 26.5 Production Readiness

- [ ] All metrics instrumented (`metrics.ts`)
- [ ] Prometheus endpoint returns valid metrics
- [ ] Structured logging on all service events
- [ ] Health check endpoint responds
- [ ] Error alerts configured
- [ ] Audit trail populated on every request
- [ ] Rate limiting enforced (20 req/min/user)
- [ ] RLS policies enforced (user-scoped)
- [ ] P95 latency < 2000ms for server, < 600ms for browser
- [ ] Timeout handling (120s server, graceful degradation)
- [ ] Error messages sanitized (no internal leaks)

### 26.6 Documentation

- [ ] Validation reports for all 3 services (`reports/M31_*_VALIDATION.md`)
- [ ] `benchmark_archive.json` appended with all M31 experiment records
- [ ] API documentation (endpoint specs, response schemas)
- [ ] Security/privacy documentation (claims, disclaimers, data handling)
- [ ] Deployment guide (staging → beta → production)

---

## 27. Recommended Next Mission

### M32 — Implement Tier-1 Shared Service Layer

After M31 (this roadmap), the immediate next mission is **M32: implement the shared Tier-1 Service Layer infrastructure**.

**M32 objectives:**
1. Create the ServiceRegistry pattern (`src/lib/ai/decoders/registry.ts`)
2. Extend NeuralVectorIndex to support result tables (`src/lib/vector-search/tier1-index.ts`)
3. Create the `service_provenance` tracking system
4. Implement the M31 migration (3 result tables + audit log + indexes)
5. Wire auth/rate-limit/CORS to the new `/api/joint2312/` route group
6. Add metrics for all 3 services to `metrics/index.ts`
7. Implement the first service: Subject Identity & Cohort Similarity (full API + validation)

**M32 success criteria:**
- `POST /api/joint2312/similarity/search` works end-to-end
- `joint_embeddings_2312` embeddings reused across services (no re-embedding)
- R@5=0.8527 reproduced on 50-subject LOSO via the API
- All 3 services' shared infrastructure tested and validated
- `benchmark_archive.json` appended with `m31-subject-identity` record

After M32, M33 implements Cognitive State Intelligence and M34 implements Anomaly Detection, following the exact patterns established in M32.

---

*Report generated: 2026-08-13 · Neuro-Fabric Core M31 Planning Mission*
*All findings sourced from repository code, M25-M30 reports, and benchmark_archive.json (29 experiments)*
*No production code was modified during this planning mission*
