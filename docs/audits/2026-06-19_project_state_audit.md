# NeuroWeave / Neuro Fabric — Project State Audit

- **Date:** 2026-06-19
- **Scope:** project-wide (architecture, AI/EEG stack, data, deployment,
  product readiness). Read-only audit; no source files modified.
- **Inputs cross-referenced:**
  - `docs/AUDIT_REPORT.md` (2026-06-06 baseline)
  - `docs/REALITY_CHECK.md`, `docs/BLUEPRINT_PREPARATION.md`
  - `docs/architecture.md`, `docs/ai-layer-architecture.md`
  - `docs/adr/0001-braindecode-execution-strategy.md`
  - `docs/audits/2026-06-17_*` (8 reports incl. strategic-progress-audit)
  - `docs/audits/2026-06-19_eegconformer-{live-audit,routing-fix,runtime-verification}.md`
  - `docs/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md`
  - `docs/benchmarks/2026-06-17_maturity-benchmark.md`
  - Working tree: `src/lib/ai/**`, `src/routes/**`, `supabase/migrations/**`,
    `scripts/export_braindecode_eegconformer.py`, `public/models/eegconformer.onnx`

---

## 1. Executive Summary

Two months of work have moved NeuroWeave from a frontend-only EEG demo
(A1, 2026-06-06) through a persistence-backed authenticated app (A2,
mid-June) into a platform with a **functioning AI Foundation Layer** and
the first **live neural embedding model** in production routing. As of
2026-06-19 the EEGConformer ONNX artefact is shipped, served from
`public/models/eegconformer.onnx`, registered as
`braindecode-eegconformer-prod`, and confirmed end-to-end in the browser
(`fellBack: false`, `dim = 32`, `‖v‖ = 1.0000`) after a WASM-path fix to
the ONNX adapter.

The single biggest bottleneck identified by the 2026-06-17 strategic
audit — _"the trained EEGConformer ONNX file does not exist"_ — is now
resolved. However, the embedding-quality audit run on 2026-06-19 is
**INCONCLUSIVE**: on a synthetic motor-imagery probe the model showed a
negligible intra/inter-class separation (Cohen's d = 0.027, 1-NN cosine
= 30 % vs chance 25 %). The probe is plausibly OOD, but it means the
production embedding space has not been _empirically_ shown to be
discriminative on the device the user runs.

**Net verdict:** the platform has crossed the _technical_ foundation
threshold but has **not** crossed the _scientific validation_
threshold. It is suitable for prototype and research-platform use today,
gated for pilot, and not ready for an MVP that promises clinical or
decision-grade output.

**Final readiness score: 63 / 100** (up from 58 / 100 on 2026-06-17;
see §11 for derivation).

---

## 2. Progress Timeline

```text
2026-06-06   A1  AUDIT_REPORT.md            22/100   frontend-only, PCA + heuristics
2026-06-17   A2  delta-audit.md             41/100   Postgres+RLS+roles, ML scaffolds
2026-06-17   A3  strategic-progress-audit   58/100   AI Foundation Layer, ONNX runtime,
                                                     registry, validation, vector bridge,
                                                     EEGConformer wired but artefact missing
2026-06-19   A4  eegconformer-live-audit    →        artefact shipped under public/models/,
                                                     but DEFAULT_PREFERRED still pointed at
                                                     legacy id → silent PCA fallback
2026-06-19   A5  eegconformer-routing-fix   →        DEFAULT_PREFERRED flipped to
                                                     braindecode-eegconformer-prod
2026-06-19   A6  runtime-verification       →        WASM path pinned to jsdelivr CDN;
                                                     end-to-end PASS, dim=32, no fallback
2026-06-19   A7  embedding-quality          →        INCONCLUSIVE — separation margin 0.008;
                                                     model effectively behind feature flag
2026-06-19   *   THIS REPORT                63/100   first run with a live neural embedder
```

---

## 3. Architecture Assessment

### 3.1 Stack snapshot (verified against working tree)

- **Framework:** TanStack Start v1 + Vite 7 + React 19, Tailwind v4.
  File-based routing under `src/routes/` (24 route files including
  `_authenticated/` gate, `api/` server routes).
- **Backend:** Lovable Cloud (Supabase). 7 SQL migrations; tables
  include eeg_analyses, experiments, role tables. RLS + GRANTs present.
- **AI layer:** `src/lib/ai/{adapters,artifacts,benchmark,embeddings,
inference,models,validation,vector-bridge}` with 26/26 tests passing
  prior to this audit window.
- **Adapters registered:** `pca-legacy-v1`, `pytorch-export-placeholder`,
  `eegpt-placeholder`, `braindecode-eegnetv4-default`,
  **`braindecode-eegconformer-prod`** (live).
- **Inference orchestrator:** `src/lib/ai/inference/embed-eeg.ts`
  with cascading fallback EEGConformer → optional ONNX → PCA.
- **Server runtime:** TanStack `createServerFn` + `requireSupabaseAuth`
  middleware; no Edge Functions for app-internal logic.

### 3.2 Strengths

1. **Adapter pattern is real**, not aspirational. ONNXRuntime-web
   genuinely runs in the browser (verified 2026-06-19 with
   `fellBack: false`).
2. **Defence in depth on fallbacks** — every neural path has PCA as a
   verified terminal, with structured `ai.embed.*` logging on every
   transition.
3. **Validation discipline** — `validateEmbedding` (NaN/Inf/dim/zero) +
   L2 normalisation are mandatory, not opt-in.
4. **Vector bridge with model-id tagging** — `NeuralVectorIndex`
   prevents cross-model vector contamination, which makes A/B and
   rollback trivial.
5. **Benchmark harness** (`benchmarkAdapter`, `benchmarkAll`) records
   p50/p95/heap, ready to back SLOs.
6. **Auth + RLS posture** matches the role-table best practice (no
   role-on-profile, `has_role` SECURITY DEFINER, `_authenticated`
   route gate).
7. **Documentation density is unusually high** for a project this size
   — every architectural step has an audit, ADR, or roadmap.

### 3.3 Weaknesses & risks

1. **WASM dependency on a public CDN**
   (`cdn.jsdelivr.net/.../onnxruntime-web@1.26.0`). Single point of
   failure for inference; no SLA, CSP-incompatible in strict
   deployments.
2. **Model artefact is repo-bundled** (~3.2 MiB in `public/`). Adds
   weight to every cold load; no content-hashed URL, no signed-URL
   delivery, no SHA enforcement at load time despite
   `src/lib/ai/artifacts/` having the schema.
3. **No CI/CD pipeline** — tests pass locally but nothing gates a PR.
4. **No upload size cap, no rate limiting** on `/api/eeg/upload`.
5. **Vector index is in-memory only** — pgvector schema not wired.
   Loss on reload, no cross-device retrieval.
6. **DFT band-power is O(M²)** in `embeddings/features.ts`; latency
   ceiling visible at long windows.
7. **Cognitive decoder is heuristic** — `attention/workload/arousal`
   ratios; no trained model, despite TF.js / ONNX scaffolding.
8. **Dataset ingestion** (Sleep-EDF, CHB-MIT, TUH) is not wired;
   training/eval depend on synthetic or off-platform data.
9. **Architectural drift on hosting:** the 2026-06-17 deployment
   roadmap recommended **Option A — Lovable Cloud Storage public
   bucket with content-hashed path** for the ONNX artefact; the
   actual deployment took **Option B — app-bundled `public/models/`**,
   the option the roadmap explicitly flagged "only for offline demos".

---

## 4. AI / EEG Stack Assessment

| Layer                        | State                           | Evidence                                                    |
| ---------------------------- | ------------------------------- | ----------------------------------------------------------- |
| Signal I/O (EDF/CSV/NPY)     | Implemented                     | parsers + tests                                             |
| IIR bandpass / notch         | Implemented                     | `signal/` modules                                           |
| Band-power features          | Implemented (slow DFT)          | `embeddings/features.ts`                                    |
| Artifact rejection + quality | Implemented                     | `preprocessing/artifact-rejection`, `signal-quality/`       |
| Synthetic generator          | Implemented                     | `synthetic/`                                                |
| PCA embedder                 | Implemented                     | `pca-adapter.ts`                                            |
| ONNX runtime adapter         | Implemented (CDN-pinned WASM)   | `onnx-adapter.ts`                                           |
| Braindecode bridge           | Implemented                     | `braindecode-onnx-bridge.ts`                                |
| EEGConformer (prod)          | **Live**                        | `public/models/eegconformer.onnx` 3 360 306 B; runtime PASS |
| EEGNet v4 default            | Registered, untested in browser | `braindecode-eegnetv4-default`                              |
| EEGPT                        | Stub only                       | `eegpt-adapter.ts`                                          |
| Validation + L2 norm         | Implemented                     | `validation/`                                               |
| Benchmark harness            | Implemented                     | `benchmark/`                                                |
| Vector index (in-memory)     | Implemented                     | `vector-bridge/`                                            |
| Vector index (pgvector)      | **Missing**                     | no migration                                                |
| Cognitive decoder (trained)  | **Missing**                     | heuristic ratios                                            |
| EEG reconstruction           | Route scaffold only             | `src/routes/synthetic.tsx`                                  |
| EEG2IMG                      | Route scaffold only             | `src/routes/eeg2image.tsx`                                  |
| Empirical embedding quality  | **Inconclusive**                | synthetic probe only                                        |
| Real dataset evaluation      | Not done in-platform            | needs Colab/off-platform                                    |

**Routing chain (verified 2026-06-19):**

```
embedEEG()  →  braindecode-eegconformer-prod   (default)
            →  <opts.onnxModelId> (if provided)
            →  pca-legacy-v1                   (terminal, always available)
```

---

## 5. Deployment Assessment

- **Hosting:** Cloudflare Workers (TanStack Start template). Server
  fns and SSR run under `nodejs_compat`.
- **Public artefacts:** `public/models/eegconformer.onnx` ships in the
  static bundle. Egress charged per page load until a content-hashed
  bucket is adopted.
- **External runtime dep:** ORT WASM binaries fetched from jsdelivr
  CDN at first inference. Functional but operationally fragile (CDN
  outage → silent PCA fallback worldwide).
- **No CI** — neither `vitest run` nor typecheck blocks merges.
- **No CDN cache headers** documented for the model artefact.
- **Secrets posture:** no service-role key in client code (verified by
  file inspection); `supabaseAdmin` import isolated to `.server.ts`.
- **Email/auth:** Supabase auth integrated, Google OAuth instructions
  in place; `attachSupabaseAuth` wired in `src/start.ts`.

**Deployment readiness: usable for preview / demo. Not yet
production-hardened.**

---

## 6. Product Readiness Assessment

| Tier                          | Verdict                 | Why                                                                                                                                                               |
| ----------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prototype**                 | ✅ Ready                | Live neural embedder, end-to-end pipeline, auth, persistence.                                                                                                     |
| **Research platform**         | ✅ Ready (with caveats) | Adapter pattern + benchmark + structured logs make it a fine harness for ML researchers; however dataset loaders are missing, so research is BYO-data.            |
| **Pilot deployment**          | ⚠️ Conditional          | Embedding quality unproven on real data; CDN-WASM dependency; no rate limiting; in-memory vectors. Acceptable for an internal closed pilot, not a customer pilot. |
| **MVP (paid users)**          | ❌ Not ready            | No CI, no rate limiting, no upload cap, no SLO instrumentation, no cognitive decoder, no pgvector persistence, no validated discriminative embedding space.       |
| **Clinical / decision-grade** | ❌ Out of scope         | Would require IRB / regulatory pathway, none of which exists.                                                                                                     |

---

## 7. Risk Matrix

| #   | Risk                                                             | Likelihood  | Impact                | Mitigation                                                                                              |
| --- | ---------------------------------------------------------------- | ----------- | --------------------- | ------------------------------------------------------------------------------------------------------- |
| R1  | jsdelivr CDN outage → all inference falls back to PCA silently   | Medium      | High                  | Vendor ORT `.wasm` into `public/ort/`; pin `wasmPaths` to self-host.                                    |
| R2  | Embedding space not discriminative on real EEG                   | Medium      | Critical              | Run BCI-IV-2a holdout eval off-platform; gate model behind a flag tied to recall@10 vs PCA.             |
| R3  | No CI → regressions ship undetected                              | High        | Medium                | Add `bunx vitest run` + typecheck workflow on PR.                                                       |
| R4  | Unbounded upload / no rate limit                                 | Medium      | High                  | Add size cap + per-user quota on `/api/eeg/upload`.                                                     |
| R5  | Repo-bundled ONNX inflates app weight, no SHA verify             | High        | Low–Medium            | Move artefact to content-hashed Cloud Storage URL; record SHA in `artifacts/index.ts`.                  |
| R6  | In-memory vector index loses state on reload                     | High        | Medium                | Migrate to pgvector with cosine ANN.                                                                    |
| R7  | Heuristic cognitive decoder marketed as ML                       | Medium      | High (trust)          | Either retire feature, label "heuristic v0", or train decoder.                                          |
| R8  | Architectural drift between blueprint and implementation         | Medium      | Medium                | Either update the blueprint or schedule reconciliation work; do not let docs and code diverge silently. |
| R9  | Worker runtime ≠ Node — Node-only deps would break prod silently | Low (today) | High (when triggered) | Existing rules in `server-runtime` directive; codify with a CI guard.                                   |

---

## 8. Gap Analysis — claims vs implementation

### Claims supported by code today

- Browser-side ONNX inference is real (`onnx-adapter.ts`, runtime PASS).
- Model registry is multi-model and id-tagged.
- PCA fallback is verified terminal.
- Embedding validation is mandatory.
- Auth is enforced via TanStack `_authenticated` gate + Supabase RLS.
- EEGConformer is live in default routing.

### Claims still aspirational

- "Foundation models in the browser" — only EEGConformer is live; EEGPT
  is a stub; EEGNetv4 has a registration but no in-browser verification.
- "Cognitive intelligence (attention, workload, arousal)" — heuristic
  feature ratios, no trained decoder.
- "EEG → image reconstruction" — route scaffolds only, no model.
- "Vector search at scale" — works in-memory only.
- "Real datasets (Sleep-EDF, CHB-MIT, TUH)" — not ingested.
- "Discriminative neural embedding space" — not empirically supported on
  real EEG data.

---

## 9. Updated Blueprint (reconciled with reality)

The original Neuro Fabric blueprint (`docs/BLUEPRINT_PREPARATION.md`)
remains coherent in its **layered intent** (signal → embedding →
representation → cognitive → reconstruction → cross-modal). The drift
is in **delivery vehicles**, not in the architectural shape:

| Blueprint assumption                                        | Reality                            | Reconciliation                                             |
| ----------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| Foundation model served from object storage with signed URL | App-bundled in `public/models/`    | Move to Cloud Storage with content-hashed path before MVP. |
| ORT runtime self-hosted                                     | Pulled from jsdelivr CDN           | Vendor `.wasm` into `public/ort/`.                         |
| pgvector persistence                                        | In-memory only                     | Add migration + ANN index.                                 |
| Trained cognitive decoder                                   | Heuristic                          | Train a small decoder (TF.js or ONNX).                     |
| Multiple foundation models                                  | One live (EEGConformer), two stubs | Either ship EEGPT/EEGNet or remove the registrations.      |
| Real datasets                                               | Synthetic only                     | Wire Sleep-EDF (smallest realistic loader).                |

**Verdict:** the blueprint is still the right north star; the
implementation has reached its first layer (signal → embedding) end to
end and is _not yet_ into the second (representation persistence) or
third (cognitive decoding).

---

## 10. Revised 4-Month Roadmap

Goal: convert the current "live but unvalidated" platform into a
defensible **internal pilot** by month 4.

### Month 1 — Validate & harden the live path

- Off-platform: full BCI-IV-2a holdout eval (recall@10 vs PCA,
  intra/inter-class margin, classifier-head accuracy). Sign-off gate
  before any further EEGConformer claims.
- Vendor ORT WASM into `public/ort/`; remove jsdelivr dependency.
- Move `eegconformer.onnx` to Lovable Cloud Storage with content-hashed
  path; enforce SHA at adapter load.
- Add CI (`bunx vitest run`, typecheck) on PR.
- Add upload size cap + per-user rate limit on `/api/eeg/upload`.

### Month 2 — Persistence & retrieval

- pgvector migration + cosine ANN index, model-id tagged.
- Backfill existing in-memory vectors.
- Recall@10 dashboard wired to the benchmark harness.
- Replace O(M²) DFT with FFT in `embeddings/features.ts`.

### Month 3 — Cognitive decoder + dataset loader

- Train a small attention/workload decoder on a public dataset; ship as
  a second registered ONNX model (`cognitive-decoder-v1`).
- Retire the heuristic ratios behind a feature flag.
- Implement Sleep-EDF loader end-to-end (one real dataset is enough to
  retire the "no real datasets" risk).

### Month 4 — Pilot hardening

- SLO dashboards (P50/P95/fallback rate) productionised, alerting.
- Canary → beta → GA rollout per the existing
  `eegconformer-deployment-roadmap.md` Phase 4.
- Internal pilot with >= 3 design partners; recall@10 + qualitative
  feedback as exit criteria.
- Reassess EEGPT and reconstruction tracks against pilot signal.

Explicitly **out of scope** for this 4-month window: EEG2IMG,
generative reconstruction, multi-tenant scaling beyond pilot,
regulatory pathway.

---

## 11. Final Readiness Score

| Component                     |  Weight |  Score |   Contribution |
| ----------------------------- | ------: | -----: | -------------: |
| Signal processing             |      10 | 8 / 10 |            8.0 |
| Embedding (technical)         |      12 | 9 / 10 |           10.8 |
| Embedding (empirical quality) |      10 | 3 / 10 |            3.0 |
| Foundation models breadth     |      10 | 4 / 10 |            4.0 |
| Cognitive decoder             |       8 | 2 / 10 |            1.6 |
| Reconstruction / EEG2IMG      |       4 | 1 / 10 |            0.4 |
| Persistence (Postgres+RLS)    |       8 | 8 / 10 |            6.4 |
| Vector retrieval              |       8 | 4 / 10 |            3.2 |
| Auth & security posture       |       8 | 7 / 10 |            5.6 |
| Deployment & ops              |       8 | 5 / 10 |            4.0 |
| Observability                 |       6 | 7 / 10 |            4.2 |
| Documentation                 |       4 | 9 / 10 |            3.6 |
| Tests / CI                    |       4 | 5 / 10 |            2.0 |
| Datasets & evaluation         |       6 | 2 / 10 |            1.2 |
| Scalability primitives        |       4 | 2 / 10 |            0.8 |
| **Total**                     | **100** |      — | **≈ 63 / 100** |

- Previous (2026-06-17 strategic audit): **58 / 100**
- Current (2026-06-19): **63 / 100** — Δ **+5**

The +5 reflects exactly two facts: (a) the EEGConformer artefact is now
live in default routing (was the single biggest bottleneck), and (b)
ONNX runtime executes in the browser end-to-end. The score is held back
from a larger jump by the inconclusive embedding-quality result, the
CDN-pinned WASM, the missing CI, and the unchanged state of cognitive
decoding, datasets, and pgvector persistence.

---

## 12. Preservation Statement

No source files, migrations, or earlier audits were modified by this
report. All baseline reports remain in place under `docs/` and
`docs/audits/`.
