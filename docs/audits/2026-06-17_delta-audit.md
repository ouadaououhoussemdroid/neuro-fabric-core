# NeuroWeave / Neuro-Fabric — Delta Audit Report

**Audit Date:** 2026-06-17  
**Baseline:** `docs/AUDIT_REPORT.md` (2026-06-06, commit 2ee39e44)  
**Current:** working tree on this date  
**Scope:** read-only delta audit. No code, migrations, or dependencies were modified.

---

## Executive Summary

Since the 2026-06-06 baseline, the project has moved from a **frontend-only signal-processing demo** to an **authenticated, persistence-backed EEG analysis app** with the beginnings of an ML training surface (ONNX, TF.js decoder, Pyodide/MOABB hooks). Real progress is concentrated in three areas:

1. **Persistence layer.** Four new public tables (`profiles`, `eeg_analyses`, `experiments`, `experiment_runs`) plus role-based profiles (`researcher_profiles`, `enterprise_profiles`) with RLS, GRANTs, and an `app_role` enum.
2. **Authentication & dashboards.** `_authenticated` route gate, three role-specific dashboards, and an analyses dashboard wired to the new tables.
3. **EEG pipeline hardening.** Real artifact rejection module, signal-quality metrics, synthetic EEG generator, model registry, training pipeline scaffold, ONNX trainer hook, Pyodide/MOABB hook, TF.js decoder.

The AI claims gap from the baseline is **partially closed**: TF.js + ONNX surfaces now exist as code, but the production decoder path is still the heuristic spectral ratios. Reconstruction (EEG→image) is exposed as a route but remains a UI scaffold.

| Score                         | Baseline | Current      | Δ   |
| ----------------------------- | -------- | ------------ | --- |
| **Overall Platform Maturity** | 22 / 100 | **41 / 100** | +19 |
| **Production Readiness**      | 15 / 100 | **34 / 100** | +19 |

---

## Progress Since Last Audit

| Area                            | Baseline finding               | Resolved?  | Evidence                                                                                                    |
| ------------------------------- | ------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------- |
| No data persistence             | Results computed but discarded | ✅ Yes     | `eeg_analyses` table + insert in `/api/eeg/upload`                                                          |
| No authentication integrated    | Supabase imported, unused      | ✅ Yes     | `_authenticated/route.tsx`, `signin/signup/reset-password` routes, `auth-middleware.ts`, `auth-attacher.ts` |
| No file size limit / DoS vector | Open upload                    | ⚠️ Partial | Auth required; explicit size cap still not visible in `upload.ts`                                           |
| No rate limiting                | Missing                        | ❌ No      | No middleware added                                                                                         |
| Zero ML libraries               | Hard "no ML"                   | ⚠️ Partial | ONNX trainer hook, TF.js decoder file, Pyodide hook present; runtime models still placeholders              |
| Autoencoder = PCA wrapper       | Mock                           | ❌ No      | `embeddings/autoencoder.ts` unchanged                                                                       |
| No artifact rejection           | Missing                        | ✅ Yes     | `preprocessing/artifact-rejection.ts` (96 LOC)                                                              |
| No signal quality               | Missing                        | ✅ Yes     | `signal-quality/index.ts` (91 LOC)                                                                          |
| No experiment tracking          | Missing                        | ✅ Yes     | `experiments`, `experiment_runs` tables + `experiments.tsx` route                                           |
| No model versioning             | Missing                        | ⚠️ Partial | `model-registry/index.ts` exposes `ACTIVE_DECODER` only                                                     |
| Sleep-EDF / CHB-MIT loaders     | Missing                        | ❌ No      | Still absent                                                                                                |
| FFT instead of naive DFT        | Performance issue              | ❌ No      | `features.ts` still O(M²) DFT                                                                               |

---

## 1. Architecture Review

|                           | Baseline                               | Current                                                                                                                                                                                                                      |
| ------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime                   | TanStack Start, no backend persistence | Same + Lovable Cloud (Postgres + RLS)                                                                                                                                                                                        |
| Auth                      | None integrated                        | `_authenticated` gate + 3 role dashboards + middleware                                                                                                                                                                       |
| Server fns                | None app-internal                      | `auth-attacher` + `auth-middleware` registered in `src/start.ts`                                                                                                                                                             |
| HTTP API                  | `/api/eeg/upload` only                 | Same; now authenticated + persistence-backed                                                                                                                                                                                 |
| New modules               | —                                      | `synthetic/`, `signal-quality/`, `model-registry/`, `training/pipeline`, `vector-search/`, `decoder/tfjs-decoder`, `preprocessing/artifact-rejection`, hooks `use-onnx-trainer`, `use-pyodide`, `use-moabb`, `use-telemetry` |
| New routes                | —                                      | `experiments`, `training`, `synthetic`, `eeg2image`, `embeddings`, `models`, `onnx`, `mne`, `playground`, `studio`, `datasets`, `developers`, `research`, `architecture`, `_authenticated/dashboard.*`                       |
| Removed                   | —                                      | None observed                                                                                                                                                                                                                |
| Architectural regressions | —                                      | None                                                                                                                                                                                                                         |

**Architecture Maturity Score: 5.5 / 10** (baseline 3.0).

---

## 2. Database Review

|                     | Baseline                                                                         | Current                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Public tables       | `waitlist` only (4 cols, 1 policy)                                               | 7 tables: `waitlist`, `profiles`, `enterprise_profiles`, `researcher_profiles`, `eeg_analyses`, `experiments`, `experiment_runs` |
| Migrations          | 1                                                                                | 7                                                                                                                                |
| Roles               | none                                                                             | `app_role` enum (`individual`/`researcher`/`enterprise`)                                                                         |
| RLS                 | partial                                                                          | Enabled on every new table; policies scope to `auth.uid()`                                                                       |
| FKs / integrity     | none                                                                             | `eeg_analyses.user_id → auth.users`, `experiment_runs.experiment_id → experiments`, `experiment_runs.analysis_id → eeg_analyses` |
| Updated-at triggers | none                                                                             | Present on `experiments`                                                                                                         |
| Missing             | analytics tables, audit log, model registry table, dataset catalog, vector index | same gaps                                                                                                                        |

Pre-existing linter warn (`waitlist` `WITH CHECK (true)`) is intentional for public signup.

**Database Maturity Score: 5 / 10** (baseline 1.0).

---

## 3. EEG Pipeline Review

| Component                  | Baseline  | Current                                                             | Status           |
| -------------------------- | --------- | ------------------------------------------------------------------- | ---------------- |
| EDF parser                 | Real      | Real (unchanged)                                                    | Implemented      |
| CSV parser                 | Real      | Real                                                                | Implemented      |
| NPY parser                 | Real      | Real                                                                | Implemented      |
| Bandpass / notch filters   | Real      | Real                                                                | Implemented      |
| Segmentation               | Real      | Real                                                                | Implemented      |
| Normalization              | Real      | Real                                                                | Implemented      |
| **Artifact rejection**     | Missing   | `preprocessing/artifact-rejection.ts` w/ thresholds + contamination | Implemented      |
| **Signal quality**         | Missing   | `signal-quality/index.ts`                                           | Implemented      |
| Feature extraction         | Naive DFT | Naive DFT (unchanged)                                               | Partial (no FFT) |
| ICA                        | Missing   | Missing                                                             | Missing          |
| Re-referencing (CAR, REST) | Missing   | Missing                                                             | Missing          |
| Performance                | O(M²) DFT | O(M²) DFT                                                           | Partial          |

**EEG Readiness Score: 6.5 / 10** (baseline 4.5).

---

## 4. Embeddings Review

|                     | Baseline               | Current                                                     |
| ------------------- | ---------------------- | ----------------------------------------------------------- |
| Band-power features | Real                   | Real                                                        |
| PCA                 | Real (power-iteration) | Real (unchanged)                                            |
| Autoencoder         | Mock (PCA wrapper)     | **Still mock**                                              |
| Vector storage      | None                   | `vector-search/index.ts` (48 LOC) + `cosine.ts` (in-memory) |
| pgvector            | Not used               | Not used; `eeg_analyses.embedding` stored as JSON/array     |
| Retrieval           | None                   | In-memory cosine top-K                                      |

**Embedding Readiness Score: 4 / 10** (baseline 3.0).

---

## 5. Reconstruction (EEG→Image) Review

|                          | Baseline | Current                                             |
| ------------------------ | -------- | --------------------------------------------------- |
| Route                    | none     | `/eeg2image` + `recon-showcase.tsx`                 |
| Pipeline                 | none     | UI showcase only — no diffusion / GAN / VAE backend |
| Scientific completeness  | n/a      | None                                                |
| Engineering completeness | n/a      | UI scaffold                                         |

**Reconstruction Readiness Score: 1 / 10** (baseline 0).

---

## 6. Synthetic EEG Review

|                       | Baseline | Current                                                                               |
| --------------------- | -------- | ------------------------------------------------------------------------------------- |
| Generator             | none     | `lib/synthetic/index.ts` (116 LOC) — band-mixed sinusoids + noise                     |
| Route                 | none     | `/synthetic`                                                                          |
| Validation            | none     | None                                                                                  |
| Plausibility controls | none     | Hardcoded SNR/band ratios; no spectral validation, no cross-subject variability model |

**Synthetic EEG Readiness Score: 3 / 10** (baseline 0).

---

## 7. Security Review

| Item                           | Baseline | Current                                                                                                   |
| ------------------------------ | -------- | --------------------------------------------------------------------------------------------------------- |
| Auth on upload                 | ❌       | ✅ Bearer-token validation in `api/eeg/upload.ts`                                                         |
| RLS on user data               | n/a      | ✅ All new tables                                                                                         |
| Roles stored separately        | n/a      | ⚠️ Role stored on `profiles.role` (baseline guidance prefers a dedicated `user_roles` table — minor risk) |
| Service-role key exposure      | n/a      | Not exposed (per stack rules)                                                                             |
| File-size cap                  | ❌       | ❌                                                                                                        |
| Rate limiting                  | ❌       | ❌                                                                                                        |
| CSRF / origin checks           | ❌       | ❌                                                                                                        |
| Webhook signature verification | n/a      | n/a (no webhooks yet)                                                                                     |
| Secrets handling               | n/a      | Env-only, no leaks observed                                                                               |

**Security Score: 5 / 10** (baseline 2.0).

---

## 8. Scalability Review

| Bottleneck                         | Baseline | Current                              |
| ---------------------------------- | -------- | ------------------------------------ |
| O(M²) DFT in hot path              | Present  | Present                              |
| Single-window pooling              | Present  | Present                              |
| In-memory vector search            | n/a      | Present (won't scale past ~10⁴ rows) |
| No queue / background jobs         | Present  | Present                              |
| Edge worker CPU limit on `/upload` | Present  | Present                              |
| Postgres indexes on new tables     | n/a      | Basic FKs only; no embedding index   |

**Scalability Score: 3 / 10** (baseline 2.5).

---

## 9. Technical Debt Review

**Debt removed:** missing tables for routes that already referenced them; broken `requireSupabaseAuth` direct call; TS2367 in preprocessing; missing `ACTIVE_DECODER` export; missing experiments / analyses tables.

**Debt added:**

- Two overlapping `eeg_analyses` migrations (`20260607000000_eeg_analyses.sql` and `20260607151032_…`) — schema lineage is confusing and the earlier one uses `INT` for `file_size_bytes`.
- `experiment_runs` evolved across two migrations (initial + `20260617180002` adding `user_id`, `analysis_id`, `completed_at`) — should be consolidated for new envs.
- Linear autoencoder still misnamed (it is PCA).
- TF.js decoder file exists but is not the path the API uses; risk of drift between "claimed" and "served" decoder.
- `synthetic`, `training/pipeline`, `vector-search` lack tests.

**Highest priority debt:** (1) replace naive DFT with FFT, (2) move role to dedicated `user_roles` table with `has_role()`, (3) consolidate duplicate `eeg_analyses` migration, (4) implement actual learned decoder or rename to "spectral heuristic v1" everywhere.

**Technical Debt Score: 5 / 10** (baseline 3.5; lower = more debt).

---

## 10. Feature Matrix

| Feature                              | Previous          | Current                               | Delta |
| ------------------------------------ | ----------------- | ------------------------------------- | ----- |
| EDF/CSV/NPY parsing                  | Implemented       | Implemented                           | —     |
| Bandpass/notch/segment/normalize     | Implemented       | Implemented                           | —     |
| Artifact rejection                   | Missing           | Implemented                           | ▲▲    |
| Signal quality metrics               | Missing           | Implemented                           | ▲▲    |
| ICA / re-reference                   | Missing           | Missing                               | —     |
| Feature extraction (FFT)             | Partial (DFT)     | Partial (DFT)                         | —     |
| PCA embeddings                       | Implemented       | Implemented                           | —     |
| Learned autoencoder                  | Mock              | Mock                                  | —     |
| Vector search                        | Missing           | Partial (in-memory)                   | ▲     |
| Persistent analyses                  | Missing           | Implemented                           | ▲▲    |
| Cognitive decoder (heuristic)        | Implemented       | Implemented                           | —     |
| TF.js decoder                        | Missing           | Partial (file exists, not wired)      | ▲     |
| ONNX training                        | Missing           | Partial (hook + UI)                   | ▲     |
| Pyodide / MOABB                      | Missing           | Partial (hook)                        | ▲     |
| Synthetic EEG                        | Missing           | Partial                               | ▲     |
| EEG→Image reconstruction             | Missing           | Mock (UI only)                        | ▲     |
| Auth (email + roles)                 | Missing           | Implemented                           | ▲▲    |
| Role-based dashboards                | Missing           | Implemented                           | ▲▲    |
| Experiment tracking                  | Missing           | Implemented (CRUD)                    | ▲▲    |
| Model registry                       | Missing           | Partial (constant only)               | ▲     |
| Dataset loaders (PhysioNet)          | Implemented       | Implemented                           | —     |
| Dataset loaders (TUH/BCI)            | Architecture-only | Architecture-only                     | —     |
| Dataset loaders (Sleep-EDF, CHB-MIT) | Missing           | Missing                               | —     |
| Rate limiting / size cap             | Missing           | Missing                               | —     |
| Observability / telemetry            | Missing           | Partial (`use-telemetry`, `logging/`) | ▲     |
| pgvector index                       | Missing           | Missing                               | —     |
| Background jobs                      | Missing           | Missing                               | —     |

---

## 11. Progress Scorecard

| Dimension                  | Previous % | Current % |   Net Δ |
| -------------------------- | ---------: | --------: | ------: |
| Frontend                   |         80 |        88 |      +8 |
| Backend (server fns + API) |         15 |        45 |     +30 |
| Database                   |         10 |        55 |     +45 |
| APIs                       |         25 |        45 |     +20 |
| EEG Processing             |         45 |        65 |     +20 |
| Embeddings                 |         30 |        40 |     +10 |
| Reconstruction             |          0 |        10 |     +10 |
| Synthetic EEG              |          0 |        30 |     +30 |
| Security                   |         20 |        50 |     +30 |
| Scalability                |         25 |        30 |      +5 |
| **Overall Platform**       |     **22** |    **41** | **+19** |

```text
Maturity growth (overall)
 0%                                  100%
 |#########............................|  baseline 22%
 |#################....................|  current  41%
```

---

## 12. Roadmap Alignment

**Status: Mostly On Track.**

The trajectory matches the intended Neuro-Fabric direction (auth → persistence → experiment tracking → ML surfaces → reconstruction). Divergence points:

- The "AI" surface area expanded faster than the underlying ML substance — ONNX/TF.js/Pyodide hooks exist but the served decoder is still heuristic.
- No vector database adoption (pgvector) despite embeddings being persisted, which will block retrieval scale targets.
- Reconstruction was promoted to a route before any model exists; risks user-facing credibility gap.

---

## 13. Top 20 Highest-Impact Next Tasks

| #   | Task                                                               | Impact | Difficulty | Dependencies   | Est. maturity gain |
| --- | ------------------------------------------------------------------ | ------ | ---------- | -------------- | ------------------ |
| 1   | Replace naive DFT with FFT (radix-2 / Bluestein)                   | High   | Med        | —              | +4                 |
| 2   | Move roles to `user_roles` + `has_role()` SECDEF                   | High   | Low        | migration      | +3                 |
| 3   | Consolidate duplicate `eeg_analyses` migrations                    | Med    | Low        | —              | +2                 |
| 4   | Enable pgvector and index `eeg_analyses.embedding`                 | High   | Med        | migration      | +5                 |
| 5   | File size limit + MIME validation on `/api/eeg/upload`             | High   | Low        | —              | +3                 |
| 6   | Rate limiting middleware (per user + per IP)                       | High   | Med        | KV / DO        | +3                 |
| 7   | Replace placeholder autoencoder with a real trainable model (ONNX) | High   | High       | training data  | +6                 |
| 8   | Wire `tfjs-decoder` into `/api/eeg/upload` behind a feature flag   | High   | Med        | model artifact | +4                 |
| 9   | CAR / Laplacian re-referencing                                     | Med    | Low        | —              | +2                 |
| 10  | ICA for ocular/muscle artifact removal                             | High   | High       | —              | +4                 |
| 11  | Sleep-EDF + CHB-MIT loaders                                        | Med    | Med        | mirrors        | +3                 |
| 12  | Persist experiment runs from training pipeline                     | High   | Low        | —              | +3                 |
| 13  | Background job queue for long uploads (R2 + DO/queue)              | High   | High       | infra          | +4                 |
| 14  | Real EEG→image baseline (linear decoder → image)                   | High   | High       | dataset        | +5                 |
| 15  | Synthetic EEG validation against PhysioNet spectra                 | Med    | Med        | —              | +2                 |
| 16  | Audit log table + write paths                                      | Med    | Low        | migration      | +2                 |
| 17  | Model registry table (versions, metrics, artifact URL)             | Med    | Low        | migration      | +3                 |
| 18  | Dataset catalog table + UI                                         | Med    | Low        | migration      | +2                 |
| 19  | Test suite for EEG/embeddings/synthetic                            | High   | Med        | vitest         | +3                 |
| 20  | Cron + signature-verified webhook surface under `/api/public/*`    | Med    | Med        | secret         | +2                 |

---

## 14. Evidence Index

- **New migrations:** `supabase/migrations/20260603035330_*`, `20260604031328_*`, `20260604031339_*`, `20260607000000_eeg_analyses.sql`, `20260607151032_*`, `20260617180002_*`
- **New tables (live):** `profiles`, `enterprise_profiles`, `researcher_profiles`, `eeg_analyses`, `experiments`, `experiment_runs` (verified via runtime supabase-tables context).
- **New modules:** `src/lib/eeg/preprocessing/artifact-rejection.ts`, `src/lib/signal-quality/index.ts`, `src/lib/synthetic/index.ts`, `src/lib/training/pipeline.ts`, `src/lib/vector-search/{index,cosine}.ts`, `src/lib/model-registry/index.ts`, `src/lib/decoder/tfjs-decoder.ts`.
- **New hooks:** `use-onnx-trainer.ts`, `use-pyodide.ts`, `use-moabb.ts`, `use-telemetry.ts`.
- **New routes:** `experiments`, `training`, `synthetic`, `eeg2image`, `embeddings`, `models`, `onnx`, `mne`, `playground`, `studio`, `datasets`, `developers`, `research`, `architecture`, `signin`, `signup`, `reset-password`, `_authenticated/dashboard.{individual,researcher,enterprise,analyses}`.
- **Auth wiring:** `src/integrations/supabase/auth-{middleware,attacher}.ts`, `src/start.ts`, `src/routes/_authenticated/route.tsx`.

---

## 15. Final Scores

| Score                               | Baseline |  Current |
| ----------------------------------- | -------: | -------: |
| Architecture                        |      3.0 |      5.5 |
| Database                            |      1.0 |      5.0 |
| EEG Pipeline                        |      4.5 |      6.5 |
| Embeddings                          |      3.0 |      4.0 |
| Reconstruction                      |      0.0 |      1.0 |
| Synthetic EEG                       |      0.0 |      3.0 |
| Security                            |      2.0 |      5.0 |
| Scalability                         |      2.5 |      3.0 |
| Technical Debt (higher = less debt) |      3.5 |      5.0 |
| **Overall Platform Maturity**       | **22 %** | **41 %** |
| **Production Readiness**            | **15 %** | **34 %** |
