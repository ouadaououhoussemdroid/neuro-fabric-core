# Neuro-Fabric — Strategic Vision Alignment Audit

- **Date:** 2026-06-19
- **Scope:** strategic alignment between the original Neuro-Fabric vision and the
  current implementation (read-only; no source files modified).
- **Inputs cross-referenced:**
  - `docs/AUDIT_REPORT.md`, `docs/REALITY_CHECK.md`,
    `docs/BLUEPRINT_PREPARATION.md` (2026-06-06 baseline)
  - `docs/architecture.md`, `docs/ai-layer-architecture.md`
  - `docs/adr/0001-braindecode-execution-strategy.md`
  - `docs/audits/2026-06-17_*` (8 reports incl. strategic-progress-audit)
  - `docs/audits/2026-06-19_*` (eegconformer live / routing-fix /
    runtime-verification / project_state_audit)
  - `docs/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md`
  - Working tree under `src/lib/ai/**`, `src/routes/**`, `supabase/migrations/**`

---

## 1. Original Vision Reconstruction

The "Neuro-Fabric" name and stack came with two distinct authorial layers, and
the strategic confusion in this project starts there. To audit alignment
honestly, both must be reconstructed.

### 1.A The 2026-06-06 vision (`AUDIT_REPORT`, `REALITY_CHECK`,
`BLUEPRINT_PREPARATION`)

At inception the project was framed as a **validated EEG analytics platform**
with the following intent:

1. Genuine signal processing (EDF/CSV/NPY → filters → features → embeddings).
2. **Persistence + auth + RLS** so analyses are retrievable and per-user.
3. **Model registry, experiment tracking, artefact rejection, signal quality**.
4. **Scientific validation**: ground-truth annotation, LOSO, benchmark
   comparison, statistical reporting (Phase 3, months 4-8).
5. **Trained ML** (VAE embeddings + cognitive classifier + HPO) only after
   validation passes (Phase 4, months 8-12).
6. Explicit critical decision point at **Month 4**: if validation fails,
   recalibrate heuristics — do not race ahead to ML.

This original document does **not** mention foundation models, ONNX,
Braindecode, EEGConformer, EEGPT, EEG2IMG, vector retrieval, or cross-modal
reconstruction. It is a *scientific neurotechnology* roadmap, not a
*foundation-model* roadmap.

### 1.B The "Neuro Fabric" emergent vision (mid-June 2026 onward)

Starting around 2026-06-17, audits and the project-state report restated the
vision as a **layered neural fabric**:

```
Signal  →  Embedding  →  Representation  →  Cognitive  →  Reconstruction  →  Cross-modal
```

with foundation models (EEGConformer / EEGNet / EEGPT) producing embeddings,
pgvector providing persistent representation, a trained cognitive decoder
replacing heuristics, and EEG2IMG / generative reconstruction as the
long-horizon ambition. The product framing on the marketing routes
(`/eeg2image`, `/synthetic`, `/embeddings`, `/research`) reflects this layer
vocabulary, not the 2026-06-06 vocabulary.

### 1.C Reconciled "true north"

For the purpose of this audit the **operative vision** is the union of the
two: the 2026-06-06 contract (validated science, persistence, trained ML)
treated as the *floor*, and the layered Neuro-Fabric stack treated as the
*ceiling*. Either alone would be too generous or too harsh; together they
describe what the project has actually been pitching itself as for the last
two months.

---

## 2. Alignment Matrix

Each row is a vision component. **Verdict** is one of:
✅ Advances vision, ⚠️ Partial / placeholder, ❌ Drift or absent.

| # | Vision component (source) | Implementation evidence | Verdict |
|---|---|---|---|
| V1 | Genuine signal pipeline (EDF/CSV/NPY, IIR filtfilt, segmentation) | `src/lib/eeg/parsers/*`, `preprocessing/*` | ✅ |
| V2 | Real spectral features + embeddings (PCA baseline) | `embeddings/features.ts`, `embeddings/pca.ts` | ✅ |
| V3 | Persistence (analyses, experiments, logs) | `supabase/migrations/*`, eeg_analyses + experiments tables | ✅ |
| V4 | Auth + RLS + role tables (`has_role` SECURITY DEFINER) | `_authenticated/route.tsx`, `auth-middleware`, role tables | ✅ |
| V5 | Security hardening (size cap, rate limit, NaN guards) | NaN guards present; **no upload size cap, no rate limit** on `/api/eeg/upload` | ⚠️ |
| V6 | Model versioning / registry | `src/lib/ai/models/registry.ts`, multi-adapter, version metadata | ✅ |
| V7 | Experiment tracking | `experiments` table + route, but parameters logging shallow | ⚠️ |
| V8 | Artefact rejection + signal-quality metrics | `preprocessing/artifact-rejection`, `signal-quality/` | ✅ |
| V9 | Ground-truth annotation UI | **Not built** | ❌ |
| V10 | LOSO / cross-subject validation | **Not built** in-platform | ❌ |
| V11 | Benchmark vs Sleep-EDF / CHB-MIT / TUH | TUH/PhysioNet/BCI-IV-2a *loaders*; no eval harness wired | ⚠️ |
| V12 | Statistical reporting (CIs, p-values, effect sizes) | **Not built** | ❌ |
| V13 | Trained cognitive decoder (attention/workload/arousal) | **Heuristic ratios only** in `decoder/index.ts` | ❌ |
| V14 | Trained deep embedding model (VAE / self-supervised) | **Not built**; PCA + ONNX-bridged EEGConformer instead | ⚠️ (substituted) |
| V15 | HPO / Bayesian optimisation | **Not built** | ❌ |
| V16 | ML infrastructure (training pipeline) | `training/` package + scripts + notebook + ONNX export | ✅ |
| V17 | Foundation-model adapters (ONNX / Braindecode / EEGPT) | `onnx-adapter.ts`, `braindecode-onnx-bridge.ts`, EEGPT stub | ✅ for ONNX/Braindecode; ⚠️ EEGPT stub |
| V18 | Live foundation model in production routing | **EEGConformer live** (`braindecode-eegconformer-prod`, dim=32, fellBack=false) | ✅ |
| V19 | Empirical embedding quality (discriminative on real EEG) | Synthetic probe only; separation margin 0.008 | ❌ |
| V20 | Persistent representation (pgvector + ANN) | **In-memory only** | ❌ |
| V21 | Reconstruction layer (EEG2IMG, generative) | Route scaffold only (`/eeg2image`, `/synthetic`) | ⚠️ (UI shell) |
| V22 | Cross-modal embeddings | Not present | ❌ |
| V23 | Validation primitives (NaN/Inf/dim, L2 norm, mandatory) | `validation/`, used by `embed()` facade | ✅ |
| V24 | Observability (structured logs, benchmark harness, SLOs) | `logging/`, `benchmark/`; **no SLO dashboard** | ⚠️ |
| V25 | Operational hardening (CI, signed artefact URLs, self-hosted WASM) | None of these in place | ❌ |

**Tally:** ✅ 10 · ⚠️ 7 · ❌ 8 (out of 25).

---

## 3. Drift Analysis

Drift is defined as: implementation effort that does **not** advance any
vision component, *or* effort that explicitly substitutes a vision component
with a generic alternative.

### 3.1 Hard drift (work that the original vision would not have asked for)

- **D1 — Marketing-route surface area.** Many routes
  (`/research`, `/datasets`, `/playground`, `/studio`, `/onnx`, `/mne`,
  `/training`, `/eeg2image`, `/synthetic`, `/architecture`, `/developers`,
  `/pricing`) are content/landing pages, not vision-bearing features. The
  2026-06-06 vision is silent on a marketing site; the layered Neuro-Fabric
  framing tolerates them only as showcases.
- **D2 — In-memory vector index productionised** instead of being treated as
  a stop-gap. Both visions assume durable representation; the
  `vector-bridge/` work has been polished (model-id tagging, tests) without
  ever migrating to pgvector.
- **D3 — Heuristic cognitive decoder kept in the product surface.** The
  2026-06-06 doc explicitly identified this as the thing to *replace*; the
  current code still ships the same beta/(alpha+theta) ratios behind the
  same "attention/workload/arousal" labels.
- **D4 — Repo-bundled ONNX in `public/models/`** instead of content-hashed
  Cloud Storage with SHA verification (the deployment roadmap's Option A).
  Engineering convenience over the architectural intent.
- **D5 — CDN-pinned ORT WASM** (jsdelivr) accepted as a production
  dependency. A vision-aligned implementation would self-host.

### 3.2 Soft drift (vision-aligned work that displaced higher-priority work)

- **S1 — EEGConformer end-to-end** (export → registration → routing → runtime
  fix → quality probe) consumed the bulk of the last sprint. It advances V17
  and V18 but at the cost of V9–V12 and V20, which the original blueprint
  treated as *prerequisites* for trained-ML claims.
- **S2 — Adapter scaffolding for EEGPT** without a validation gate. The
  registration pattern is real, but EEGPT remains a stub with no roadmap
  date.
- **S3 — Maturity benchmarks and audit volume.** 12+ audit/roadmap documents
  in two weeks. Documentation density is high (and rated 9/10 in the
  project-state audit) but no audit has yet *gated* a release.

### 3.3 Substitutions (vision items quietly replaced by adjacent work)

| Vision item | Substituted by | Aligned? |
|---|---|---|
| V14 trained deep embedding (VAE) | EEGConformer ONNX (third-party, pre-trained) | Partially — different mechanism, same architectural slot. |
| V13 trained cognitive classifier | Heuristic ratios + "to be trained later" note | No — slot still empty. |
| V11 benchmark on Sleep-EDF/CHB-MIT/TUH | Synthetic probe + BCI-IV-2a notebook (off-platform) | No — never closed the loop in-product. |
| V20 pgvector persistence | In-memory `NeuralVectorIndex` with model-id tagging | No — durability missing. |

---

## 4. Vision Preservation Score

Definition: weighted fraction of the operative vision (§1.C) that is
materially advanced by current code, weighted by criticality.

| Layer | Weight | Score (/10) | Contribution |
|---|---:|---:|---:|
| Signal pipeline (V1-V2) | 10 | 9 | 9.0 |
| Persistence + auth + RLS (V3-V4) | 12 | 8 | 9.6 |
| Security hardening (V5) | 6 | 4 | 2.4 |
| Model registry + adapters (V6, V16, V17) | 12 | 8 | 9.6 |
| Live foundation model (V18) | 10 | 8 | 8.0 |
| Empirical quality (V19) | 12 | 3 | 3.6 |
| Trained cognitive decoder (V13) | 10 | 1 | 1.0 |
| Validation + benchmark (V11, V12, V23, V24) | 10 | 5 | 5.0 |
| Persistent representation (V20) | 8 | 2 | 1.6 |
| Reconstruction / cross-modal (V21, V22) | 4 | 1 | 0.4 |
| Ops hardening (V25) | 6 | 2 | 1.2 |
| **Total** | **100** | — | **≈ 51 / 100** |

**Vision Preservation Score: 51 / 100.**

The score is materially below the project-state audit's 63/100 *engineering
readiness* score because vision preservation is harsher: it discounts work
that is engineering-mature but does not move the project toward its
stated layered-fabric / validated-science end state.

---

## 5. Strategic Drift Score

Definition: fraction of effort and surface area that does not measurably
advance the operative vision. Inverse of preservation, but weighted by
observable build effort, not by intent.

| Drift bucket | Severity (/10) | Weight | Contribution |
|---|---:|---:|---:|
| D1 marketing surface area | 4 | 10 | 4.0 |
| D2 in-memory vector polish vs pgvector | 7 | 15 | 10.5 |
| D3 heuristic decoder still shipped | 8 | 15 | 12.0 |
| D4 repo-bundled artefact vs hashed bucket | 5 | 10 | 5.0 |
| D5 CDN-pinned WASM | 6 | 10 | 6.0 |
| S1 EEGConformer ahead of validation | 5 | 20 | 10.0 |
| S2 EEGPT scaffolding without gate | 3 | 10 | 3.0 |
| S3 audit/doc volume without release gates | 3 | 10 | 3.0 |
| **Total** | — | **100** | **≈ 53.5 / 100** |

**Strategic Drift Score: 54 / 100** (rounded).

Interpretation: roughly half the recent build energy has gone into
vision-adjacent or vision-displacing work. The project is **not** dominated
by drift, but it is **not** dominated by alignment either.

---

## 6. Recommended Corrections

### 6.1 Accelerate (highest impact on alignment restoration)

1. **A1. Empirical validation on a real holdout (BCI-IV-2a, Sleep-EDF).**
   Without this, V18 is theatre. Off-platform Colab is acceptable for the
   first pass; results must gate any further EEGConformer marketing
   claim.
2. **A2. pgvector migration with model-id tagging and ANN index.**
   Closes V20, unlocks recall@10 SLO (V11/V24), and is the *only* way the
   layered fabric becomes more than a diagram.
3. **A3. Train and ship a small cognitive decoder.** Even a logistic
   classifier on band-power features beats heuristics for V13. Retire the
   heuristic ratios from the public surface or hard-label them "v0
   heuristic" until the trained version exists.
4. **A4. Self-host ORT WASM + content-hash the ONNX artefact in Cloud
   Storage with SHA enforcement.** Closes D4, D5, and removes a silent
   global PCA-fallback risk.

### 6.2 Delay (defer until alignment is restored)

1. **B1. EEGPT integration.** Keep the stub and registration; do not
   invest engineering until V18 has cleared empirical validation and V20
   is durable.
2. **B2. EEG2IMG / generative reconstruction track.** Route scaffolds may
   stay; do not build models. The reconstruction layer cannot meaningfully
   precede a discriminative embedding space.
3. **B3. Further marketing-route polish.** Freeze new content routes until
   the validated stack catches up.

### 6.3 Remove (or relabel)

1. **C1. Heuristic cognitive metrics marketed as "AI"** — relabel to
   "spectral indicator (heuristic)" everywhere they surface in the UI, or
   feature-flag off until V13 lands.
2. **C2. EEGPT registration without a forward plan** — either schedule it
   or drop the registration so the registry honestly reflects shipped
   capability.
3. **C3. Audit/roadmap inflation** — consolidate the 12+ recent audit
   files behind a single living index; new audits should only be cut when
   they gate a release.

### 6.4 Redesign

1. **R1. Replace the in-memory `NeuralVectorIndex`** with a thin
   pgvector-backed implementation that preserves the same TS interface
   (model-id tagged, fellBack-aware). The wrapper-shape is correct; the
   storage is wrong.
2. **R2. Reissue the deployment roadmap as Option A** (Cloud Storage +
   content-hashed URL). The current Option B (`public/models/`) was
   explicitly flagged "offline demo only" in the 2026-06-17 roadmap.
3. **R3. Make the embedding-quality probe a CI-gating fixture** keyed to
   recall@10 vs PCA on a real labelled snippet. Without a gate, "live"
   silently regresses to "live and broken."

---

## 7. Does the Current 4-Month Roadmap Maximize Vision Progress?

The revised 4-month roadmap in the 2026-06-19 project-state audit is:

- **M1** — validate live path, vendor WASM, content-hash artefact, CI,
  upload caps.
- **M2** — pgvector + cosine ANN, FFT-replace DFT.
- **M3** — train cognitive decoder, ship Sleep-EDF loader, retire
  heuristics behind a flag.
- **M4** — pilot hardening, SLO dashboards, design partners.

Mapping each month onto the alignment matrix:

| Month | Vision items advanced | Drift items addressed | Verdict |
|---|---|---|---|
| M1 | V19, V24, V25 | D4, D5, S3 | Vision-positive (validation + ops). |
| M2 | V20, V11/V24 | D2 | Vision-positive (representation layer). |
| M3 | V13, V11 | D3, C1, C3 | **Most vision-positive month.** |
| M4 | V18, V24 | — | Engineering maturity, not new vision. |

**Verdict.** The current 4-month roadmap **does** advance the vision; it is
roughly 70 % vision-positive and 30 % engineering maturity. However, the
ordering is suboptimal: M3 (cognitive decoder + real dataset) is the highest
vision-leverage block and should be pulled forward to overlap M2, even if
pgvector slips by a week. The current ordering risks spending M1 entirely on
ops hardening *before* the embedding space is empirically defensible — that
is engineering maturity in service of an unvalidated core.

**Recommended re-sequencing:**
- M1 keeps validation + WASM self-host; defers CI-as-gate to M2.
- M2 begins pgvector **and** in parallel begins decoder training.
- M3 finishes decoder, retires heuristics, ingests Sleep-EDF.
- M4 unchanged.

---

## 8. Executive Verdict

- **Vision Preservation Score:** 51 / 100.
- **Strategic Drift Score:** 54 / 100.
- **Net direction over the last two weeks:** weakly vision-positive
  (EEGConformer is a real V18 win) but eroded by D2/D3/D4/D5 and by spending
  most of the sprint on a single layer (embedding) without closing the
  layers immediately above and below it.

### Is Neuro-Fabric still becoming Neuro-Fabric?

**Conditionally yes.** The platform has not regressed into a generic EEG
analytics CRUD app — the adapter pattern, validation discipline, model
registry, and live foundation-model routing are all genuinely
vision-bearing. But it has also not yet *crossed* into a fabric:
representation is still volatile, the cognitive layer is still heuristic,
and the embedding space is still empirically unproven. If the next 90 days
execute roughly the recommended re-sequencing (validate → persist → train
decoder → retire heuristics) the trajectory holds. If the next 90 days
continue to optimise the embedding layer in isolation, the project will
complete its drift into "an EEG analytics platform with one nice ONNX model"
and the Neuro-Fabric framing will become unjustifiable.

The decision point is operational, not architectural. The architecture is
still pointed at the right star.

---

## 9. Preservation Statement

No source files, migrations, prior audits, or roadmaps were modified by this
report. All baseline material remains in place under `docs/` and
`docs/audits/`.