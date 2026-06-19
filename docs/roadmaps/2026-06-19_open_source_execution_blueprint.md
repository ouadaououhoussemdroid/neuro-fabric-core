# Neuro-Fabric Core — Open-Source Execution Blueprint

- **Date:** 2026-06-19
- **Type:** Execution plan (not an audit). Builds directly on
  `docs/audits/2026-06-19_project_state_audit.md` (readiness 63/100) and
  `docs/audits/2026-06-19_vision_alignment_audit.md` (preservation 51 /
  drift 54).
- **Constraint:** open-source only. Every recommended dependency below is
  inspected for license, repo, and maintenance posture.
- **Scope:** the next 30 / 90 / 180 days of engineering, plus the Top-25
  prioritised task list and the shortest-path answer.

---

## 0. How to read this blueprint

Each task in §2 follows a fixed schema:

```
T-### — <title>
  Objective       : what we ship
  Why it matters  : alignment with V## vision items / drift items
  Dependencies    : tasks or assets required first
  Difficulty      : Low / Medium / High / Research
  Effort          : ideal engineer-days (1 dev, no context-switch)
  Stack           : libraries / models / tools (with §3 reference)
```

Difficulty is *technical risk* (unknown unknowns). Effort is *wall-clock
work* assuming the dependencies exist. They are independent: a task can be
Low-difficulty / High-effort (rote migration) or High-difficulty /
Low-effort (one tricky line of code).

---

## 1. Strategic frame

The 2026-06-19 alignment audit identified the binding constraints:

1. **Empirical validation gap** (V19) — live EEGConformer is unproven on
   real holdout data.
2. **Representation volatility** (V20) — in-memory vector index instead of
   pgvector.
3. **Cognitive-layer placeholder** (V13) — heuristic ratios still shipped.
4. **Ops fragility** (V25, D4, D5) — CDN-pinned ORT WASM, repo-bundled ONNX,
   no CI gate.

This blueprint sequences work to clear those four constraints first, then
open the upper layers of the Neuro-Fabric stack (knowledge graph,
foundation-model research, reconstruction) once the floor is sound.

---

## 2. Task catalogue

### A. EEG acquisition layer

**T-001 — Hardware-agnostic acquisition adapter**
- Objective: a single TS interface (`AcquisitionSource`) producing
  `EEGSignal` chunks from file uploads, BrainFlow streams, and LSL.
- Why: V1/V2 currently only ingest files; a fabric needs live sources.
- Dependencies: existing `src/lib/eeg/parsers/*`.
- Difficulty: Medium. Effort: 3 d.
- Stack: BrainFlow (§3-C1), LSL bridge via WebSocket gateway (§3-B2).

**T-002 — EDF+ / BDF / GDF parser hardening**
- Objective: support EDF+ annotations, BDF (24-bit), and GDF for BCI-IV
  competition data without round-tripping through Python.
- Why: V11 (benchmarks) blocked on robust loaders.
- Dependencies: `edf-decoder` upgrade or replacement.
- Difficulty: Medium. Effort: 4 d.
- Stack: `edfdecoder` (§3-A1) primary; fall back to Pyodide+MNE (§3-D1)
  for GDF.

### B. Real-time streaming layer

**T-003 — WebSocket EEG gateway (server route)**
- Objective: `/api/public/stream/:source` upgrades to WS, fans out chunks
  with sequence numbers and a `model_id` header.
- Why: precondition for live decoding, T-013 dashboards.
- Dependencies: T-001.
- Difficulty: Medium. Effort: 3 d.
- Stack: TanStack Start server route + native `WebSocket` (Workers).

**T-004 — Lab Streaming Layer (LSL) bridge**
- Objective: tiny Python sidecar that bridges LSL → WS gateway. Ships as
  optional Docker image for partners.
- Why: industry-standard EEG transport; needed for academic pilots.
- Dependencies: T-003.
- Difficulty: Low. Effort: 2 d.
- Stack: `pylsl` (§3-B1).

### C. BrainFlow integration

**T-005 — BrainFlow Node binding wrapper**
- Objective: server-side BrainFlow consumer that publishes to the WS gateway.
- Why: covers OpenBCI, Muse, Ganglion, Cyton, synthetic boards in one API.
- Dependencies: T-003.
- Difficulty: Medium. Effort: 4 d.
- Stack: `brainflow` Node bindings (§3-C1); synthetic board for CI.

### D. MNE integration

**T-006 — Pyodide-MNE preprocessing parity harness**
- Objective: golden-file tests that compare TS filters / segmentation
  against MNE on the same EDF input.
- Why: closes a long-standing parity question; gates V23.
- Dependencies: existing `use-pyodide` hook.
- Difficulty: Medium. Effort: 3 d.
- Stack: MNE-Python via Pyodide (§3-D1); Vitest snapshot fixtures.

### E. Embedding infrastructure

**T-007 — FFT replacement of naive DFT in `bandPowerFeatures`**
- Objective: O(N log N) spectrum, real-time-safe.
- Why: M2 roadmap item; required before live decoding at >250 Hz.
- Dependencies: none.
- Difficulty: Low. Effort: 1 d.
- Stack: `fft.js` (§3-E1) or `kissfft-wasm` for SIMD.

**T-008 — Self-hosted ORT WASM bundle**
- Objective: ship `ort-wasm-simd-threaded.wasm` from our own origin with
  SHA-384 integrity; remove jsdelivr dependency.
- Why: closes D5; eliminates silent PCA fallback on CDN outage.
- Dependencies: none.
- Difficulty: Low. Effort: 1 d.
- Stack: `onnxruntime-web` static assets (§3-P1) copied via Vite plugin.

**T-009 — Content-hashed ONNX artefact in object storage**
- Objective: move `public/models/eegconformer.onnx` to Cloud Storage,
  reference by `sha256-…` URL, verify at load.
- Why: closes D4; immutable, cache-friendly, attestable.
- Dependencies: T-008 (same release).
- Difficulty: Low. Effort: 1 d.
- Stack: Supabase Storage bucket + SubtleCrypto.

**T-010 — EEGConformer empirical validation on BCI-IV-2a holdout**
- Objective: notebook + CI fixture; report intra/inter-class cosine,
  separation margin, recall@10 vs PCA baseline.
- Why: clears V19; gates further EEGConformer marketing claims.
- Dependencies: T-002 (GDF loader).
- Difficulty: Research. Effort: 5 d.
- Stack: Braindecode (§3-J1), MOABB (§3-K1), scikit-learn.

### F. Similarity search

**T-011 — `pgvector` migration of `NeuralVectorIndex`**
- Objective: TS interface preserved; storage becomes `embeddings` table
  with `vector(32)` + `ivfflat` index, model-id tagged.
- Why: closes V20; M2 keystone.
- Dependencies: Lovable Cloud (already on).
- Difficulty: Medium. Effort: 4 d.
- Stack: `pgvector` (§3-G1) via Supabase migration.

**T-012 — Recall@10 SLO harness**
- Objective: nightly job samples labelled snippets, computes recall@10
  against pgvector; alerts on regression.
- Why: turns V24 into a release gate.
- Dependencies: T-010, T-011.
- Difficulty: Medium. Effort: 3 d.
- Stack: pg_cron + server route under `/api/public/cron/recall`.

### G. Vector database options

See §3-G for the full matrix. Default: **pgvector** (already paid for via
Lovable Cloud). Lightweight alt: **hnswlib-node** for in-process bench.
Long-term scalable: **Qdrant** self-hosted, switched in behind the same
`NeuralVectorIndex` interface.

### H. Knowledge graph options

**T-013 — Concept graph schema (subject → session → window → embedding → label)**
- Objective: minimal property-graph schema sitting on top of Postgres
  (using `ltree` + `embeddings` FK) before considering a dedicated graph DB.
- Why: makes "Neuro-Fabric" literal; enables provenance queries.
- Dependencies: T-011.
- Difficulty: Medium. Effort: 4 d.
- Stack: Postgres `ltree` + recursive CTEs (§3-H1). Apache AGE (§3-H2)
  reserved for the long-term scalable path.

### I. Brain representation layer

**T-014 — Subject-level embedding aggregation (median + ICA basis)**
- Objective: from per-window 32-D vectors, derive a per-subject signature
  with stability metrics across sessions.
- Why: representation persistence is meaningless without aggregation.
- Dependencies: T-011.
- Difficulty: Medium. Effort: 3 d.
- Stack: numeric.js / ndarray-fft (already TS), optional Pyodide for ICA.

### J. Foundation-model research stack

**T-015 — Braindecode model zoo registration**
- Objective: register EEGNetv4, ShallowFBCSPNet, Deep4Net alongside
  EEGConformer; export each to ONNX via the existing script.
- Why: gives the registry comparative power; enables ablations.
- Dependencies: existing export script.
- Difficulty: Medium. Effort: 4 d.
- Stack: Braindecode (§3-J1), PyTorch 2.x, ONNX opset 17.

**T-016 — EEGPT honest stub or removal**
- Objective: either schedule EEGPT integration with a concrete weight
  source, or remove the placeholder adapter (per C2 in alignment audit).
- Why: registry should reflect shipped reality.
- Dependencies: none.
- Difficulty: Low. Effort: 0.5 d.
- Stack: existing adapter scaffold.

### K. Benchmarking infrastructure

**T-017 — MOABB-driven evaluation harness**
- Objective: Colab-runnable + CI-runnable script that scores any
  registered model on BCI-IV-2a, BCI-IV-2b, PhysioNetMI.
- Why: V11 closure; reusable for every future model.
- Dependencies: T-015.
- Difficulty: Medium. Effort: 4 d.
- Stack: MOABB (§3-K1), scikit-learn, MLflow logging.

### L. Explainability stack

**T-018 — Saliency over EEGConformer (Captum)**
- Objective: per-window integrated-gradients map exposed on `/embeddings`
  route; overlays on topomap.
- Why: investor + scientific story; turns the embedding from a black box
  into a defensible artefact.
- Dependencies: T-010.
- Difficulty: Research. Effort: 5 d.
- Stack: Captum (§3-L1) at export time; precomputed attributions cached
  per artefact hash.

### M. Dataset management

**T-019 — Dataset manifest + DVC-lite metadata table**
- Objective: `datasets` table (name, license, sha256, source URL,
  preprocessing hash); UI list under `/datasets`.
- Why: reproducibility; required by any audit-bearing benchmark.
- Dependencies: none.
- Difficulty: Low. Effort: 2 d.
- Stack: Postgres + signed download URLs; DVC (§3-M1) reserved for
  long-term scalable path.

### N. Training infrastructure

**T-020 — Reproducible training container**
- Objective: Dockerfile that pins `training/requirements.txt`, exposes
  `make train MODEL=eegconformer DATASET=bciiv2a`, writes artefacts to a
  hashed path.
- Why: training currently runs only in a notebook on a laptop.
- Dependencies: T-019.
- Difficulty: Medium. Effort: 3 d.
- Stack: Docker, PyTorch Lightning (§3-N1), Hydra configs.

**T-021 — MLflow tracking server (local-first, optional remote)**
- Objective: every training run logs params, metrics, artefacts, ONNX
  export hash; UI accessible to researchers.
- Why: V7 (experiment tracking) is currently shallow.
- Dependencies: T-020.
- Difficulty: Low. Effort: 2 d.
- Stack: MLflow (§3-N2), SQLite backend by default.

### O. Model registry

**T-022 — Registry ↔ MLflow ↔ Storage three-way sync**
- Objective: the TS registry pulls model metadata from MLflow + verifies
  artefact hash in Storage. One source of truth.
- Why: closes the loop between training and serving.
- Dependencies: T-009, T-021.
- Difficulty: Medium. Effort: 3 d.
- Stack: MLflow REST API + Storage signed URLs.

### P. ONNX deployment stack

**T-023 — `onnx-simplifier` + `onnxoptimizer` in export pipeline**
- Objective: every exported ONNX is simplified and shape-inferred before
  upload; smaller graphs, faster cold start.
- Why: cuts EEGConformer cold start; required for mobile.
- Dependencies: T-015.
- Difficulty: Low. Effort: 1 d.
- Stack: `onnx-simplifier` (§3-P2), `onnxoptimizer` (§3-P3).

**T-024 — WebGPU execution provider feature flag**
- Objective: opt-in WebGPU EP in `onnxruntime-web`; CPU/WASM fallback
  unchanged.
- Why: 5–20× speedup on supported browsers; enables real-time decoding.
- Dependencies: T-008.
- Difficulty: Medium. Effort: 2 d.
- Stack: `onnxruntime-web` WebGPU build (§3-P1).

### Q. Research platform tooling

**T-025 — Trained cognitive decoder v0 (logistic on band-power)**
- Objective: replace heuristic ratios with a calibrated logistic
  regression trained on a public attention/workload dataset; expose
  confidence intervals.
- Why: clears V13 and C1 in one stroke.
- Dependencies: T-019.
- Difficulty: Medium. Effort: 4 d.
- Stack: scikit-learn → ONNX via `skl2onnx` (§3-N3).

**T-026 — Notebook portal (read-only nbviewer of `training/notebooks/`)**
- Objective: serve executed notebooks under `/research/notebooks/:id`.
- Why: makes the research surface real instead of marketing-only.
- Dependencies: none.
- Difficulty: Low. Effort: 2 d.
- Stack: `jupyter nbconvert --to html` build step + static serve.

**T-027 — CI security & quality gates**
- Objective: GitHub Actions running `bun run test`, `bun run build`,
  `eslint`, `tsc --noEmit`, plus the recall@10 fixture from T-012.
- Why: turns audits into release gates (closes S3, D-ops).
- Dependencies: T-012.
- Difficulty: Low. Effort: 1 d.
- Stack: GitHub Actions; later mirror to Forgejo for sovereignty.

**T-028 — Upload hardening (size cap, rate limit, content sniff)**
- Objective: enforce 50 MB cap, IP+user rate-limit, magic-number check
  on `/api/eeg/upload`.
- Why: V5 gap explicitly flagged.
- Dependencies: none.
- Difficulty: Low. Effort: 1 d.
- Stack: TanStack server route + `@upstash/ratelimit`-style algorithm
  implemented against Postgres.

---

## 3. Open-source stack catalogue

For each entry: **license · repo · maintenance · why · alternatives ·
compatibility**.

### 3-A. Acquisition / parsers

**A1 — edfdecoder**
- MIT · https://github.com/jodogne/edfdecoder · last release 2024, low
  activity but stable spec.
- Why: pure JS, runs in Workers, no native deps.
- Alternatives: `edfjs` (less complete), Pyodide+MNE (heavy).
- Compatibility: drop-in for current `parsers/edf.ts`.

### 3-B. Streaming

**B1 — pylsl** — MIT · https://github.com/labstreaminglayer/pylsl · active.
Why: canonical LSL client. Alternatives: `liblsl` C++ direct (more work).
Compatibility: runs in optional Python sidecar, not in the Worker.

**B2 — ws** (Node) — MIT · https://github.com/websockets/ws · very active.
Why: only needed if we run a Node bridge; Workers use native WebSocket.

### 3-C. BrainFlow

**C1 — brainflow** — MIT · https://github.com/brainflow-dev/brainflow ·
very active, multi-language. Why: single API for ~30 devices.
Alternatives: per-vendor SDKs (fragmented). Compatibility: Node binding
only; not Worker-safe — must run in a sidecar.

### 3-D. MNE

**D1 — MNE-Python** — BSD-3 · https://github.com/mne-tools/mne-python ·
very active. Why: gold standard for EEG preprocessing. Compatibility: via
Pyodide in-browser (already wired) or Python sidecar.

### 3-E. DSP

**E1 — fft.js** — MIT · https://github.com/indutny/fft.js · stable,
low-maintenance. Why: pure JS radix-4 FFT, fast enough for 64ch @ 250Hz.
Alternatives: `kissfft-wasm` (faster, WASM overhead), `dsp.js` (older).
Compatibility: zero-dep, works in Worker + browser.

### 3-F. (covered under E and P)

### 3-G. Vector DB

**G1 — pgvector** — PostgreSQL License ·
https://github.com/pgvector/pgvector · very active. Why: already
available via Lovable Cloud; ivfflat + hnsw indexes.
Alternatives matrix:
- Lightweight: **hnswlib-node** (Apache-2.0,
  https://github.com/yoshoku/hnswlib-node) for in-process eval.
- Long-term scalable: **Qdrant** (Apache-2.0,
  https://github.com/qdrant/qdrant) — self-hostable, gRPC, payload
  filtering. **Weaviate** (BSD-3) and **Milvus** (Apache-2.0) are
  alternatives but heavier ops.
Compatibility: pgvector slots behind current `VectorIndex` interface;
Qdrant via REST adapter only when scale demands.

### 3-H. Knowledge graph

**H1 — Postgres `ltree` + recursive CTEs** — PostgreSQL License · ships
with Postgres. Why: zero new infra, sufficient for subject→session→window
hierarchies.

**H2 — Apache AGE** — Apache-2.0 ·
https://github.com/apache/age · active. Why: Cypher on Postgres; the
upgrade path when graph queries get gnarly. Alternatives: Neo4j Community
(GPL-3), JanusGraph (Apache-2.0, heavy), Memgraph (BSL).

### 3-I. (representation — pure code, no new deps)

### 3-J. Foundation-model research

**J1 — Braindecode** — BSD-3 ·
https://github.com/braindecode/braindecode · very active. Why: already
the source of EEGConformer; provides EEGNet, ShallowFBCSPNet, Deep4Net.
Alternatives: `torcheeg` (Apache-2.0,
https://github.com/torcheeg/torcheeg) — broader model zoo, less
established. Compatibility: PyTorch 2 → ONNX export already proven.

**J2 — EEGPT (reference impl)** — research code, license per upstream.
Pending T-016 decision.

### 3-K. Benchmarking

**K1 — MOABB** — BSD-3 · https://github.com/NeuroTechX/moabb · active.
Why: standard motor-imagery / P300 / SSVEP benchmark suite. Alternatives:
roll-your-own (no).

### 3-L. Explainability

**L1 — Captum** — BSD-3 · https://github.com/pytorch/captum · active
(PyTorch org). Why: integrated gradients, saliency, layer attribution on
PyTorch models. Alternatives: `shap` (MIT) for tabular features.

### 3-M. Dataset management

**M1 — DVC** — Apache-2.0 · https://github.com/iterative/dvc · very
active. Why: dataset/version pinning. Alternatives: LakeFS (Apache-2.0,
heavy), plain git-lfs (cheap, less metadata). For now we use a
Postgres-resident manifest and reserve DVC for the long-term scalable
path.

### 3-N. Training

**N1 — PyTorch Lightning** — Apache-2.0 ·
https://github.com/Lightning-AI/pytorch-lightning · very active. Why:
removes boilerplate, plays well with Hydra and MLflow.

**N2 — MLflow** — Apache-2.0 ·
https://github.com/mlflow/mlflow · very active. Why: tracking + registry
with a permissive license and SQLite-friendly local mode. Alternatives:
`aim` (Apache-2.0, lighter), `wandb` (proprietary — excluded by mandate).

**N3 — skl2onnx** — Apache-2.0 ·
https://github.com/onnx/sklearn-onnx · active. Why: lets the cognitive
decoder ship through the existing ONNX runtime path.

### 3-O. (registry — covered by N2 + existing TS registry)

### 3-P. ONNX deployment

**P1 — onnxruntime-web** — MIT ·
https://github.com/microsoft/onnxruntime · very active. Why: incumbent;
WebGPU EP arriving. Alternatives: `tract` (Apache-2.0/MIT,
https://github.com/sonos/tract) — WASM-only, smaller, no WebGPU.

**P2 — onnx-simplifier** — Apache-2.0 ·
https://github.com/daquexian/onnx-simplifier · active.

**P3 — onnxoptimizer** — Apache-2.0 ·
https://github.com/onnx/optimizer · maintained by ONNX org.

### 3-Q. Research platform tooling

- **JupyterLab** (BSD-3) for authoring.
- **nbconvert** (BSD-3) for the static portal (T-026).
- **Hydra** (MIT, https://github.com/facebookresearch/hydra) for config.
- **Forgejo** (MIT/GPL, https://codeberg.org/forgejo/forgejo) reserved as
  sovereign CI mirror.

---

## 4. Roadmaps

### 4.1 30-day (Days 1–30)

Goal: **make the embedding floor honest.**

| Day | Tasks | Outcome |
|---|---|---|
| 1–3 | T-008, T-009 | ORT WASM self-hosted, ONNX hash-pinned. |
| 4–6 | T-007, T-028 | FFT in place; uploads hardened. |
| 7–11 | T-010 | EEGConformer evaluated on BCI-IV-2a; numbers in repo. |
| 12–17 | T-011 | pgvector live behind existing TS interface. |
| 18–20 | T-012, T-027 | Recall@10 SLO + CI gates wired. |
| 21–25 | T-025 | Cognitive decoder v0 trained, exported, served. |
| 26–30 | T-016, T-026 | EEGPT honesty pass; notebook portal live. |

Exit criteria: V19, V20, V25 all upgraded from ⚠️/❌ to ✅; readiness
score target ≥ 72.

### 4.2 90-day (Days 31–90)

Goal: **make the fabric multi-source and multi-model.**

- T-001, T-003, T-005, T-004 — acquisition + streaming + BrainFlow + LSL.
- T-002, T-006 — parser hardening + MNE parity harness.
- T-015, T-023, T-017 — Braindecode zoo + ONNX optimisation + MOABB.
- T-020, T-021, T-022 — training container + MLflow + registry sync.
- T-014 — subject-level aggregation.

Exit criteria: at least three foundation models head-to-head on MOABB
with reproducible numbers; live LSL demo runs end-to-end; readiness
target ≥ 80.

### 4.3 180-day (Days 91–180)

Goal: **make the upper layers real.**

- T-013 — concept graph schema, with provenance queries on the dashboard.
- T-018 — Captum saliency surfaced on `/embeddings`.
- T-024 — WebGPU EP behind a flag; real-time decoding on supported
  browsers.
- Reconstruction track (currently V21/V22) revisited only here, and only
  as a discriminative-conditioning experiment (no generative claims yet).
- DVC migration (§3-M1) if dataset count > 10.
- Qdrant adapter (§3-G) only if pgvector recall SLO is exceeded.

Exit criteria: defensible Neuro-Fabric story end-to-end (acquire → embed
→ persist → decode → explain → query); readiness target ≥ 88.

---

## 5. Top-25 highest-impact tasks (ranked)

Each row scored 1–5 on Scientific (S), Engineering (E), Investor (I),
Algeria-Builder (A). Total is unweighted sum, ties broken by S then I.

| Rank | Task | S | E | I | A | Σ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | T-010 EEGConformer empirical validation | 5 | 4 | 5 | 4 | 18 |
| 2 | T-011 pgvector migration | 4 | 5 | 5 | 4 | 18 |
| 3 | T-025 Trained cognitive decoder v0 | 5 | 4 | 5 | 4 | 18 |
| 4 | T-012 Recall@10 SLO harness | 5 | 5 | 3 | 4 | 17 |
| 5 | T-017 MOABB evaluation harness | 5 | 4 | 4 | 4 | 17 |
| 6 | T-008 Self-hosted ORT WASM | 3 | 5 | 4 | 5 | 17 |
| 7 | T-009 Content-hashed ONNX artefact | 3 | 5 | 4 | 5 | 17 |
| 8 | T-015 Braindecode model zoo | 5 | 4 | 4 | 3 | 16 |
| 9 | T-018 Captum saliency | 5 | 3 | 5 | 3 | 16 |
| 10 | T-027 CI/CD release gates | 3 | 5 | 4 | 4 | 16 |
| 11 | T-005 BrainFlow integration | 4 | 4 | 5 | 3 | 16 |
| 12 | T-003 WebSocket EEG gateway | 4 | 5 | 4 | 3 | 16 |
| 13 | T-021 MLflow tracking | 4 | 5 | 3 | 4 | 16 |
| 14 | T-013 Concept graph schema | 5 | 4 | 4 | 3 | 16 |
| 15 | T-020 Reproducible training container | 4 | 5 | 3 | 4 | 16 |
| 16 | T-022 Registry ↔ MLflow ↔ Storage sync | 3 | 5 | 4 | 3 | 15 |
| 17 | T-006 Pyodide-MNE parity harness | 5 | 4 | 2 | 4 | 15 |
| 18 | T-002 EDF+/BDF/GDF parser hardening | 4 | 4 | 3 | 4 | 15 |
| 19 | T-007 FFT replacement | 3 | 5 | 3 | 4 | 15 |
| 20 | T-024 WebGPU EP flag | 3 | 5 | 4 | 3 | 15 |
| 21 | T-028 Upload hardening | 2 | 5 | 4 | 4 | 15 |
| 22 | T-014 Subject-level aggregation | 5 | 3 | 3 | 3 | 14 |
| 23 | T-019 Dataset manifest | 4 | 4 | 3 | 3 | 14 |
| 24 | T-004 LSL bridge | 4 | 3 | 3 | 4 | 14 |
| 25 | T-026 Notebook portal | 3 | 3 | 4 | 4 | 14 |

T-001, T-016, T-023 fall just below the cut and are tracked in §2.

---

## 6. Shortest realistic path (open-source only)

**Question.** "If the goal is to build the original Neuro-Fabric vision
using only open-source technologies, what is the shortest realistic path
from the current repository state to that objective?"

**Answer.** A six-step critical path of roughly **9 engineer-weeks** of
focused work:

1. **Honesty pass (week 1).** Self-host ORT WASM, hash-pin the ONNX
   artefact, harden uploads, FFT-replace the DFT. (T-007, T-008, T-009,
   T-028.) These remove the silent failure modes that would invalidate
   every later claim.
2. **Empirical floor (week 2).** Run T-010 on BCI-IV-2a using
   Braindecode + MOABB. Publish the numbers, good or bad, in the repo.
   Without this step, every later milestone is unfalsifiable.
3. **Representation persistence (week 3).** Migrate the in-memory index
   to **pgvector** (T-011) behind the existing `VectorIndex` interface,
   then wire **recall@10** as a CI-gating SLO (T-012, T-027). This is the
   single highest-leverage move because it unlocks subject-level history,
   provenance queries, and the concept graph.
4. **Cognitive layer (weeks 4–5).** Train a small **scikit-learn**
   classifier on a public workload/attention dataset, export via
   **skl2onnx**, serve it through the same ONNX runtime path, and retire
   the heuristic ratios behind a feature flag (T-025, T-019). The
   "AI-driven cognitive metric" claim becomes literally true.
5. **Acquisition + streaming (weeks 6–7).** Add the WebSocket gateway
   (T-003), the **BrainFlow** sidecar (T-005), and the **LSL** bridge
   (T-004). The platform now ingests live EEG from real hardware, not
   just file uploads.
6. **Comparative depth + explainability (weeks 8–9).** Register the rest
   of the **Braindecode** zoo (T-015), run **MOABB** head-to-head
   (T-017), and overlay **Captum** saliency on the `/embeddings` route
   (T-018). The fabric now has multiple models, a benchmark story, and a
   defensible explanation surface.

Everything beyond this nine-week path — knowledge-graph upgrade to AGE,
Qdrant, DVC, EEGPT, reconstruction — is *optionality*, not *vision
completion*. The original Neuro-Fabric vision (acquire → embed →
persist → decode → explain) is reachable with the open-source stack
already catalogued in §3, using only software whose licenses are
permissive (BSD-3, MIT, Apache-2.0, PostgreSQL License) and whose
maintenance posture is currently healthy.

The binding constraint is not technology. It is the discipline to do
steps 1–3 before steps 4–6, and to refuse to ship steps 4–6 until step 2
has produced numbers worth defending.

---

## 7. Preservation statement

No source files, migrations, prior audits, or roadmaps were modified by
this blueprint. All baseline material remains in place under `docs/`,
`docs/audits/`, and `docs/roadmaps/`.