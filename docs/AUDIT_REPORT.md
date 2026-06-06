# NeuroWeave — Comprehensive Audit Report

**Date:** June 6, 2026
**Repository:** neuro-fabric-core
**Stack:** TanStack Start v1 · React 19 · Vite 7 · Tailwind v4 · Supabase (Lovable Cloud)

---

## 1. Executive Summary

NeuroWeave is a brain-data platform marketed around EEG ingestion, neural
embeddings, cognitive decoding, synthetic EEG generation, and visual
reconstruction. The repository ships a **production-quality marketing
front-end and auth/dashboard substrate**, plus a **functional but baseline
EEG processing core** (parsers, preprocessing, spectral embeddings, baseline
decoder, brute-force vector search). The headline "neural-to-image
reconstruction" surface is a **visual demo** without a real model.

| Layer | State |
|---|---|
| Marketing site / landing | ✅ Implemented |
| Auth (email + Google) | ✅ Implemented |
| Role-based dashboards | ✅ Implemented |
| EEG parsers (EDF/CSV/NPY) | ✅ Implemented (pure-JS) |
| Preprocessing (filters, segmentation, normalize) | ✅ Implemented |
| Spectral embeddings + linear AE | 🟡 Partial (baseline, not learned at scale) |
| Cognitive decoder | 🟡 Partial (heuristic, not trained) |
| Synthetic EEG | ✅ Implemented (1/f + band oscillators) |
| Vector search | 🟡 Partial (in-memory brute force) |
| EEG → image reconstruction | ❌ Mock (procedural visual only) |
| Public dataset loaders (PhysioNet, BCI-IV-2a, TUH) | 🟡 Partial (URL builders; no caching/ETL) |
| Persistence of uploads/embeddings | ❌ Missing |
| API auth / rate limiting / quotas | ❌ Missing |
| Observability / metrics export | 🟡 Partial (console logger only) |
| Billing / credits | ❌ Missing |

**Overall maturity: pre-production beta (v0.9).** Front-end is shippable;
back-end ML is research-grade scaffolding suitable for demos.

---

## 2. Current Architecture

- **Framework:** TanStack Start v1 (SSR, file-based routing in `src/routes/`).
- **Runtime:** Edge-compatible Worker for server functions and `/api/*` routes.
- **DB / Auth / Storage:** Supabase via Lovable Cloud.
- **Server logic:** Two surfaces — `createServerFn` (RPC) and `createFileRoute` `server.handlers` (HTTP). Currently only one real HTTP endpoint exists (`/api/eeg/upload`).
- **Client:** React 19, TanStack Query, shadcn/ui primitives, custom `ui-bits` (`GlassCard`, `Eyebrow`, `StatPill`), bespoke neural particle background.
- **Design system:** Tokens in `src/styles.css` (oklch), dark theme, cyan/teal `--neuro` accent.

---

## 3. Frontend Status — ✅ Implemented

Routes present and rendering:

| Route | Purpose | Status |
|---|---|---|
| `/` | Landing / hero | ✅ |
| `/about`, `/architecture`, `/developers`, `/research`, `/pricing` | Marketing | ✅ |
| `/embeddings`, `/playground`, `/studio`, `/synthetic`, `/upload`, `/eeg2image` | Product surfaces | ✅ UI · 🟡 wiring varies |
| `/signin`, `/signup`, `/reset-password` | Auth | ✅ |
| `/_authenticated/dashboard/{individual,researcher,enterprise}` | Role dashboards | ✅ |
| `/sitemap.xml` | SEO | ✅ |

Highlights: `EEGLive` (real canvas multi-band animation),
`ReconstructionShowcase` (procedural visual demo), `NeuralBackground`,
`DashboardShell`. Design tokens enforced via Tailwind v4 / `styles.css`.

**Gaps:** several product routes (`/eeg2image`, parts of `/studio`,
`/playground`) render demo visuals without calling the real backend
pipeline.

---

## 4. Backend Status — 🟡 Partially Implemented

- **Server functions:** only `src/lib/api/example.functions.ts` scaffold; no domain RPCs (uploads, embeddings, projects, datasets, billing).
- **HTTP routes:** `/api/eeg/upload` is real end-to-end (parse → preprocess → embed → decode). No auth/rate-limit.
- **Auth attacher / middleware:** wired (`requireSupabaseAuth`, `attachSupabaseAuth`).
- **No background jobs, no queues, no webhooks, no cron.**

---

## 5. Database Status — 🟡 Partially Implemented

Tables present: `profiles`, `researcher_profiles`, `enterprise_profiles`, `waitlist`. All have RLS + grants. `handle_new_user` trigger populates role-specific tables via `SECURITY DEFINER`.

**Missing tables for the advertised product:**
`projects`, `datasets`, `uploads`, `signals`, `embeddings` (with pgvector),
`api_keys`, `usage_events`, `credits`, `invitations`, `team_members`,
`reconstructions`, `model_runs`, `audit_log`.

No `pgvector` extension enabled; vector search is in-memory only.

---

## 6. API Status — 🟡 Partial / ❌ Missing

| Endpoint | State |
|---|---|
| `POST /api/eeg/upload` | ✅ Real (no auth, no quota) |
| Public REST surface (developer docs imply one) | ❌ Missing |
| API keys / tokens | ❌ Missing |
| Webhooks | ❌ Missing |
| Rate limiting / quotas / billing meter | ❌ Missing |
| Versioning (`/v1/*`) | ❌ Missing |

---

## 7. EEG Pipeline Status — ✅ Implemented (baseline)

- **Parsers:** EDF, CSV, NPY — pure-JS, no native deps.
- **Preprocessing:** bandpass + notch IIR filters, normalization, windowed segmentation, `PreprocessingReport` with per-step timings.
- **Loaders:** PhysioNet eegmmidb (URL enumeration, real fetch), BCI-IV-2a (operator-supplied mirror), TUH (architecture-only stub returning `[]` until mirror+index supplied).

**Gaps:** no artifact rejection (ICA/ASR), no re-referencing, no resampling, no montage handling, no per-channel quality metrics, no persistence of preprocessed signals.

---

## 8. Embeddings Status — 🟡 Partial

- Real **band-power features** per window (`features.ts`).
- **PCA** and a **linear autoencoder** (`autoencoder.ts`, `pca.ts`) fit on-the-fly per request.
- `embedSignal` returns either `raw-bandpower` or a freshly-fit `linear-ae` projection — **no persistent learned model, no pretraining, no cross-subject alignment**, no contrastive / self-supervised training.
- Advertised "768-d brain vector" / "nwf-7b-embed" is **marketing copy**; actual dimensionality is configurable (default 64) and produced by a per-request linear AE.

---

## 9. Reconstruction Status — ❌ Mock

`ReconstructionShowcase` is a procedural visual (radial gradients + scanlines + diffusion-progress animation). There is **no model**, no CLIP alignment, no diffusion call, no image output. Pipeline labels (`nwf-7b-embed`, `nw-vision-v1 · 40 steps`) are decorative.

---

## 10. Synthetic EEG Status — ✅ Implemented

`generateSyntheticEEG` produces multi-channel signals with 1/f (Voss-McCartney pink noise) + δ/θ/α/β/γ oscillators, seeded RNG, configurable channels/fs/duration/band weights. Output is a real `EEGSignal` and flows through the same pipeline.

**Gaps:** no event/label generation, no subject-conditioning, no artifacts (eye blink, EMG), no head model.

---

## 11. Security Review

**Implemented**
- RLS + grants on every public table; `user_roles`-style separation (`app_role` enum, profile-only role flag).
- `handle_new_user` is `SECURITY DEFINER` with `search_path=public` and `EXECUTE` revoked from public.
- Auth middleware + auth attacher wired correctly.
- Service-role client isolated in `*.server.ts`.

**Risks**
- 🔴 `POST /api/eeg/upload` has **no authentication, no rate limit, no size cap** — DoS / cost attack vector. (Body fully buffered into memory.)
- 🟠 No CSRF/origin checks on the upload route.
- 🟠 No audit log; admin actions not traceable.
- 🟠 No API key system → cannot revoke programmatic access.
- 🟠 `profiles.role` is the source of truth for role-based routing; if RLS or trigger were misconfigured a user could self-elevate via `raw_user_meta_data`. Mitigated today by trigger logic but lacks defense-in-depth (no separate `user_roles` table + `has_role()` SECURITY DEFINER).
- 🟡 No password policy enforcement surfaced in UI; relies on Supabase defaults.

---

## 12. Scalability Review

- **Compute:** all EEG math runs on the edge Worker per request. Large EDF files (>50 MB) will blow CPU/memory budgets. No streaming.
- **Vector search:** in-memory `VectorIndex` — O(n) per query, ephemeral, single-replica. Not viable past ~10k vectors.
- **DB:** no `pgvector`, no partitions, no indexes beyond defaults.
- **Storage:** no object storage configured; uploads are processed then discarded.
- **Queues / jobs:** none; long-running training/inference would need an external worker (currently absent).
- **CDN / caching:** Vite default; no API response caching, no preprocessed-signal cache.

---

## 13. Technical Debt

- Marketing claims (768-d embeddings, 7B model, vision diffusion) diverge from implementation → product/UX risk.
- `routeTree.gen.ts` historically conflicted with manual edits — keep auto-generated only.
- Logger writes to console only; no structured sink.
- Test suite absent — no unit tests for parsers, filters, embeddings, RLS.
- No CI gates documented.
- `src/lib/api/example.functions.ts` is a placeholder still present.
- `tuh` loader is a stub that ships in the public bundle.

---

## 14. Missing Components

1. Persistence: uploads, signals, embeddings, reconstructions tables + storage buckets.
2. `pgvector` + ANN index for embeddings.
3. API key issuance, rotation, scoping, usage metering.
4. Billing / credits ledger; Stripe or Paddle integration.
5. Team / workspace model for Enterprise (invites, seats, roles beyond enum).
6. Background job runner (training, batch embedding, dataset ingestion).
7. Real reconstruction model + CLIP alignment + image storage.
8. Cross-subject alignment / pretrained embedding checkpoint distribution.
9. Artifact rejection (ICA/ASR), resampling, montage tools.
10. Observability: metrics, traces, error reporting sink (Sentry-class).
11. Admin console + audit log.
12. Test suite + CI.
13. Documented public REST API + OpenAPI spec.
14. Email templates, transactional email infra.

---

## 15. Mock vs Real Implementations

| Surface | Real | Mock |
|---|---|---|
| Landing / dashboards | ✅ | — |
| Auth + RLS | ✅ | — |
| `/api/eeg/upload` | ✅ | — |
| EDF/CSV/NPY parsing | ✅ | — |
| IIR filters, segmentation | ✅ | — |
| Band-power features | ✅ | — |
| Linear AE / PCA embedding | ✅ (per-request, untrained at scale) | "768-d nwf-7b-embed" label |
| Cognitive decoder | ✅ heuristic (Pope index, ratios) | "trained classifier" framing |
| Synthetic EEG | ✅ | — |
| Vector search | ✅ in-memory brute force | "production ANN" framing |
| EEG → image | — | ✅ procedural visual |
| Dashboard stats (API calls, credits, latency) | — | ✅ hard-coded values |
| TUH loader | architecture | ✅ returns `[]` without operator config |
| Live ops widget | animation only | ✅ |

---

## 16. Recommended Roadmap

**Phase 1 — Harden the existing surface (1–2 weeks)**
- Add auth + per-user rate limit + max upload size to `/api/eeg/upload`.
- Replace dashboard mock stats with real `usage_events` aggregates.
- Introduce a separate `user_roles` table + `has_role()` SECURITY DEFINER; migrate role checks off `profiles.role`.
- Add Sentry-class error reporting; ship structured logs.
- Add unit tests for parsers, filters, embedding math; add CI.

**Phase 2 — Persistence + Vector Store (2–3 weeks)**
- Enable `pgvector`; create `uploads`, `signals`, `embeddings`, `projects`, `datasets` tables + RLS + grants.
- Add Supabase Storage bucket for raw EEG; stream uploads instead of buffering.
- Move embedding compute to a background job (separate Worker / queue); persist results.
- Replace in-memory `VectorIndex` with `pgvector` ANN search.

**Phase 3 — Productize the API (2–3 weeks)**
- API key issuance, scopes, rotation, audit log.
- Usage metering → credits ledger → Stripe billing.
- Public OpenAPI spec; `/v1/*` routes; SDK examples in `/developers`.

**Phase 4 — Real models (multi-quarter)**
- Pretrained cross-subject embedding checkpoint (distribute as static asset + worker).
- Trained cognitive-state classifier replacing the heuristic decoder.
- Reconstruction: CLIP-aligned latent decoder + hosted diffusion (out-of-Worker GPU service); honest "research preview" framing until shipped.

**Phase 5 — Enterprise (parallel)**
- Workspaces, invites, seats, SSO/SAML.
- Admin console, audit log UI, per-workspace quotas.

---

*End of report.*
