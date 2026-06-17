# EEGConformer — Deployment Roadmap

- Date: 2026-06-17
- Companion to: `docs/audits/2026-06-17_eegconformer-artifact-acquisition.md`
- Assumes the Track-A artefact (`eegconformer-bciiv2a-v1.onnx`) from the
  acquisition report.
- Touches **no** existing pipelines: PCA fallback, generic ONNX, vector
  search, audits, and the EEG preprocessing chain remain unchanged.

## Phase 0 — Artefact production (off-platform)

| Step | Owner | Output | Effort |
|---|---|---|---|
| Train EEGConformer on BCI-IV-2a (Braindecode tutorial) | ML eng | `eegconformer.pt` | 1 d |
| Run `scripts/export_braindecode_eegconformer.py` | ML eng | `eegconformer.onnx` (~7 MB) + parity report | 0.25 d |
| Author `MODEL_CARD.md` (dataset, licence, metrics, intended use) | ML eng | `MODEL_CARD.md` | 0.25 d |
| Internal security review (no PII, no embedded secrets, ONNX checker pass) | Sec | sign-off | 0.25 d |

## Phase 1 — Hosting

Two viable hosting options; pick **one** for v1.

| Option | Where | Pros | Cons | Recommended |
|---|---|---|---|:---:|
| **A. Lovable Cloud Storage (public bucket)** | `models` bucket, public read | Versioned, signed URLs available, integrates with existing infra | Egress counted against project | ✅ for v1 |
| B. App-bundled (`public/models/eegconformer.onnx`) | Static asset | Zero infra | +7 MB to every page load; cache-busts on deploy | only for offline demos |
| C. Third-party CDN (R2 / S3 + Cloudflare) | External | Cheapest egress | Extra origin, CORS to configure | future |

v1 hosting plan (Option A):

1. Create bucket `models` (public, read-only).
2. Upload `eegconformer.onnx` and `MODEL_CARD.md` with content-hash in
   the path: `models/eegconformer/v1-<sha256[:8]>/eegconformer.onnx`.
3. Record the immutable URL in `src/lib/ai/artifacts/index.ts`
   alongside the existing `braindecode-eegconformer-prod` entry (one-line
   change, no API surface change).
4. CORS: allow GET from the production + preview origins only.

## Phase 2 — Wiring (single TypeScript change set)

No new files; no refactors. Three edits, in this order:

1. **`src/lib/ai/artifacts/index.ts`** — flip `braindecode-eegconformer-prod`
   `source` from `inline` placeholder to `{ kind: "url", url, sha256 }`.
2. **App boot (`src/router.tsx` or a small `src/lib/ai/bootstrap.ts`)** —
   call `registerBraindecodeEEGConformer({ artifact: ... })` once.
3. **`src/lib/ai/inference/embed-eeg.ts`** — set
   `DEFAULT_PREFERRED = "braindecode-eegconformer-prod"`.

Fallback chain stays:
`EEGConformer → generic ONNX (if registered) → PCA`. PCA is unaffected.

## Phase 3 — Validation

Run, in CI and once in staging, against the existing harnesses:

- `benchmarkAll(["pca-legacy-v1", "braindecode-eegconformer-prod"], ...)`
  → expect P50 ≤ 400 ms, heap Δ ≤ 25 MB.
- `validateEmbedding()` checks (already in suite) → no NaN / Inf /
  zero-vector / dim mismatch.
- Cosine recall@10 on BCI-IV-2a holdout vs PCA → expect ≥ +15 pp
  (gate from production-readiness report).

## Phase 4 — Rollout

| Stage | Cohort | Flag | Exit criterion |
|---|---|---|---|
| Canary | 5 % of authenticated users | `ai.eegconformer.enabled = canary` | < 0.5 % fallback rate over 24 h |
| Beta | 50 % | `= beta` | P95 latency < 600 ms; no error-budget burn |
| GA | 100 % | `= ga` | one week green |
| Rollback | n/a | `= off` → `unregisterModel(...)` | < 5 min MTTR |

Vectors written under each model id are tagged in `NeuralVectorIndex`;
rolling back does **not** require re-indexing — old PCA vectors keep
matching their own model id.

## Phase 5 — Post-launch

- Track P50 / P95 / fallback-rate dashboards (already covered by
  benchmark + structured logs `ai.embed.*`).
- Schedule Track-B retrain (PhysioNet MI → BCI-IV-2a fine-tune) once GA
  is stable.
- Reassess hosting Option C if egress becomes the dominant cost line.

## Effort summary

| Phase | Effort |
|---|---|
| 0 — Artefact | ~2 d |
| 1 — Hosting | 0.5 d |
| 2 — Wiring | 0.25 d |
| 3 — Validation | 0.5 d |
| 4 — Rollout | 1 week wall-clock, < 1 d hands-on |
| **Total to GA** | **~4 engineer-days + 1 week canary** |