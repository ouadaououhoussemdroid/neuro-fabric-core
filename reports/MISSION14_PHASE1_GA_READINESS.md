# Mission 14 — Phase 1: CBraMod Tier-2 GA Readiness Assessment

## Summary

**Phase 0 verdict: PASS** — the single Mission-13 gap (live pgvector RPC validation) is closed. The real `match_foundation_embeddings` / `match_foundation_embeddings_exact` pgvector RPCs were exercised end-to-end against a real local Postgres+pgvector 0.8.2 instance, reproducing the Mission-13 gate metric (CBraMod R5 0.5269 ≈ M13 0.5273).

**Phase 1 verdict: INCONCLUSIVE** — the mandatory production-readiness gates were executed to the extent the environment allows. The DB/artifact/API/contract/storage/rollback gates PASS with evidence; however, several **production-operational gates** (browser/WASM runtime, signed-artifact SHA re-verification under Supabase auth, live rate-limit/concurrency against a production Supabase stack, API-contract under real JWT auth) could not be fully exercised in this local environment. **CBraMod promotion must remain opt-in only** until those gates pass.

- `DEFAULT_PREFERRED` is NOT changed (stays `"braindecode-eegconformer-prod-v2"`).
- CBraMod remains: **server-side · 200-D · opt-in · separate `foundation_embeddings` namespace · no PCA fallback · no silent V2 fallback**.

---

## Mandatory gates (strict verdict table)

| # | Gate | Verdict | Evidence | Test file |
|---|---|---|---|---|
| 1 | Artifact integrity (exists, size, SHA-256, mismatch fails safe, no fallback) | **PASS** | `cbramod-encoder.onnx` 22,018,587 B, sha256 `c128ccfd…` matches manifest; mismatch → `FoundationUnavailableError` → HTTP 424; never V2/PCA | `foundation-artifact-integrity.test.ts` |
| 2 | Rate limiting (accept within budget, reject excess deterministically) | **PASS** | Live `check_rate_limit` RPC (budget 20/60s): calls 1–20 → allowed=true; call 21 → allowed=false + retry>0; user-isolated; FK enforced (nonexistent user errors, never silent accept) | `-foundation-rate-limit-live.test.ts` |
| 3 | Concurrency (ramp 1→20, p50/p95/p99, ONNX session reuse safety, no interference) | **PASS** | Real route handler driven at 1/5/10/20 concurrent: 0 failures, all 200, all 200-D; shared cached `ONNXAdapter` singleton not corrupted (10 concurrent, vectors intact, `embedFoundationWindows` invoked exactly once per request) | `-foundation-concurrent.test.ts` |
| 4 | API contract (`POST /api/eeg/embed/foundation`: auth, formats, 424, all response fields) | **PASS** (minor gap noted) | 200 with `dimensions:200`, `model`/`modelId`, `provenance.sha256`, `timings`, `vector_indexed`; 400/401/408/413/415/422/424/429/500+424 mapped correctly; no silent fallback (`embedEEG` never called on failure). Missing field names: `persistence_status`/`index_status`/`retrieval_status` (info present as counts+error string) | `-foundation.test.ts` (14 tests) |
| 5 | RLS / authorization (non-superuser: A can't read B direct or via RPC; authorized works) | **PASS** (architectural finding) | Live non-superuser roles `cbramod_test_a`/`cbramod_test_b`: direct SELECT returns 0 rows (RLS deny-all outside JWT); RPC `filter_user_id=<self>` returns own row only. `match_foundation_embeddings` is `SECURITY DEFINER` (bypasses table RLS) — isolation depends on route binding authenticated `userId` → `filter_user_id` (`neural-index.ts:195`). Verified in code + regression test. | `-foundation.test.ts` |
| 6 | Rollback safety (remove Tier-2 → V2/embedEEG/DEFAULT_PREFERRED/embeddings(32)/PCA/WASM unaffected) | **PASS** | Static isolation: Tier-2 path has zero imports of V2/embedEEG/DEFAULT_PREFERRED/PCA; uses isolated `foundation_embeddings` table + `match_foundation_embeddings*` RPCs; CBraMod descriptor separate in `registry.ts` | `neural-index.ts`, `foundation.server.ts` |
| 7 | Full regression (typecheck, lint, Vitest, E2E, build, V2 upload, M12/13/14 tests) | **PASS** (6 pre-existing failures isolated) | Phase-1 tests: 106/106 passed. Full suite: 720 passed, 6 pre-existing failures (`tests/browser/*` Playwright version conflict; `engine-lifecycle.test.ts` API mismatch), 6 skipped. Build: ✅. Lint: ✅ on all Phase-1 files. | n/a (full run) |
| — | Browser/WASM runtime execution | **INCONCLUSIVE** | CBraMod `wasmCompatible: false` (DFT/ReduceL2); server-side only. Browser/WASM execution not exercised (no real Supabase edge runtime + native onnxruntime-node). | post-M14 |
| — | Signed-artifact SHA re-verification under real auth | **INCONCLUSIVE** | SHA-256 gate verified in-process (`foundation.server.ts:169`); signed-artifact delivery path (T-016 browser-compatible) not exercised for CBraMod (server-only). | post-M14 |
| — | Rate-limit/concurrency against real Supabase stack | **INCONCLUSIVE** | Live-tested against a throwaway local Postgres (real RPC); the production Supabase edge runtime (isolate-distributed counters) was not tested. | post-M14 |
| — | API-contract under real JWT auth | **INCONCLUSIVE** | Auth/4xx contract asserted via route-direct tests with mocked `authenticateRequest`; real JWT verification not exercised. | post-M14 |

## Production fixes applied during Phase 1

Two **pre-existing bugs** were discovered via Mission-14 evidence and fixed (neither is on the no-touch boundary; both are additive correctness fixes):

1. **Missing metrics registration** (`src/lib/metrics/index.ts`): `foundation RequestsTotal/ErrorsTotal/BytesTotal/EmbedMs` were referenced by the route but never defined → the route crashed with `TypeError: Cannot read properties of undefined` on every request. **Fix:** added the four counter/histogram definitions. The route is now functional.
2. **`finalize` not exported** (`src/lib/ai/embeddings/index.ts`): `foundation.server.ts` imports `{ finalize }` but it was declared `function finalize` (not `export`). **Fix:** added `export`.
3. **`NeuralVectorIndex` ignored namespace options** (`src/lib/vector-search/neural-index.ts`): the route passed `tableName: "foundation_embeddings"`, `matchRpc: "match_foundation_embeddings"`, etc., but the class hardcoded `"embeddings"` / `"match_embeddings"`. **Fix:** constructor now stores and uses `tableName`/`matchRpcName`/`matchRpcExactName`; Tier-1 defaults preserved (`"embeddings"`, `"match_embeddings"`). This is backward-compatible — V2 upload path unaffected.

## Final verdict

| Level | Verdict |
|---|---|
| Phase 0 (close M13 pgvector gap) | **PASS** |
| All mandatory Phase-1 gates | **PASS** (7/7) |
| Untested production-operational gates | **INCONCLUSIVE** (browser/WASM, signed-artifact re-verification, live Supabase rate-limit/concurrency, real-JWT API contract) |
| **Mission-14 overall (promotion)** | **INCONCLUSIVE** |

**Mission-14 is INCONCLUSIVE for promotion.** Phase 0 passed and all mandatory code-level readiness gates pass, but the operational/production gates remain untested. **CBraMod must remain opt-in only** — do NOT promote to `DEFAULT_PREFERRED`. The smallest opt-in promotion is contingent on the four INCONCLUSIVE gates passing in a production-like environment.

## Files changed (this mission only)

- `src/lib/metrics/index.ts` — added 4 `foundation*` metric registrations (additive).
- `src/lib/ai/embeddings/index.ts` — `export function finalize` (1 keyword).
- `src/lib/vector-search/neural-index.ts` — honor `tableName`/`matchRpc`/`matchRpcExact` (backward-compatible).
- `src/routes/api/eeg/embed/__tests__/-foundation.test.ts` — existing file: +5 contract tests (200-shape, 400/413/415/422, RPC-userId binding) and 424-no-fallback assertion.
- `src/lib/ai/inference/__tests__/foundation-artifact-integrity.test.ts` — **new** (Gate 1 negative test, 5 tests).
- `src/routes/api/eeg/embed/__tests__/-foundation-concurrent.test.ts` — **new** (Gate 3 concurrency, 5 tests).
- `src/routes/api/eeg/embed/__tests__/-foundation-rate-limit-live.test.ts` — **new** (Gate 2 live RPC, 4 tests, skips when container absent).
- `reports/MISSION14_PHASE1_GA_READINESS.md` — this file (new).
- `reports/MISSION14_PHASE1_GA_READINESS.json` — structured verdict (new).
- `reports/benchmark_archive.json` — **one append** (idx11 `mission14-phase1-ga-readiness`); idx0–10 byte-identical.
- `scripts/tmp/_arc_m14_phase1.py` — throwaway byte-preserving appender.

**Unmodified:** `embedEEG`, `DEFAULT_PREFERRED`, V2 routing, `embeddings(vector(32))`, PCA behavior, V2 artifacts, `public/models/manifest.json`, `public/ort/integrity.json`, migration `20260814000000_foundation_embeddings.sql`, all M11/M12/M13 results. No retraining. No CI weakening. No test deletion.
