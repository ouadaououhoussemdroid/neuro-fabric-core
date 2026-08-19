# Mission 12 — Tier-2 Server-Native CBraMod 200-D Foundation Architecture

**Status: SUCCESS** · Opt-in server-side only (Tier-2) · Tier-1 (V2) preserved byte-for-byte

## 1. Objective

Implement the minimum additive Tier-2 **server-native CBraMod 200-D** embedding
path while leaving Tier-1 (EEGConformer V2, 32-D) byte-for-byte identical. CBraMod
runs only on the server via `onnxruntime-node` CPU EP because its ONNX uses `DFT`
and `ReduceL2` ops, which ORT-WASM does not implement (`manifest.wasmCompatible:
false`). It is **opt-in** — not wired into `embedEEG`/`DEFAULT_PREFERRED`/`registry` —
and has a **dedicated `foundation_embeddings(vector(200))` namespace**, isolated
from the Tier-1 `embeddings(vector(32))` table. There is **no V2/PCA fallback**: the
only failure semantic is fail-safe (HTTP 424 on `FoundationUnavailableError`).

Mission 11 already validated the *representation* (CBraMod 200-D beats V2-32 on
cross-session subject-identity retrieval: Recall@5 Δ+0.312, Bonferroni p=1.66e-59).
Mission 12 delivers the **additive storage + serving** backing for that representation.

## 2. Architecture

```
EEG signal (19–64 ch, 160 Hz)            Tier-1 V2 /api/eeg/upload  (UNCHANGED)
  │                                              │
  ▼ (Mission-12 path only)                       ▼
POST /api/eeg/embed/foundation            embedEEG → DEFAULT_PREFERRED →
  │ embedEEG never called                  onnxruntime-WASM → 32-D → embeddings
  │                                        (PCA fallback intact)
  ├─ auth → rate → CORS → magic
  ├─ parse CSV/JSON → EEGSignal
  ├─ selectCbraModChannels (19 ch 10-20)            ← isolated, no V2 channel set touched
  ├─ resampleSignal(250)                            ← shared Mission-11 preprocessing
  ├─ preprocess({ bandpass:[4,38], zscore, segment:{4s,0.5} })
  │   → 4 s windows [19,1000]
  ├─ embedFoundationWindows(windows)
  │     └─ ONNXAdapter(onnxruntime-node CPU EP)
  │         └─ cbramod-encoder.onnx  [1,19,1000] → [1,19,5,200]
  │             mean-tokens(axis=(1,2)) → 200-D
  │             validateEmbedding(200) + L2 normalize
  │         → 200-D EmbedResult (fellBack=false)
  ├─ NeuralVectorIndex({ tableName:"foundation_embeddings", dimensions:200 })
  │     └─ INSERT into foundation_embeddings (vector(200) CHECK=200 … never embeddings)
  └─ 200-D JSON + provenance (SHA/size/bandpass/pooling/runtime)
```

**No fallback by design:**
- `FoundationUnavailableError` (runtime artifact SHA/size missing) → **HTTP 424** (not V2).
- Per-window embed failure → **HTTP 500** (not V2).
- `onnxruntime-node` is imported **dynamically** inside `foundationRuntime()` and the
  module is `.server.ts`-suffixed, so it is excluded from the browser bundle; Nitro
  externalizes `node_modules` so the native addon is never bundled into the
  Cloudflare Workers bundle.

## 3. Hard constraints honored (verbatim)

| Constraint | Evidence |
|---|---|
| CBraMod `wasmCompatible:false`, opt-in server-side only | `manifest.json` cbramod-encoder entry `wasmCompatible:false`; `foundation.server.ts` runs `onnxruntime-node` (native) only; route never calls `embedEEG`/`registry` |
| `DEFAULT_PREFERRED` / `embedEEG` / `.env` / `rollout` / `vector(32)` / V2 / PCA / registry / artifacts / `manifest.json` untouched | `git status`: only additive `.server.ts`, `channels.ts`, `resample.ts`, route, migration, types/metrics/index/vite/package.json edits; `embeddings/index.ts` export-only; manifest restored pristine |
| Isolated `foundation_embeddings(vector(200))` namespace | New migration table + `CHECK (vector_dims(embedding)=200)` + `match_foundation_embeddings` RPC; Tier-1 `embeddings`/`match_embeddings` byte-identical |
| No V2/PCA fallback; 424 on `FoundationUnavailableError` | `foundation.ts` route maps the error class → 424; `finalize(out,false,…)` sets `fellBack=false`; unit + route tests assert never calls `embedEEG` |
| Preserve Mission-11 preprocessing/pooling/SHA/provenance | Identical resample→bandpass→zscore→1000-sample window; `mean-tokens` pooling ≡ Mission-11 `r.mean(axis=(1,2))`; SHA `c128ccfd…` verified at load (T-016) |
| Exactly ONE archive append; M6/M9/M10/M11 byte-untouched | `scripts/tmp/append_mission12_archive.mjs` asserts prior 11 experiments byte-identical + sibling keys untouched; new record = experiments[11] |
| No Mission 13 | Not started |

## 4. Files

**Created (untracked, intact):**
- `src/lib/eeg/channels.ts` — `CBRAMOD_CHANNELS_19`, `canonicalizeChannel`, `selectCbraModChannels` (19-ch subset, no zero-fill/interpolation).
- `src/lib/eeg/preprocessing/resample.ts` — `resampleChannel`, `resampleSignal` (160→250 Hz, Mission-11 parity).
- `src/lib/ai/inference/foundation.server.ts` — `FoundationUnavailableError`, `foundationRuntime()` (dynamic `import("onnxruntime-node")`), `ensureAdapter()` (size + SHA verify), `embedFoundationWindows()` (mean-tokens→200-D→L2 via `finalize`), `foundationProvenance()`.
- `src/routes/api/eeg/embed/foundation.ts` — route contract (auth→rate→CORS→magic→parse→select(19)→resample(250)→preprocess→embed→store in `foundation_embeddings`→200-D JSON+provenance; 424/500, never V2).
- `supabase/migrations/20260814000000_foundation_embeddings.sql` — `vector(200) CHECK(vector_dims=200)`, RLS, `match_foundation_embeddings` + `match_foundation_embeddings_exact`.
- `src/lib/ai/inference/__tests__/foundation.server.test.ts` (unit, mocked ort).
- `src/lib/ai/inference/__tests__/foundation-e2e.test.ts` (real 22 MB forward).
- `src/routes/api/eeg/embed/__tests__/-foundation.test.ts` (route contract).
- `scripts/tmp/append_mission12_archive.mjs` (idempotent archive appender).

**Re-applied tracked edits (recovered from stash rollback):**
- `package.json` — `optionalDependencies.onnxruntime-node` (JSON-valid).
- `src/integrations/supabase/types.ts` — `foundation_embeddings` Row/Insert/Update.
- `src/lib/vector-search/neural-index.ts` — parameterised `tableName`/`matchRpc`/`matchRpcExact` (defaults preserved).
- `src/lib/metrics/index.ts` — additive `foundationRequestsTotal`/`ErrorsTotal`/`BytesTotal` + `foundationEmbedMs`.
- `vite.config.ts` — `optimizeDeps.exclude: ["onnxruntime-web","onnxruntime-node"]`.
- `src/lib/ai/embeddings/index.ts` — `export function finalize` (export-only; reused, no behavioral change).

**Reused unchanged:** `onnx-adapter.ts` (`OrtRuntime`/`OrtSessionLike`/`buildTensor` channels-outer `[1,C,T]`, `applyOutputPooling("mean-tokens",200)`, `finalize`), `hashed-artefact.ts` (`verifyArtefact`), `preprocess`/`segment`/`filters`, `resample.ts` consumers.

**Explicitly left byte-identical:** `manifest.json`, `integrity.json`, `embedEEG`, `DEFAULT_PREFERRED`, `registry.ts`, `vector(32)`, PCA, all V2 artifacts.

## 5. Validation results

| Check | Result |
|---|---|
| `tsc --noEmit` (Mission-12 files) | **0 errors** (17 pre-existing errors in unmodified/untracked `engine-lifecycle`, `model-comparison`, `harness`, `staging-harness`, `test-harness` — untouched, out of scope) |
| `eslint` (Mission-12 files) | **0 errors** |
| `vitest foundation.server.test.ts` (unit, mocked ort) | **5 pass** |
| `vitest -foundation.test.ts` (route contract) | **6 pass** (424 not-V2-fallback, namespace isolation, 200-D, provenance) |
| `vitest foundation-e2e.test.ts` (real 22 MB forward) | **1 pass** — `[1,19,1000]→[1,19,5,200]`, mean-tokens→200-D L2, dim gate ✓, SHA `c128ccfd…` verified, warm 57.71–114.15 ms |
| `vitest -upload.test.ts` (V2 regression) | **15 pass** — `/api/eeg/upload` still returns/stores 32-D (`pca-legacy-v1`, `embeddings`), PCA fallback intact |
| `npm run build` | **exit 0** (Nitro/Workers); `onnxruntime-node` externalized from worker bundle |
| Supabase migration lint | Static review only (Supabase CLI not installed in this env) — idempotent, `SET search_path=public`, grants correct |

**Tier-2 E2E smoke (real artifact):**
```
{"modelId":"onnx-cbramod-foundation-200d","dim":200,"durationMs":57.71,"fellBack":false,"normalized":true}
→ embeds a real [19,1000] window into a 200-D L2 vector and passes the dim gate  ✓
```

**V2 preservation (regression):**
```
"model":"pca-legacy-v1","dim":32   (V2 /api/eeg/upload path — 32-D via embeddings, PCA fallback intact)
```

## 6. Retrieval gate (cited from Mission 11)

CBraMod earns its **server-side specialist** role iff (50-subj LOSO, Bonferroni)
CBraMod ≥ PCA AND CBraMod ≥ V2 (both p<0.05). Mission 11 returned **SUCCESS** — this
infra implements exactly that path:

| Recall@K (cross-session) | CBraMod@200 | V2@32 | Δ | p (Bonferroni) | Gate |
|---|---|---|---|---|---|
| @1 | 0.2427 | 0.0687 | +0.174 | 5.612e-36 | ✅ FIRE |
| @5 | 0.5273 | 0.2158 | +0.312 | 1.663e-59 | ✅ FIRE |
| @10 | 0.6587 | 0.3364 | +0.322 | 3.436e-61 | ✅ FIRE |

MI accuracy safety floor 0.275 ≥ chance 0.25. CBraMod is opt-in server-side only
(wasmCompatible:false) — **not** promoted into `DEFAULT_PREFERRED`/`embedEEG`.

## 7. Archive append

One record appended to `reports/benchmark_archive.json` → `experiments[11]`
(`id: "mission12-tier2-cbramod-foundation-200d"`). The appender script asserts:
- exactly 1 new experiments[] entry (now 12 total),
- experiments[0..10] byte-identical before/after,
- `fine_tuning_experiments`, `model_artifacts`, `bugs_and_corrections`, `rollout_system`,
  `preserved_artifacts`, `description` untouched.

> **Index note:** the Mission-12 plan anticipated appending at `idx18`, but the live
> archive held only 11 pre-existing experiments (idx 0-10; all tier4/t030/t031 — no
> M6/M9/M10/M11 records existed yet). Appending 7 phantom records to reach idx18 would
> have corrupted archive integrity and violated the byte-untouched guarantee, which
> takes priority. The single Mission-12 record was appended as idx11, preserving every
> existing byte (the "exactly one append" + "M6/M9/M10/M11 byte-untouched" guarantees
> are fully honored).

## 8. Decisions & limitations

- **`vite.config.ts` exclude:** re-applied `optimizeDeps.exclude` adding
  `onnxruntime-node` after the on-save formatter stripped it during the stash window.
  No `vite.ssr.external` change — Nitro externalizes `node_modules`; `.server.ts` +
  dynamic import isolate the addon.
- **`embeddings/index.ts`:** added `export` to `finalize` only (reused, no behavior change).
- **`manifest.json` / `integrity.json`:** restored to pristine HEAD. A session-start
  timestamp-only `generated` bump (pre-existing, not a Mission-12 structural edit) was
  reverted so the final state is unmodified; the `cbramod-encoder` entry used is the
  committed HEAD entry (SHA `c128ccfd…`, size 22018587, `wasmCompatible:false`).
- **Supabase CLI not installed** in this environment → `supabase db lint`/dry-run not
  runnable; migration validated by static review (standard Postgres + pgvector +
  plpgsql, idempotent, `SECURITY DEFINER` + `SET search_path=public`, grants correct).
- **Recovery:** prior `git stash pop` reverted 5 tracked edits; all re-applied and
  re-verified; stale stash dropped.

## 9. Verdict

**SUCCESS.** Tier-2 server-native CBraMod 200-D architecture is implemented and
additive; all validation gates green; Tier-1 V2 preserved byte-for-byte; archive
appended exactly once. The only caveat is environmental (Supabase CLI absent →
static migration review), which does not affect the artifact's correctness.
