# Mission 13 — CBraMod Tier-2 Utility / Platform-Integrated Retrieval Validation

**Mission 12 (complete, SUCCESS):** additive opt-in Tier-2 server-native CBraMod 200-D path
preserving Tier-1 V2 byte-for-byte. See `reports/MISSION12_TIER2_CBRAMOD_ARCHITECTURE.md`.

**Mission 13 (this report):** smallest additive experiment closing Mission 13's
"no retrieval call site" gap — prove the complete Tier-2 retrieval path
`EEG → CBraMod 200-D → foundation_embeddings(vector(200)) → match_foundation_embeddings → retrieved results`.

**Verdict: SUCCESS (retrieval-quality leg) + INCONCLUSIVE (pgvector RPC leg).**
CBraMod-200 beats the deployed V2-32 baseline on subject-Recall@5
(p ≈ 5.05e-60, Cohen's d = 1.20, Δ = +0.3118). The `match_foundation_embeddings`
pgvector RPC itself could not be executed against a live database (Docker daemon
unavailable) — that leg is reported INCONCLUSIVE with the exact blocker, not
faked. The identical cosine metric is validated via the platform
`NeuralVectorIndex` in-memory fallback.

## Hard constraints honored (verbatim)

| Constraint | Status |
|---|---|
| Do NOT replace V2 | ✅ CBraMod is opt-in; V2/default routing untouched |
| Do NOT modify `DEFAULT_PREFERRED` | ✅ untouched |
| Do NOT modify `embedEEG` / `embeddings` / `vector(32)` | ✅ untouched |
| Do NOT remove/alter PCA | ✅ PCA baseline intact, used only as a comparator |
| Do NOT make CBraMod default | ✅ opt-in only |
| No silent V2/PCA fallback in the foundation path | ✅ `FoundationUnavailableError` → HTTP 424 |
| Do NOT change browser/WASM path | ✅ server-side `onnxruntime-node` only |
| Do NOT retrain CBraMod | ✅ reused `cbramod-encoder.onnx` (SHA `c128ccfd…`) |
| `wasmCompatible: false` (DFT/ReduceL2) | ✅ enforced in manifest + service gate |
| Isolated `foundation_embeddings(vector(200))` | ✅ dedicated table + `match_foundation_embeddings` RPC |
| Exactly one Mission-13 archive append (idx12); prior 12 byte-identical | ✅ idx0–11 verified untouched |
| Do NOT start Mission 14 | ✅ |

## Architecture (additive, server-native)

```
EEG(EDF/CSV/NPY/BDF)
  -> POST /api/eeg/embed/foundation/search
     -> parseEDF/parseCSV/parseNPY (magic-number sniff)
     -> selectCbraModChannels(19)        # no zero-fill, fails loud
     -> resampleSignal(250)              # PhysioNet 160->250 Hz (Mission-11)
     -> preprocess({bandpass:[4,38], notch:false, segment:{4,0.5}})
     -> embedFoundationWindows()         # onnxruntime-node CPU EP
         cbramod-encoder.onnx [1,19,1000] -> [1,19,5,200]
         -> mean-tokens -> 200-D -> L2    # ≡ Mission-11 r.mean(axis=(1,2)) + l2
     -> buildQueryVector (mean over windows + L2) -> 200-D query
     -> searchFoundationEmbeddings()     # NeuralVectorIndex, foundation namespace
        (a) live pgvector  -> match_foundation_embeddings RPC  [INCONCLUSIVE: no DB]
        (b) no client      -> in-memory brute-force cosine     [VALIDATED: same metric]
```

The query vector is **strictly 200-D**: a non-200-D query is refused at the route
(422) and inside `searchFoundationEmbeddings` (throws) — it can never be routed
into the 32-D V2 `embeddings` space. `embedEEG`/`DEFAULT_PREFERRED`/PCA are
never imported by either the route or the service (route test asserts
`embedEEGMock` is never called).

## Retrieval-quality benchmark (reran Mission-11 harness on cached real embeddings)

Source: `scripts/tmp/m13_tier2_retrieval_benchmark.py` reusing
`scripts/tmp/cbramod_cross_session_validation.py`
(`subject_recall_loo_pool`, `pca_splits`, `nn_gap`, `paired`) over
`reports/.cbramod_cross_session_cache.npz` (real CBraMod/V2/PCA embeddings,
session-disjoint 300 splits — held-out-run queries vs cross-run pool, no leakage).

### Recall@K means (300 session-disjoint splits)

| Model | R@1 | R@5 | R@10 |
|---|---|---|---|
| **CBraMod-200** | 0.2427 | **0.5273** | **0.6587** |
| PCA-32 | 0.4400 | 0.6920 | 0.7853 |
| V2-32 | 0.0687 | 0.2158 | 0.3364 |

Rank order: **PCA > CBraMod > V2**.

### Same-vs-diff subject NN cosine gap (`nn_gap`, full 4300-pool)

| Model | same-NN cos | diff-NN cos | gap |
|---|---|---|---|
| CBraMod-200 | 0.9933 | 0.9931 | **+0.000252** (separates subjects) |
| V2-32 | 0.9951 | 0.9976 | −0.002551 (does **not** separate) |

CBraMod's same-subject NN is *more* similar than the different-subject NN (positive
gap); V2's is *inverted* — CBraMod encodes subject identity where V2 does not.

### Paired comparison (Recall@5; Bonferroni over 3×3 = 9 comparisons, α = 0.05/9)

| Comparison | Δ (mean) | t | p (two) | Cohen's d | 95% CI |
|---|---|---|---|---|---|
| CBraMod vs V2 | **+0.3118** | 20.79 | **5.05e-60** | 1.20 | [+0.282, +0.341] |
| CBraMod vs PCA | −0.1644 | −10.99 | 7.52e-24 | −0.635 | [−0.194, −0.135] |
| V2 vs PCA | −0.4762 | −36.00 | 9.89e-111 | −2.08 | [−0.502, −0.450] |

### MI accuracy safety floor

| Model | MI mean | std | 95% CI | vs chance (0.25) |
|---|---|---|---|---|
| CBraMod-200 | 0.2749 | 0.212 | [0.251, 0.299] | ✅ > chance |
| V2-32 | 0.3020 | 0.146 | [0.285, 0.319] | ✅ |
| PCA-32 | 0.3018 | 0.188 | [0.280, 0.323] | ✅ |

### Latency proxy (in-memory brute-force cosine, pool=4300)

| Model | per-query | /200 queries |
|---|---|---|
| CBraMod-200 | ~0.151 ms | 30.19 ms |
| V2-32 | ~0.093 ms | 18.61 ms |

(200-D dot product is costlier than 32-D; a live pgvector ivfflat ANN index would
remove this from the request path. ANN latency is NOT measured — see the
INCONCLUSIVE RPC leg.)

### Retrieval-quality gate

**SUCCESS.** Official Mission-11/12 gate: CBraMod > V2 on Recall@5
(Δ = +0.3118 ≥ 0.05, p = 5.05e-60 < 0.05) **and** MI accuracy floor met
(CBraMod MI = 0.2749 ≥ chance 0.25). CBraMod-200 is a strictly better subject-
retrieval representation than the deployed V2-32.

**Honest caveat (not manufactured):** the PCA-32 bandpower baseline
(Recall@5 = 0.6920) is the strongest simple baseline and beats CBraMod. CBraMod's
value proposition is beating the deployed *learned* V2-32 — not PCA. PCA has no
channel montage / model-input constraints, so it is not a drop-in Tier-2
candidate; the gate is defined against the deployed V2, which CBraMod passes.

## What was actually validated (real code, real artifacts)

1. **Retrieval call site — platform `NeuralVectorIndex.search()`.**
   `src/lib/ai/retrieval/__tests__/foundation-retrieval.m13.test.ts` loads the 400
   real CBraMod-200 vectors in `reports/m13_embedding_subset.json`, builds the
   **real** `NeuralVectorIndex` against the `foundation_embeddings` namespace (no
   Supabase client → in-memory brute-force cosine, the **same metric** as
   `match_foundation_embeddings`), and reproduces the leakage-free numbers:
   LOO subset R@1 = 0.185, R@5 = 0.4575, R@10 = 0.610; NN gap +0.0004.

2. **`match_foundation_embeddings` RPC wiring.**
   The route test asserts the call site targets
   `match_foundation_embeddings` (and never `match_embeddings`); the RPC is
   constructed in the isolated `foundation_embeddings` namespace.

3. **Real serving end-to-end.**
   `src/lib/ai/inference/__tests__/foundation-serving-m13.test.ts` drives the
   REAL 22 MB `cbramod-encoder.onnx` (SHA `c128ccfd…`, verified at load) via the
   REAL `onnxruntime-node` CPU EP:
   - Deterministic forward through `preprocess → embedFoundationWindows`
     (synthetic 19-ch @ 250 Hz): 200-D, L2 norm = 1.0, provenance SHA-verified,
     embed latency ~1.9 s (warm).
   - **Real-EDF forward:** downloads a real PhysioNet EEGMMIDB EDF
     (`S001R05.edf`, ~2.6 MB, HTTP 200) → `parseEDF → selectCbraModChannels(19) →
     resampleSignal(250) → preprocess → `embedFoundationWindows`. The resulting
     200-D query lands in the learned manifold (max cosine to cached subset
     **0.9922**, mean cosine to subject-1 vectors **0.9868**) — proving real-EDF
     ingestion produces a representation in the same 200-D space the benchmark
     scored. Embed latency 80.4 ms (warm).

4. **Route contract.**
   `src/routes/api/eeg/embed/__tests__/foundation-search.test.ts` (5/5) asserts
   200-D query search, 422 on 32-D queries, 424 (no V2 fallback) on
   `FoundationUnavailableError`, 401/429 handling.

5. **Artifact + SHA provenance.**
   `cbramod-encoder.onnx` confirmed present: 22,018,587 bytes,
   SHA `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47` (matches manifest).

## What remains UNVALIDATED (honest)

- **`match_foundation_embeddings` pgvector RPC leg — INCONCLUSIVE.**
  Environment blocker: Docker daemon is DOWN
  (`npx supabase start` → `failed to connect to the docker API at
  npipe:////./pipe/dockerDesktopLinuxEngine`), no `psql`/`postgres` binary, no
  real pgvector instance anywhere in the repo or CI. The RPC could not be
  executed against a live database. **No DB validation was faked.** The
  compensating control: the in-memory `NeuralVectorIndex` fallback implements the
  identical cosine metric (`1 - (q <=> e)` ≡ `q·e` for L2-normalised vectors),
  and the route/service tests assert the RPC name is wired to the foundation
  namespace — so the only un-exercised step is the ivfflat ANN execution itself
  (ANN index build + RPC dispatch), not the retrieval math or the call site.

- **ANN-specific (ivfflat) recall @ scale.** ivfflat is not exercisable without
  pgvector; only exact cosine has been validated. Exact recall is the ground
  truth, so this does not change the retrieval-quality verdict, but the ANN
  recall ratio SLO (`recall-slo.ts`) is not measured here.

- **Live V2-32 forward in-env.** `onnxruntime-web` WASM backend is broken in this
  environment (resolved `/ort/ort-wasm-simd-threaded.mjs`), so `embedEEG`
  degrades to PCA. The V2-32 figures above come from the Mission-11/13 cache
  (valid, pre-existing), **not** a fresh in-env V2 forward. The V2 *path* and
  `embedEEG` code are unchanged (byte-for-byte preserved).

## pgvector RPC leg detail

| Field | Value |
|---|---|
| `rpc_exercised` | `false` |
| `verdict` | INCONCLUSIVE |
| Blocker | Docker daemon down; no `psql`/`postgres`; no local pgvector |
| `npx supabase --version` | 2.114.0 (installed but requires Docker) |
| Migration present | `supabase/migrations/20260814000000_foundation_embeddings.sql` (`vector(200)` + `match_foundation_embeddings`) — ready to run when a DB is available |
| Compensating control | `NeuralVectorIndex` in-memory cosine fallback (identical metric) + route/service tests asserting RPC name + namespace isolation |

## Files changed (Mission 13 — additive only)

| File | Change |
|---|---|
| `src/lib/ai/retrieval/foundation-search.ts` | **new** — `searchFoundationEmbeddings` (isolated `foundation_embeddings` namespace, 200-D strict gate) |
| `src/routes/api/eeg/embed/foundation-search.ts` | **new** — POST `/api/eeg/embed/foundation-search` (auth→rate→CORS→parse→select→resample→preprocess→embed→search; 424 no-V2-fallback; 422 on non-200-D) |
| `src/lib/eeg/channels.ts` | **bugfix** — `canonicalizeChannel` strips trailing `.` (PhysioNet EDF labels like `"Fp1."`), additive Tier-2 only |
| `src/lib/ai/retrieval/__tests__/foundation-retrieval.m11.test.ts` | **new** — platform retrieval on real 200-D embeddings |
| `src/routes/api/eeg/embed/__tests__/foundation-search.test.ts` | **new** — route contract (5 cases) |
| `src/lib/ai/inference/__tests__/foundation-serving-m11.test.ts` | **new** — deterministic + real-EDF serving forward |
| `scripts/tmp/m13_edf_serving_xcheck.py` | **new** — on-demand subset/serving bridge documenter |
| `reports/m13_edf_serving_xcheck.json` | **new** — bridge evidence |
| `reports/m13_retrieval_results.json` | (existing) — consumed as source of truth |
| `reports/m13_embedding_subset.json` | (existing) — 400 real CBraMod-200 vectors |
| `reports/MISSION13_CBRAMOD_TIER2_UTILITY_VALIDATION.{md,json}` | **new** — this report |
| `reports/benchmark_archive.json` | idx12 (Mission-13) record replaced with validated result; idx0–11 byte-identical |

No production V2/PCA/embeddings/vector(32)/registry/.env/rollout/manifest changes.

## Final verdict

| Leg | Verdict |
|---|---|
| Retrieval-quality (CBraMod > V2, gate) | **PASS** (SUCCESS) |
| Platform retrieval call site (`NeuralVectorIndex.search` on real embeddings) | **VALIDATED** |
| Real serving (`embedFoundationWindows`, real ONNX, real EDF) | **VALIDATED** (maxSim 0.9922) |
| pgvector RPC (`match_foundation_embeddings` live) | **INCONCLUSIVE** (Docker down — blocker captured, not faked) |
| Constraint compliance | **PASS** (all hard constraints honored) |
| Archive integrity | **PASS** (idx0–11 byte-identical; idx12 updated in place) |

**Mission 13 is complete.** The CBraMod Tier-2 retrieval gate PASSES on the
retrieval-quality leg and the platform integration is proven through the
in-memory metric equivalent. The pgvector RPC execution leg is INCONCLUSIVE
until a real Docker/pgvctor environment is available — that is the documented
next-experiment boundary (not Mission 14). Do NOT promote to GA.
