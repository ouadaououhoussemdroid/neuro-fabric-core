# Model Role & Product Evaluation — Experiment Plan

**Mission 8 — read-only analysis.** No V2 change, no `DEFAULT_PREFERRED`/`.env`/rollout change, no model deployment, no retraining, no artifact modification (`reports/MODEL_ROLE_EXPERIMENT_PLAN.md` is the only file created). This plan is grounded in `reports/MODEL_ROLE_PRODUCT_ARCHITECTURE.md` (Mission 7) + the Mission 6 archive record `cbramod-remap-50subj` (in `reports/benchmark_archive.json`, 16 experiments), `reports/MODEL_STRATEGY_OTHER_MODELS.md`, and the live registry/routing code.

**Governing principle:** a model that cannot run in the browser may still have a valuable **server-side** role — but only if it provides something the 32-D browser path (EEGConformer V2 + PCA) cannot. The experiments below test *product roles*, not "is model X > V2 at MI Recall@K."

---

## 1. Product capability map

| # | Product capability | Implemented? | Production-consumed? | Latency req | Ch/sr/dim | Browser / server | Batch / real-time | Realistic candidate model |
|---|---|---|---|---|---|---|---|---|
| 1 | Browser / real-time EEG embeddings | ✅ | ✅ (`/api/eeg/upload` → `embedEEG`) | P50<400 / P95<600 ms (FFox gate) | 22·250·1000 → 32-D | browser + SSR | real-time | **EEGConformer v2** (only one) |
| 2 | Wearable / edge inference | ⚠️ transport only | ❌ | ms | device-native | browser (WASM) | real-time | none viable (V2 is the edge ceiling; EEGPT 25 MB too big) |
| 3 | Server-side high-quality embeddings | planned | ❌ | minutes acceptable | native-d | server | batch | **CBraMod 200-D**, EEGPT 2048-D (gated) |
| 4 | Batch/offline representation generation | planned | ❌ | hours | native-d | server | batch | CBraMod, EEGPT, LaBraM, FEMBA (see §5 gates) |
| 5 | Similarity / retrieval (interactive) | ✅ | ✅ (pgvector `vector(32)` + Recall@10 cron) | <600 ms P95 | 32-D | server | real-time | **V2/PCA 32-D** only (ANN contract) |
| 6 | Cognitive-state classification | ✅ | ✅ (inlined in `/api/eeg/upload`) | sub-10 ms | 5 band-power features | both | real-time | `cognitive-decoder-v0` (band-power logistic) |
| 7 | Sleep analysis | ⚠️ loader only | ❌ | batch | 7–8 ch (sleep montage) | server | batch | **none** — no sleep-staging model; only `sleep-edf` loader |
| 8 | EEG → image / reconstruction | ❌ | ❌ | batch | — | server | batch | **none** — `/eeg2image.tsx` is a static demo |
| 9 | Synthetic neurodata | ✅ | ⚠️ (synthetic front-end only) | — | configurable | both | batch/realtime | `synthetic.ts` (not a task model) |
| 10 | Long-context EEG analysis | ❌ | ❌ | — | — | — | — | **none** |
| 11 | Research / foundation representation | ⚠️ (foundations registered, unrouted) | ❌ | — | native-d | server | batch | CBraMod/EEGPT/LaBraM/FEMBA (pending §5) |
| 12 | Future specialist APIs | — | — | — | — | — | — | (see §4) |

---

## 2. Model capability matrix

| Model | Artifact / sha | In `[1,C,T]` | ch·sr·T | Out dim | Output type | WASM? | CPU lat | Size | Evidence | Prod usage | Best plausible role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EEGConformer v2** `braindecode-eegconformer-prod-v2` | `eegconformer_finetuned.onnx` / `18644de1…` | `[1,22,1000]` | 22·250·1000 | 32 | embedding (+4 logits, unused) | ✅ self-contained | ~8 ms / 131 ms warm-browser | 3.2 MB | 0.325 @50 (T-032/033) | GA default, `embedEEG` gate | Browser/real-time embedding · Retrieval |
| **PCA** `pca-legacy-v1` | (code) | bandpower 110 | 22·250·1000 | 32 | embedding | js | ~0 ms | ~0 | 0.313–0.320 @50 | Terminal fallback | Fallback / instant retrieval |
| EEGConformer v1 `braindecode-eegconformer-prod` | `eegconformer.onnx`+`onnx.data` / `31cd3651…`(+ext `892b5a77…`) | `[1,22,1000]` | 22·250·1000 | 32 | embedding | ⚠️ external-data | 6.9 ms | 3.0+3.2 MB | 0.251–0.283 @50 | Rollback only | Rollback artifact (not browser-safe) |
| **CBraMod** `onnx-cbramod` | `cbramod-encoder.onnx` / `c128ccfd…` | `[1,19,1000]` | 19·250·1000 | 19000 raw / **200** (T-030 mean-tokens) | embedding | ❌ DFT,ReduceL2 | 53.6 ms | 22.0 MB | 0.3043 @50 (M6); 0.3233 @10 (T-030) | Not routed | Server high-cap representation |
| **EEGPT** `onnx-eegpt` | `eegpt-encoder-int8.onnx` / `a92daf44…` | `[1,62,1000]` | 62·250·1000 | 2048 | embedding (ViT mean-tokens) | ✅ INT8 | 0.83–4.8 s | 24.9 MB | 0.3067 @10 | Experimental (unrouted) | Server 2048-D backbone (if 62→22 viable) |
| LaBraM `onnx-labram` | `labram-encoder.onnx` / `61f28d12…` | `[1,16,1600]` | 16·250·1600 | 200 | embedding (ViT) | ✅ | 68 ms | 22.2 MB | 0.2533 @10 **< PCA** | Experimental (unrouted) | Drop (below-PCA + 16-ch mismatch) |
| FEMBA-tiny `onnx-femba-tiny` | `femba-tiny-encoder-adapter.onnx` / `e0242279…` | `[1,22,1280]` | 22·200·1280 | 30800 | embedding (Mamba) | ✅ | 220 ms | 32.2 MB | 0.2400 @10 **< PCA** | Experimental (unrouted) | Drop (below-PCA + 30K-D projection liability) |
| EEGNetv4 `braindecode-eegnetv4-default` | (none) | `[1,22,256]` | 22·128·256 | 16 | embedding | n/a | — | — | none | Not loadable | Drop (stub: no artifact, no bridge) |
| Shallow/Deep4 `braindecode-*-default` | (none) | `[1,22,1125]` | 22·250·1125 | 40/200 | embedding | n/a | — | — | none | Not loadable | Drop (stub) |
| PyTorch export `pytorch-export-placeholder` | (none) | — | — | — | — | — | — | — | none | Not loadable (`implemented:false`) | Drop (stub) |
| `cognitive-decoder-v0` | `cognitive-decoder-v0.onnx` / `ea4f216c…` | `[1,5]` bandpower | — | 3 | classification (att/work/ars) | ✅ | <1 ms | 1.3 KB | T-025 | ✅ prod (with heuristic fallback) | Cognitive classification (decoupled) |

**Stubs vs real (the spec demands this distinction):**
- **Real, runnable artifacts:** v2, PCA, CBraMod, EEGPT, LaBraM, FEMBA, cognitive decoder (and v1 as a rollback artifact). All have on-disk `.onnx` matching a `manifest.json` SHA-256 and a registered adapter that `load()`s.
- **Stubs/placeholders:** `pytorch-export-placeholder` (`implemented:false`), `braindecode-eegnetv4/shallowfbcspnet/deep4net-default` (no ONNX artifact; `BraindecodeAdapter.load()` throws "runtime not available" because no Pyodide+torch bridge is registered — see `braindecode-adapter.ts:186–193`).

---

## 3. Browser vs server architecture

**Browser / interactive path (locked):** `embedEEG()` at GA → `InferenceEngine` (persistent, LRU 2, SHA-verified, mutex) → V2 32-D → L2-normalized → `NeuralVectorIndex.add()` (pgvector `vector(32)` + CHECK) + `eeg_analyses`. PCA is the JS fallback (`runtime:"js"` → runs everywhere). COOP/COEP/CORP (`server.ts`) for `SharedArrayBuffer`; ORT WASM self-hosted `/ort/`; `numThreads` pinned to 1 (threading negative). This path is correct and **must not change**.

**Server / batch path (the one with headroom):** Nitro SSR reuses the **same** `embedEEG()` singleton for `/api/eeg/upload`; the same `ONNXAdapter`/registry can produce **native-dim** embeddings via `embed({modelId:"onnx-cbramod"})` (CPU EP, no WASM needed for non-wasm models). The DB already supports native-dim persistence without schema change: `eeg_analyses.embedding FLOAT8[]` has **no dimension CHECK** (only the pgvector `embeddings` table is locked to `vector(32)`). So a server foundation embedder can persist its native 200/2048/30800-D vectors to `eeg_analyses` today; only the **ANN index** needs a model-namespaced, dimension-specific companion (see §4).

**API boundary:** `embedEEG()` (32-D, browser↔SSR, GA-gated, latency-bound) vs a proposed server-only `embedFoundation({modelId})` for native-D offline use, reachable from `/api/eeg/upload` options or a new `/api/eeg/embed/foundation?model=…` route. The boundary is clean because server ingress already exists and the gateway already stamps `model_id`.

---

## 4. Model-to-role mapping (KEEP / SERVER SPECIALIST / BROWSER / RESEARCH / DROP / BLOCKED)

| Model | Classification | Rationale |
|---|---|---|
| **EEGConformer v2** | **KEEP** (browser interactive · 32-D retrieval) | Only wasm-compatible model that fits `vector(32)` + latency gate; GA default. Do not touch. |
| PCA | **KEEP** (fallback) | Always-on JS floor; 32-D aligned; zero-dependency fallback. |
| EEGConformer v1 | **DROP** (as a product model) | External-data artifact is not browser-WASM-safe; kept only as a rollback file. |
| **CBraMod** | **SERVER SPECIALIST** (pending P0 validation) | Server-only (DFT/ReduceL2); 200-D native output has *representational headroom* V2's 32-D lacks; 19-ch channel set is the study, not a blocker, for a representation role. |
| EEGPT | **BLOCKED → SERVER SPECIALIST (conditional)** | 62-ch contract vs 22-ch pipeline must be de-risked first (62→22 remap feasibility, P1). If it survives → 2048-D server backbone. If not → DROP. |
| LaBraM | **DROP** | 22 MB, 10-subj acc **below PCA** (0.2533 vs 0.313+), 16-ch mismatch. No headroom argument survives. |
| FEMBA-tiny | **DROP** | Below-PCA (0.2400), 30,800-D (a 940× projection to 32 annihilates it), 22-ch match is its only advantage and not enough. |
| EEGNet/Shallow/Deep4 | **DROP** | Stubs: no artifact, no bridge, `load()` throws. |
| PyTorch export | **DROP** | `implemented:false` stub. |
| cognitive decoder | **KEEP** | Prod cognitive classification; decoupled by design. |

---

## 5. Why Mission 6 was insufficient to judge CBraMod's overall product value

Mission 6 (`cbramod-remap-50subj`, archived as experiment #16 in `benchmark_archive.json`) ran CBraMod as a **19→22 native-montage MI retrieval** study: CBraMod@19-ch native (mean-tokens pooled to 200-D) vs V2@22-ch vs PCA, on the **same 50-subject LOSO** protocol, scored by **accuracy / Recall@1/5/10 / Fisher** with Bonferroni correction. Result: CBraMod acc 0.3043 < V2 0.3250 and tied PCA 0.3065; p=0.353 (vs V2), 1.0 (vs PCA) → negative on that task.

That result is **valid for exactly one question**: *"Does CBraMod-200 beat V2-32 (or PCA) when both are crammed into a 32-D-style MI Recall@K retrieval task?"* It is **insufficient** to judge CBraMod's total product value because:

1. **Wrong product role.** CBraMod is server-only (`wasmCompatible:false`, DFT+ReduceL2) with a 200-D native output. Cramming its 200-D representation into a 32-D cosine-Recall@K MI task **censors its representational richness** — it is evaluated as a *32-D interactive replacer*, which it can never be (not wasm-compatible). The fair question is *"what can CBraMod provide as a native 200-D server embedding that V2's 32-D cannot?"* — a within/between-class separation or subject-embedding task in 200-D, not a 32-D MI task.
2. **Wrong metric.** MI Recall@K nearest-centroid rewards a model whose 4-class MI head is optimized (V2's logits head + 32-D is literally trained for MI). CBraMod is a *representation* model — its value is separation geometry, which Recall@1@32-D does not surface.
3. **Dimensionality mismatch.** 200-D cosine Recall@K and 32-D cosine Recall@K are not equivalent tasks; Mission 6 mean-token-pooled CBraMod to 200-D but still ran it in a 32-D-shaped evaluation, so the dimensionality advantage was never tested natively.
4. **No native-dim store existed.** To judge CBraMod as a server representation specialist you must measure it in its *native* 200-D space (Fisher, silhouette, subject-level retrieval Recall@K at 200-D) against a V2 that is also free to use its 32-D — i.e., test CBraMod's *extra degrees of freedom*, not a head-to-head on V2's home turf.

**Therefore:** Mission 6 proves CBraMod is **not a browser replacer for V2** (correct) but **does not prove it has no server-side representation role**. §6 of the Mission 7 audit reaches the same conclusion; this plan's P0 (§6.1) is the decisive experiment for that separate question.

---

## 6. Candidate server-side roles

1. **CBraMod 200-D server representation / offline similarity** — Persist native 200-D to `eeg_analyses.embedding` (FLOAT8[], no schema change) + a model-namespaced `vector(200)` ANN companion. Use case: cohort phenotyping, offline clustering, retrospective similarity search where 32-D V2 is known to collapse (intra≈inter cos 0.907/0.904). CBraMod's raw-Fourier+attention features may separate classes *that V2's 32-D head collapses* — that is the capability V2 cannot provide.
2. **EEGPT 2048-D backbone** — Same native-store pattern at `vector(2048)`. Use case: a high-capacity similarity/retrieval service or a transfer encoder for downstream fine-tuning. **Blocked** unless the 62→22 channel projection is validated (ViT channel-patch attention). If the projection discards signal, this role evaporates and there is no 62-ch product to serve it.
3. **LaBraM 200-D** — Same pattern, but 16-ch → 22-ch remaps to *fewer* channels (signal gain, not loss); however acc is already below-PCA at n=10, so the role has no evidence basis. **DROP** unless a 16-ch-native product is defined (none exists).
4. **FEMBA 30800-D** — 22-ch matches the pipeline, but 30,800-D is a storage/routing liability and a 940× projection to 32 is destructive; below-PCA at n=10. A 30K-dim server store is only justified if a concrete external-analysis export use case carries the budget. **DROP** unless scoped.
5. **Sleep specialist** — The `sleep-edf` loader exists and labels `task:"sleep-staging"`, but there is **no sleep-staging model**. Any model could attempt it, but none is trained for it; V2 is MI-only. This is a *missing-model* opportunity, not a role for an existing model. **BLOCKED** on a trained sleep model.

---

## 7. Candidate experiments

### P0 — CBraMod native 200-D server representation validation *(highest product value)*
1. **Product question:** Can CBraMod's native 200-D embedding provide richer offline similarity / class separation than V2's 32-D — i.e., does it earn a server-side specialist role V2 cannot fill?
2. **Why necessary:** Mission 6 tested the wrong role (32-D MI Recall@K). CBraMod's 200-D representational headroom was never exercised natively.
3. **Metric (success):** Primary — within-class/between-class separation (Fisher ratio *and* silhouette) + subject-level retrieval **Recall@K in native 200-D**, where CBraMod 200-D ≥ V2 32-D by a meaningful, Bonferroni-corrected margin (Δ ≥ 0.05, p<0.05). Guardrail — MI accuracy of a 200-D nearest-centroid classifier ≥ V2-32-D (CBraMod must not be *worse* on V2's home task).
4. **Data/protocol:** PhysioNet EEGMMIDB S001–S050 LOSO (already available; same as Mission 6), 4-class MI (runs 5/6), identical preprocessing (1000-sample @ 250 Hz, bandpower 4–38 Hz, z-score), CBraMod 200-D via mean-tokens pooling (T-030 convention), V2 32-D, train-only candidate pools, Bonferroni across the separation metrics. Persist CBraMod 200-D to a **model-namespaced** store: `eeg_analyses.embedding` (FLOAT8[], no schema change) + a new `vector(200)` ANN companion table (separate from the `vector(32)` interactive index — do **not** touch `vector(32)`/DEFAULT_PREFERRED/.env).
5. **Success gate:** CBraMod 200-D separation/Recall@K ≥ V2-32-D (Δ≥0.05, p<0.05 Bonferroni) **and** guardrail met → implement `/api/eeg/embed/foundation?model=cbramod` writing `vector(200)`. **Failure gate:** below threshold → CBraMod has no proven server role; **DROP** from routing.
6. **Production decision:** server-specialist route + native store, or retire CBraMod.
7. **Browser/server:** server only. **Compute:** M (offline; 53.6 ms × 50 subj ≈ minutes).
8. **Prod-safety:** zero change to V2/GA/registry/`.env`/`DEFAULT_PREFERRED`. Adds a parallel native store + one new server route — implementation deferred until P0 proves value.

### P1 — EEGPT 62→22 remap viability *(gate for any EEGPT server role)*
1. **Question:** does EEGPT's 2048-D ViT representation survive dropping 40 of 62 channels?
2. **Why:** EEGPT's only plausible server role requires its 2048-D capacity; but its 62-ch contract vs the 22-ch pipeline means either (a) the product must become 62-ch (none exists) or (b) 62→22 projection. If (b) destroys the signal, EEGPT has no role.
3. **Metric:** cos-sim(62-ch, 22-ch projection) ≥ 0.90 (representation retained) **and** 22-ch MI acc ≥ V2-32-D.
4. **Data:** PhysioNet EEGMMIDB (62-ch subset where present) + BCI-IV-2a (standard 22-ch) under identical LOSO.
5. **Success:** both thresholds → proceed to a P1b EEGPT 2048-D server-representation experiment (same native-store pattern as P0). **Failure:** EEGPT → DROP.
6. **Browser/server:** server. **Compute:** M.

### P1b — INT8-QDQ V2 Firefox latency ablation *(the actual GA gate, not a role)*
1. **Question:** does INT8-QDQ V2 clear Firefox P50<400 / P95<600 ms while preserving parity?
2. **Why:** GA is not honestly green on Firefox; the 1.1 MB INT8 artifact already exists in `public/models/_bench/` (`reports/int8_v2_quantization.json`: embed cos 0.9985, logits 0.9993, all 19 ops WASM-compatible). This is a latency fix, not a new model.
3. **Metric:** Firefox gate (P50<400 / P95<600) + cos-sim ≥ 0.99 vs FP32 + fallback rate < 0.5%.
4. **Protocol:** swap the `_bench` artifact into the wasm-smoke + `v2-firefox-latency-gate.test.ts` harness via the existing `runtimeProvider` override (no production file swap).
5. **Success:** gate cleared → promote INT8 artifact to `eegconformer_finetuned.onnx` (manifest SHA update only) as the v2 GA artifact. **Failure:** worker-spin-up is the ceiling, not arithmetic → stop.
6. **Browser/server:** browser (Chromium+Firefox). **Compute:** 1–2 days.

### DROP candidates (no experiment)
- **LaBraM, FEMBA:** below-PCA at n=10 + channel/dim mismatches — a representation experiment would have to beat PCA to be interesting; current evidence says it does not.
- **EEGNet/Shallow/Deep4, PyTorch export:** stubs — no artifact to benchmark.
- **Contrastive FT of V2 (T-034):** already ran (`t034_contrastive` 0.274, `t034_aug_contrastive` 0.259) and **lost to V2 0.325** — re-running is duplicate work.
- **EEG2Image reconstruction:** the route is a static demo; implementing a reconstructor is engineering, and the 4-class MI logits head that could feed it is unused — defer until a reconstruction *product* is defined.

---

## 8. P0 / P1 / P2 prioritization

| Priority | Experiment | Can it prove a server-side capability V2 cannot provide? |
|---|---|---|
| **P0** | CBraMod native 200-D server representation (§7.1) | ✅ Yes — richer separation geometry for offline/cohort retrieval where V2's 32-D collapses; server-only, so it opens a capability the browser path *cannot* offer. |
| **P1** | EEGPT 62→22 remap viability (§7.2) | ✅ Yes, *if* it survives — 2048-D backbone is headroom V2 lacks. |
| **P1** | INT8-QDQ V2 Firefox latency (§7.3) | ⚠️ No (not a role) — but it is the **production GA gate**. Run in parallel; it unblocks honest GA regardless of role outcomes. |
| **P2** | Cross-domain generalization of V2 | ⚠️ Indirect — proves whether V2's accuracy ceiling is data- or head-limited (informs the whole hierarchy). |
| **DROP** | LaBraM / FEMBA / stubs / contrastive re-run | ❌ No plausible distinct capability vs V2/PCA. |

Priority rule (per spec): experiments are ranked by their ability to show a model providing a **server-side capability V2 cannot provide**. CBraMod P0 is strictly this; EEGPT P1 is conditional on it; the INT8 gate is a production blocker rather than a role, so it is P1-by-necessity.

---

## 9. Explicit success / failure gates

- **CBraMod P0 success:** 200-D Fisher/silhouette **or** subject-Recall@K ≥ V2-32-D (Δ ≥ 0.05, Bonferroni p < 0.05) **and** 200-D MI nearest-centroid acc ≥ V2-32-D. → Build `/api/eeg/embed/foundation?model=cbramod` + `vector(200)` store.
- **CBraMod P0 failure:** below thresholds. → CBraMod has no proven server role; **DROP** from routing; preserve Mission 6 negative.
- **EEGPT P1 success:** cos(62→22) ≥ 0.90 **and** 22-ch acc ≥ V2. → Proceed to EEGPT 2048-D server-rep experiment.
- **EEGPT P1 failure:** → DROP EEGPT (no 62-ch product).
- **INT8 P1b success:** Firefox P50<400 / P95<600 **and** cos ≥ 0.99 vs FP32 **and** fallback < 0.5%. → Promote INT8 artifact to `eegconformer_finetuned.onnx` (+manifest SHA).
- **INT8 P1b failure:** → worker lifecycle is the ceiling; stop; document.

All gates are run with V2/GA/registry/`.env`/`DEFAULT_PREFERRED` **unchanged** (read-only, harness-only artifact swaps).

---

## 10. Final recommendation

- **V2 stays** as the browser/interactive 32-D embedding + GA default. No changes.
- **PCA stays** as the terminal 32-D fallback.
- **CBraMod is the only model with a plausible distinct server role** (native 200-D representation), but its value is **unproven** because Mission 6 tested the wrong role. Run **P0** (native 200-D representation experiment) to decide. If it wins → a server `/api/eeg/embed/foundation` route + a `vector(200)` companion store (the `vector(32)` interactive index is untouched). If it loses → DROP CBraMod.
- **EEGPT is BLOCKED** on the 62→22 remap (P1). **LaBraM/FEMBA/v1-stubs = DROP** (below-PCA + mismatches).
- The architecture the experiments assume is the **separate-embedding-space** design from §3 of Mission 7: browser path stays 32-D; server foundation models persist **native-dim** to `eeg_analyses.embedding` (already dim-agnostic) and use a **model-namespaced ANN companion** (`vector(200)`/`vector(2048)`) — never a projection into `vector(32)`.
- **Do not run any experiment until the role is decided** (spec: "determine what the model would actually be used for before benchmarking").

---

## 11. Exact next mission proposal

**Mission 9 — Implement & run the P0 CBraMod server-representation experiment** (the P1b INT8 latency gate can run in parallel or first, since it unblocks GA). Concretely, after explicit approval:

1. **Read-only first:** add a `vector(200)` model-namespaced ANN companion (new table, e.g. `foundation_embeddings(model_id, embedding vector(200), …)`, partitioned by model) — no change to `embeddings` (`vector(32)`) or `eeg_analyses` (already FLOAT8[].
2. **Server route:** a thin `/api/eeg/embed/foundation?model=cbramod` (server-only, CPU EP; CBraMod is non-WASM so it can only live here anyway) that calls the registered `onnx-cbramod` adapter, mean-tokens-pools to 200-D, SHA-verifies, and writes to the `vector(200)` store + `eeg_analyses`.
3. **Run P0** (physionet S001–S050 LOSO, native 200-D separation + subject-Recall@K vs V2-32-D, Bonferroni).
4. **Gate outcome:** success → keep/expand the server foundation path (generalize to EEGPT 2048-D); failure → drop CBraMod and close the server-foundation thread.

Until that approval, **do not** add the store, the route, or run the benchmark. This document is the read-only plan.

*End of Mission 8. No production code, artifact, `.env`, `DEFAULT_PREFERRED`, rollout stage, registry entry, or embedding contract was modified.*
