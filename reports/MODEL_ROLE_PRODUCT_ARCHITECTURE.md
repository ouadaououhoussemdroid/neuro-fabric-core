# Neuro Fabric — Model Role & Product Capability Architecture

**Mission 7 — Model Role & Product Capability Mapping.**
Read-only investigation. Per the mission constraints: no production source edits, no change to `.env`, no change to `DEFAULT_PREFERRED`, no rollout-stage change, no model deployment, no retraining, no artifact modification, no expensive 50-subject experiments. The only file created by this mission is this one.

**Codebase state:** live working tree against `HEAD = b9164a6` (many Mission–5 source files are modified in the tree — this audit reflects the **live code**, not the archived docs). All SHAs/sizes are from `public/models/manifest.json` + `ls public/models` (verified on disk).

> The governing strategic rule, taken literally: **a model that cannot run in the browser may still be extremely valuable on the server — for a *different* product capability.** The question is never "is model X better than V2?" It is **"what product role, if any, does this model uniquely enable?"**

---

## STEP 1 — Actual product capabilities

Status legend: **IMPLEMENTED** (real code path, real model, real storage contract), **PARTIAL** (transport/data only, or scaffold, or model present but not wired), **ABSENT** (no route, model, or storage).

### 1.1 EEG embeddings / representation learning — IMPLEMENTED

- **Route:** `POST /api/eeg/upload` (`src/routes/api/eeg/upload.ts:280–283`).
- **What `embedEEG()` (production gate) is responsible for** (`src/lib/ai/inference/embed-eeg.ts:38–116`): resolve `DEFAULT_PREFERRED = "braindecode-eegconformer-prod-v2"`; apply the GA cohort (`isEEGConformerEnabledForUser(userId)`); route the chosen non-PCA model through the **process-wide `InferenceEngine`** (LRU `maxLoaded=2`, per-model async mutex, first-load dedup); SHA-256 verify at first load only (`adapter.load()` → `verifyRemoteArtifact`); on any failure `disposeModel()` + per-call `embed()` facade → `fallbackChain` → PCA. Returns `{ vector: number[], dim, modelId, fellBack, normalized }`.
- **Storage contract:** `embeddings.embedding vector(32)` + `CHECK (vector_dims(embedding)=32)` (migration `20260808010000_embedding_dimension_contract.sql`), enforced client-side by `NeuralVectorIndex.add()` (`DimensionMismatchError`) and by `validateEmbedding` (rejects NaN/Inf/zero/dim-mismatch) before L2-normalization in `finalize()`.
- **Runtime:** browser (`onnxruntime-web` WASM, self-hosted `/ort/`, COOP/COEP/CORP for `SharedArrayBuffer`) **and** Nitro SSR (same singleton, process-wide LRU, disposed on SSR lifecycle). v2 persistent-session cut Chromium tail ~1589→131 ms P95; threading is net-negative for the 3.2 MB model.
- **Latency gate:** `tests/browser/v2-firefox-latency-gate.test.ts:111–159` — P50<400 ms / P95<600 ms on Firefox (Chromium passes; Firefox is the open GA evidence item).
- **Input/output:** `[1,22,1000]@250Hz` → `embedding[1,32]` (+ a `logits[1,4]` output that is **never consumed**, see §6.3).
- **Browser/server:** both.

### 1.2 Cognitive-state decoding — IMPLEMENTED (decoupled from embeddings)

- **Entry:** `decodeCognitiveState(signal)` (`src/lib/decoder/index.ts:47–83`) → `trained-decoder.ts`.
- **Model:** `cognitive-decoder-v0.onnx` (1,333 bytes; manifest sha `ea4f216c…`) — a **5-band-power-feature logistic regression** (`trained-decoder.ts:38–54, 137–151`: input `[1,5]` of δ/θ/α/β/γ, output `[attention, workload, arousal]` + 0.08-margins CIs), with a spectral-ratio heuristic fallback (`baseline-spectral-v1`).
- **Critically:** it consumes `bandPowerFeatures` directly from `src/lib/embeddings/features.ts` (FFT-based, T-007) — **NOT** the EEGConformer 32-D embedding. The two pipelines share a preprocessing back-end but are architecturally independent.
- **Storage:** persisted into `eeg_analyses.{attention,workload,arousal,bandpass_low,…}` via the upload route (FLOAT-ish columns). No separate API route (inlined in `/api/eeg/upload`).
- **Latency:** sub-10 ms (5-feature logistic), heuristic ~0 ms.
- **Browser/server:** both (ONNX/WASM + heuristic).

### 1.3 Real-time / streaming inference — PARTIAL

- **Transport exists:** WebSocket gateway `GET /api/public/stream/:source` (`src/routes/api/public/stream/-$source.ts`, `StreamGateway` in `src/lib/eeg/stream-gateway.ts`). Fans out `EEGStreamFrame { seq, model_id, source, channels, sampleRate, data }` per chunk; per-peer sequence counters; `model_id` header stamped on every frame.
- **Acquisition sources:** operator-registered (`file:…`, `brainflow:…`, `lsl:…`) via `brainflow-adapter.ts` (BrainFlow + LSL); `src/lib/eeg/loaders/` enumerates datasets.
- **Gap:** the gateway **only transports** raw `data: number[][]` chunks — it does **not** call `embedEEG`/`InferenceEngine` server-side per frame. The default `model_id` is a **stale literal** `"eegconformer-v1"` (line 47/21) that is **not** a valid registry id (`braindecode-eegconformer-prod`); at GA the real id is `braindecode-eegconformer-prod-v2`. No continuous real-time embedding API is wired. → *PARTIAL: fan-out transport implemented; per-frame inference not implemented; model_id default is stale.*
- **Browser/server:** server (Nitro WebSocket).

### 1.4 EEG similarity / retrieval — IMPLEMENTED

- **Store:** Supabase pgvector `embeddings` table, ivfflat index, RPCs `match_embeddings` (ANN) + `match_embeddings_exact` (brute-force). `NeuralVectorIndex` (`neural-index.ts`) writes via `add()` (best-effort, reports `vector_indexed=false` on failure) and searches via `search()/nearest()`. In-memory `VectorIndex` brute-force fallback when no Supabase client.
- **Explorer:** `GET /embeddings` (`src/routes/embeddings.tsx`) reads ANN via `match_embeddings`.
- **SLO:** nightly cron `/api/public/cron/recall` (`recall-slo.ts`) — Recall@10 vs brute-force ground truth; FAIL if `ann < 0.85` or `ann / bruteForce < 0.95` (`recall-slo.ts:54–58`).
- **Browser/server:** server (ANN RPC + cron); the in-memory fallback can run browser-side.

### 1.5 EEG2Image / reconstruction — ABSENT

- `/eeg2image.tsx` is a **static frontend concept demo** ("cycles through fixed sample outputs… not connected to a live image-generation model"). No backend route, no model artifact, no decoder. `src/components/recon-showcase.tsx` is the associated scaffold. → ABSENT.

### 1.6 Sleep analysis — PARTIAL

- `src/lib/eeg/loaders/sleep-edf.ts` — a **data loader** (`sleepEDF`, PhysioNet Sleep-EDF SC 1.0.0, `task: "sleep-staging"`, hypnogram labels) + `sleep-edf.test.ts`. **No** sleep-staging model, **no** route, **no** decoder. → PARTIAL (data loader only).

### 1.7 Synthetic EEG / neurodata generation — IMPLEMENTED

- `src/lib/synthetic/index.ts` — 1/f pink noise (Voss-McCartney, 16 octaves) + δ/θ/α/β/γ band-bumps, seeded (mulberry32), eyes-closed-resting defaults. Output is a normal `EEGSignal` that feeds the **same** preprocess → embed → decode pipeline. `/synthetic` frontend. → IMPLEMENTED (generation only; not a task model).

### 1.8 Long-context / sequence analysis — ABSENT

- Preprocessing slices fixed 4 s windows (`upload.ts:263`, `preprocess/segment`). No recurrent/sequence/Transformer-LM model, no context-window contract, no route. The only "recurrent" references are FEMBA's INT8 `recurrent scan loop` *quantization* issue (model-internal), not a product. → ABSENT.

### 1.9 Foundation-model representation — PARTIAL

- v2 EEGConformer is wired into production (`embedEEG` GA gate). EEGPT (`onnx-eegpt`), LaBraM (`onnx-labram`), FEMBA (`onnx-femba-tiny`), CBraMod (`onnx-cbramod`), and v1 Conformer (`braindecode-eegconformer-prod`) are **registered + on-disk + real ONNX** but **not routed** — reachable only as manual `embed({modelId})` targets. EEGNetv4/ShallowFBCSPNet/Deep4Net are registered but have no Pyodide bridge → `load()` fails. → PARTIAL.

### 1.10 Acquisition / data surface (supporting)

- Loaders: `src/lib/eeg/loaders/` — `physionet` (eegmmidb, 109×14 EDF), `bci-competition` (IV 2a), `sleep-edf`, `tuh` (scaffold). `src/lib/eeg/parsers/` — EDF/BDF, CSV, NPY. `src/lib/eeg/preprocessing/` — biquad zero-phase bandpass + notch + z-score + segment.

---

## STEP 2 — Audit every available model

Attributes: artifact / sha (on disk) / input / ch·sr / window / output / trained-on / wasm / server / CPU latency / size / registry status / prod status / consumed in `src/` / evidence / limitations / adapter? / satisfies `vector(32)` directly?

| # | Model | Artifact / sha | Input | Ch·sr·T | Out dim | Trained on | wasm? | CPU lat | Size | Registry | Prod? | Consumed in src/? | Evidence | Limitations | Adapter? | Fits vector(32)? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **PCA** `pca-legacy-v1` | (code, no file) | bandpower 110 | 22·250·1000 | 32 | unsup (band-power) | js | ~0 ms | ~0 | registered | ✅ fallback | yes (terminal) | 0.313–0.320 @50 (T-032) | weak (linear), not neural | PCAEmbeddingAdapter | ✅ yes |
| 2 | **EEGConformer v2** `braindecode-eegconformer-prod-v2` | `eegconformer_finetuned.onnx` / `18644de1…` | `[1,22,1000]` | 22·250·1000 | **32** | PhysioNet EEGMMIDB (FT) | ✅ self-contained | ~8 ms warm / 131 ms warm-browser | 3.2 MB | registered + GA gate | ✅ GA default | ✅ embedEEG DEFAULT_PREFERRED | 0.325 @50 (T-032/033); parity cos 1.0 | Firefox tail latency; intra≈inter (collapse) | ONNXBraindecodeBridge | ✅ yes |
| 3 | EEGConformer **v1** `braindecode-eegconformer-prod` | `eegconformer.onnx`+`onnx.data` / `31cd3651…`(+ext `892b5a77…`) | `[1,22,1000]` | 22·250·1000 | 32 | BCI-IV-2a | ⚠️ external-data | 6.9 ms | 3.0+3.2 MB | registered (rollback) | rollback only | no | 0.251–0.283 @50; 0.317 @10 | external-data unsafe in WASM | ONNXBraindecodeBridge | ✅ 32-D but not routed |
| 4 | **CBraMod** `onnx-cbramod` | `cbramod-encoder.onnx` / `c128ccfd…` | `[1,19,1000]` | 19·250·1000 | 19000 (raw) / 200 (T-030 mean-tokens) | — (MIT pretrain) | ❌ DFT,ReduceL2 | 53.6 ms | 22.0 MB | registered | no (server) | no (manual embed) | 0.3043 @50 (Mission 6); 0.3233 @10 (T-030) | 19-ch≠22-ch; not-WASM; 19000-D native | ONNXAdapter | ❌ (19000/200≠32) |
| 5 | **EEGPT** `onnx-eegpt` | `eegpt-encoder-int8.onnx` / `a92daf44…` | `[1,62,1000]` | 62·250·1000 | 2048 | braindecode/eegpt-pretrained (25k) | ✅ INT8 | 830 ms–4.8 s | 24.9 MB | registered (experimental) | no | no | 0.3067 @10 | 62→22 channel drop dubious; slow/large; 2048≠32 | ONNXAdapter (via EEGPTAdapter) | ❌ (2048≠32) |
| 6 | **LaBraM** `onnx-labram` | `labram-encoder.onnx` / `61f28d12…` | `[1,16,1600]` | 16·250·1600 | 200 | labram-pretrained (MIT) | ✅ | 68 ms | 22.2 MB | registered (experimental) | no | no | 0.2533 @10 **below PCA** | 16-ch≠22; below-PCA; 200≠32 | ONNXAdapter | ❌ (200≠32) |
| 7 | **FEMBA-tiny** `onnx-femba-tiny` | `femba-tiny-encoder-adapter.onnx` / `e0242279…` | `[1,22,1280]` | 22·200·1280 | 30800 | PulpBio/FEMBA (Apache-2.0) | ✅ | 220 ms | 32.2 MB | registered (experimental) | no | no | 0.2400 @10 **below PCA** | 30800-D is a 940× projection non-starter to 32 | ONNXAdapter | ❌ (30800≠32) |
| 8 | EEGNetv4 `braindecode-eegnetv4-default` | (none) | `[1,22,256]` | 22·128·256 | 16 | untrained zoo | n/a | — | — | registered | no | no | none | no artifact; no Pyodide bridge→load fails | BraindecodeAdapter (default bridge) | n/a (not loadable) |
| 9 | ShallowFBCSPNet `braindecode-shallowfbcspnet-default` | (none) | `[1,22,1125]` | 22·250·1125 | 40 | untrained zoo | n/a | — | — | registered | no | no | none | no artifact; bridge fails | BraindecodeAdapter | n/a |
| 10 | Deep4Net `braindecode-deep4net-default` | (none) | `[1,22,1125]` | 22·250·1125 | 200 | untrained zoo | n/a | — | — | registered | no | no | none | no artifact; bridge fails | BraindecodeAdapter | n/a |
| 11 | PyTorch export `pytorch-export-placeholder` | (none) | — | — | — | — | — | — | — | registered | no | no | none | `implemented:false`; load→NotImplementedError | PyTorchExportAdapter | n/a (stub) |
| 12 | Cognitive decoder `cognitive-decoder-v0` | `cognitive-decoder-v0.onnx` / `ea4f216c…` | `[1,5]` bandpower | — | 3 (att/work/ars) | `scripts/train_cognitive_decoder.py` | ✅ | <1 ms | 1.3 KB | NOT in model registry (separate decoder) | ✅ used (with heuristic fallback) | ✅ decodeCognitiveState | T-025 (calibrated) | tiny; 5-feature only; decoupled from embed | ONNXAdapter (onnxruntime-web) | N/A (not an embedding model) |

**Scratches/ablation NOT routed, not in registry:** `eegconformer_finetuned.onnx.bak`, `eegconformer_intermediate.onnx`, `eegconformer_finetuned_intermediate.onnx` (all sha `b3029ca2…`/`31cd3651…`, intermediates/checkpoints); `public/models/_bench/eegconformer_finetuned_int8.onnx` (1.1 MB INT8-QDQ, sha `59e9555a…`, embed cos 0.9985/logits 0.9993 vs FP32, `reports/int8_v2_quantization.json`).

**Consumption audit (the decisive question):** `embedEEG` (production gate) routes **only** to `braindecode-eegconformer-prod-v2` at GA, falling back to `pca-legacy-v1`. No other foundation model is wired into any route. The cognitive decoder is the only other model invoked in production (and it is band-power, not embedding, based). The 4-class MI **logits** head on the v2 artifact is generated by `applyOutputPooling`/ONNX but **no `src/` file reads `logits`** (grep confirms `logits` appears in `src/` only at `registry.ts:258,272`, optional plumbing). So the classification head is **dead output** — produced but unconsumed.

---

## STEP 3 — Model → role matrix

Possible roles: **A** Browser real-time embedding · **B** Server high-capacity representation · **C** Cognitive classification · **D** Sleep specialist · **E** EEG2Image reconstruction · **F** Synthetic generation · **G** Long-context analysis · **H** Retrieval embedding · **I** Fallback/safety. A model may have multiple roles or none.

| Model | A browser real-time | B server high-cap | C cognitive classification | D sleep | E recon | F synthetic | G long-context | H retrieval (32-D ANN) | I fallback | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| **EEGConformer v2** | ✅ **primary** | — | (logits head unused) | — | — | — | — | ✅ primary (32-D vector(32)) | — | Keep; the only browser+ANN model |
| **PCA** | ✅ instant | — | — | — | — | — | — | ✅ (fallback ANN) | ✅ terminal | Keep as safety floor |
| EEGConformer v1 | ⚠️ ext-data (unsafe) | — | — | — | — | — | — | ✅ 32-D (rollback) | — | Rollback artifact only |
| **CBraMod** | ❌ (DFT/ReduceL2) | ⚠️ **candidate** (200-D native) | — | — | — | — | — | ❌ (19000≠32) | — | Server rep candidate (see §7) |
| EEGPT | ❌ (24.9 MB, 0.8–4.8 s) | ⚠️ **candidate** (2048-D) | — | — | — | — | — | ❌ (2048≠32) | — | Server rep IF 62→22 remap survives |
| LaBraM | ⚠️ (22 MB, 68 ms) | ⚠️ (200-D) | — | — | — | — | — | ❌ (200≠32) | — | Below-PCA; low value |
| FEMBA-tiny | ❌ (32 MB, 220 ms) | ⚠️ (30800-D, 22-ch match) | — | — | — | — | — | ❌ (30800≠32) | — | Only 22-ch match; 940× projection kills it |
| EEGNet/Shallow/Deep4 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Drop (stubs) |
| PyTorch export | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Drop (stub) |
| Cognitive decoder | — | — | ✅ **primary** (band-power) | — | — | — | — | — | — | Keep; decoupled |

---

## STEP 4 — Correct architecture (proposed)

**Current architecture (what exists):** a *single* 32-D embedding path. `embedEEG` → 32-D vector → `eeg_analyses` + pgvector `embeddings` (vector(32)) → ANN retrieval + Recall SLO. Cognitive decoding is a separate band-power path. No server representation path; no native-dim storage; the gateway ships raw frames (no real-time embed).

**Proposed (smallest sensible multi-model architecture):**

1. **Keep V2 as the browser/default embedding model.** ✅ V2 is 32-D, wasm-compatible, SHA-locked, GA, and the only model that natively satisfies `vector(32)` + the latency gate. **Do not replace it.** The 32-D contract is a *browser latency + 32-D ANN* contract — it is correct to keep it narrow and fast.

2. **PCA remains the terminal fallback.** Correct and minimal. Keep `pca-legacy-v1` exactly as-is (it pad/truncates to 32 and is JS, so it runs everywhere including SSR when WASM is absent).

3. **Add a server-side native-dim representation layer — as a SEPARATE embedding space, not a projection to 32.** This is the key decision the spec asks for. **Do not force 200/2048/30800-D foundation models into `vector(32)`.** A 940× projection (FEMBA 30800→32) or even 2048→32 destroys the representation and gains nothing over PCA. Instead:
   - Persist native-dim embeddings to `eeg_analyses.embedding` (`FLOAT8[]`) — **already dimension-agnostic** (no CHECK) — tagged by `embedding_model` + `embedding_dimensions` (the upload route already writes both fields). This requires **no schema change** to `eeg_analyses`.
   - For server-side ANN over a native space, create a **model-namespaced store separate from `vector(32)`**: e.g. a `vector(2048)` column on a `embeddings_eegpt` table, or a `model_id`-partitioned `foundation_embeddings` table with a per-model dim column, or a `vector(N)` column per model family. pgvector supports mixed dims across columns/tables; it does **not** across rows in one column — so the unit of separation is the table/column, not the row.
   - API boundary: keep `embedEEG()` (32-D, interactive) on the browser↔SSR shared path; add a **server-only** foundation embed call (e.g. `embedFoundation({modelId, input})` reusing `ONNXAdapter`/the registered adapter, CPU EP) that writes to the native store. This boundary is clean because the gateway/stream and the upload route are already server-only entry points.

4. **Browser and server embeddings do NOT need to share a vector space.** The browser/retrieval role (32-D, ANN, Recall@10 SLO) and the offline-representation role (native-D, phenotyping/clustering/export) serve different product capabilities. Conflating them via an arbitrary projection is the wrong move. If cross-role retrieval is ever genuinely needed, **train a 32-D projection head** for that purpose — never an untrained PCA-down-projection of a 30K-dim model.

5. **Cognitive decoding stays decoupled.** The trained-logistic-v0 operates on band-power features; it is correct to keep it independent of the embedding head. The dead 4-class MI logits head should be **decided**: either productize it (a real MI classifier product) or stop exporting it from the bridge (`logitsOutputName` is optional and already unset for v1/v2).

6. **Real-time streaming** needs wiring: the gateway should either (a) stamp the correct model id (`braindecode-eegconformer-prod-v2`), or (b) optionally run the shared `InferenceEngine` per chunk and tag frames with the resulting `model_id`. Out of scope to implement; flagged as a follow-up.

**Answering the 10 architecture questions:**
- *Should V2 remain the browser/default?* **Yes, and GA.**
- *Should there be server-side foundation models?* **Yes — conditionally, with native-dim storage.**
- *Which are suitable for server inference?* **CBraMod** (2.2 MB, 53.6 ms, 200-D/19000-D native — best size/latency trade for a server rep); **EEGPT** (2048-D, but 25 MB + ~1 s and a 62→22 channel problem); **LaBraM/FEMBA** (below-PCA accuracy + 16/22-ch mismatch + huge dims).
- *Which are specialized rather than replacements?* **CBraMod** = server high-capacity rep; **EEGPT** = server 2048-D rep backbone; cognitive decoder = cognitive classification. None are browser replacers for V2.
- *Where should PCA stay?* Terminal fallback only (correct).
- *Separate embedding space for server foundation?* **Yes — strongly recommended** (see #3 above). The 32-D contract must not be violated by a 30K-D model.
- *Would native-D models need separate storage?* **Yes** — a per-model `vector(N)` store, partitioned by model, distinct from the interactive `vector(32)` index.
- *Browser and server same vector space?* **No** — separate roles, separate spaces; project only if a cross-role task is defined.
- *API boundary browser↔server?* `embedEEG()` (interactive, 32-D, shared) vs a server-only `embedFoundation()` for native-D; the gateway/stream route is the natural server ingress.
- *Which model for each future capability?* (a) interactive embedding/retrieval → V2+PCA 32-D; (b) cognitive decode → trained-logistic-v0; (c) offline phenotyping/similarity → CBraMod 200-D (server); (d) high-capacity transfer backbone → EEGPT 2048-D (server, pending 62→22 feasibility); (e) sleep staging → FUTURE model on the sleep-edf loader; (f) synthesis → synthetic.ts; (g) reconstruction → FUTURE decoder (not the static demo).

---

## STEP 5 — Ranked experiment roadmap (do NOT run in this mission)

**P0 — EEGPT**
- *Model:* EEGConformer v2 (FP32 → INT8-QDQ ablation, `_bench/eegconformer_finetuned_int8.onnx`, sha `59e9555a…`).
- *Role tested:* Browser real-time embedding (the GA gate itself, not a new model).
- *Exact task:* Clear the Firefox P50<400 / P95<600 ms latency gate with parity preserved.
- *Dataset:* deterministic synthetic (wasm-smoke) + the existing gate signal; parity vs FP32 on PhysioNet-derived vectors.
- *Why:* **Firefox tail latency is the only open GA evidence item.** v2 is otherwise production-ready; this is the bottleneck blocking honest GA on Firefox. It is a latency fix, not a "new model."
- *Required metric:* Firefox P99/P95<600, P50<400; embed cos-sim ≥ 0.99 vs FP32; fallback rate < 0.5%.
- *Browser/server:* browser (Chromium + Firefox).
- *Expected compute cost:* 1–2 days (artifact exists; harness + gate test exist; only measurement).
- *Decision it enables:* Promote INT8 artifact to `public/models/eegconformer_finetuned.onnx` (manifest SHA update) as the v2 GA artifact — **only** if the gate clears.
- *This is the Mission 8 recommendation.*

**P1 — CBraMod server representation (the RIGHT role, not the MI-classifier role)**
- *Model:* CBraMod (`onnx-cbramod`, 200-D mean-tokens).
- *Role tested:* Server high-capacity representation / B. This deliberately **re-tests the correct product role** — Mission 6 bench-marked CBraMod vs v2/PCA on 32-D-style MI Recall@K (the wrong role for a 200-D model). Here the question is whether 200-D CBraMod embeddings separate classes / subjects better than 32-D V2 on a **representation-quality** task.
- *Exact task:* Persist CBraMod 200-D to a native-dim store (FLOAT8[] or a `vector(200)` table); measure within-class cohesion vs between-class separation (Fisher, silhouette) and subject-level retrieval Recall@K, vs V2-32-D on PhysioNet EEGMMIDB.
- *Dataset:* PhysioNet EEGMMIDB S001–S050 (already available).
- *Why:* Highest point estimate of any foundation model (0.3233@10; 0.3043@50). But its 200-D richness is wasted on a 32-D cosine-Recall@K MI task — that's why Mission 6 was negative. A 200-D-native task is the fair test.
- *Required metric:* representation separation metric where CBraMod 200-D ≥ v2 32-D; AND MI accuracy where 200-D ≥ v2 (sanity).
- *Browser/server:* server only (DFT/ReduceL2).
- *Expected compute cost:* M (offline; 53.6 ms CPU × 50 subj ≈ minutes).
- *Decision it enables:* If CBraMod 200-D beats v2 32-D on a representation task → deploy a server-side CBraMod embed + `vector(200)` ANN store (new capability: offline cohort phenotyping). If not → retire CBraMod from consideration.

**P1 — EEGPT 62→22 remap feasibility (gate before server rep)**
- *Model:* EEGPT (`onnx-eegpt`, INT8, 2048-D).
- *Role tested:* Server 2048-D backbone, conditional on remap survivability.
- *Exact task:* 62-ch vs 22-ch-input comparison under identical LOSO; does dropping 40 channels destroy the ViT-channel-patch representation?
- *Dataset:* PhysioNet EEGMMIDB (62-ch montage where available).
- *Why:* EEGPT's only shot at a server role depends on whether the 62→22 projection is survivable; the 24.9 MB / ~1 s footprint already disqualifies browser use.
- *Metric:* cos-sim(62-ch, 22-ch) ≥ 0.90; 22-ch acc ≥ v2 with p<0.05.
- *Browser/server:* server.
- *Compute:* M.
- *Decision:* if feasible → server 2048-D rep; if not → DROP EEGPT.

**P2 — Cross-domain generalization (data, not head)**
- *Model:* EEGConformer v2.
- *Role:* Validate that v2's ~PCA accuracy on MI is data-limited, not head-limited (T-033 says 32-D is not the bottleneck).
- *Task:* held-out BCI-IV-2a runs 4–7 (4-class MI, 9 subjects × 4 sessions × 100 trials) + a non-MI task, zero-shot from the EEGMMIDB checkpoint.
- *Why:* v2 vs PCA is not significant (p≈0.07–0.62); the representation-collapse (intra≈inter cos) suggests narrow MI overfit. This tests generalization, not a new architecture.
- *Metric:* LOSO acc on BCI-IV-2a vs PhysioNet; drop < 5 pp = generalizes.
- *Browser/server:* server (offline).
- *Compute:* L (needs BCI-IV-2a access + eval harness).
- *Decision:* if it generalizes → keep v2 as the backbone and invest in more data; if it collapses → reframe v2 as MI-specific and invest in a general-purpose rep.

**DROP — no reason to spend resources:**
- **LaBraM:** 22 MB, 200-D, 10-subj acc **below PCA** (0.2533 vs 0.313+). 16-ch mismatch. No plausible role.
- **FEMBA-tiny:** 32 MB, 30,800-D, below-PCA (0.2400), 22-ch match is its only advantage — and projecting 30,800→32 annihilates it. A 30K-dim server store is a storage/liability with no evidence it helps.
- **EEGNetv4/ShallowFBCSPNet/Deep4Net:** no artifacts, no bridge → `load()` fails. Drop from the routing surface.
- **`pytorch-export-placeholder`:** stub. Drop.
- **Contrastive FT of v2 (T-034):** already run (`t034_contrastive` 0.274, `t034_aug_contrastive` 0.259) — both **lost to v2 0.325**. Re-proposing is duplicate work.
- **EEG2Image reconstruction as a model task:** the route is a static demo; "implement a reconstructor" is engineering, not a model role. Defer until a reconstruction product is defined.

---

## STEP 6 — Re-evaluate the four foundation models by ROLE (not "better than v2")

### CBraMod `onnx-cbramod` — 19→22 remap already answered the wrong question

- **Mission 6 result (preserved):** native-19-ch CBraMod on the 50-subj LOSO MI Recall@K task: acc **0.3043** vs v2 **0.3250**, vs PCA 0.3065; Bonferroni p = 0.353 (vs v2), 1.0 (vs PCA) → **did not beat v2/PCA → not promoted.**
- **Why it is the wrong role:** Mission 6 tested CBraMod as a 19→22 *remap* competing with v2 on the **same 32-D-style MI classification / Recall@K** task. But CBraMod's native output is **200-D** (mean-tokens) / 19000-D (raw) — its representation richness is **wasted** in a 32-D cosine-Recall@K MI task. That is like benchmarking a 300-D word2vec against a 50-D SVD on a 50-D slot — the dimension ceiling, not the model, is the bottleneck.
- **Correct role:** Server-side **200-D representation** for offline similarity/clustering/phenotyping (a B role), stored in a native-dim (`vector(200)`) store, *not* projected to 32.
- **Evidence sufficient?** No — the 50-subj negative was on the wrong role/metrics. The n=10 T-030 0.3233 point estimate (highest of any model) is encouraging but underpowered and on the MI task, not a representation task.
- **Next experiment:** the P1 in §5 (200-D-native representation-separation vs v2 32-D). **Do not re-run the 19→22 remap / MI-Recall@K.**
- **Verdict:** not a replacer (confirmed), **but a conditional server-representation specialist** pending the P1 experiment. 19-ch vs 22-ch pipeline mismatch is accepted as the *study*, not a blocker, once the role is representation rather than 32-D MI.

### EEGPT `onnx-eegpt` — server backbone, gated on 62→22 feasibility

- **Evidence:** n=10 acc 0.3067 (above v1, not significant); INT8 parity cos 0.999, 24.9 MB, 830 ms–4.8 s.
- **Correct role:** Server 2048-D representation / transfer backbone (C-adjacent / B). 62-ch is its **channel contract**.
- **62→22 problem:** dropping ~64% of channels on a ViT that attends over channel patches is scientifically dubious — likely destroys the signal. This must be validated (P1) before any server role.
- **Verdict:** server candidate **only if** the remap feasibility experiment passes; otherwise DROP. Browser path ruled out (size/latency).

### LaBraM `onnx-labram` — below baseline, no plausible role

- **Evidence:** n=10 acc 0.2533 **< PCA** (0.313+); 16-ch (≠22); 22 MB.
- **Correct role:** 200-D representation — but it is already below the 32-D baseline at n=10, and its 16-channel footprint mismatches the 22-ch pipeline.
- **Verdict:** **DROP.** No experiment proposed; the accuracy signal + channel mismatch compound. Re-consider only if a 16-ch-native product surface appears (none exists).

### FEMBA-tiny `onnx-femba-tiny` — the only 22-ch match, but the 30,800-D liability

- **Evidence:** n=10 acc 0.2400 **< PCA**; 22-ch (matches pipeline — the only foundation model that does); 30,800-D; 220 ms; 32 MB.
- **Correct role:** high-capacity 30K-D server representation — but a 940× projection to 32 annihilates it, and there is **no evidence** a 30K-D store helps relative to 200-D CBraMod or 2048-D EEGPT.
- **Verdict:** **DROP** for now. Its 22-ch match is offset by below-PCA accuracy and a dimensionality that is a storage/routing liability. Re-open only if a concrete 30K-D server use case (e.g., raw offline export for external analysis) is defined with a budget.

---

## STEP 7 — Final decision

1. **What runs in the browser?** **EEGConformer v2 only** (32-D, GA, wasm, sha-verified) + **PCA** as the always-on JS fallback. Nothing else is browser-routed; EEGPT/LaBraM/FEMBA are too big/slow, CBraMod is non-WASM, v1 uses external-data (unsafe).
2. **What runs on the server?** Today: `embedEEG` via Nitro SSR (same v2+PCA path) for `/api/eeg/upload`; pgvector ANN; the Recall@10 SLO cron; LOSO eval API; the WebSocket streaming gateway (transport-only, raw frames). **Proposed next:** a server-only native-dim foundation embedder (CBraMod 200-D) writing to a **separate** `vector(200)`-style store — gated on the §5 P1 experiment.
3. **What remains fallback?** `pca-legacy-v1` (terminal, JS, 32-D) — correct; keep.
4. **Which model serves each product capability?** (a) interactive embedding/retrieval → V2 32-D; (b) cognitive decode → `cognitive-decoder-v0` (band-power, decoupled); (c) offline cohort phenotyping → CBraMod 200-D (proposed, pending P1); (d) transfer backbone → EEGPT 2048-D (proposed, pending 62→22 feasibility); (e) sleep staging → FUTURE model on the existing sleep-edf loader; (f) synthesis → `synthetic.ts`; (g) reconstruction → FUTURE decoder (the `/eeg2image` route is a static demo).
5. **Which models to abandon?** `pytorch-export-placeholder` (stub), `braindecode-eegnetv4/shallowfbcspnet/deep4net-default` (no artifacts/bridge → load fails), `onnx-labram` (below-PCA), `onnx-femba-tiny` (below-PCA + 30K-D liability), and the v2 **4-class MI logits head** (dead output — productize or stop exporting).
6. **Which models deserve a proper task-specific benchmark?** **CBraMod** (200-D-native representation task — the right role) and **EEGPT** (62→22 remap feasibility). LaBraM and FEMBA are below-PCA at n=10 on MI; a representation-role benchmark is only justified if/when a server 200-D / 30K-D use case is scoped (not yet).
7. **Mission 8 recommendation:** **Run the P0 experiment — INT8-QDQ latency ablation of EEGConformer v2 against the Firefox GA latency gate.** It is (a) the only open GA evidence item, (b) read-only to implement (the 1.1 MB INT8-QDQ artifact already exists in `public/models/_bench/` with cos 0.9985 parity; the Firefox latency gate test and the wasm-smoke harness already exist), (c) zero production/`.env`/`DEFAULT_PREFERRED`/registry/model change, (d) no retraining. On the **same pass**, gate MI: if it clears Firefox P50<400 / P95<600 with parity ≥ 0.99, swap the INT8 artifact into `eegconformer_finetuned.onnx` (manifest SHA update only) as the v2 GA artifact; if it fails, document the worker-spin-up ceiling and stop. After Mission 8 clears GA, run the §5 **P1 CBraMod 200-D server-representation** experiment to decide whether a second (server) embedding space is warranted.

**Major uncertainty that must be resolved before implementation:**
- **(Streaming)** `StreamGateway.defaultModelId = "eegconformer-v1"` is a stale literal and the gateway does not run `embedEEG` per frame — real-time inference is transport-only. Decide whether to (a) correct the model id, (b) wire per-chunk `InferenceEngine`, or (c) leave streaming model-tagged but not inferred.
- **(Separate embedding space)** whether a `vector(200)`-style native store is in-scope (schema/RPC work) before the CBraMod P1. The `eeg_analyses.embedding FLOAT8[]` column already accepts native dims, so *persistence* needs no change — only the *ANN index* path does.
- **(CBraMod browser claim)** the uncommitted `wasm-smoke.test.ts` Group 3 asserts CBraMod now runs in ORT-WASM ≥1.27.0, contradicting the registry/manifest `wasmCompatible:false`. Treat the registry flag as authoritative until a green Group-3 run is demonstrated; do not route CBraMod to browser on the Group-3 claim alone.

*End of audit. Read-only: no production code, artifact, `.env`, `DEFAULT_PREFERRED`, rollout stage, or registry entry was modified. The recommended experiments are deferred pending approval.*
