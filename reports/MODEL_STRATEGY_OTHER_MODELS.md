# Neuro-Fabricore Model Strategy — "Other EEG Models"
*Mission: Model Portfolio Investigation & Next-Experiment Planning*
*Scope: READ-ONLY investigation. No training, production edits, default/rollout changes, artifact modifications, GA deployment, or model modifications were performed. This file is the only change — a documentation record of the investigation (permitted by the mission constraints).*

> **Ground truth this is written against (verified in this session):**
> - Repo `worktree == HEAD b9164a6`; clean.
> - V2 artifact `public/models/eegconformer_finetuned.onnx` sha‑`18644de1…` (3,359,557 B, self‑contained, WASM‑compatible) — verified matching manifest + registry (`braindecode-eegconformer-prod-v2`).
> - V2 is registered but **not** `DEFAULT_PREFERRED` (still v1) and rollout default = `off`. **Unchanged.**

## Key evidence (consolidated)

| Model | 50‑subj Acc | 10‑subj Acc | CPU latency | Browser‑WASM latency | Size | WASM? | Trained in‑repo? |
|---|---|---|---|---|---|---|---|
| PCA bandpower | 0.313–0.320 | 0.290 | ~0 ms | ~0 (pure JS) | ~0 | ✅ | N/A (baseline) |
| EEGConformer v2 FT | **0.343** | 0.327 (held‑out) | 8.1 ms | Chr 300 ms / **FFox 1.4–2.6 s** | 3.2 MB | ✅ self‑contained | ✅ PhysioNet (3 FT runs) |
| EEGConformer v1 (prod) | 0.251–0.283 | 0.317 | 6.9 ms | Chr 216 ms / FFox ~0.9 s | 3.0 MB* | ⚠️ ext‑data | ✅ BCI‑IV‑2a |
| CBraMod | — | **0.3233** | 53.6 ms | n/a (no WASM) | 2.2 MB | ❌ DFT,ReduceL2 | preload only |
| EEGPT | — | 0.3067 | 0.8–4.8 s | 0.8–4.8 s | 24.9 MB | ✅ but huge/slow | preload only |
| LaBraM | — | 0.2533 | 68.3 ms | 68 ms/22 MB | 22.2 MB | ✅ | preload only |
| FEMBA‑tiny | — | 0.2400 | 220 ms | 220 ms | 30 MB (fp32) | ✅ | preload only |

\* v1's committed ONNX is **external‑data** format (`eegconformer.onnx` + `.onnx.data`) → not reliably browser‑WASM‑loadable (this is precisely why T‑035 re‑exported v2 as a single self‑contained file).

**Statistical reality:** v2 vs PCA at 50 subjects is *not* significant (p≈0.07–0.62, d≈0.05–0.19). v2 vs v1 **is** significant (p<0.001, d=0.70). No other model has 50‑subject validation.

---

## Per‑model assessment

Format per the mission briefing: Current evidence → Current benchmark results → Current limitations → Current role → Possible product/use case → Required next experiment → Exact dataset/protocol needed → Expected engineering effort → Evidence gate before promotion → Priority → Decision.

### 1. EEGConformer V2 (the current backbone) — must NOT be taken for granted

- **Current evidence:** 50‑subj LOSO acc 0.3428 (FT v2, 40 train) vs Original 0.2826; FT vs Original p=0.0002, d=0.701. vs PCA 0.3128, +0.030, p=0.070 (not sig). Parity PyTorch↔ONNX cos=1.0; 17 WASM ops, zero DFT/ReduceL2/Einsum (T‑035 stripped Einsum→MatMul); self‑contained 3.2 MB; sha `18644de1…`. Staging (T‑035) passes SHA‑verify + determinism on Chromium **and** Firefox, but **Firefox P50/P95 FAIL GA gates** (1447 / 1576 ms vs <400/<600 ms).
- **Current benchmark results:** 50‑subj: acc 0.325, Recall@1 0.292, Recall@5 0.779, Recall@10 0.946, Fisher 0.0072, intra‑class cos 0.907 ≈ inter‑class 0.904 (representation collapse). 4‑class MI only. CPU latency 8.1 ms.
- **Current limitations:** (a) **Firefox WASM latency** is the GA blocker; (b) representation collapse — classes barely separate (intra≈inter); (c) v2 does **not** significantly beat PCA accuracy (the "edge" is the 4‑logits head + reliability, not proven accuracy lift); (d) fine‑tuning was MI‑only (PhysioNet EEGMMIDB), no cross‑dataset validation.
- **Current role:** Interactive representation backbone (32‑D + 4‑class MI logits); fallback floor is PCA.
- **Possible product/use case:** Embedding API; in‑browser BCI inference; real‑time wearable MI classification.
- **Required next experiment:** **NOT contrastive FT** (see search‑before‑proposing). T‑034 *already* ran `t034_contrastive` (0.274) and `t034_aug_contrastive` (0.259) on the same 50‑subj protocol → both **lost to v2 (0.325)**. Re‑proposing is duplicate work. The correct next experiments are:
  1. **Firefox WASM latency fix** (8‑bit dynamic quant — retry now that Einsum is stripped; or ORT‑Web thread/build tuning). *Offline ablation, no weight change.*
  2. **Cross‑domain / cross‑dataset validation** (held‑out BCI‑IV‑2a and a non‑MI task) to test generalization — the model is MI‑specific; accuracy gains likely come from *data*, not the 32‑D head (T‑033 confirmed 32‑D is not the bottleneck).
  3. **Multi‑task head** (MI + artifact/confidence) to exploit the logits head without chasing raw MI accuracy.
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050 LOSO for the latency ablation (already have it); BCI‑IV‑2a (runs 4‑7) + a non‑MI dataset for cross‑domain.
- **Expected engineering effort:** Latency fix M‑L; cross‑domain eval M; multi‑task head M.
- **Evidence gate before promotion:** v2 **P95 < 600 ms on BOTH Chromium and Firefox** at GA; determinism + SHA‑verify preserved; fallback rate < 0.5%.
- **Priority:** **P0.**
- **Decision:** **PRODUCT (interactive backbone)** — conditional on clearing the Firefox latency gate. The model is ready; the browser tail‑latency is not.

### 2. CBraMod — `onnx-cbramod`

- **Current evidence:** 19 ch / 250 Hz / 1000, 2.2 MB, 53.6 ms (CPU), opset 17. **NOT WASM** — `DFT` + `ReduceL2` (`wasmCompatible:false`, `wasmBlockers:["DFT","ReduceL2"]`). Registry declares `embeddingDim:19000` but T‑030 mean‑pools `[B,19,5,200]→[200]`.
- **Current benchmark results:** 10‑subj acc **0.3233** (T‑030) — highest point estimate of any model, +11.5% vs EEGConformer‑v1, but **not significant** (p=0.401, d=0.28). Recall@1 0.277. **No 50‑subject validation.**
- **Current limitations:** (a) n=10 → underpowered, wide CI; (b) NOT WASM → cannot serve browser/interactive; (c) **19‑ch contract ≠ 22‑ch production pipeline** — any comparison is confounded by channel mismatch; (d) registry `embeddingDim` (19000) inconsistent with benchmarked behaviour (200).
- **Current role:** Registered, gated out of interactive routing by `wasmCompatible:false`; no server‑side path is wired yet.
- **Possible product/use case:** Server‑side analytical embedder (cohort similarity search, offline clustering) — IF the channel mismatch is neutralized.
- **Required next experiment:** **(1) 19→22 channel remap study** (project 19 ch onto the 22‑ch standard, or drop the 3 extra channels that the prod pipeline carries and run both at 19) to see whether CBraMod's 0.3233 survives realignment; **(2) 50‑subject LOSO** with train‑only PCA fit + separate candidate pools (no self‑retrieval).
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050, LOSO 50 folds, bandpass 4–38 Hz, 22 (or projected) channels. Compare acc vs PCA + v2 with paired t + Bonferroni.
- **Expected engineering effort:** M (remap projection matrix + protocol) — evaluation‑only, no retraining.
- **Evidence gate before promotion:** CBraMod acc ≥ PCA **and** ≥ v2 with p<0.05 (Bonferroni‑corrected) on 50 subjects **after** remapping; WASM blocker resolution (or server‑side routing) decided.
- **Priority:** **P2** (investigation) — but the remap+50‑subj study is **P1 conditional**: if CBraMod survives remap and beats v2 with significance, it becomes the server‑side specialist (it already has the highest point estimate); if it collapses, retire it from routing.
- **Decision:** **R&D → conditional SERVER‑SIDE.** Do not route to browser; run remap + 50‑subj before deciding.

### 3. EEGPT — `onnx-eegpt`

- **Current evidence:** 62 ch / 250 Hz / 1000, INT8, 24.9 MB, 830 ms–4.8 s latency (JSON 830 ms CPU mean; T‑030 4.82 s incl. cold). 2048‑D (mean‑pooled). WASM‑compatible in principle but **62‑ch ≠ 22‑ch** pipeline. `isExperimental:true`.
- **Current benchmark results:** 10‑subj acc 0.3067 (+5.7% vs v1, n.s. p=0.343); Recall@1 0.250. **No 50‑subject validation.**
- **Current limitations:** (a) 62→22 remapping **discards ~64% of spatial channels** — scientifically dubious; EEGPT's ViT attends over channel patches, so dropping channels may destroy its advantage; (b) latency/24.9 MB rule it out for interactive use regardless; (c) only n=10.
- **Current role:** Registered experimental encoder; not routed.
- **Possible product/use case:** Server‑side high‑dimensional (2048‑D) representation for offline similarity/search over large archives; a "quality‑at‑any‑cost" embedder.
- **Required next experiment:** **(1) 62→22 remap feasibility study** — does EEGPT retain any lift when 40 channels are dropped/projected? If remapping collapses it to ≤ v2, stop. **(2) If remap survives**, a 50‑subject LOSO + server‑batching latency amortization (batch N trials/forward).
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050, two channel configs (full 62 via standard montage where available, and 22‑ch subset) under identical LOSO; accuracy + latency per batch size.
- **Expected engineering effort:** M (remap + batch harness).
- **Evidence gate before promotion:** Remapped 22‑ch EEGPT acc ≥ v2 with p<0.05 on 50 subjects **and** batch latency < 200 ms/trial at batch≥32.
- **Priority:** **P3** — low prior (remap likely destroys the signal; 24.9 MB + ~1 s already disqualify interactive use). Only pursue if CBraMod's remap *fails* and a high‑capacity server rep is genuinely needed.
- **Decision:** **R&D.** Do not build a browser path; do not route.

### 4. LaBraM — `onnx-labram`

- **Current evidence:** 16 ch / 250 Hz / **1600** samples, 22.2 MB, 68.3 ms, ViT, `embeddingDim:200`, `wasmCompatible:true`, `experimental:true` (registry). Graph‑surgery Reshape `→ [1,16,8,200]`.
- **Current benchmark results:** 10‑subj acc **0.2533 — below PCA** (−12.6%, d=−0.61, p=0.084). No 50‑subject validation.
- **Current limitations:** (a) **below PCA on the only test it ran**; (b) 16‑ch ≠ 22‑ch (remap needed); (c) **1600‑sample window ≠ 1000‑sample production segmentation** → misaligned to the live pipeline; (d) 22 MB.
- **Current role:** Registered experimental; not routed.
- **Possible product/use case:** None until it beats PCA.
- **Required next experiment:** **(1) 16→22 channel remap + 1600→1000 window handling** (truncate/center or resample) to make it comparable; **(2) 50‑subject LOSO** under the T‑032 protocol.
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050 LOSO, bandpass 4–38 Hz, 22 channels, 1000‑sample windows, mean‑pool if sequence output.
- **Expected engineering effort:** M (remap + window contract change) — evaluation‑only.
- **Evidence gate before promotion:** Remapped LaBraM acc ≥ PCA with p<0.05 (Bonferroni) on 50 subjects.
- **Priority:** **P4.** The prior is very low — it's the model with the worst published accuracy *and* below PCA. Only worth a 50‑subject validation if someone has spare eval capacity; do **not** ship.
- **Decision:** **RETIRE from routing** (leave registered as experimental; mark `wasmCompatible` gating so it's never auto‑selected).

### 5. FEMBA‑tiny — `onnx-femba-tiny`

- **Current evidence:** 22 ch / **200 Hz** / **1280** samples, Mamba, FP32 30.7 MB (fp16 16.3 MB), 220 ms, `wasmCompatible:true`, `experimental:true`. **INT8 quantization destabilized** ("recurrent scan error compounding", per‑channel max_diff=3.23). Registry declares `embeddingDim:30800`; description says graph‑surgery Reshape `→ [1,1,22,1280]`, output `[B,80,385]`.
- **Current benchmark results:** 10‑subj acc **0.2400 — worst**, −17.2% vs PCA (d=−0.58). No 50‑subject validation. **Contract bug:** registry `embeddingDim:30800` vs benchmark mean‑pooled 385 — inconsistent; the adapter default is the raw 30800 (un‑pooled), so routing it would emit 30800‑dim vectors that break the 32‑dim pipeline.
- **Current limitations:** (a) below PCA; (b) **sample‑rate/window mismatch** (200/1280 vs prod 250/1000) → resample/truncate needed; (c) **registry vs adapter `embeddingDim` inconsistency** (30800 vs 385) — a real contract bug that must be resolved before any fair re‑benchmark; (d) INT8 unstable → can't shrink for browser; (e) 220 ms + 30 MB disqualify interactive use.
- **Current role:** Registered experimental; not routed.
- **Possible product/use case:** None until it beats PCA and the contract is fixed.
- **Required next experiment:** **(1) resolve the 30800‑vs‑385 contract** (decide: mean‑pool to 385 and update registry/adapter to agree, OR emit raw and pad the pipeline); **(2) remap 200/1280→250/1000 + 50‑subject LOSO**. **NOTE:** fixing the contract is a model‑contract change — per mission rules ("do NOT modify unrelated models just to make them 'fit'"), this stays as a **documented required pre‑condition**, not a change made now.
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050 LOSO, 22 channels, 250 Hz, 1000 samples, bandpass 4–38 Hz, L2‑normalized output of agreed dim.
- **Expected engineering effort:** S (contract fix in registry) + M (eval); but contract fix touches a registered model → flag, don't apply silently.
- **Evidence gate before promotion:** Fixed‑contract FEMBA acc ≥ PCA with p<0.05 on 50 subjects.
- **Priority:** **P4.** Worst accuracy + contract bug + large/slow. Document the negative result (INT8 instability is now recorded) and leave routed‑out.
- **Decision:** **RETIRE from routing / R&D negative control.** Keep artifact; fix contract only if/until a study is started.

### 6. EEGNetv4 / ShallowFBCSPNet / Deep4Net — BraindecodeAdapter stubs

- **Current evidence:** Registered in `registry.ts` (`braindecode-eegnetv4-default`, `-shallowfbcspnet-default`, `-deep4net-default`) but **no ONNX artefacts** exist (`public/models/` has no `eegnetv4.onnx`/`shallowfbcspnet.onnx`/`deep4net.onnx`). Export script `scripts/export_braindecode_eegconformer.py` supports `--architecture EEGNetv4|ShallowFBCSPNet|Deep4Net` but it has **not** been run. `MODEL_INVENTORY.md` (Aug 8) classifies them as stubs. Contracts: EEGNetv4 22ch/128Hz/256; Shallow 22ch/250/1125; Deep4 22ch/250/1125.
- **Current benchmark results:** None (not exported, not evaluated).
- **Current limitations:** No artefacts → cannot run; window/sample‑rate contracts diverge from the 250/1000 prod pipeline.
- **Current role:** Reserved ablation comparators (registered in tests, not routed).
- **Possible product/use case:** Architecture‑selection ablation only; EEGNetv4 could become a **cheap interactive baseline** (smallest/fastest) if it benchmarks competitively.
- **Required next experiment:** **Onnx export** of each via `--architecture`, then run the **same 50‑subject T‑032 LOSO protocol** for a fair, same‑contract ablation vs PCA + v2.
- **Exact dataset/protocol needed:** PhysioNet EEGMMIDB S001–S050, resample to 250 Hz (where 128 Hz), 4–38 Hz bandpass, 22 channels, 1000‑sample windows, train‑only PCA fit, separate candidate pools.
- **Expected engineering effort:** S (run the existing export script) + M (50‑subj eval).
- **Evidence gate before promotion:** Accuracy ≥ some threshold (e.g., ≥ 0.9× v2) at ≤ 20 ms with WASM — then consider EEGNetv4 as a "fast fallback embedder" tier.
- **Priority:** **P3.** Small effort, high information value for architecture selection; strictly ablation, not a product path yet.
- **Decision:** **ABLAION.** Do not ship; export + evaluate for architecture comparison.

### 7. Cognitive Decoder — `cognitive-decoder-v0` (separate from backbone models)

- **Current evidence:** `public/models/cognitive-decoder-v0.onnx` is **1,333 bytes** — a placeholder/stub (manifest: `wasmCompatible:true`, no `externalData`). `scripts/train_cognitive_decoder.py` exists. `src/lib/decoder/` has `features.ts`, `tfjs-decoder.ts`, `trained-decoder.ts`, `index.ts` (heuristics) + `__tests__/cognitive-decoder-integration.test.ts`. **Datasets present in repo:** PhysioNet EEGMMIDB (motor imagery — NOT cognitive states) and a **Sleep‑EDF loader** (`src/lib/eeg/loaders/sleep-edf.ts`). No labelled attention/workload/arousal dataset.
- **Current benchmark results:** None (stub).
- **Current limitations:** (a) 1,333‑byte stub, not a real model; (b) **no labelled cognitive‑state EEG dataset** in the repo (EEGMMIDB = MI, Sleep‑EDF = sleep stages — neither is "attention/workload/arousal"); (c) decoder contract (3‑D output) not agreed.
- **Current role:** Placeholder; heuristic (`index.ts`) is the live fallback.
- **Possible product/use case:** Attention/workload/arousal inference (B2B fatigue/attention monitoring).
- **Required next experiment:** (1) **Source a labelled cognitive‑state dataset** (e.g., SEED, DREAMER, or a public attention/workload set — none currently in repo); (2) train a real 3‑output model; (3) validate it beats the heuristic baseline on held‑out subjects.
- **Exact dataset/protocol needed:** A dataset with EEG + attention/workload/arousal labels, LOSO, compare decoder accuracy vs the existing heuristic in `src/lib/decoder/index.ts`.
- **Expected engineering effort:** L (dataset sourcing + training pipeline + labelling) — this is a **training‑required** item, not evaluation‑only.
- **Evidence gate before promotion:** Decoder ≥ heuristic on held‑out accuracy/Pearson, with p<0.05.
- **Priority:** **P3** (after backbone stability). It's a different product class (task head), not a backbone.
- **Decision:** **R&D (training‑required).** Do not route the stub; replace with a real model once a labelled dataset is sourced.

---

## Search‑before‑proposing: experiments already performed (do NOT re‑run)

| Proposed experiment | Already done? | Result | Action |
|---|---|---|---|
| Contrastive / augmentation FT of 32‑D head (T‑033 reco) | ✅ T‑034 | `t034_contrastive` 0.274, `t034_aug_contrastive` 0.259 — **both < v2 0.325** | ❌ Do not re‑propose. Negative result recorded. v2 32‑D ceiling is data/objective‑limited, not dimension‑limited. |
| EEGConformer 50‑subj validation | ✅ T‑031‑50SUBJ, T‑032 | v2 0.343, vs PCA n.s., vs v1 sig | — (already the baseline) |
| CBraMod 50‑subj | ❌ only 10 (T‑030) | — | ✅ to do (after remap) |
| CBraMod 19→22 remap | ❌ not done | — | ✅ to do |
| EEGPT 50‑subj / remap | ❌ not done | — | ✅ to do (if justified) |
| LaBraM 50‑subj / remap / window fix | ❌ not done | — | ✅ to do (low prior) |
| FEMBA 50‑subj / contract fix / remap | ❌ contract bug not fixed | — | ✅ to do (low prior) |
| EEGNet/Shallow/Deep4 ONNX export | ❌ not exported | — | ✅ to do (ablation) |
| Contrastive FT on a **non‑collapsed** objective / cross‑task | ❌ not done | — | (future; not the T‑033 reco as stated) |

---

## Concise report

### 1. Next Model Mission
Lock down **V2 as GA** (clear the Firefox WASM latency gate) and run the **remap + 50‑subject validation** studies for CBraMod / LaBraM / FEMBA / EEGPT — **none of which may beat V2** (V2 is the WASM‑compatible 32‑D backbone; the others are specialists at best). Treat the Cognitive Decoder as a separate, dataset‑gated training item. Do **not** re‑run contrastive FT (T‑034 already did it and lost).

### 2. Why this is first
Because V2 is the only model that is (a) WASM‑compatible, (b) self‑contained, (c) significantly better than v1, (d) contract‑matched to the prod pipeline (22/250/1000), and (e) already promoted to a registered id. The **only** thing blocking GA is a browser tail‑latency gate (Firefox) — everything else is gated behind that. The other models must first prove they survive a 22‑ch remap and then beat V2 on 50 subjects; none has.

### 3. Models to investigate next
1. **CBraMod** — highest point estimate (0.3233, 10‑subj); do the **19→22 remap + 50‑subj LOSO**. This is the single most promising "is there a server‑side specialist better than V2?" question.
2. **EEGNetv4 / ShallowFBCSPNet / Deep4Net** — export via `--architecture` + run the 50‑subj ablation. Cheap, informs architecture choice, may surface a fast small backbone.
3. **EEGPT** — only worth a 62→22 remap feasibility study **if** CBraMod's remap fails and a high‑capacity server rep is still wanted (low prior).
4. **LaBraM / FEMBA‑tiny** — 50‑subj validation *after* remap/contract fix; both are currently below PCA (low priority, likely retire).
5. **Cognitive Decoder** — source a labelled attention/workload/arousal dataset and train a real 3‑output model (training‑required; dataset‑gated).

### 4. Models to leave untouched
- **V2 weights/artifact** — do not retrain; do not re‑export until a specific ablation is scoped. The T‑034 negative result means the 32‑D head is *not* the problem.
- **LaBraM / FEMBA** routing — they are already gated out; do not route traffic. FEMBA's 30800‑vs‑385 contract bug is a *documented pre‑condition*, not a change to make now.
- **CBraMod** — do not touch the ONNX (it is correctly marked `wasmCompatible:false`); only run the remap eval.
- **PCA legacy** — leave as the SLA floor + baseline.
- **V1 production model** — leave as a reference artifact (external‑data ONNX is browser‑unsafe, which is the whole reason v2 exists); do not promote.

### 5. Experiments required
| # | Experiment | Type | Model(s) | Status |
|---|---|---|---|---|
| E1 | Firefox WASM latency fix (8‑bit quant retry / ORT‑Web tuning) | eval/abl. | V2 | P0, before GA |
| E2 | 19→22 remap ablation + 50‑subj LOSO | eval‑only | CBraMod | P1 (conditional) |
| E3 | Export EEGNetv4/Shallow/Deep4 → ONNX + 50‑subj ablation | eval‑only | EEGNet family | P3 |
| E4 | 62→22 remap feasibility + 50‑subj (batch) | eval‑only | EEGPT | P3 (only if E2 fails) |
| E5 | 16→22 remap + 1600→1000 window + 50‑subj | eval‑only | LaBraM | P4 |
| E6 | Fix FEMBA 30800↔385 contract; remap 200/1280→250/1000 + 50‑subj | eval‑only + 1 registry fix | FEMBA | P4 |
| E7 | Source labelled cognitive dataset + train 3‑output decoder | **training** | Cognitive Decoder | P3 |
| E8 | V2 cross‑domain validation (BCI‑IV‑2a holdout + non‑MI task) | eval‑only | V2 | P0 (after E1) |

### 6. Promotion criteria (per model)
- **V2 → GA:** Firefox P95 < 600 ms **and** Chromium P95 < 600 ms; determinism + SHA‑verify pass; fallback rate < 0.5%; Recall@10 maintained.
- **CBraMod → server‑specialist:** Remapped 22‑ch acc ≥ max(PCA, V2) with p<0.05 (Bonferroni) on 50 subjects; server routing wired on `wasmCompatible:false`.
- **Any challenger → interactive/browser:** WASM‑compatible + P95 latency gate + acc ≥ V2 (p<0.05, 50 subj). Currently **none** qualifies.
- **Cognitive Decoder → product:** acc/AUC vs heuristic, p<0.05 on held‑out; 3‑D output contract agreed.

### 7. Training required vs evaluation‑only
- **Evaluation‑only:** E1 (latency ablation), E2, E3, E4, E5, E6 (remap + protocol), E8 (cross‑domain validation).
- **Training‑required:** E7 (Cognitive Decoder — no labelled dataset exists yet); *optionally* future cross‑domain V2 FT (E8 follow‑up) — **not** contrastive on the same data (T‑034 already ruled that out).
- **Contract‑fix (not training):** FEMBA 30800↔385 (E6) — a one‑line registry/adapter fix, documented as a pre‑condition only.

### 8. Product opportunities enabled
1. **Embedding API (B2B)** — unlocked once V2 clears GA (E1). The pipeline (`upload → preprocess → embed() → PCA fallback`) is already wired; V2 is the upgrade path.
2. **Browser BCI inference SDK** — unlocked once V2 is self‑contained + Firefox latency fixed (E1). The WASM smoke tests + T‑035 staging prove the path works on Chromium; Firefox is the tail.
3. **Server‑side specialist route** — conditionally unlocked if CBraMod survives remap + 50‑subj (E2) and beats V2. This is the only model that *could* be a "quality‑at‑any‑cost" server embedder.
4. **Cognitive‑state decoder (attention/workload/arousal)** — unlocked only after E7 (dataset + training). Different product class from backbones.
5. **Architecture ablation service** — EEGNet export (E3) lets us offer "pick the smallest model that meets your accuracy/SLO budget" (small/fast EEGNetv4 vs accurate V2 vs high‑cap EEGPT).

### 9. Benchmark preservation plan
- **Canonical record:** `reports/benchmark_archive.json` (36 artefacts, 11 model entries, 3 FT experiments, 10 bug‑and‑corrections, rollout config) — the read‑only historical record. **Do not overwrite.**
- **Every future experiment** appends a record with: `experiment_id`, `config`, `dataset + exact splits`, `preprocessing`, `eval protocol`, `model/checkpoint/artifact (with sha)`, `metrics`, `comparison vs PCA`, `comparison vs V2`, `negative results`, `provenance`, `conclusion`.
- **Negative results are first‑class:** T‑034's contrastive‑lost‑to‑V2 result and FEMBA's INT8 instability must be cited, not hidden — they save real compute.
- **Artefact lineage:** `training/artefacts/eegconformer-physionet-v*` (v1/v2/v3 FT checkpoints + ONNX) and `eegconformer-t034-*` (aug/contrastive ablations) are the canonical weights; new FT runs get a new `eegconformer-physionet-v{n}` directory, **never** overwrite v2.
- **Protocol lock:** all 50‑subject work reuses `scripts/t032`-family protocol (LOSO 50 folds, train‑only PCA fit, separate candidate pools, no self‑retrieval, Bonferroni α). This guarantees apples‑to‑apples comparison vs the existing archive.
- **No discarding:** this report itself is the investigation record; it references existing artefact paths so nothing is lost.

---

## Prioritized roadmap (next model‑related missions, in order)

| P | Mission | Type | Depends on | Why here |
|---|---|---|---|---|
| P0 | **V2 Firefox WASM latency fix + GA promotion** | eval/abl. | — | The only blocker to shipping the backbone; everything else waits. |
| P0 | V2 cross‑domain validation (BCI‑IV‑2a + non‑MI holdout) | eval‑only | E1 | Proves V2 generalizes, not just MI‑fit; informs whether more FT is even worth it. |
| P1 | CBraMod 19→22 remap + 50‑subj LOSO | eval‑only | — | Single most promising "server specialist > V2" question. |
| P2 | EEGNetv4/Shallow/Deep4 ONNX export + 50‑subj ablation | eval‑only | — | Cheap architecture‑selection signal; may yield a fast small fallback tier. |
| P3 | EEGPT 62→22 remap feasibility (+ batch latency) | eval‑only | E2 fails | Only if we need a high‑cap server rep and CBraMod remap collapses. |
| P3 | Cognitive Decoder: source labelled dataset + train 3‑D head | **training** | — | Different product class; needs a dataset not yet in repo. |
| P4 | LaBraM 16→22 remap + 1600→1000 + 50‑subj | eval‑only | — | Currently below PCA; lowest prior. |
| P4 | FEMBA contract fix (30800↔385) + remap + 50‑subj | eval‑only + 1 registry fix | — | Worst accuracy + contract bug + large/slow; document, then retire if still < PCA. |

### Sequencing rules
- **Now (before any GA):** only P0 V2 latency fix + cross‑domain eval run.
- **After V2 GA:** P1 CBraMod remap study, then P2 EEGNet ablation, then P3 EEGPT/Cognitive‑Decoder in parallel.
- **Never pursue P3 EEGPT remap unless P1 CBraMod remap fails** (avoids duplicate "high‑cap server rep" work).
- **Do not start P4 (LaBraM/FEMBA) until P1/P2 done** — they are the lowest‑priority and both already sit below PCA.
- **No new contrastive FT on the 32‑D head** — T‑034 settled it (negative vs V2). Any future FT work targets cross‑domain data, not the representation‑loss objective.

*Repo remains clean; no production/artifact/registry/default/rollout changes were made in producing this report.*
