# Mission 6 — CBraMod 19→22 Channel Remap Study + 50-Subject LOSO Validation

**Status:** ✅ Complete — **negative result** (CBraMod does not beat V2 or PCA; not promoted/routed).
**Date:** 2026-08-14 · **Mission context:** V2 is the production GA default (`AI_EEGCONFORMER_ENABLED=ga`, 100% cohort; 24h staging soak **skipped** due to resource constraints — Mission 5). CBraMod was **not** modified, retrained, or deployed.

---

## 1. Objective & Evidence Gate

Per `reports/MODEL_STRATEGY_OTHER_MODELS.md`, CBraMod earns a **server-side specialist role** only if, on the locked T-032 50-subject LOSO protocol:

> **CBraMod acc ≥ PCA AND ≥ V2, with Bonferroni-corrected p < 0.05** (3 model-pair comparisons).

Otherwise it must **not** be promoted or routed. CBraMod remains `wasmCompatible:false` (blocks: `DFT`, `ReduceL2`) → it can only ever be server-side, never browser-side.

---

## 2. Remap Design (Option A — native montage from a shared raw source)

Both backbones are **fixed-shape ONNX**; neither can ingest the other's channel count, and retraining is forbidden:

- CBraMod ONNX: input `eeg[1, 19, 1000]`, **19 channels** fixed.
- EEGConformer v2 ONNX: input `input[1, 22, 1000]`, **22 channels** fixed.

So the remap study is **NOT** "make CBraMod 22-channel." It is a fair, leakage-free comparison in which **each model uses its native montage, both sourced from the same 64-channel PhysioNet raw trial**:

| Channel set | Count | Members |
|---|---|---|
| CBraMod-native (19) | 19 | FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, FZ, CZ, PZ |
| V2-prod (22) | 22 | FP1, FP2, F5, F6, F3, F4, F1, F2, FC5, FC6, FC3, FC4, C5, C6, C3, C4, T7, T8, P7, P8, P5, P6 |
| Shared (intersection) | 10 | FP1, FP2, F3, F4, C3, C4, T7, T8, P7, P8 |
| CBraMod-only | 7 | O1, O2, F7, F8, FZ, CZ, PZ |
| V2/prod-only | 12 | F5, F6, F1, F2, FC5, FC6, FC3, FC4, C5, C6, P5, P6 |
| Zero-filled | 0 | — |
| Interpolated | 0 | — |

**Assumptions (documented, marked explicitly):**
1. CBraMod and V2 are fixed-shape ONNX (19 / 22 inputs); retraining is forbidden → native-montage comparison is the only feasible fair design.
2. All 19 CBraMod channels and all 22 prod channels exist in the PhysioNet 64-channel layout, so no channel is dropped, zero-filled, or interpolated for either model — the 19↔22 gap *is* the remap study.
3. Preprocessing is identical for all (select channels → resample 160→250 Hz → bandpass 4–38 Hz → z-score per channel → central 1000-sample/4s window); only channel selection differs.
4. Metrics (nearest-centroid acc, Recall@K, Fisher) use cosine similarity and are dimension-agnostic; CBraMod is pooled to **200-D** (mean-tokens over `[1,19,5,200]`, matching `benchmark.ts`/T-030), V2/PCA are 32-D.
5. PCA is fit **per LOSO fold on training data only** (train-only, no leakage), `random_state=42` — stricter than T-032's global PCA fit; documented as the fair-comparison correction.
6. Recall@K uses a **train-only candidate pool with self-retrieval exclusion** (T-032 leakage fix, Bug T-030-1).
7. A clean 50-fold LOSO was used (T-032's `run_loso_evaluation` has a duplicate-append artifact inflating 50 folds to 100; this study uses a clean loop reusing T-032's metric functions).

---

## 3. Dataset & Protocol

- **Dataset:** PhysioNet EEGMMIDB 1.0.0, subjects **S001–S050** (50 subjects), runs **5 & 6** (4-class motor imagery: left_hand, right_hand, feet, tongue).
- **Trials:** 1,493 total (S014 had fewer events: 23 trials; all others 30). Label distribution: `[367, 376, 380, 370]`.
- **Protocol:** LOSO cross-validation, **50 folds** (one held-out subject per fold); nearest-centroid classification in cosine space; Recall@1/5/10 with train-only pool; Fisher's LDA on full-dataset embeddings.
- **Stats:** pairwise paired t-tests on 50 per-fold accuracies, **Bonferroni-corrected** over 3 model pairs (α = 0.05/3 ≈ 0.0167); plus Cohen's d.

---

## 4. Results

### 4.1 LOSO accuracy / retrieval (mean ± 95% CI over 50 folds)

| Model | Accuracy | ±95% CI | Recall@1 | Recall@10 | Macro-F1 | Fisher |
|---|---|---|---|---|---|---|
| **CBraMod @19 (200-D)** | **0.3043** | [0.2816, 0.3271] | 0.2646 | 0.9393 | 0.2119 | 0.0023 |
| **EEGConformer v2 @22 (32-D)** | **0.3250** | [0.3042, 0.3457] | 0.2922 | 0.9456 | 0.2854 | 0.0072 |
| **PCA bandpower @22 (32-D)** | **0.3065** | [0.2833, 0.3296] | 0.2742 | 0.9063 | 0.2315 | 0.0035 |

Chance (4-class) = 0.25. All models are modestly above chance; the PhysioNet 4-class task is a known hard benchmark (BCI-IV-2a pretrained baseline reached 0.587 val / 0.578 holdout on its native task).

### 4.2 Statistical comparisons (paired t-test on per-fold accuracy, Bonferroni-corrected ×3)

| Comparison | Δ (a−b) | t | p | Bonf. p | Cohen's d | Significant? |
|---|---|---|---|---|---|---|
| CBraMod vs V2 | −0.0206 | −1.593 | 0.118 | **0.353** | −0.225 | ❌ No |
| CBraMod vs PCA | −0.0021 | −0.200 | 0.842 | **1.000** | −0.028 | ❌ No |
| V2 vs PCA | +0.0185 | +1.323 | 0.192 | 0.576 | +0.187 | ❌ No (exploratory) |

Recall@1 deltas (Bonferroni p): CBraMod−V2 = −0.028 (p=0.227); CBraMod−PCA = −0.0097 (p=1.0).

### 4.3 Class separability (Fisher LDA, full dataset, cosine pairs)

| Model | Intra cos (mean) | Inter cos (mean) | Separation margin | Fisher |
|---|---|---|---|---|
| CBraMod | 0.9647 | 0.9644 | +0.00035 | 0.0023 |
| EEGConformer v2 | 0.9072 | 0.9037 | +0.00341 | 0.0072 |
| PCA bandpower | 0.0176 | 0.0021 | +0.0155 | 0.0035 |

V2 has the highest Fisher score (largest intra/inter separation). CBraMod's intra/inter cosine is nearly identical (margin 0.00035) — class clusters barely separate, consistent with its low accuracy.

### 4.4 Latency (onnxruntime CPU EP; server-side — CBraMod is not WASM-compatible)

| Model | Warm per-trial latency |
|---|---|
| CBraMod @19 | **60.74 ms** |
| EEGConformer v2 @22 | **7.42 ms** |
| PCA bandpower | **12.72 ms** (fit+transform estimate) |

---

## 5. Decision

**Evidence gate:** CBraMod acc ≥ PCA **AND** CBraMod acc ≥ V2, with Bonferroni p < 0.05.

- CBraMod (0.3043) is **below** V2 (0.3250) and **essentially tied** with PCA (0.3065) — and **neither** comparison is statistically significant after Bonferroni correction (p = 0.353 and 1.000, respectively).
- Cohen's d is small/negligible (−0.225 vs V2, −0.028 vs PCA).

> **→ DO NOT promote or route CBraMod. Negative result.**

CBraMod is **not** granted a server-side specialist role. The existing CBraMod artifact (`public/models/cbramod-encoder.onnx`, SHA `c128ccfd…`) is preserved unmodified; no code change routes traffic to it. V2 remains the production GA default (FP32, `AI_EEGCONFORMER_ENABLED=ga`, 100% cohort, PCA fallback + rollback intact).

---

## 6. Constraints Compliance

All Mission 6 hard constraints were honored (verified, no violations):

| Constraint | Status |
|---|---|
| V2 production path / rollout / `DEFAULT_PREFERRED` / `.env` unchanged | ✅ unchanged (`ga`) |
| No retrain of V2 (or CBraMod / any other model) | ✅ none |
| EEGPT / LaBraM / FEMBA / PCA code or artifacts untouched | ✅ untouched |
| CBraMod not deployed (stays `wasmCompatible:false`, not wired into `embedEEG`) | ✅ not deployed |
| Existing CBraMod artifact preserved (read-only) | ✅ SHA `c128ccfd…` verified at load |
| Canonical FP32 V2 artifact preserved (no INT8 introduced) | ✅ SHA `18644de1…` |
| 24h staging soak NOT faked / NOT marked complete | ✅ skipped (Mission 5), not claimed |
| All provenance archived | ✅ see §7 |

---

## 7. Provenance & Artifacts

- **Eval script:** `scripts/tmp/cbramod_remap_50subj.py` (reuses T-032 helpers via import; clean 50-fold LOSO; train-only PCA; Bonferroni).
- **Results JSON (machine-readable):** `reports/cbramod_remap_50subj_results.json` (full per-fold arrays, CIs, pairwise stats, provenance).
- **This report:** `reports/CBRAMOD_REMAP_50SUBJ_REPORT.md`.
- **Archive entry:** appended to `reports/benchmark_archive.json` → `experiments[]` id `cbramod-remap-50subj`; latent `promote_ga.sh` [3/6] manifest-traversal bug recorded in `bugs_and_corrections` (id `MISSION5-1`).
- **Dataset:** PhysioNet EEGMMIDB S001–S050 runs 5,6, cached at `/tmp/eegmmidb` (downloaded fresh in this run; 102 EDF files).
- **Model SHAs:** CBraMod `c128ccfdee06…` (verified); V2 `18644de187e9…` (verified, unchanged).
- **Git HEAD:** `b9164a664fce039df24c23656427a30c3a966926` (snapshot at run time).
- **Runtime:** onnxruntime 1.28.0 (CPU EP), MNE 1.12.1, scipy 1.18.0, scikit-learn 1.9.0, Python 3.13.

---

## 8. Next Mission

CBraMod is closed as a **negative** candidate under the documented evidence gate — no server-side specialist role, no routing. Next candidate per `MODEL_STRATEGY_OTHER_MODELS.md` would be re-evaluating whether any of EEGPT/LaBraM/FEMBA (WASM-compatible) can close the gap to V2 under the same 50-subject LOSO, or whether additional data/compute is required. V2 (GA) + PCA fallback remain the production ladder.
