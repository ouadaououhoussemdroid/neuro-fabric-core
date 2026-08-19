# Mission 27 — Augmented Joint-2312 with EEGPT-2048

## Status: **COMPLETED**

> **Verdict:** STRONG_SUCCESS — Joint-2312 significantly improves retrieval over Joint-264

---

## 1. Objective

Test whether adding EEGPT-2048 as a 4th fusion block to the production Joint-264
(CBraMod-200 + V2-32 + PCA-32) improves session-disjoint EEG representation retrieval.

**Motivation (from M26 Extended):** EEGPT-2048's 2048-D representation is non-inferior
to the production Joint-264 on the 50-subject retrieval protocol (R@5: 0.8118 vs 0.7858,
p=0.021). Since EEGPT matches Joint-264 standalone, it may provide complementary
representation information when fused, potentially exceeding both individually.

**Primary question:** Does Joint-2312 (learned 4-block weights) improve retrieval over
Joint-264 (fixed 3-block weights)?

---

## 2. Method

### Architecture

```
CBraMod-200 (200-D)  ──┐
V2-32 (32-D)          ──┤
PCA-32 (32-D)         ──┼── concat ──→ 2312-D
EEGPT-2048 (2048-D)   ──┘              ↓
                                    block weights
                                    ↓
                              L2-normalized
```

### Weight Learning (M18 methodology, extended to 4 blocks)

| Step | 3-block (M18) | 4-block (M27) |
|------|--------------|---------------|
| Input | 264-D joint | 2312-D joint |
| Train/test | 50-fold LOSO | 50-fold LOSO |
| Feature scaling | StandardScaler (train-only) | StandardScaler (train-only) |
| Classifier | RidgeClassifier (train-only) | RidgeClassifier (train-only) |
| Coef aggregation | mean(abs(coef)) per block | mean(abs(coef)) per block |
| Normalization | Non-negative, sum=1 | Non-negative, sum=1 |
| Block L2-norm | Before weighting | Before weighting |
| Final L2-norm | After weighting | After weighting |

### Constraints Honored

- **No training/fine-tuning:** All embeddings pre-computed and cached
- **No model/ONNX modification:** EEGPT, CBraMod, V2 artifacts read-only
- **No production changes:** No modifications to joint.server.ts, routes, or DB
- **No leakage:** Weights learned from training subjects only (49 of 50 per fold)
- **Bonferroni correction:** α = 0.05/4 = 0.0125 (4 comparisons)

---

## 3. Results

### Retrieval Quality (50 subjects, 300 splits)

| Model | Dim | R@1 | R@5 | R@10 | MRR |
|-------|-----|-----:|-----:|-----:|-----:|
| CBraMod-200 | 200 | 0.2427 | 0.5276 | 0.6587 | 0.3775 |
| V2-32 | 32 | 0.0687 | 0.2158 | 0.3364 | 0.1568 |
| PCA-32 | 32 | 0.4856 | 0.7404 | 0.8264 | 0.6016 |
| EEGPT-2048 | 2048 | 0.5391 | 0.8118 | 0.8867 | 0.6584 |
| Joint-264 (M18) | 264 | 0.5284 | 0.7858 | 0.8616 | 0.6425 |
| **Joint-2312 (learned)** | **2312** | **0.6438** | **0.8527** | **0.9060** | **0.7361** |
| Joint-2312 (fixed w) | 2312 | 0.6147 | 0.8376 | 0.8996 | 0.7146 |

### Baseline Reproduction (M18/M26 verification)

| Model | Recomputed R@5 | Expected (M18/M26) | Match? |
|-------|--------------:|----------------:|:------:|
| Joint-264 | 0.7858 | 0.7856 | ✅ |
| EEGPT-2048 | 0.8118 | 0.8118 | ✅ |
| CBraMod-200 | 0.5276 | 0.5276 | ✅ |
| V2-32 | 0.2158 | 0.2158 | ✅ |

---

## 4. Statistical Comparisons (paired t-test, Bonferroni α=0.0125)

### Primary: Joint-2312 (learned) vs baselines

| Comparison | ΔR@5 | p-value | Cohen's d | 95% CI (diff) | Sig.? |
|------------|-----:|--------:|----------:|---------------|:-----:|
| Joint-2312 vs Joint-264 | +0.0669 | 4.80e-28 | 0.704 | [+0.0558, +0.0776] | ✅ SIG |
| Joint-2312 vs EEGPT-2048 | +0.0409 | 9.22e-06 | 0.261 | [+0.0213, +0.0589] | ✅ SIG |
| Joint-2312 vs PCA-32 | +0.1122 | 2.77e-35 | 0.819 | [+0.0962, +0.1278] | ✅ SIG |
| Joint-2312 vs CBraMod-200 | +0.3251 | 6.27e-74 | 1.422 | [+0.2982, +0.3505] | ✅ SIG |

### Secondary: Fixed-weight vs Learned-weight Joint-2312

| Comparison | ΔR@5 | p-value | Sig.? |
|------------|-----:|--------:|:-----:|
| Fixed vs Learned | -0.0151 | 3.22e-06 | ✅ SIG |

---

## 5. Learned Block Weights Analysis

| Block | Mean Weight | Std | CV | Min | Max |
|-------|----------:|-----:|-----:|-----:|-----:|
| CBraMod-200 | 0.3062 | 0.0016 | 0.0052 | 0.3023 | 0.3089 |
| V2-32 | 0.1434 | 0.0015 | 0.0109 | 0.1381 | 0.1469 |
| PCA-32 | 0.1519 | 0.0007 | 0.0047 | 0.1504 | 0.1538 |
| EEGPT-2048 | 0.3985 | 0.0014 | 0.0035 | 0.3951 | 0.4012 |

> **CV (coefficient of variation) = std/mean.** Low CV indicates stable weight assignment
> across folds, suggesting the learned weights are robust rather than fold-specific noise.

---

## 6. Answering the Key Questions

1. **Joint-2312 R@1/R@5/R@10/MRR:** R@1=0.6438,
   R@5=0.8527,
   R@10=0.9060,
   MRR=0.7361

2. **Does Joint-2312 beat Joint-264?** ΔR@5 = +0.0669
   (Joint-2312 0.8527 vs Joint-264 0.7858)

3. **Is the improvement statistically significant?** p = 4.80e-28,
   ✅ Yes, significant after Bonferroni.
   Cohen's d = 0.704

4. **Does EEGPT provide complementary information?** Yes — Joint-2312 outperforms both Joint-264 and EEGPT-2048 alone, indicating complementary signal.

5. **What learned block weights were obtained?**
   CBraMod=0.3062, V2=0.1434, PCA=0.1519, EEGPT=0.3985

6. **Are the weights stable across folds?**
   Yes — low CV across all blocks

7. **Does EEGPT improve the representation or merely add redundant dimensions?**
   EEGPT provides complementary information

8. **Is Joint-2312 better than EEGPT-2048 alone?**
   Yes (ΔR@5 = +0.0409)

9. **Is Joint-2312 better than Joint-264 enough to justify productionization?**
   Yes — significant improvement

10. **Recommended next mission:** M28: Productionize Joint-2312 (extend M25 joint.server.ts with EEGPT block)

---

## 7. Verification

| Check | Status |
|-------|--------|
| EEGPT SHA-256 verified | ✅ |
| CBraMod SHA-256 verified | ✅ |
| V2 SHA-256 verified | ✅ |
| Trial alignment (4500/4500) | ✅ |
| No train/test leakage | ✅ |
| Block dims: 200/32/32/2048 | ✅ |
| Joint-2312 dim = 2312 | ✅ |
| Per-block L2 normalization | ✅ |
| Final L2 normalization | ✅ |
| Deterministic inference | ✅ |
| Weight learning train-only | ✅ |
| Bonferroni correction (α=0.0125) | ✅ |
| Historical records preserved | ✅ |
| M26 results preserved | ✅ |

---

## 8. Artifacts

| Artifact | Path |
|----------|------|
| This report | `reports/MISSION27_AUGMENTED_JOINT_2312_REPORT.md` |
| Results JSON | `reports/m27_augmented_joint_2312_results.json` |
| Evaluation script | `scripts/tmp/m27_augmented_joint_2312.py` |
| EEGPT cache | `reports/.m26_eegpt_50subj_cache.npz` |
| Cross-session cache | `reports/.cbramod_cross_session_cache.npz` |
| M18 results (reference) | `reports/m18_learned_joint_embedding_results.json` |
| M26 results (preserved) | `reports/m26_eegpt_50subj_retrieval_results.json` |
| M25 record (reference) | benchmark_archive.json → `m25-joint-264-production` |

### Provenance

- **EEGPT:** SHA `a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36` (verified ✅)
- **CBraMod:** SHA `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47` (verified ✅)
- **V2:** SHA `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931` (verified ✅)
- **M18 block weights:** CBraMod=0.6216, V2=0.1619, PCA=0.2165 (3-block, production Joint-264)
- **M27 learned 4-block weights:** CBraMod=0.3062, V2=0.1434, PCA=0.1519, EEGPT=0.3985

---

## 9. Constraints Honored

| Constraint | Status |
|-----------|--------|
| No training / fine-tuning | ✅ |
| No model modification | ✅ |
| No ONNX modification | ✅ |
| No artifact change | ✅ |
| No production rollout changes | ✅ |
| No historical benchmark rewrite | ✅ |
| No M26 results rewrite | ✅ |
| Train-only weight learning | ✅ |
| Session-disjoint evaluation | ✅ |

*Total evaluation time: 294.4s*
