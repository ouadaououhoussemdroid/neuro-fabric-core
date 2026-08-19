# Mission 20: Robust Validation of the Best Learned 264-D EEG Embedding

## Executive Summary

**Experiment:** Independent replication and robustness validation of the Mission 18 block-weighted 264-D embedding (R@5=0.7856) and Mission 19 C-shrinkage embedding (R@5=0.7860) on the 50-subject session-disjoint LOSO protocol.

**Verdict: PASS** — M18 block-weighted 264-D is robustly superior to raw concatenation (ΔR@5=+0.0271pp, p=4.45×10⁻⁹, d=0.088, Bonferroni α=0.025). M19 C-shrinkage does **not** significantly improve over M18 (ΔR@5=+0.0004pp, p=0.157, d=0.021). M18 block weighting is the canonical research embedding.

```
FINAL SUMMARY — Mission 20 (Independent Replication)
Embedding                          R@1    R@5    R@10   MRR
─────────────────────────────────────────────────────────────
Raw 264-D concat                   0.4891 0.7584 0.8364 0.6100
M18 block-weighted (3 params)      0.5271 0.7856 0.8622 0.6419  ← CANONICAL
M19 C-shrinkage (264 params)       0.5267 0.7860 0.8620 0.6414

Primary  (M18 vs Raw):     ΔR@5 = +0.0271, p = 4.45e-9,   d = 0.088  ✅ SIGNIFICANT
Secondary (M19 vs M18):    ΔR@5 = +0.0004, p = 0.157,    d = 0.021  ❌ NOT SIGNIFICANT

Decision: PASS (M18 robustly superior to raw; M19 adds no significant gain)
Recommendation: KEEP M18 — M19 C-shrinkage does not significantly improve (p=0.157)
```

**Key findings:**
1. ✅ **M18 block-weighted 264-D robustly outperforms raw 264-D** — exactly reproduced (R@5=0.7856, p=4.45×10⁻⁹)
2. ✅ **M19 C-shrinkage does NOT significantly improve over M18** — exactly reproduced (R@5=0.7860, p=0.157, d=0.021)
3. ✅ **Block weights are extremely stable across folds** — CV = 0.002-0.009 (near-zero variation)
4. ✅ **No representation zeroed out** — all blocks stay above 5% in every fold
5. ✅ **Reproducibility verified** — seed=42 gives identical results on re-run
6. ✅ **All SHAs verified** — CBraMod `c128ccfd…`, V2 `18644de1…` unchanged
7. ✅ **Weight-performance correlation is negligible** — learned weights do NOT correlate with fold performance (|r| < 0.08 for all blocks)

---

## 1. Experiment Protocol

### Objective

Independently replicate and robustly validate the Mission 18 block-weighted 264-D embedding and Mission 19 C-shrinkage embedding. Confirm that M18's advantage over raw concatenation is real and stable, and that M19 provides no statistically significant improvement.

### Dataset

- **Dataset:** PhysioNet EEGMMIDB 1.0.0
- **Subjects:** S001–S050 (50 subjects)
- **Runs:** 5, 6, 7, 8, 9, 10 (6 runs per subject, 15 trials each)
- **Total trials:** 4,500
- **Classes:** 4-class motor imagery (left hand, right hand, feet, tongue)

### Cross-Validation

- **50-fold LOSO** (one held-out subject per fold)
- **Session-disjoint retrieval:** Query = one run (15 trials) of held-out subject; Pool = all other trials (4,485 trials)
- **300 session-disjoint query configurations** (6 runs × 50 subjects), each with 15 individual trial-level evaluations = 4,500 total splits

### Embedding Construction

For every LOSO fold:
1. Load verified CBraMod-200, V2-32, PCA-32 from cache (`.cbramod_cross_session_cache.npz`)
2. Construct raw 264-D concatenation: `[CBraMod_200 ‖ V2_32 ‖ PCA_32]`, each block L2-normalized, then global L2-normalized
3. **M18**: Learn 3 block weights via `RidgeClassifier` on training subjects only (49 subjects)
4. **C-shrinkage**: Learn 264 per-dimension weights via 50/50 interpolation of Ridge per-dim and block-expanded weights (train-only)
5. Apply weights to ALL embeddings (train + test), then L2-normalize globally
6. Evaluate via session-disjoint nearest-neighbor retrieval (cosine similarity)

### Constraints

- **Seed:** 42
- **All weight learning:** Train subjects only per fold (no test-subject leakage)
- **Bonferroni correction:** 2 comparisons (primary: M18 vs Raw, secondary: M19 vs M18), α = 0.025
- **Bootstrap:** 2,000 resamples, seed=42

### SHA Verification

| Model | Artifact Path | SHA-256 | Verified |
|---|---|---|---|
| CBraMod | `public/models/cbramod-encoder.onnx` | `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47` | ✅ |
| V2 (EEGConformer) | `public/models/eegconformer_finetuned.onnx` | `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931` | ✅ |

---

## 2. Results

### 2.1 Candidate Embeddings

| Candidate | Dim | Params | R@1 | R@5 | R@10 | MRR | 95% CI (R@5) |
|---|---|---|---|---|---|---|---|
| **A. Raw 264-D concat** | 264 | 0 | 0.4891 | 0.7584 | 0.8364 | 0.6100 | [0.7456, 0.7707] |
| **B. M18 block-weighted** | 264 | 3 | 0.5271 | 0.7856 | 0.8622 | 0.6419 | [0.7731, 0.7980] |
| **C. M19 C-shrinkage** | 264 | 264 | 0.5267 | 0.7860 | 0.8620 | 0.6414 | [0.7740, 0.7984] |

### 2.2 Primary Comparison: M18 vs Raw 264-D

| Metric | Value |
|---|---|
| ΔR@1 | +0.0380 |
| ΔR@5 | **+0.0271** |
| ΔR@10 | +0.0258 |
| ΔMRR | +0.0319 |
| t-statistic | 5.878 |
| p-value | **4.45×10⁻⁹** |
| Cohen's d | **0.088** |
| 95% CI of difference | [+0.0184, +0.0362] |
| Significant after Bonferroni (α=0.025)? | **✅ Yes** |

### 2.3 Secondary Comparison: M19 C-shrinkage vs M18

| Metric | Value |
|---|---|
| ΔR@5 | +0.0004 |
| t-statistic | 1.414 |
| p-value | **0.157** |
| Cohen's d | **0.021** |
| 95% CI of difference | [0.000, +0.0011] |
| Significant after Bonferroni (α=0.025)? | **❌ No** |

### 2.4 Cross-Check with Historical M18/M19 Results

| Metric | M18/M19 Historical | M20 (this study) | Match? |
|---|---|---|---|
| M18 R@5 | 0.7856 | 0.7856 | ✅ Exact |
| M19 R@5 | 0.7860 | 0.7860 | ✅ Exact |
| M19 vs M18 p-value | 0.1573 | 0.1573 | ✅ Exact |
| M19 vs M18 Cohen's d | 0.0211 | 0.0211 | ✅ Exact |

All historical M18 and M19 values are **exactly reproduced** — confirming full reproducibility.

---

## 3. Fold-Level Weight Statistics (M18 Block Weights)

The M18 method learns 3 block weights per LOSO fold via RidgeClassifier. Here are the statistics across 50 folds:

| Block | Mean | Std | Median | Min | Max | CV | 95% CI |
|---|---|---|---|---|---|---|---|
| **CBraMod** | 0.6216 | 0.0015 | 0.6219 | 0.6179 | 0.6250 | 0.0024 | [0.6190, 0.6239] |
| **V2** | 0.1619 | 0.0014 | 0.1618 | 0.1576 | 0.1658 | 0.0085 | [0.1600, 0.1653] |
| **PCA** | 0.2165 | 0.0010 | 0.2166 | 0.2145 | 0.2188 | 0.0045 | [0.2147, 0.2182] |

**Key observations:**
- Weights match M18 historical values exactly (0.6216, 0.1619, 0.2165)
- **Extremely stable** across folds — CV is 0.24-0.85% (essentially zero variation)
- **No pathological weights**: no fold has any block below 1% or above 99%
- All representations remain meaningfully weighted in every fold

### Weight-Performance Correlation

| Block | Pearson r | p-value | Interpretation |
|---|---|---|---|
| CBraMod | -0.064 | 0.657 | Negligible |
| V2 | +0.078 | 0.588 | Negligible |
| PCA | -0.013 | 0.930 | Negligible |

**The learned block weights do NOT correlate with per-fold retrieval performance.** This means the optimal weighting is a globally stable solution — the same weights (0.62/0.16/0.22) work across all folds. There is no fold-specific adaptation needed.

---

## 4. Robustness Analysis

### 4.1 Fold-Level Dominance (M18 vs Raw)

| Metric | Value |
|---|---|
| Folds where M18 beats raw | 32/50 (64.0%) |
| Folds where M18 = raw | 10 (20.0%) |
| Folds where M18 loses to raw | 8/50 (16.0%) |
| Mean per-fold ΔR@5 | +0.0271 |
| Median per-fold ΔR@5 | +0.0111 |
| Min per-fold ΔR@5 | -0.078 |
| Max per-fold ΔR@5 | +0.133 |

M18 beats raw in 64% of folds, and the mean improvement is positive and significant. The 8 folds where M18 loses are within the expected distribution (the mean ΔR@5 is +0.027 with a CI that doesn't cross zero).

### 4.2 Bootstrap Confidence Intervals

| Method | R@5 | 95% CI (bootstrap) |
|---|---|---|
| Raw 264-D concat | 0.7584 | [0.7456, 0.7707] |
| M18 block-weighted | 0.7856 | [0.7731, 0.7980] |
| M19 C-shrinkage | 0.7860 | [0.7740, 0.7984] |

The CIs for M18 and raw concat **do not overlap** — confirming the improvement is robust. The CIs for M18 and M19 overlapping substantially confirms no significant difference.

### 4.3 Reproducibility

Re-running the entire M18 evaluation with seed=42 produces **identical** R@5 = 0.7856 (exact match to 15 decimal places). The experiment is fully deterministic and reproducible.

### 4.4 Representation Zeroing Check

| Block | Min weight (across all folds) | Max weight | All above 5%? |
|---|---|---|---|
| CBraMod | 0.6179 | 0.6250 | ✅ Yes |
| V2 | 0.1576 | 0.1658 | ✅ Yes |
| PCA | 0.2145 | 0.2188 | ✅ Yes |

No representation is ever zeroed out — all three contribute meaningfully to the joint embedding in every fold.

---

## 5. Fisher Discriminant Analysis

| Model | Fisher Ratio | Intra-class cos | Inter-class cos | Separation margin |
|---|---|---|---|---|
| Raw 264-D concat | 0.7022 | — | — | — |
| CBraMod-200 (raw) | 0.8300 | — | — | — |
| V2-32 (raw) | 0.2772 | — | — | — |
| PCA-32 | 0.7522 | — | — | — |

All representations encode subject identity above chance, with CBraMod having the highest Fisher ratio among individual representations.

---

## 6. Leakage Audit

| Check | Status |
|---|---|
| RidgeClassifier fit on train subjects only | ✅ Verified |
| StandardScaler fit on train subjects only | ✅ Verified |
| Test embeddings used only in retrieval pool | ✅ Verified |
| No test-subject weights influence pool | ✅ Verified |
| 50-folds LOSO with session-disjoint splits | ✅ Verified |
| SHA-256 verification of all model artifacts | ✅ Verified |
| No channel interpolation or zero-filling | ✅ N/A (cached embeddings) |

The code path for weight learning (`evaluate_candidate()`) calls `learn_block_weights(joint_raw[train_mask], subj_ids[train_mask])` where `train_mask` excludes the held-out subject. The StandardScaler and RidgeClassifier are fit ONLY on training subjects. Test embeddings are used exclusively in the retrieval pool.

---

## 7. Scientific Answer

### Question 1: Does M18 weighted 264-D consistently outperform raw 264-D?

**Yes, robustly.** ΔR@5 = +0.0271 (p = 4.45×10⁻⁹, d = 0.088, Bonferroni α=0.025). M18 beats raw in 64% of folds, and the bootstrap CIs do not overlap.

### Question 2: Are learned block weights stable across LOSO folds?

**Extremely stable.** Coefficient of variation is 0.24% (CBraMod), 0.85% (V2), and 0.45% (PCA). The weights are essentially constant across folds.

### Question 3: Is the M18 improvement robust under bootstrap CIs and paired testing?

**Yes.** Bootstrap 95% CI for M18 R@5: [0.7731, 0.7980] vs raw [0.7456, 0.7707] — no overlap. Paired t-test: p = 4.45×10⁻⁹.

### Question 4: Does M19 C-shrinkage provide meaningful improvement over M18?

**No.** ΔR@5 = +0.0004pp, p = 0.157, d = 0.021. The improvement is numerically positive but not statistically significant. Bootstrap CI of the difference includes zero [0.000, +0.0011].

### Question 5: Is M18 the canonical research embedding?

**Yes.** M18 block-weighted 264-D is:
- Significantly better than raw concat (p = 4.45×10⁻⁹)
- Not significantly beaten by M19 C-shrinkage (p = 0.157)
- The most parameter-efficient (3 params vs 264 for shrinkage)
- Weight-stable across folds (CV < 1%)
- Fully reproducible (seed=42 exact match)

### Question 6: Is weighting global or fold-specific?

**Global.** Weight-performance correlation is negligible (|r| < 0.08 for all blocks, all p > 0.58). The same global weights (CBraMod=0.62, V2=0.16, PCA=0.22) are optimal across all folds.

---

## 8. Conclusion

**Mission 20 validates Mission 18's block-weighted 264-D embedding as the canonical research embedding.**

### Decision: **PASS**

1. ✅ **M18 block-weighted 264-D robustly outperforms raw concat** — independently reproduced (R@5=0.7856, p=4.45×10⁻⁹)
2. ✅ **M19 C-shrinkage does not significantly improve** — independently reproduced (R@5=0.7860, p=0.157, d=0.021)
3. ✅ **Block weights are globally stable** — CV < 1% across folds, no representation zeroed
4. ✅ **M18 is parameter-efficient** — 3 parameters suffice, 264 parameters provide no significant gain
5. ✅ **Reproducible** — seed=42 gives identical results
6. ✅ **All SHAs verified** — no artifact tampering

### Recommendation: **KEEP M18 as the canonical research embedding**

M18's block-weighted 264-D concatenation (CBraMod=0.62, V2=0.16, PCA=0.22) is:
- The best validated actual embedding vector
- Statistically robust (significant improvement over raw concat)
- Parameter-efficient (3 vs 264 parameters)
- Globally stable (weights don't need fold-specific adaptation)
- Fully reproducible and leakage-free

M19 C-shrinkage (264 parameters) should **not** replace M18 — it provides no statistically significant improvement (p=0.157) at 88× the parameter cost.

---

## 9. Constraints Honored

| Constraint | Status |
|---|---|
| No model retraining (CBraMod, V2, or any other model) | ✅ |
| No artifact modification | ✅ |
| No ONNX modification | ✅ |
| No DEFAULT_PREFERRED change | ✅ |
| No V2 or PCA change | ✅ |
| No production code changes | ✅ |
| LOSO leakage-free | ✅ |
| Session-disjoint evaluation | ✅ |
| Train-only fitting | ✅ |
| Bonferroni correction | ✅ |
| Seed 42 reproducible | ✅ |
| SHA verified | ✅ |
| Prior archive records byte-preserved | ✅ |
| No CBraMod production promotion | ✅ |
| No similarity-score fusion | ✅ |
| Actual embedding vectors only | ✅ |

---

## 10. Deliverables

| Artifact | Path |
|---|---|
| Experiment script | `scripts/tmp/m20_embedding_robustness.py` |
| Results JSON | `reports/m20_embedding_robustness_results.json` |
| This report | `reports/MISSION20_EMBEDDING_ROBUSTNESS_REPORT.md` |
| Cache | `reports/.m20_embedding_robustness_cache.npz` |
| Archive append script | `scripts/tmp/_arc_m20.py` |

---

*Generated by `scripts/tmp/m20_embedding_robustness.py`*
*All model artifacts verified (SHA256 match). Embeddings loaded from verified caches.*
*Experiment conducted on 50 subjects × 6 runs × 15 trials = 4,500 trials, with 50-fold LOSO and session-disjoint evaluation.*
*Runtime: ~24s total (raw: 3.1s, M18: 7.5s, M19: 13.1s, stats: ~2s).*
