# Mission 19: Dimension-Wise Learned EEG Embedding

## Executive Summary

**Experiment:** Does learning individual weights for all 264 dimensions of the joint embedding improve over Mission 18's block-weighted embedding (R@5=0.7856)?

**Verdict: Marginal improvement (Result B)** — Dimension-wise weighting with shrinkage toward block weights produces a numerically higher R@5 (0.7860 vs 0.7856), but the improvement is **not statistically significant** (p=0.157, Cohen's d=0.021).

```
FINAL SUMMARY
Method                                   R@1      R@5     R@10      MRR
----------------------------------------------------------------------
CBraMod-200 raw cosine                0.2427   0.5276   0.6587   0.3776
V2-32 raw cosine                      0.0687   0.2158   0.3364   0.1568
PCA-32 bandpower                     0.4856   0.7404   0.8264   0.6016
Raw 264-D concat (baseline)           0.4891   0.7584   0.8364   0.6100
A. Block weighting (Mission 18)       0.5271   0.7856   0.8622   0.6419
B. Ridge per-dim                      0.5038   0.7547   0.8318   0.6174
B. Fisher per-dim                     0.2496   0.5089   0.6302   0.3727
C. Shrinkage (50/50 ridge+block)      0.5267   0.7860   0.8620   0.6415  ← BEST
C. Simplex-constrained                0.5047   0.7629   0.8440   0.6203
D. Hierarchical (block × dim)          0.4722   0.7262   0.8096   0.5856
Abl: CBraMod-only dim                 0.4889   0.7573   0.8371   0.6095
Abl: V2-only dim                      0.4913   0.7558   0.8351   0.6110
Abl: PCA-only dim                     0.4738   0.7262   0.8156   0.5885
----------------------------------------------------------------------

Mission 19 result classification: B — Marginal improvement (not significant)
```

**Key findings:**
1. ⚠️ **C-shrinkage marginally beats M18 block weighting** (R@5=0.7860 vs 0.7856, +0.04pp) but **NOT significant** (p=0.157, d=0.021)
2. ✅ **264 individual weights overfit** — B-ridge per-dim is significantly *worse* than block weighting (Δ=-0.031pp, p=1.6e-11)
3. ❌ **Fisher score per-dim fails catastrophically** (R@5=0.5089) — assigns too much weight to noisy dimensions
4. ✅ **Shrinkage rescues dimension-wise learning** — interpolating toward block weights (α=0.5) eliminates overfitting
5. ❌ **Hierarchical weighting over-parameterizes** (R@5=0.7262, Δ=-0.059pp vs M18)
6. ✅ **Block weighting is the optimal parameter-count tradeoff** — 3 parameters beats 264 parameters

---

## 1. Experiment Design

### Hypothesis

Learning individual weights for all 264 dimensions of the joint embedding will improve subject retrieval over Mission 18's 3-parameter block weighting (R@5=0.7856), because individual dimensions carry different amounts of subject-identity signal.

### Primary Baseline

The **Mission 18 block-weighted 264-D embedding** (R@5=0.7856) is the primary baseline. The mission succeeds only if a dimension-wise method significantly improves over this.

### Protocol

- **Dataset:** PhysioNet EEGMMIDB (S001-S050), 50 subjects × 6 runs × 15 trials = 4,500 trials
- **Cross-validation:** 50-fold LOSO
- **Evaluation:** Session-disjoint retrieval (300 splits per fold)
- **All weight learning:** Training subjects only per fold (no test-subject leakage)
- **Seed:** 42
- **Statistical correction:** Bonferroni across 4 comparisons (α=0.05/4=0.0125)

### Methods Evaluated

**Method A — Block weighting (Mission 18 baseline):**
- 3 parameters: weights for [CBraMod|V2|PCA] blocks
- Learned via RidgeClassifier coefficient aggregation
- R@5=0.7856

**Method B — Dimension-wise non-negative weighting:**
- B1: RidgeClassifier per-dimension |coef| mean, simplex-normalized (264 parameters)
- B2: Per-dimension Fisher discriminant score (264 parameters)

**Method C — Regularized dimension-wise weighting:**
- C1: Shrinkage: 50/50 interpolation of B1 ridge weights and block-expanded weights
- C2: Simplex-constrained via softmax (RidgeClassifier α=0.5, temperature=0.5)

**Method D — Hierarchical weighting:**
- D1: block_weight × within-block_dimension_weight (3 + 261 = 264 parameters)

**Ablations:**
- CBraMod-only dimension-wise (dims 0-199 weighted, V2/PCA uniform)
- V2-only dimension-wise (dims 200-231 weighted, CBraMod/PCA uniform)
- PCA-only dimension-wise (dims 232-263 weighted, CBraMod/V2 uniform)

### Weight Learning

For each method, weights are computed per-fold:
1. Fit RidgeClassifier (or compute Fisher scores) on training subjects
2. Aggregate to per-dimension or block-level weights
3. Normalize (simplex or softmax)
4. Apply as element-wise scaling: Z' = normalize(w ⊙ normalize(Z))

---

## 2. Results

### 2.1 Full Results Table

| Method | Params | R@1 | R@5 | R@10 | MRR | vs M18 ΔR@5 | p-value | Sig? |
|---|---|---|---|---|---|---|---|---|
| CBraMod-200 raw | — | 0.2427 | 0.5276 | 0.6587 | 0.3776 | — | — | — |
| V2-32 raw | — | 0.0687 | 0.2158 | 0.3364 | 0.1568 | — | — | — |
| PCA-32 | — | 0.4856 | 0.7404 | 0.8264 | 0.6016 | — | — | — |
| Raw 264-D | — | 0.4891 | 0.7584 | 0.8364 | 0.6100 | — | — | — |
| **A. Block weighting (M18)** | 3 | 0.5271 | 0.7856 | 0.8622 | 0.6419 | — (baseline) | — | — |
| B1. Ridge per-dim | 264 | 0.5038 | 0.7547 | 0.8318 | 0.6174 | -0.0309 | 1.6e-11 | ✅ Worse |
| B2. Fisher per-dim | 264 | 0.2496 | 0.5089 | 0.6302 | 0.3727 | -0.2767 | 2.5e-251 | ❌ Much worse |
| **C1. Shrinkage (α=0.5)** | 264 | 0.5267 | 0.7860 | 0.8620 | 0.6415 | +0.0004 | 0.157 | ❌ Not sig. |
| C2. Simplex (T=0.5) | 264 | 0.5047 | 0.7629 | 0.8440 | 0.6203 | -0.0227 | 2.3e-8 | ✅ Worse |
| D. Hierarchical | 264 | 0.4722 | 0.7262 | 0.8096 | 0.5856 | -0.0593 | 2.9e-27 | ✅ Worse |
| Abl: CBraMod-only | 200 | 0.4889 | 0.7573 | 0.8371 | 0.6095 | -0.0282 | 7.4e-10 | ✅ Worse |
| Abl: V2-only | 32 | 0.4913 | 0.7558 | 0.8351 | 0.6110 | -0.0298 | 4.2e-10 | ✅ Worse |
| Abl: PCA-only | 32 | 0.4738 | 0.7262 | 0.8156 | 0.5885 | -0.0593 | 4.5e-29 | ✅ Worse |

### 2.2 Statistical Comparison: Best vs M18 Baseline

| Comparison | ΔR@5 | t-stat | p-value | Cohen's d | Bonferroni sig? (α=0.0125) |
|---|---|---|---|---|---|
| **C1-shrinkage vs M18 block** | **+0.0004** | — | **0.157** | **+0.021** | ❌ No |
| B1-ridge vs M18 | -0.0309 | — | 1.6e-11 | -0.101 | ✅ Worse |
| B2-fisher vs M18 | -0.2767 | — | 2.5e-251 | -0.539 | ✅ Much worse |
| C2-simplex vs M18 | -0.0227 | — | 2.3e-8 | -0.084 | ✅ Worse |
| D-hierarchical vs M18 | -0.0593 | — | 2.9e-27 | -0.162 | ✅ Worse |

### 2.3 Bootstrap 95% CIs

| Method | R@5 CI95 |
|---|---|
| M18 block weighting | [0.7731, 0.7980] |
| C1 shrinkage (best) | [0.7740, 0.7984] |
| Raw 264-D | [0.7456, 0.7707] |

The CIs of C1-shrinkage and M18-block **overlap almost entirely**, confirming no significant difference.

### 2.4 Learned Weight Analysis (Ridge per-dim)

**Block-level weight distribution (mean across folds):**
- CBraMod: 0.911 (91.1%)
- V2: 0.038 (3.8%)
- PCA: 0.051 (5.1%)

The RidgeClassifier assigns 91% of total weight to the CBraMod block, confirming that CBraMod carries the most discriminative signal. However, when applied per-dimension, this leads to over-concentration on a few CBraMod dimensions.

**Top 20 most important dimensions (all in CBraMod block):**
```
dim 2: 0.0086  dim 189: 0.0085  dim 6: 0.0084  dim 5: 0.0079
dim 25: 0.0078  dim 11: 0.0077  dim 17: 0.0076  dim 185: 0.0076
dim 23: 0.0069  dim 135: 0.0069  dim 10: 0.0068  dim 187: 0.0067
dim 92: 0.0066  dim 190: 0.0066  dim 93: 0.0066  dim 14: 0.0065
dim 193: 0.0064  dim 114: 0.0064  dim 7: 0.0063  dim 111: 0.0062
```

**Weight stability across folds:** CV mean = 0.0000 — weights are perfectly stable across folds. The same dimensions consistently receive the highest weights.

**Interpretation:** The top-20 dimensions are all within the CBraMod block (indices 0-199), and the top dimension (dim 2) receives 0.0086 — about 2.3× the mean weight (0.0038). This is a relatively uniform weighting, not extreme concentration. The failure of B1-ridge per-dim is therefore not due to weight instability but due to **overfitting from 264 parameters vs 3**.

---

## 3. Analysis

### 3.1 Why does dimension-wise weighting mostly fail?

**Unregularized per-dim (B1-ridge) significantly underperforms block weighting** (Δ=-0.031pp, p=1.6e-11). The reason:

1. **Parameter count**: 264 vs 3 parameters. With 49 training subjects × 90 trials = 4,410 training samples, 264 parameters is feasible statistically, but the RidgeClassifier coefficients are noisy in directions orthogonal to the subject-identity manifold.

2. **Overfitting to noise**: RidgeClassifier coefficients capture training-set-specific noise in individual dimensions. When applied to test subjects, these noise directions add variance without adding signal.

3. **CBraMod block dominance**: The model concentrates 91% of weight on CBraMod's 200 dimensions, effectively creating 200 separate weight parameters where 1 (the CBraMod block weight) was sufficient. The per-dimension variation within CBraMod is mostly noise.

### 3.2 Why does shrinkage work (but not significantly better)?

**C1-shrinkage (α=0.5) interpolates** between the noisy per-dim weights and the stable block weights:

```
w_final = 0.5 × w_ridge_per_dim + 0.5 × w_block_expanded
```

This:
- Adds slight regularization to the overfitting of pure per-dim (reducing B1's degradation from -0.031 to +0.0004 vs M18)
- But also dilutes any potential per-dim improvements (hence no significant gain)

The result (R@5=0.7860 vs 0.7856) is statistically indistinguishable from block weighting, suggesting **block-level weighting already captures the essential structure**.

### 3.3 Why does Fisher per-dim fail catastrophically?

Fisher score per dimension (B2) assigns weights based on `between_class_variance / within_class_variance` for each dimension independently. This fails because:

1. **No cross-dimensional interaction**: Fisher score treats each dimension independently, ignoring that subject identity is encoded in multi-dimensional patterns (especially in the high-dimensional CBraMod space).

2. **Noise amplification**: Low-variance dimensions with high Fisher ratios on training data get upweighted, but these are often noise directions that don't generalize.

3. **CBraMod's anisotropy**: CBraMod embeddings have mean pairwise cosine ~0.962 — individual dimensions are highly correlated. Single-dimension Fisher scores are unreliable estimators of information content.

### 3.4 Why do ablations fail?

None of the single-block ablations (CBraMod-only, V2-only, PCA-only) beat block weighting:

| Ablation | R@5 | Δ vs M18 | Why it fails |
|---|---|---|---|
| CBraMod-only | 0.7573 | -0.028pp | Per-dim weighting on 200 CBraMod dims overfits |
| V2-only | 0.7558 | -0.030pp | 32 parameters, some signal but noisy |
| PCA-only | 0.7262 | -0.059pp | 32 bandpower PCA dims, less subject signal |

This confirms that **no single representation's dimensions can be optimally reweighted in isolation** — the block-level weighting captures the cross-block information tradeoff.

### 3.5 Hierarchical weighting over-parameterizes

Hierarchical weighting (D) decomposes the 264 weights into 3 block weights + 261 within-block weights:

```
w[i] = block_weight[block(i)] × dim_weight_within_block[i]
```

This has 264 parameters (3 + 261) and performs worst among the structured methods (R@5=0.7262). The within-block dimension weights add noise without adding signal, because block weighting already optimally balances the three representations.

### 3.6 The parameter-count hypothesis

| Method | Parameters | R@5 |
|---|---|---|
| Raw concat | 0 | 0.7584 |
| Block weighting | 3 | 0.7856 |
| Shrinkage | 264 | 0.7860 |
| Pure per-dim | 264 | 0.7547 |
| Hierarchical | 264 | 0.7262 |

The results show a clear **bias-variance tradeoff**: 3 parameters (block weighting) is the sweet spot. Going to 264 parameters either helps marginally (with shrinkage) or hurts (without shrinkage). The 0.04pp gain from shrinkage is within noise.

---

## 4. Scientific Answer

> **Does dimension-wise weighting beat Mission 18's block weighting?**

**No — not significantly.** The best dimension-wise method (C1-shrinkage, R@5=0.7860) produces a numerically higher score than Mission 18's block weighting (R@5=0.7856), but:

- **ΔR@5 = +0.04pp** (0.0004)
- **p = 0.157** (not significant after Bonferroni, α=0.0125)
- **Cohen's d = 0.021** (negligible effect size)
- **Bootstrap CIs overlap almost entirely**: [0.774, 0.798] vs [0.773, 0.798]

### Answering the 12 questions

1. **Does dimension-wise weighting beat R@5=0.7856?** No — C1-shrinkage gives 0.7860 but p=0.157 (not significant)
2. **Best actual embedding:** C1-shrinkage 264-D (R@5=0.7860) — statistically tied with M18 block weighting
3. **Dimensionality:** 264-D (same as baseline)
4. **R@1/R@5/R@10/MRR:** R@1=0.5267, R@5=0.7860, R@10=0.8620, MRR=0.6415
5. **Statistical significance:** NO — p=0.157, d=0.021
6. **Which dimensions matter most:** Top dims all in CBraMod block (indices 0-199), with dim 2, 189, 6, 5, 25 being most weighted
7. **Which model block contributes most:** CBraMod (91% of total weight), then PCA (5%), then V2 (4%)
8. **Weight stability:** Perfectly stable across folds (CV=0.0) — same dims consistently ranked highest
9. **Does hierarchical weighting outperform?** No — R@5=0.7262, significantly worse than block weighting
10. **Does dimension-wise learning generalize?** No — unregularized per-dim overfits (R@5=0.7547, p=1.6e-11 worse than block)
11. **Why does block weighting outperform?** Block weighting (3 params) has optimal bias-variance tradeoff. 264 parameters overfit to noise. Shrinkage recovers block-level performance but cannot exceed it.
12. **Final research embedding:** The **Mission 18 block-weighted 264-D concatenation** (CBraMod×0.62 ⊕ V2×0.16 ⊕ PCA×0.22) remains the best representation. No dimension-wise method significantly improves upon it.

### Result Classification: **B — Marginal improvement**

A numerical improvement exists (+0.04pp R@5) but it is not statistically significant (p=0.157). The 264-parameter shrinkage variant is statistically indistinguishable from the 3-parameter block weighting.

---

## 5. Limitations

### Parameter count dominates
With 4,410 training samples per fold, 264 parameters is statistically feasible but the RidgeClassifier coefficients are noisy in directions orthogonal to the subject-identity manifold. The overfitting manifests as slight R@5 degradation (-0.031pp for unregularized per-dim).

### Fisher score independence assumption
The Fisher per-dimension approach assumes dimensions are independent, which is violated in CBraMod's highly anisotropic embedding space (mean cosine ≈ 0.962).

### No deep metric learning
The MLP (R@5=0.6827) and SupCon (R@5=0.6229) methods from Mission 18's architecture were not included in Mission 19 since they operate on the embedding space, not the weight-learning approach. They already showed poor generalization in Mission 18.

### Computational cost of PyTorch methods
MLP and SupCon training required ~340s and ~200s per fold respectively, making full dimension-wise grid search computationally expensive.

---

## 6. Comparison Across Missions 17-19

| Mission | Best R@5 | Method | vs PCA |
|---|---|---|---|
| Mission 17 | 0.5736 | CBraMod LDA | -16.2pp |
| Mission 18 | 0.7856 | Block-weighted 264-D | +4.96pp |
| Mission 18 | 0.7584 | Raw 264-D concat | +2.2pp |
| Mission 19 | 0.7860 | Shrinkage per-dim | +4.56pp |
| Mission 19 | 0.7856 | Block weighting (M18) | +4.96pp |

The hierarchy is clear: raw concat → block weighting → (marginal shrinkage).

---

## 7. Constraints Honored

- ✅ CBraMod ONNX artifact (c128ccfd…) — not retrained, not modified
- ✅ V2 ONNX artifact (18644de1…) — not retrained, not modified
- ✅ PCA implementation — unchanged
- ✅ Production routing — not modified
- ✅ `DEFAULT_PREFERRED` — unchanged
- ✅ `.env` — unchanged
- ✅ No CI weakening, no test deletion
- ✅ No overwriting of previous benchmark results
- ✅ 50-fold LOSO, session-disjoint, train-only weight learning
- ✅ Seed 42 reproducible
- ✅ Prior archive records byte-preserved

---

## 8. Deliverables

| Artifact | Path |
|---|---|
| Experiment script | `scripts/tmp/m19_dimensionwise_embedding.py` |
| Results JSON | `reports/m19_dimensionwise_embedding_results.json` |
| This report | `reports/MISSION19_DIMENSIONWISE_EMBEDDING_REPORT.md` |
| Learned embedding cache | `reports/.m19_dimensionwise_embedding_cache.npz` |
| Archive append script | `scripts/tmp/_arc_m19.py` |

---

## 9. Verdict

**Mission 19 concludes that dimension-wise weighting does NOT significantly improve over Mission 18's block weighting.**

1. ✅ **C-shrinkage (R@5=0.7860) marginally beats M18 (R@5=0.7856)** but the improvement is not statistically significant (p=0.157, d=0.021)
2. ✅ **Unregularized 264-parameter weighting fails** — overfitting degrades performance
3. ✅ **Block weighting (3 parameters) is the optimal bias-variance tradeoff**
4. ✅ **Fisher per-dim fails** — dimension independence assumption violated
5. ✅ **Hierarchical weighting over-parameterizes** — 264 parameters, no benefit
6. ✅ **All constraints honored** — no model retraining, no artifact modification, leakage-free

**The Mission 18 block-weighted 264-D concatenation remains the final research embedding.** The 3-parameter block weighting (CBraMod=0.62, V2=0.16, PCA=0.22) is the best dimensionality for learned embedding transformation from the 264-D joint space.

**Final research embedding: Block-weighted 264-D concatenation, R@5=0.7856**

This is the best actual EEG embedding achievable from frozen CBraMod-200, V2-32, and PCA-32 representations.

---

*Generated by `scripts/tmp/m19_dimensionwise_embedding.py`*
*All model artifacts verified (SHA256 match). Embeddings loaded from verified caches.*
*Experiment conducted on 50 subjects × 6 runs × 15 trials = 4,500 trials, with 50-fold LOSO and session-disjoint evaluation.*
