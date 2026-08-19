# Mission 26 Reassessment — Fair Retrieval Evaluation of EEGPT

## Status: **EXTEND (Reopen)**

> **Decision: B — Reopen/extend M26.** EEGPT's original FAIL was based solely on MI classification (Gate B). All other backbone models were validated primarily on session-disjoint retrieval (R@K/MRR). On that identical protocol, EEGPT-2048 is non-inferior to the production Joint-264 (R@5=0.9511 vs 0.9467, p=0.716) and significantly better than CBraMod and V2. The MI-only gate was a methodological error — MI was a guardrail, not the primary representation metric.

---

## 1. Reassessment Objective

M26 originally evaluated EEGPT **only** on MI classification (4-class MI accuracy on 10 subjects). However, every other backbone model in the pipeline — CBraMod, V2, PCA, Joint-264 — was validated primarily on **session-disjoint retrieval** (R@1/R@5/R@10/MRR), with MI accuracy serving as a secondary safety-floor guardrail (signal presence ≥ chance=0.25, not a competitiveness gate).

This reassessment closes that methodological gap by running EEGPT through the **identical retrieval protocol** (M13/M18) used to justify all other models, enabling a direct apples-to-apples comparison.

---

## 2. Experimental Setup

### Dataset
- **PhysioNet EEGMMIDB 1.0.0** (S001–S010), runs {5, 6, 7, 8, 9, 10}
- **900 trials** (10 subjects × 6 runs × 15 trials per run)
- **60 session-disjoint splits** (10 subjects × 6 runs)
- For each split: query = 15 trials from one (subject, run); pool = all other 885 trials
- MI labels verified to match cache exactly (trial alignment confirmed)

### EEGPT Configuration (identical to M26 production path)
- **Model**: EEGPT ViT-INT8, SHA `a92daf44…` (verified ✓)
- **Input**: 62-channel layout, 22-channel production subset + zero-fill
- **Preprocessing**: 250 Hz, bandpass [1, 40] Hz, 1000 samples, z-score per channel
- **Pooling**: mean-token across 31 patch tokens → 2048-D
- **Inference time**: 900 trials in 895s (995ms/trial, no batch support)

### Baselines (from cache, same 10-subject subset, same trial alignment)
- CBraMod-200, V2-32, PCA-32, Joint-264 (block-weighted, M18 weights [0.62, 0.16, 0.22])

### Statistical Framework
- Paired t-test across 60 session-disjoint splits
- Bonferroni correction: α = 0.05/4 = 0.0125 (4 comparisons: EEGPT vs CBraMod, V2, PCA, Joint-264)
- Cohen's d for effect size
- 95% bootstrap CI (2000 resamples, seed=42)

---

## 3. Results

### 3.1 Retrieval Metrics (10 subjects, 60 splits)

| Model | Dim | R@1 | R@5 | R@10 | MRR |
|-------|-----|-----:|-----:|-----:|-----:|
| **EEGPT-2048** | 2048 | 0.7811 | **0.9511** | 0.9722 | **0.8534** |
| Joint-264 | 264 | 0.7989 | 0.9467 | **0.9733** | 0.8632 |
| PCA-32 | 32 | 0.7844 | 0.9189 | 0.9544 | 0.8443 |
| CBraMod-200 | 200 | 0.5522 | 0.8467 | 0.9189 | 0.6790 |
| V2-32 | 32 | 0.1722 | 0.5044 | 0.6967 | 0.3382 |

### 3.2 Statistical Comparisons (EEGPT vs baselines, Bonferroni α=0.0125)

| Comparison | ΔR@5 | p-value | Cohen's d | Significant? |
|------------|------:|--------:|----------:|:-------------|
| EEGPT vs CBraMod-200 | +0.1044 | 8.44×10⁻⁷ | 0.711 | ✅ **PASS** |
| EEGPT vs V2-32 | +0.4467 | 7.13×10⁻²⁸ | 2.571 | ✅ **PASS** |
| EEGPT vs PCA-32 | +0.0322 | 0.048 | 0.261 | ❌ FAIL (not sig.) |
| EEGPT vs Joint-264 | +0.0044 | 0.716 | 0.047 | ❌ FAIL (non-sig.) |

**MRR comparisons:**

| Comparison | ΔMRR | p-value | Cohen's d | Significant? |
|------------|------:|--------:|----------:|:-------------|
| EEGPT vs CBraMod-200 | +0.1743 | 1.28×10⁻¹⁰ | 1.005 | ✅ **PASS** |
| EEGPT vs V2-32 | +0.5152 | 1.25×10⁻³¹ | 3.032 | ✅ **PASS** |
| EEGPT vs PCA-32 | +0.0091 | 0.632 | 0.062 | ❌ FAIL (not sig.) |
| EEGPT vs Joint-264 | -0.0099 | 0.576 | -0.073 | ❌ FAIL (non-sig.) |

---

## 4. Key Findings

### Finding 1: EEGPT is non-inferior to Joint-264 on retrieval

EEGPT-2048's R@5 = 0.9511 is **statistically indistinguishable** from Joint-264's R@5 = 0.9467 (p = 0.716, Cohen's d = 0.047). This is remarkable: a single 2048-D ViT model matches the carefully learned block-weighted fusion of CBraMod-200 + V2-32 + PCA-32.

On MRR, EEGPT (0.8534) and Joint-264 (0.8632) are also statistically indistinguishable (p = 0.576).

### Finding 2: EEGPT significantly outperforms CBraMod and V2

EEGPT significantly outperforms both CBraMod-200 (ΔR@5 = +0.104, p = 8.4×10⁻⁷) and V2-32 (ΔR@5 = +0.447, p = 7.1×10⁻²⁸) on retrieval. This positions EEGPT as a superior standalone representation.

### Finding 3: EEGPT vs PCA-32 — numerically better, not statistically significant

EEGPT's R@5 = 0.9511 vs PCA-32's R@5 = 0.9189 (Δ = +0.032, p = 0.048). The un-corrected p-value is borderline (0.048), but it does NOT survive Bonferroni correction (α = 0.0125). On this 10-subject subset, EEGPT and PCA-32 are statistically tied for retrieval quality.

> **Note**: It is scientifically important that PCA-32 — a trivial 32-D bandpower baseline — is competitive with EEGPT's 2048-D ViT. The high dimensionality of EEGPT does not translate to a statistically significant retrieval advantage over a simple 32-D baseline on 10 subjects. This tempers any conclusion that EEGPT is categorically superior.

### Finding 4: MI accuracy is a guardrail, not a gate

M26's Gate B used MI classification accuracy as a competitiveness gate (EEGPT ≥ V2's 0.3428). However:

- **MI was never the primary metric** for model promotion. CBraMod's promotion gate (M11/M13) explicitly stated: *"MI accuracy is used ONLY as a signal-presence safety guardrail (CBraMod MI acc >= chance 0.25), NOT as a competitiveness gate."*
- EEGPT's 10-subject MI accuracy (0.2833) is **above chance** (0.25), satisfying the safety-floor requirement.
- The V2 baseline (0.3428) was computed on **50 subjects**, while EEGPT was evaluated on **10 subjects** — an apples-to-oranges comparison that inflated the apparent gap.

### Finding 5: The 10-subject caveat

All results here are on 10 subjects (60 splits), not 50 subjects (300 splits). With fewer subjects, the retrieval pool is smaller (885 vs 4485 pool trials), making the task easier. The M13/M18 results on 50 subjects (CBraMod R@5=0.528, V2 R@5=0.216, PCA R@5=0.692, Joint-264 R@5=0.786) are lower in absolute terms than the 10-subject subset. EEGPT's true 50-subject retrieval quality may be lower than the 0.9511 observed here.

---

## 5. Decision: **EXTEND M26**

### Rationale

The original M26 FAIL decision was **premature and methodologically inconsistent**:

| Aspect | M26 Original | Reassessment |
|--------|-------------|-------------|
| EEGPT metric | MI classification only | MI + retrieval (R@5=0.9511) |
| Baselines | V2 50-subj MI only | All models, 10-subj retrieval |
| Gate type | MI competitiveness | Retrieval competitiveness |
| Protocol alignment | ❌ EEGPT-only MI | ✅ Same protocol as all other models |

**Counterargument to the original FAIL**: The original report stated "EEGPT is dropped as a server-backbone candidate. Close the EEGPT remap thread." This was based on:
1. Gate B (MI accuracy 0.2833 < V2 0.3428) — but V2 was 50-subj, EEGPT was 10-subj
2. No retrieval evaluation — despite retrieval being the sole metric used to justify every other model

**Reassessment conclusion**: EEGPT's 2048-D representation matches Joint-264 on the canonical retrieval protocol. The original MI-only gate does not support dropping EEGPT as a representation backbone. The MI accuracy gap (0.2833 vs 0.3428) is:
- On different subject counts (10 vs 50)
- Using MI as a competitiveness metric, when MI was defined as a guardrail
- Against a metric not used to validate any other model

### Recommendation

**B. Reopen/extend M26.** Specifically:

1. **Extend M26** with the full 50-subject retrieval evaluation (feasible: ~62 min inference, EEGPT cache now available for 10 subjects as proof-of-concept).
2. **Re-weight the gates**: Gate A (representation preservation, cos=0.9747) ✅ PASS. The retrieval evaluation replaces Gate B as the competitiveness gate, with MI as a secondary safety-floor guardrail (EEGPT MI ≈ 0.2833 ≥ chance 0.25 ✅).
3. **Explore EEGPT as a 4th fusion block**: Since EEGPT (2048-D) matches Joint-264 (264-D) on retrieval, augmenting Joint-264 with EEGPT as a weighted 4th block may improve fusion quality further.

### Limitations (unchanged)

1. **10 subjects** (not 50): The 10-subject retrieval pool (885 trials) is smaller than the 50-subject pool (4485), making retrieval easier. Full 50-subject evaluation recommended.
2. **22-channel zero-fill**: EEGPT is evaluated with 22-channel production montage (40/62 channels zeroed), matching the production deployment path. Gate A confirmed cos=0.9747 (representation preserved).
3. **No batch inference**: EEGPT processes `[1, 62, 1000]` only (no batch dimension), at ~995ms/trial. 50-subject full evaluation would take ~62 minutes.

---

## 6. Artifacts

| Artifact | Path |
|----------|------|
| Results JSON | `reports/m26_retrieval_reassessment_results.json` |
| EEGPT embeddings cache | `reports/.m26_eegpt_retrieval_cache.npz` |
| Evaluation script | `scripts/tmp/m26_retrieval_reassessment.py` |
| Original M26 report | `reports/MISSION26_EEGPT_62TO22_REMAP_REPORT.md` |
| Original M26 results | `reports/m26_eegpt_remap_results.json` |
| Cross-session cache | `reports/.cbramod_cross_session_cache.npz` |

### Provenance

- **EEGPT**: SHA `a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36` (verified ✓)
- **CBraMod**: SHA `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47` (verified ✓)
- **V2**: SHA `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931` (verified ✓)
- **Trial alignment**: MI labels from PhysioNet extraction match cache order exactly (900/900)

### Constraints Honored

| Constraint | Status |
|-----------|--------|
| No training | ✅ |
| No fine-tuning | ✅ |
| No model modification | ✅ |
| No ONNX modification | ✅ |
| No artifact replacement | ✅ |
| No production rollout changes | ✅ |
| No historical benchmark rewrite | ✅ |

---

## 7. Summary Table: Retrieval Quality Comparison

```
R@5 (higher is better; Bonferroni-corrected significance vs EEGPT):

  1.00 ──────────────────────────────────────
       │              EEGPT-2048 ──● (0.9511)
       │              Joint-264  ──● (0.9467)
       │              PCA-32    ──○ (0.9189)  not sig. (p=0.048)
       │              CBraMod-200 ──▲ (0.8467) ** sig. (p=8.4e-7)
       │              V2-32     ──▼ (0.5044)  ** sig. (p=7.1e-28)
  0.50 ──────────────────────────────────────
       │              (50-subj M13 values for context: Joint-264=0.7856,
       │               PCA-32=0.6920, CBraMod=0.5276, V2=0.2158)
  0.00 ──────────────────────────────────────
```

EEGPT-2048 on 10-subject retrieval (R@5=0.9511) is non-inferior to Joint-264 (R@5=0.9467), the current production best. The original M26 FAIL decision based solely on MI classification is not supported by the fair retrieval evaluation.
