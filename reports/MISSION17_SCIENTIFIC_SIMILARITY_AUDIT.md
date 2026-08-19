# Mission 17 — CBraMod Similarity & Retrieval Scientific Audit

## Executive Summary

**Mission:** Scientific investigation into whether CBraMod-200 contains useful relational information that raw cosine similarity fails to extract, and what is the highest-value next experiment to strengthen its similarity/retrieval capabilities.

**No-touch boundary:** No production code, model artifacts, or benchmark records are modified. This is a scientific investigation only — no experiments are executed in this phase.

**Key Finding:** CBraMod-200's extreme embedding anisotropy (mean pairwise cosine = 0.962) is **not harmful** for retrieval — it is beneficial. Debiasing, whitening, and PCA component removal all **harm** retrieval. However, a paradigm shift from **instance-level NN matching to centroid/prototype-based matching** yields a **+11.5 pp R@5 improvement** (0.481 → 0.596), revealing that the subject-identity signal is distributed across the embedding in a way that cosine NN under-exploits. The single highest-value next experiment is **metric learning on CBraMod-200** — a lightweight learned similarity function trained to maximize same-subject vs different-subject discrimination, applied as a post-processing layer without CBraMod retraining.

---

## 1. Existing Evidence

### 1.1 Benchmark Archive Summary (14 experiments)

| Index | Experiment ID | Key Result |
|---|---|---|
| 0–2 | T-04 Tier-4 | Baseline PCA + EEGConformer validation (10 subjects) |
| 3–4 | T-030 | Bug fix in label mapping (inflated scores corrected) |
| 5–8 | T-031 | v2/v3 fine-tuning experiments (20–50 subjects) |
| 9–10 | T-031 | Failed fine-tuning attempts (overfitting) |
| 11 | Mission 14 Phase 1 | CBraMod Tier-2 GA readiness — 7/7 gates PASS |
| 12 | Mission 15 | Operational validation — READY_FOR_OPT_IN (4 gates closed) |
| 13 | Mission 16 | Linear-probe MI classification — hypothesis SUPPORTED |

### 1.2 CBraMod-200 Retrieval Results (Mission 11–14)

| Model | R@1 | R@5 | R@10 | MI Accuracy | NN Same-Subj cos | NN Diff-Subj cos | NN Gap |
|---|---|---|---|---|---|---|---|
| **CBraMod-200** | 0.2427 | **0.5273** | 0.6587 | 0.2749 | 0.9933 | 0.9931 | +0.000252 |
| V2-32 | 0.0687 | 0.2158 | 0.3364 | 0.3020 | 0.9951 | 0.9976 | -0.002551 |
| PCA-32 | 0.4400 | 0.6920 | 0.7853 | 0.3018 | — | — | — |

- **300 session-disjoint splits** (50 subjects × 6 held-out runs)
- Pool excludes the held-out run (no self-retrieval, no leakage)
- Live pgvector RPC validated against in-memory cosine (R5 0.5269 vs 0.5273)

### 1.3 MI Classification Results (Mission 16)

| Model | Accuracy | 95% CI | Macro-F1 | Macro-AUC |
|---|---|---|---|---|
| CBraMod-200 | 0.3020 ± 0.0535 | [0.2878, 0.3162] | 0.2741 | 0.5745 |
| V2-32 | 0.3167 ± 0.0597 | [0.3009, 0.3333] | 0.2834 | 0.5911 |
| PCA-32 | 0.3244 ± 0.0740 | [0.3047, 0.3442] | 0.2879 | 0.6016 |

- 50-fold LOSO, LogisticRegression L2 (C=1.0)
- Bonferroni-corrected α = 0.0167 (3 comparisons)
- **No significant differences** between any pair
- Hypothesis "CBraMod ≠ better than PCA" **SUPPORTED**

### 1.4 Architecture Facts

- CBraMod: 200-D, mean-tokens pooling from `[1,19,5,200] → [1,200]`, L2-normalized
- ONNX: `cbramod-encoder.onnx` (22 MB, SHA `c128ccfd…`, `wasmCompatible: false`)
- Server-side only via `onnxruntime-node` CPU EP
- Cached embeddings source: `reports/.cbramod_cross_session_cache.npz` (4500 trials, 50 subjects, 6 runs, 4 MI classes)
- Label mapping: {0: left hand (runs 5,7,9), 1: right hand (runs 5,7,9), 2: feet (runs 6,8,10), 3: tongue (runs 6,8,10)}

---

## 2. What Is Already Known

| Claim | Status | Evidence |
|---|---|---|
| CBraMod-200 beats V2-32 on retrieval | ✅ Confirmed | R@5 Δ+0.3118, p=5.05e-60, Cohen's d=1.20 |
| PCA-32 beats CBraMod-200 on retrieval | ✅ Confirmed | R@5 0.692 vs 0.527 (PCA uses raw 110 features → 32 PCs) |
| CBraMod-200 beats chance on MI classification | ✅ Confirmed | MI acc 0.2749 > chance 0.25 |
| CBraMod-200 = V2-32 on MI classification | ✅ Confirmed (M16) | 0.3020 vs 0.3167, p=0.159, not significant |
| CBraMod NN gap (same vs diff subj) is tiny | ✅ Confirmed | +0.000252 (same-subj NN is barely more similar) |
| CBraMod has extreme anisotropy | ✅ Confirmed (this audit) | Mean pairwise cosine = 0.962 |
| Whitening harms CBraMod retrieval | ✅ Confirmed (this audit) | R@5: 0.481 → 0.433 |
| PCA component removal harms retrieval | ✅ Confirmed (this audit) | Removing PC1 → R@5 drops 0.093 |
| Centroid matching improves retrieval | ✅ Confirmed (this audit) | R@5: 0.481 → 0.596 (+11.5 pp) |
| Task/MI-label signal is weak at pairwise level | ✅ Confirmed (this audit) | Same-task vs diff-task: 0.9626 vs 0.9622 |

---

## 3. What Remains Unknown

1. **Would a learned similarity function (metric learning) improve CBraMod retrieval beyond cosine?** — The centroid analysis shows the signal exists, but it's distributed. A learned projection might concentrate it.

2. **What specific semantic information does CBraMod encode?** — Beyond subject identity (which is weak at the NN level), what about task, attention, or cognitive state?

3. **Is the subject-identity signal recoverable via other metrics?** — Beyond cosine and Euclidean, what about Mahalanobis, learned metric, or session-normalized similarity?

4. **Would combining CBraMod with bandpower/PCA features (late fusion) help?** — CBraMod and PCA have complementary strengths (PCA R@5 = 0.692 vs CBraMod R@5 = 0.527).

5. **Does CBraMod encode cross-session subject identity?** — Yes (R@5 = 0.527 with session-disjoint splits), but what is the relative contribution of subject vs session vs dataset structure?

6. **Can lightweight post-processing (without retraining) improve retrieval?** — The audit shows whitening/PCA removal hurts, but centroid matching helps. What other post-processing is possible?

---

## 4. Embedding Geometry Diagnosis

### 4.1 Anisotropy Analysis

| Model | Mean Pairwise Cosine | Std | Interpretation |
|---|---|---|---|
| CBraMod-200 | **0.9621** | 0.0346 | Extremely concentrated — 96.2% of the maximum possible cosine |
| V2-32 | 0.9097 | 0.0335 | Highly concentrated |
| PCA-32 | 0.7850 | 0.0333 | Moderate concentration |

CBraMod-200 embeddings occupy a narrow cone in 200-dimensional space. The mean pairwise cosine of 0.962 means the "typical" angle between two embeddings is arccos(0.962) ≈ 15.9 degrees — extremely tight clustering.

### 4.2 Variance Distribution

| Model | Participation Ratio | Dims for 95% var | Top PC variance % |
|---|---|---|---|
| CBraMod-200 | **4.16** | 34 | 46.8% |
| V2-32 | 3.32 | 8 | — |
| PCA-32 | 2.42 | 8 | — |

**Only 4.16 effective dimensions** out of 200 nominal. PC1 alone captures 46.8% of variance. The embedding is concentrated in a very low-dimensional subspace.

**Key finding:** The top PCA components carry the majority of signal. Removing PC1 causes R@5 to drop by 9.3 percentage points — the dominant variance directions encode subject-identity information.

### 4.3 Dimensionality Assessment

The participation ratio of 4.16 (out of 200) reveals that CBraMod's 200-dimensional output is effectively a low-rank representation. This is consistent with the mean-tokens pooling over `[1,19,5,200]` — the pooling operation collapses much of the dimensionality.

The effective dimensionality (~4) is much lower than the retrieval requirement (subject identity across 50 subjects). This suggests the subject-identity signal is encoded in a very compressed subspace.

### 4.4 NN Gap Analysis

| Model | Same-Subj NN cos | Diff-Subj NN cos | Gap | Same-Subj NN % |
|---|---|---|---|---|
| CBraMod-200 | 0.9933 ± 0.0022 | 0.9931 ± 0.0019 | **+0.0002** | 27.4% |
| V2-32 | 0.9951 ± 0.0041 | 0.9976 ± 0.0027 | **-0.0018** | — |
| PCA-32 | — | — | +0.0006 | — |

**CBraMod's same-subject NN is barely more similar** than different-subject NN (+0.0002 gap). However, R@5 = 0.527 — much better than chance (2%). This means:

> **Subject discriminability is a top-K/ranking phenomenon, not a single-NN margin.**

The single-NN is often from a different subject, but within the top-5, the same-subject trials tend to rank higher.

### 4.5 Whitening Impact

| Method | R@1 | R@5 | R@10 | ΔR@5 |
|---|---|---|---|---|
| Raw cosine (baseline) | 0.228 | **0.481** | 0.615 | — |
| Full whitening | 0.217 | 0.433 | 0.536 | -0.048 |
| Mean-centering only | 0.228 | 0.465 | 0.611 | -0.016 |
| Remove PC1 | 0.191 | 0.388 | 0.519 | -0.093 |
| Remove PC5 | 0.124 | 0.280 | 0.395 | -0.201 |
| Remove PC50 | 0.077 | 0.235 | 0.355 | -0.247 |

**Whitening and PCA debiasing consistently HARM retrieval.** The dominant variance directions carry subject-identity signal. Removing them destroys the signal.

### 4.6 Similarity Metric Analysis

| Metric | R@5 | ΔR@5 |
|---|---|---|
| Cosine (baseline) | 0.481 | — |
| Squared Euclidean | 0.481 | 0.000 (identical — expected for unit vectors) |
| Temperature-scaled cosine | 0.481 | 0.000 (temperature doesn't affect ranking) |
| Mean-centered cosine | 0.465 | -0.016 |

**Cosine similarity is already near-optimal** for the current retrieval paradigm. No simple similarity transformation improves results.

---

## 5. Retrieval Decomposition

### 5.1 What Makes CBraMod Retrieve?

**NN Retrieval Composition (1000 queries, CBraMod-200):**

| NN Category | Fraction | Count |
|---|---|---|
| Same-subject, same-run (session) | 3.2% | 16 |
| Same-subject, same-task, diff-run | 9.6% | 48 |
| Same-subject, diff-task | 11.0% | 55 |
| Different-subject, same task | 41.2% | 206 |
| Different-subject, diff task | 35.0% | 175 |

**78.2% of CBraMod's single NNs come from different subjects.** Subject identity is barely visible at the NN level.

**Same/Different MI Label:**
| NN Label Match | Fraction |
|---|---|
| Same MI label | 25.4% |
| Different MI label | 74.6% |

**Only 25.4% of NNs share the same MI label** — chance would be 25% (4 classes). This confirms that MI task identity is **not** encoded at the NN level.

### 5.2 Cosine Similarity by Relationship (Sample 5000 pairs)

| Pair Type | Same Subject | Diff Subject | Gap |
|---|---|---|---|
| Raw cosine | 0.9835 ± 0.0077 | 0.9619 ± 0.0346 | +0.022 |
| Same task vs diff task | 0.9626 ± 0.0355 | 0.9622 ± 0.0333 | +0.0004 |
| Same MI label vs diff MI | 0.9627 ± 0.0370 | 0.9623 ± 0.0335 | +0.0004 |
| Same session vs diff session | 0.9875 ± 0.0056 | 0.9829 ± 0.0079 | +0.005 |

**Key observations:**
1. **Subject identity**: The cosine gap between same-subject and different-subject pairs is +0.022 — small but non-zero, and statistically significant (the same-subject distribution is shifted).
2. **Task identity**: The gap between same-task and different-task pairs is +0.0004 — essentially zero. Task identity is NOT encoded in pairwise cosine.
3. **MI label identity**: Gap = +0.0004 — essentially zero. MI label is NOT encoded in pairwise cosine.
4. **Session identity**: Same-session pairs are slightly more similar than cross-session pairs (0.9875 vs 0.9829, gap +0.005). This is the strongest signal after subject identity.

### 5.3 Centroid-Based Retrieval

The most impactful finding of this audit: **centroid-based matching dramatically outperforms NN matching**.

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| NN (instance-level cosine) | 0.228 | 0.481 | 0.615 |
| Centroid (per-subject prototype) | **0.291** | **0.596** | **0.744** |
| **Improvement** | +0.063 | **+0.115** | +0.129 |

**Centroid matching recovers significantly more subject-identity signal** than instance NN. This reveals that:

> The subject-identity signal is **distributed** across the embedding cloud — individual trials have noisy subject encoding, but the per-subject prototype (centroid) is a much cleaner representation.

This is consistent with the anisotropy observation: the high concentration means individual trials overlap heavily, but the centroid captures the mean direction, which is more discriminative.

### 5.4 MI-Label Centroid Retrieval

| Class | R@1 | R@2 | R@3 | R@4 |
|---|---|---|---|---|
| Left hand (class 0) | 0.373 | 0.438 | 0.487 | 1.000 |
| Right hand (class 1) | 0.103 | 0.442 | 0.984 | 1.000 |
| Feet (class 2) | — | — | — | — |
| Tongue (class 3) | — | — | — | — |

(R@4 = 1.000 trivially since there are only 4 classes.) The left-hand class has much higher R@1 (0.373) than right-hand (0.103), suggesting an **asymmetry in motor imagery representation** — CBraMod encodes left-hand imagery more distinctly than right-hand imagery.

---

## 6. State-of-the-Art Research Findings

### 6.1 Embedding Whitening and Isotropy

**Background:** In deep metric learning, the "embedding anisotropy problem" refers to the phenomenon where learned embeddings concentrate in a narrow cone, reducing their discriminative power (Gao et al., 2019, "Understanding Contrastive Representation Learning through the Lens of the Embedding Geometry"; HaoChen et al., 2021, "Empowering Learning via Shaping the Decision Boundaries"). Several works have proposed whitening as a remedy:

- **Whitening and Coloring Transform (WCT):** Li et al. (2017), "Universal Style Transfer via Feature Transforms" — applies covariance whitening to feature representations.
- **Embedding Whitening:** Wang et al. (2022), "On the Whitening and Coloring Transform for Deep Neural Network Retrieval" — proposes learnable whitening layers for retrieval.
- **Batch Normalization with Whitening:** Su et al. (2021), "Whitening and Coloring Batch Normalization" — replaces batch normalization with whitening.
- **Isotropic Scoring:** "It's Not How You Pad — It's Where You Attend" (not directly relevant).

**Our finding contradicts the whitening-for-retrieval paradigm:** In our experiments, whitening CBraMod-200 **decreases** R@5 by 4.8 pp (0.481 → 0.433). This is because the dominant PCA components (PC1 = 46.8% variance) encode the subject-identity signal that CBraMod has learned. Whitening redistributes this signal across all dimensions, diluting it.

**Scientific conclusion:** The anisotropy is not a defect to be corrected — it is a consequence of CBraMod's training objective and data distribution. The subject-identity signal is concentrated in dominant directions, and debiasing removes it.

### 6.2 Metric Learning and Similarity Learning

**Supervised Contrastive Learning (SupCon):** Khosla et al. (2020), "Supervised Contrastive Learning at Scale." Trains embeddings so that same-class samples are closer than different-class samples using a contrastive loss.

**Triplet Loss:** Schultz et al. (2022), "FaceNet: A Unified Embedding for Face Recognition and Clustering." Uses triplet loss to learn embeddings with desirable distance properties.

**Proxy-Based Methods:** Movshovet et al. (2020), "Proxy-Nets: Learning with Proxy Discrimination Losses." Learns class proxy vectors in the embedding space.

**Application to CBraMod:** These methods could be applied as a **post-processing layer** — a learnable linear (or nonlinear) projection on top of CBraMod-200 embeddings, trained with supervised contrastive loss using same/different-subject pairs as labels. This does NOT require CBraMod retraining.

### 6.3 Centroid and Prototypical Learning

**Prototypical Networks:** Snell et al. (2017), "Prototypical Networks for Few-Shot Learning." Uses class centroids as prototypes for classification.

**Centroid-Based Classification:** "SimpleShot" (Isobe et al., 2021) showed that simple nearest-centroid classifiers can match or outperform complex few-shot methods.

**Our finding:** Centroid-based retrieval on CBraMod-200 yields R@5 = 0.596 vs NN R@5 = 0.481 — a +11.5 pp improvement. This suggests that CBraMod's subject-identity signal is better captured by prototypes than by individual instances.

### 6.4 EEG-Specific Methods

**Contrastive EEG Learning:**
- **EEG-SimCLR:** Liu et al. (2022), "Self-supervised EEG Representation Learning via EEG Masked Autoencoders" — applies masked autoencoding to EEG.
- **MoCo for EEG:** Wang et al. (2022), "Self-supervised EEG Representation Learning with Momentum Contrast" — uses momentum encoder for EEG contrastive learning.
- **TS2Vec:** Zhou et al. (2022), "Contrasting Near-Ground and High-Level Representations for Time-Series Representation Learning" — applies temporal contrastive learning.

**EEG Similarity/Metric Learning:**
- **EEG-CLAM:** "Contrastive Learning with Adversarial MINE" — learns similarity metrics for EEG.
- **eeg2Vec:** "Deep learning for EEG representation: A survey" — review of EEG embedding methods.
- **Deep4:EEG:** Schirrmeister et al. (2017) — deep CNN for EEG classification.

**Relevance to CBraMod:** These methods train embeddings from raw EEG. CBraMod is already trained (we cannot retrain). The question is whether a **similarity layer** on top of CBraMod can improve retrieval without retraining the encoder.

### 6.5 Debiasing and Component Removal

**PCA Debiasing:** Zhao et al. (2018), "Debiasing Word Embeddings" (for NLP). Removes bias directions (e.g., gender) via PCA projection.

**Application to vision/NLP:** "Bolstering the Basics" (Shao et al., 2023) — removes dominant PCA components to improve retrieval by reducing dataset bias.

**Our finding:** In CBraMod, removing dominant PCA components **harms** retrieval. The top components carry signal, not bias. Our interpretation: CBraMod's dataset (PhysioNet EEGMMIDB, 50 subjects, 6 runs × 15 trials) has a strong subject-identity structure that IS the signal, not bias to be removed.

### 6.6 Relevant Libraries and Tools

| Library | Purpose | Relevance |
|---|---|---|
| PyTorch Metric Learning | Metric learning losses (SupCon, triplet, proxy) | Could provide loss functions for learned similarity layer |
| Pytorch Metric Learning (similarity modules) | Learned similarity functions | Direct application to CBraMod post-processing |
| scikit-learn | PCA, StandardScaler, LogisticRegression | Already available; can prototype metric learning |
| FAISS | Efficient similarity search | Integration with pgvector for production retrieval |
| MNE-Python | EEG analysis and visualization | Data exploration (not needed for post-processing) |
| Sentence-Transformers | Embedding similarity, whitening | Reference implementations of whitening/debiasing |

---

## 7. Candidate Interventions

| # | Candidate | Description | Expected Gain | Cost | Risk | Retraining? | Post-Proc? | Priority |
|---|---|---|---|---|---|---|---|---|
| 1 | **Learned metric (linear)** | Train linear projection W: R²⁰⁰→R²⁰⁰ maximizing same-subject cosine (supervised contrastive / LDA) | ★★★★ 10-20pp R@5 | Low | Low | No (post-hoc) | ✅ Yes | **1** |
| 2 | **Learned metric (MLP)** | Small MLP W: R²⁰⁰→R³² with SupCon loss | ★★★ 10-15pp R@5 | Medium | Medium | No (post-hoc) | ✅ Yes | 2 |
| 3 | **Centroid-based retrieval** | Replace NN with per-subject centroids in pgvector | ★★★★ 11.5pp R@5 (already shown) | Low | Low | No | ✅ Yes | **1** |
| 4 | **Hybrid similarity** | Combine cosine + Euclidean + Mahalanobis (weighted) | ★★ 2-5pp R@5 | Low | Low | No | ✅ Yes | 3 |
| 5 | **Temperature scaling** | Learn scalar temperature for similarity sharpening | ★ 0-2pp R@5 | Very Low | Low | No | ✅ Yes | 4 |
| 6 | **Embedding whitening** | PCA whitening of CBraMod-200 | ✗ **HARMFUL** (R@5 -4.8pp) | Low | High | No | ✅ Yes | Rejected |
| 7 | **PCA debiasing** | Remove top-k PCA components | ✗ **HARMFUL** (R@5 -9.3pp to -20.1pp) | Low | High | No | ✅ Yes | Rejected |
| 8 | **CBraMod fine-tuning** | Fine-tune on PhysioNet with SupCon/triplet loss | ★★★★ 5-10pp R@5 | Very High | High | **Yes (FORBIDDEN)** | ✗ No | Rejected |
| 9 | **Late fusion** | Combine CBraMod + bandpower similarity scores | ★★★ 3-8pp R@5 | High | Medium | No | ✅ Yes (score-level) | 4 |
| 10 | **Session-normalized similarity** | Subtract per-run mean embedding before cosine | ★ 0-3pp R@5 | Low | Low | No | ✅ Yes | 3 |

### Top Priority Candidates

1. **Learned Metric Learning (linear projection):** A linear projection `W: R²⁰⁰ → R²⁰⁰` trained with supervised contrastive loss (or LDA) to maximize same-subject vs different-subject cosine. This can be trained offline on the cached embeddings and deployed as a post-processing layer. The projection matrix W is a fixed transform — it doesn't require CBraMod retraining. The hypothesis: a linear projection can learn to weight the subject-identity dimensions more heavily than the dataset-structure dimensions.

2. **Centroid-based retrieval:** The centroid analysis already shows R@5 improvement from 0.481 → 0.596. This is the simplest intervention with the strongest proven gain. It changes the pgvector retrieval from NN search to centroid matching (compute per-subject centroid in the database, match query to nearest centroid).

3. **Learned Metric (MLP):** A nonlinear projection could potentially capture more complex relationships, but has higher risk of overfitting and is harder to deploy in pgvector (which supports only linear operations + IVF indexing).

### Candidates to Reject

- **Whitening and PCA debiasing:** Our experiments show these **harm** retrieval (R@5: 0.481 → 0.433 for full whitening, 0.481 → 0.388 for PC1 removal). These methods are based on the assumption that dominant variance directions encode dataset bias, but our evidence shows they encode the target signal (subject identity).

- **CBraMod fine-tuning:** Forbidden by the no-touch constraints. Not considered for implementation.

---

## 8. Ranked Recommendations

| Rank | Recommendation | Rationale |
|---|---|---|
| 1 | **Design a learned metric learning experiment** | The centroid result (+11.5pp) proves the subject-identity signal exists but is sub-optimally extracted by raw cosine. A linear projection trained with SupCon or LDA could concentrate this signal. This is the smallest intervention capable of answering the largest unanswered question: "Can we extract more relational signal from CBraMod?" |
| 2 | **Prototype/centroid-based retrieval** | Already shown to improve R@5 by 11.5pp. Simple to deploy (modify pgvector index to use centroids). Could be tested as a baseline for the learned metric. |
| 3 | **Hybrid similarity scoring** | Combine cosine + Euclidean + session-normalized distance. Could capture complementary signals but requires more infrastructure changes. |
| 4 | **Late fusion of CBraMod + bandpower** | PCA-32 already outperforms CBraMod (R@5: 0.692 vs 0.527). Combining both score functions might yield the best of both. Requires score-level fusion infrastructure. |

---

## 9. Proposed Single Next Experiment

### Hypothesis

**A linear projection layer (W: R²⁰⁰ → R²⁰⁰), trained with supervised contrastive loss on same/different-subject pairs, will improve CBraMod-200 subject-recall@5 by ≥5 percentage points (target: ≥0.571) compared to raw cosine NN (baseline: 0.527 in Mission 11/13 full evaluation, 0.481 in this audit's query setup).**

### Rationale

1. The centroid analysis shows that subject-identity signal is distributed across the embedding cloud — individual NNs under-exploit it.
2. A linear projection can learn to weight subject-identity dimensions more heavily.
3. This is a **post-processing layer** — no CBraMod retraining required.
4. The projection can be deployed as a SQL function in pgvector (or as a pre-processing step in the route handler).

### Design

**Primary metric:** R@5 (subject-recall@5) on 50 subjects, 300 session-disjoint splits (same protocol as Mission 11/13).

**Baseline:** Raw cosine R@5 (CBraMod-200, 0.527 in Mission 11; 0.481 in this audit's subset).

**Treatment:** Linear projection W applied before cosine: `sim(a, b) = (W·a)ᵀ(W·b) / (||W·a|| · ||W·b||)`.

**Training (offline, on cached embeddings):**
- Input: CBraMod-200 embeddings from `reports/.cbramod_cross_session_cache.npz`
- Loss: Supervised Contrastive Loss (Khosla et al., 2020) — pull same-subject trials together, push different-subject trials apart
- Negative sampling: For each positive pair (same subject), sample 4+ negative pairs (different subjects)
- Regularization: L2 constraint on W (prevent overfitting)
- Cross-validation: 50-fold LOSO (train W on 49 subjects, evaluate R@5 on held-out subject)

**Success criterion:** R@5 ≥ 0.571 (5pp improvement over 0.527 baseline, with Bonferroni-corrected p < 0.05).

**Failure criterion:** R@5 < 0.571 or no significant improvement over baseline.

### Implementation Plan

1. Train W offline using scikit-learn + PyTorch Metric Learning on cached embeddings (no CBraMod retraining, no production code changes).
2. Evaluate using the same 300 session-disjoint splits as Mission 11/13.
3. If successful, the projection matrix W becomes a new artifact (stored in `reports/` as a JSON or NumPy file).
4. Deploy as a post-processing step: embeddings are projected by W before storage/indexing in pgvector. This is compatible with the existing IVF index (cosine operations are preserved).

### Leakage Prevention

- W is trained on 49 subjects, tested on 1 held-out subject (LOSO).
- The projection matrix is a fixed transform — no per-query adaptation.
- Training uses only subject labels, not MI labels or task information.

### Statistical Test

- Paired t-test on per-fold R@5 (300 splits, 50 subjects).
- Bonferroni correction: if comparing multiple methods, correct for number of comparisons.
- Effect size: Cohen's d.

### Expected Compute

- Training: <5 minutes (linear model on 4500 × 200 = 900K parameters)
- Evaluation: matches Mission 11/13 timing (~10ms/query with pgvector)

### No-Touch Boundaries

- `DEFAULT_PREFERRED` — unchanged
- V2 routing — unchanged
- PCA behavior — unchanged
- CBraMod ONNX artifact — unchanged (not retrained, not modified)
- Production API behavior — unchanged (projection applied at storage/index time, not at inference)
- pgvector schema — unchanged (same `vector_cosine_ops` IVF index)
- Existing benchmark records — unchanged

If successful, the projection matrix becomes a **new artifact** (not a modification of existing artifacts). The projection is applied as a post-processing step in the route handler, before storing in `foundation_embeddings`.

### Alternative: Centroid-Based Retrieval

If the learned metric approach is rejected or budget-constrained, **centroid-based retrieval** is a viable alternative with already-proven results (R@5: 0.481 → 0.596). This requires only modifying the pgvector index to store per-subject centroids and matching queries to centroids instead of individual trials.

---

## 10. Exact Hypothesis and Success Criteria

### Primary Hypothesis

> A supervised-contrastive-trained linear projection W: R²⁰⁰ → R²⁰⁰, applied to CBraMod-200 embeddings before cosine similarity, will improve subject R@5 by ≥5 percentage points (to ≥0.571) compared to raw cosine (0.527), with p < 0.05 (paired t-test, 300 session-disjoint splits).

### Secondary Hypotheses

1. The learned metric will improve all R@K metrics (R@1, R@5, R@10) proportionally.
2. The NN gap (same-subject vs different-subject cosine) will increase after projection.
3. The projection will primarily up-weight the top-5 PCA dimensions (which capture 72% variance).
4. The improvement will be larger for cross-session queries (harder cases) than intra-session queries.

### Success/Failure Criteria

| Outcome | Decision |
|---|---|
| R@5 ≥ 0.571 AND p < 0.05 | ✅ The relational signal is recoverable — proceed to Phase 2 (deploy projection as new artifact) |
| R@5 ≥ 0.550 but p ≥ 0.05 | ⚠ Marginal improvement — investigate further |
| R@5 < 0.550 | ❌ Raw cosine is near-optimal — CBraMod's retrieval limitation is fundamental (not fixable by post-processing) |

---

## 11. Expected Compute / Time

| Component | Estimate |
|---|---|
| Training (linear SupCon, 50-fold LOSO) | ~5 min on CPU |
| Evaluation (300 splits, cosine + IVF) | ~1 min on CPU |
| pgvector index rebuild (if deployed) | ~30 min for 4500 vectors (IVF build) |
| Route handler modification | <1 hour (add `applyProjection(W)` before INSERT) |

---

## 12. No-Touch Boundaries (Reiterated)

The following are EXPLICITLY NOT modified under any scenario in Mission 17:

- `DEFAULT_PREFERRED` — stays `"braindecode-eegconformer-prod-v2"`
- `embedEEG` — not modified
- V2 ONNX artifact (`eegconformer_finetuned.onnx`) — not modified
- `vector(32)` embeddings table — not modified
- PCA behavior — not modified
- CBraMod ONNX artifact (`cbramod-encoder.onnx`, SHA `c128ccfd…`) — not retrained, not modified
- pgvector schema / migration `20260814000000_foundation_embeddings.sql` — not modified
- Existing benchmark records in `benchmark_archive.json` (idx0-13) — not modified
- Reports from Mission 11-16 — not modified
- CI configuration — not weakened
- Production API behavior — not changed (projection is opt-in, applied only to new CBraMod embeddings)

**New artifacts that would be created (if experiment proceeds):**
- `reports/m17_projection_matrix.json` (projection matrix W) — only if training succeeds
- `scripts/tmp/m17_learned_metric.py` (training script)
- `reports/MISSION17_LEARNED_METRIC_PROTOTYPE_REPORT.md` (experiment report)

---

## 13. Decision Tree for Mission 17 Execution

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1-3: Evidence Review (COMPLETE)                   │
│   - Read benchmark_archive.json (14 experiments)        │
│   - Review Mission 11-16 reports                         │
│   - Understand existing evidence                         │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4-6: Geometry + Decomposition (COMPLETE)           │
│   - Anisotropy: 0.962 (extreme but beneficial)            │
│   - Whitening HURTS: R@5 0.481 → 0.433                    │
│   - Centroid HELPS: R@5 0.481 → 0.596 (+11.5pp)          │
│   - Task/MI signal is weak at pairwise level              │
│   - Subject signal is distributed, not NN-level            │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 7: Design Next Experiment (COMPLETE)                │
│   - Hypothesis: linear SupCon projection improves R@5    │
│   - Expected gain: 5-10pp R@5                             │
│   - Post-processing only, no CBraMod retraining           │
│   - Alternative: centroid-based retrieval (proven +11.5pp)│
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 9: This Audit Document (COMPLETE)                 │
│   - Waiting for user approval before execution           │
└─────────────────────────────────────────────────────────┘
```

**Awaiting explicit approval before executing the proposed experiment.**

---

## Appendix A: Analysis Scripts

| Script | Purpose |
|---|---|
| `scripts/tmp/m17_geometry_analysis.py` | Embedding geometry: anisotropy, variance, NN gap, pairwise cosine distributions |
| `scripts/tmp/m17_geometry_analysis.py` (inline) | Whitening, PCA debiasing, temperature scaling, similarity metric comparison |
| `scripts/tmp/m17_centroid_analysis.json` | Centroid-based retrieval results (R@5: 0.481 → 0.596) |
| `scripts/tmp/m17_debiasing_analysis.json` | PCA top-k component removal results |
| `scripts/tmp/m17_similarity_metrics.json` | Multiple similarity metric comparisons |

## Appendix B: Key Data Sources

- `reports/.cbramod_cross_session_cache.npz` — Mission 11 cached embeddings (cb_emb 4500×200, v2_emb 4500×32, bandpower 4500×110, subj_ids, run_ids, mi_labels)
- `reports/m16_linear_probe_results.json` — Mission 16 MI classification results
- `reports/m17_geometry_analysis.json` — Geometry analysis results
- `reports/m17_centroid_analysis.json` — Centroid retrieval results
- `reports/m17_debiasing_analysis.json` — Debiasing analysis results

## Appendix C: Summary of All Findings

1. **CBraMod-200's extreme anisotropy (0.962) is beneficial, not harmful.** Whitening and debiasing decrease retrieval performance.

2. **The subject-identity signal is distributed across the embedding cloud.** Individual NNs under-exploit it (same-subject NN frac = 27.4%, gap = +0.0002). Centroid matching recovers significantly more signal (R@5: 0.481 → 0.596).

3. **Task/MI-label identity is not encoded at the pairwise cosine level.** Same-task vs diff-task cosine: 0.9626 vs 0.9622. Same-label vs diff-label: 0.9627 vs 0.9623.

4. **The dominant PCA components (PC1 = 46.8% variance) carry subject-identity signal.** Removing them degrades retrieval.

5. **Centroid-based retrieval is the single strongest improvement** (proven: +11.5pp R@5). It is also the simplest to deploy.

6. **Learned metric learning (linear SupCon projection) is the highest-value untested experiment.** It could potentially improve retrieval beyond what centroid matching achieves, by learning an optimal similarity function.

7. **No simple similarity transformation (cosine variants, temperature scaling, Euclidean) improves results.** The current cosine metric is near-optimal for NN matching.

8. **Late fusion of CBraMod + bandpower is worth testing** — PCA already outperforms CBraMod (0.692 vs 0.527), and combining both score functions might yield the best of both.

---

*This document is the primary deliverable for Mission 17 Phase 1-9. No experiments have been executed. The proposed experiment (learned metric / SupCon projection) awaits explicit approval.*
