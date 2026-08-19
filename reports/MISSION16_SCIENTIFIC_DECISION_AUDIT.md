# Mission 16 — Scientific Decision Audit

**Phase:** Decision/audit only — no repository modifications, no execution.  
**Objective:** Determine the highest-value next scientific/engineering question for Neuro Fabric based on accumulated Missions 11–15 evidence.

---

## 1. Current Evidence Summary

### 1.1 Model landscape (PhysioNet EEGMMIDB, 4-class MI: left hand, right hand, feet, tongue)

| Model | Dim | Retrieval R5 | MI Acc (nc) | Latency | WASM | Trained on |
|---|---|---|---|---|---|---|
| **PCA bandpower** | 32 | **0.692** | 0.302 | ~0 ms | yes | handcrafted |
| **CBraMod-200** | 200 | 0.527 | 0.275 | 64–155 ms | NO (DFT/ReduceL2) | BCI-IV-2a (pretrained) |
| **EEGConformer V2-32** | 32 | 0.216 | 0.302 | 6.73 ms | yes | 40 PhysioNet subjects |
| **EEGConformer V1-32** | 32 | 0.253 | 0.317 | 9.8 ms | yes | BCI-IV-2a (pretrained) |
| EEGPT | 2048 | 0.283 | 0.307 | 4820 ms | yes (INT8) | unknown |
| LaBraM | 200 | 0.253 | 0.253 | 76 ms | yes | BCI-IV-2a (pretrained) |
| FEMBA-tiny | 385 | 0.240 | 0.240 | 960 ms | yes | BCI-IV-2a (pretrained) |

*nc = nearest-centroid classifier. MI accuracy is a safety floor in Mission 11, a benchmark in T-030. R5 = subject-Recall@5.*

### 1.2 What was proven (by mission)

| Mission | What was tested | Key result |
|---|---|---|
| **M11** | Cross-session subject-identity retrieval (50 subj, runs 5-10, 300 splits, session-disjoint, no leakage) | CBraMod-200 beats V2-32 on Recall@K (R5: 0.527 vs 0.216, Δ=+0.312, p=5e-60, d=1.20). CBraMod still below PCA (R5: 0.692). MI safety floor: 0.275 ≥ 0.25. |
| **M12** | Tier-2 architecture (additive, isolated `foundation_embeddings(vector(200))`, no V2 fallback, 424 on `FoundationUnavailableError`) | SUCCESS — architecture ships, zero V2/PCA/embeddings(32) imports |
| **M13** | Platform retrieval call site + real EDF serving | `NeuralVectorIndex.search` validated on real embeddings; real EDF → 200-D lands in learned manifold (maxCos 0.9922). pgvector RPC: INCONCLUSIVE (Docker down, M13), then closed in M14. |
| **M14** | Live pgvector RPC + Phase 1 GA-readiness code-level gates (7/7 PASS) | Live RPC reproduces M13 gate (CBraMod R5 0.5269 ≈ M13 0.5273). CBraMod retrieval gate: PASS. 4 operational gates: INCONCLUSIVE (not production-like). |
| **M15** | Close 4 INCONCLUSIVE operational gates with production-like validation | All 4 gates PASS (real JWT auth, real rate-limit RPC, real ONNX, real artifact corruption). Verdict: **READY_FOR_OPT_IN**. |
| **T-030** | Fix 9 benchmark bugs; re-benchmark all 6 models, 10 subjects | After fixes: CBraMod 0.323, PCA 0.290, V2 0.317 (MI nearest-centroid). No model significantly beats PCA (CBraMod p=0.401). Root cause: bugs (leakage, buggy PCA, wrong preprocessing, missing pooling). |
| **T-031** | Fine-tune EEGConformer V1→V2→V3 on 6→14→20→40 subjects | V2 (40 subj) significant at 50-subject all-LOSO (0.343 vs 0.283, p=0.0002) but NOT at 10-subject strict hold-out (0.327 vs 0.280, p=0.143). V3 (30 subj) not significant. |

### 1.3 Cached evidence available for M16

`reports/.cbramod_cross_session_cache.npz` (from M11):

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `cb_emb` | (4500, 200) | float32 | CBraMod-200-D, L2-normalised |
| `v2_emb` | (4500, 32) | float32 | V2-32-D, L2-normalised |
| `bandpower` | (4500, 110) | float32 | PCA bandpower features (5 bands × 22 ch) |
| `subj_ids` | (4500,) | int64 | Subject IDs 1-50 |
| `run_ids` | (4500,) | int64 | Run IDs 5-10 |
| `mi_labels` | (4500,) | int64 | 4-class MI labels 0-3 |
| `cbramod_sha256` | () | U64 | `c128ccfd…` |
| `v2_sha256` | () | U64 | `18644de1…` |

All embeddings are **pre-computed, cached, and reproducible** — no new ONNX inference needed for M16.

---

## 2. What Is Already Proven

### 2.1 Subject-identity retrieval (the Mission 11 gate)

- CBraMod-200 beats V2-32 on cross-session subject-identity retrieval (R5: 0.527 vs 0.216, Δ=+0.312, p=5e-60, d=1.20, Bonferroni-corrected over 3 Recall metrics)
- The gap is **highly significant** with a **large effect size** — this is real geometry, not noise
- CBraMod's same-subject NN cosine (0.9933) > diff-subject NN cosine (0.9931) — positive gap, V2 has inverted gap
- PCA-32 is the strongest baseline (R5: 0.692), beating both CBraMod and V2

### 2.2 Operational readiness (Mission 15)

- **Artifact integrity**: SHA-256 + size gate → 424 on mismatch, no fallback, real ONNX 200-D inference confirmed
- **Rate limiting**: 20/60s per user, atomic UPSERT race-free, 50 concurrent → exactly 20×200 + 30×429
- **Concurrency**: ramp 1→50, ONNX session reuse (cold ~930ms, warm ~280ms), no memory corruption
- **API contract**: all 10 status codes (200/400/401/408/413/415/422/424/429/500) validated with real JWT auth
- **RLS/authorization**: real GoTrue JWT validation, non-superuser isolation, filter_user_id binding
- **Rollback safety**: Tier-2 path has zero imports of V2/embedEEG/DEFAULT_PREFERRED/PCA
- **Build**: `vite build` succeeds, `onnxruntime-node` externalized from worker bundle
- **V2 regression**: V2 upload path untouched (15/15 tests pass)

### 2.3 What was NOT proven

- **Linear-probe classification**: Only nearest-centroid (a weak linear classifier) was used for MI accuracy. No logistic regression, SVM, or ridge classifier on frozen features.
- **General-purpose representation quality**: CBraMod-200 has only been evaluated on subject-identity retrieval. No evidence of utility on standard MI classification with a proper linear probe.
- **CBraMod vs PCA on classification**: T-030 showed CBraMod MI acc 0.323 vs PCA 0.290 (nearest-centroid, p=0.401), but this used nearest-centroid, not a linear probe, and was only 10 subjects.

---

## 3. What Remains Unknown

### 3.1 The fundamental gap

CBraMod-200's value is established ONLY on **subject-identity retrieval** (cross-session, cross-task). The retrieved result is:

> CBraMod-200 beats V2-32 on retrieval, but is below PCA-32.

This raises a critical question: **Is CBraMod-200 a general-purpose Foundation representation, or a subject-identity retrieval specialist?**

The answer matters because:
- A **Foundation Model** should provide value across multiple downstream tasks (MI classification, sleep staging, attention decoding, etc.)
- A **retrieval specialist** is valuable only within its narrow domain (subject-identity retrieval)
- The product decision (opt-in vs. broader Foundation Model role) depends on which category CBraMod falls into

### 3.2 Specific unknown questions

1. **Linear decodability**: Can a linear probe on frozen CBraMod-200 features classify 4-class MI above chance/baseline? (Nearest-centroid gave 0.275 — barely above chance 0.25.)

2. **CBraMod vs PCA on classification**: Does CBraMod-200 + linear probe beat PCA-32 + linear probe on MI classification? (PCA bandpower features are already designed for spectral MI decoding.)

3. **CBraMod vs V2 on classification**: Does CBraMod-200 + linear probe beat V2-32 + linear probe? (V2 was fine-tuned on 40 PhysioNet subjects; CBraMod was pretrained on BCI-IV-2a.)

4. **Dimensionality effect**: Does the 200-D vs 32-D gap explain the retrieval gap? Would projecting CBraMod to 32-D close the gap with PCA?

5. **Cross-task generalization**: Has CBraMod-200 been validated on any task beyond subject-identity retrieval? (No.)

---

## 4. What Should NOT Be Repeated

| Already answered | Evidence |
|---|---|
| CBraMod beats V2 on retrieval | M11: R5 0.527 vs 0.216, p=5e-60, d=1.20 |
| PCA beats CBraMod on retrieval | M11/M13/M14: PCA R5 0.692 vs CBraMod 0.527, p≪0.05 |
| Artifact SHA verification → 424 on mismatch | M15 Phase 4: 5 tests, all pass, byte-for-byte restoration |
| Rate limiting (20/60s, per-user, race-free) | M15 Phase 2: 5 tests, 50 concurrent → 20×200 + 30×429 |
| Concurrency (ramp 1→50, session reuse safe) | M15 Phase 3: 7 tests, all pass |
| API contract (all 10 status codes, no fallback) | M15 Phase 5: 16 tests, all pass |
| Real JWT auth + RLS isolation | M15 Phase 1: 8 tests, all pass |
| Real pgvector RPC works | M14 Phase 0: live DB, R5 0.5269 ≈ M13 0.5273 |
| V2 regression intact | M15 Phase 6: 15/15 V2 upload tests pass, build succeeds |

---

## 5. Candidate Next Experiments (Ranked)

### Candidate A: Linear-Probe MI Classification Benchmark (LOSO, 50 subjects)

**Description**: Freeze CBraMod-200, V2-32, and PCA-32 (train-only PCA per fold) embeddings. Train a linear probe (multinomial logistic regression with L2 regularization, or linear SVM) on 49 subjects, test on 1 (50-fold LOSO). Same dataset as M11 (PhysioNet EEGMMIDB S001-S050). Use cached embeddings from M11 (no retraining, no new inference).

**Hypothesis**: CBraMod-200 + linear probe will NOT beat PCA-32 + linear probe on 4-class MI classification, because PCA bandpower features are already specifically designed for spectral MI decoding (5 frequency bands × 22 channels capture the canonical MI ERD/ERS patterns). This would confirm CBraMod is a retrieval specialist, not a general foundation model.

**Variant hypothesis**: If CBraMod-200 + linear probe DOES beat PCA, it would indicate CBraMod captures more generalizable EEG structure beyond bandpower features, strengthening the foundation model case.

**Distinguishing power**: High — directly tests the foundation model hypothesis using the standard evaluation protocol (linear probe on frozen features).

**Implementation cost**: Low — `sklearn.linear_model.LogisticRegression` on cached numpy arrays. ~100 lines of Python.

**Risk**: Very low — uses cached embeddings, no production changes, no retraining.

### Candidate B: CBraMod-200-to-32-D Projection Analysis

**Description**: Project CBraMod-200 to 32-D via PCA (train-only per fold), then re-run both retrieval and linear-probe MI classification. Tests whether the dimensionality gap (200 vs 32) explains CBraMod's retrieval disadvantage vs PCA.

**Hypothesis**: Projecting CBraMod to 32-D will close some or all of the retrieval gap with PCA-32, suggesting the gap is partly dimensionality-driven (curse of dimensionality in cosine similarity at 200-D vs 32-D).

**Distinguishing power**: Medium — tests dimensionality as a variable, but doesn't address foundation model utility.

**Implementation cost**: Low — PCA projection on cached embeddings.

**Risk**: Very low.

### Candidate C: Multi-Task Downstream Evaluation

**Description**: Beyond MI classification, test CBraMod-200 + linear probe on sleep staging, ERP component classification, and attention decoding using additional publicly available EEG datasets (e.g., Sleep-EDF, ERP_CORE).

**Hypothesis**: CBraMod-200 generalizes across EEG paradigms, demonstrating foundation model utility beyond the MI domain.

**Distinguishing power**: High for foundation model status — generalizes beyond MI.

**Implementation cost**: Medium — requires additional dataset downloads and preprocessing.

**Risk**: Low — additive benchmark.

### Candidate D: CBraMod Retrieval on Cross-Task Generalisation

**Description**: Using the M11 protocol (runs 5-10, 6 sessions), test whether CBraMod's subject-identity retrieval advantage holds when the query run's task is DIFFERENT from the pool's task for the same subject (already done in M11 — this IS the protocol).

**Assessment**: **Already answered.** M11 already uses cross-task retrieval (each run has different MI task). No new insight to gain without changing the dataset or task.

### Candidate E: Full-Stack Production Load Test

**Description**: Load-test the deployed `/api/eeg/embed/foundation` endpoint against a real Supabase project (not local) with 100+ concurrent users.

**Assessment**: **Not a scientific question.** This is an engineering/scalability test, not a representation-quality question. Can be deferred until after the scientific decision is made.

### Candidate F: CBraMod vs V2 Feature Attribution Analysis

**Description**: Use saliency maps or feature attribution (Integrated Gradients) to analyze which input channels/features drive CBraMod vs V2 predictions.

**Assessment**: **Low priority.** We already know CBraMod uses 19 channels and V2 uses 22. The channel overlap table is known. Attribution wouldn't resolve the foundation model question.

### Candidate G: CBraMod as Conditioning Signal for Synthetic Data

**Description**: Test whether CBraMod-200 embeddings can condition a synthetic EEG generator (e.g., "generate EEG similar to subject X's embedding").

**Assessment**: **High implementation cost, speculative.** Requires training a generative model. Risk of producing misleading conclusions about CBraMod's representation quality (generation ≠ classification/retrieval).

---

## 6. Ranked Recommendation

| Rank | Experiment | Scientific Value | Product Value | Distinguishes from PCA/V2 | Reproducible | Cost | Risk | Statistically Valid |
|---|---|---|---|---|---|---|---|---|
| **1** | A: Linear-Probe MI Classification | **High** | **High** | **Yes** | **Yes** | **Low** | **Very Low** | **Yes** |
| 2 | B: 200→32 Projection Analysis | Medium | Medium | Partial | Yes | Low | Very Low | Yes |
| 3 | C: Multi-Task Downstream Eval | High | High | Yes | Medium | Medium | Low | Yes |
| 4 | D: Cross-task retrieval (repeat) | — | — | — | — | — | — | — |
| 5 | E: Production Load Test | Low | Medium | No | Low | High | Medium | N/A |
| 6 | F: Feature Attribution | Low | Low | No | Medium | Medium | Low | Partial |
| 7 | G: Synthetic Data Conditioning | Speculative | Medium | Unclear | Low | High | Medium | No |

### Recommended: **Candidate A — Linear-Probe MI Classification Benchmark**

**Why this is the highest-value next question:**

1. **It directly tests the Foundation Model hypothesis.** The "Foundation Model" label implies general-purpose representation quality. The standard test is: freeze features, train a linear probe, evaluate on downstream tasks. CBraMod has only been tested on subject-identity retrieval — a narrow niche. A linear probe on MI classification is the canonical downstream task for motor imagery EEG.

2. **It uses existing cached data.** The M11 cache (`reports/.cbramod_cross_session_cache.npz`) already has CBraMod-200, V2-32, and PCA bandpower (110 features) embeddings for 4500 trials across 50 subjects with MI labels. No new ONNX inference, no retraining.

3. **It can decisively distinguish CBraMod from PCA and V2.** 
   - If CBraMod-200 LP > PCA-32 LP: CBraMod has general-purpose representation value beyond handcrafted spectral features → strengthens foundation model case
   - If CBraMod-200 LP ≤ PCA-32 LP: CBraMod is a retrieval specialist, not a foundation model → opt-in only, no broader promotion
   - If CBraMod-200 LP > V2-32 LP: CBraMod's learned representation is better than the deployed V2 for MI classification → potential production value
   - If CBraMod-200 LP ≤ V2-32 LP: CBraMod's advantage is retrieval-specific, not classification-general

4. **It is more valuable than repeating CBraMod vs V2 retrieval.** Retrieval advantage is already proven (M11: ΔR5=+0.312, p≪0.001, d=1.20). The linear probe tests a DIFFERENT evaluation dimension — classification decodability — which has not been tested with a proper linear probe at scale. It also tests CBraMod vs PCA directly on classification (not just retrieval), which is the critical comparison given PCA's strong retrieval performance.

5. **It is statistically sound and reproducible.** 50-fold LOSO with paired t-tests + Cohen's d + bootstrap CIs + Bonferroni correction. Same protocol as T-032/T-030 (train-only PCA fit per fold, seed 42). Dataset already downloaded and cached.

---

## 7. Proposed Mission 16 Experiment Design

### 7.1 Exact Hypothesis

**Primary**: CBraMod-200 + linear probe will NOT significantly outperform PCA-32 + linear probe on 4-class MI classification (LOSO, 50 subjects). The null hypothesis (CBraMod-200 LP ≤ PCA-32 LP) cannot be rejected, confirming CBraMod is a retrieval specialist rather than a general foundation model.

**Secondary A**: CBraMod-200 + linear probe will NOT significantly outperform V2-32 + linear probe on 4-class MI classification.

**Secondary B**: CBraMod-200 + linear probe will NOT significantly outperform V2-32 + linear probe.

### 7.2 Dataset and Evaluation Protocol

- **Dataset**: PhysioNet EEGMMIDB S001-S050, runs 5-6 (same as T-032 for MI classification compatibility)
  - **Note**: M11 cache used runs 5-10 (6 runs). For MI classification with 4-class labels, runs 5 (left/right hand) and 6 (feet/tongue) are sufficient and match T-032 protocol. If the cache only has runs 5-6 data, we use that. If it has all 6 runs, we can use runs 5-6 for consistency with T-032.
- **Splits**: 50-fold LOSO (leave-one-subject-out). Train on 49 subjects (4410 trials), test on 1 (90 trials).
- **Leakage prevention**: PCA fit on train-only data per fold. No test-set embeddings in training pool.
- **Seed**: 42 (matching T-032/T-030 convention)

### 7.3 Models/Baselines to Compare

| Model | Embedding | Dim | Freeze | Linear Probe |
|---|---|---|---|---|
| CBraMod-200 | `cb_emb` from cache | 200 | Yes | LogisticRegression(L2, C=1.0) |
| V2-32 | `v2_emb` from cache | 32 | Yes | LogisticRegression(L2, C=1.0) |
| PCA-32 | `bandpower` (110) → PCA(32) | 32 | N/A (feature-based) | LogisticRegression(L2, C=1.0) |

**PCA baseline**: Apply PCA(32) to the 110-dim bandpower features, fit on train-only (per fold), seed=42. Same as T-032.

### 7.4 Metrics

- Primary: **accuracy** (mean ± std across 50 folds)
- Secondary: macro-F1, AUC (one-vs-rest), Recall@1/5/10 (for representation quality)
- Latency: training time + inference time per fold (already cached, so ~0)
- 95% bootstrap CIs (percentile method, 10000 resamples, seed 42)

### 7.5 Statistical Tests

- **Paired t-test** on per-fold accuracies: CBraMod-200 LP vs PCA-32 LP, CBraMod-200 LP vs V2-32 LP, V2-32 LP vs PCA-32 LP
- **Bonferroni correction**: 3 comparisons, α = 0.05/3 = 0.01667
- **Cohen's d** effect sizes
- Bootstrap 95% CIs for all metrics

### 7.6 Success/Failure Criteria

| Outcome | Interpretation |
|---|---|
| CBraMod-200 LP > PCA-32 LP (p<0.0167, Bonferroni) | CBraMod has general-purpose representation value; strengthen foundation model case |
| CBraMod-200 LP ≤ PCA-32 LP (p≥0.0167) | CBraMod is a retrieval specialist; opt-in only |
| CBraMod-200 LP > V2-32 LP (p<0.0167) | CBraMod representation is better than deployed V2 for MI |
| CBraMod-200 LP ≤ V2-32 LP (p≥0.0167) | V2's MI classification advantage holds; CBraMod's value is retrieval-specific |

### 7.7 Expected Engineering Changes

**None required for the experiment itself** — uses cached embeddings + sklearn. After the experiment:

- New script: `scripts/tmp/m16_linear_probe_benchmark.py` (read-only, uses cached embeddings)
- New results: `reports/m16_linear_probe_results.json`
- New report: `reports/MISSION16_LINEAR_PROBE_REPORT.md`
- One archive append to `reports/benchmark_archive.json`

### 7.8 No-Touch Boundaries

- `DEFAULT_PREFERRED`: unchanged
- `embedEEG`: unmodified
- V2 routing/artifacts: unchanged
- PCA code: unaffected (only used as a cached baseline, not modified)
- CBraMod model weights: untouched (read-only cache)
- All Mission 11–15 results: byte-for-byte preserved
- `benchmark_archive.json`: only one append (new experiment record)

### 7.9 Why This Experiment > Repeating CBraMod vs V2 Retrieval

| Aspect | Repeating retrieval | Linear-probe MI classification |
|---|---|---|
| Uses already-proven result | No — retrieval advantage proven (p=5e-60) | Yes — classification decodability is UNKNOWN |
| Tests foundation model hypothesis | No — retrieval is a niche task | Yes — classification is the canonical downstream task |
| Distinguishes from PCA | No — retrieval vs PCA already known (PCA wins) | Yes — classification vs PCA is UNKNOWN |
| Implementation cost | Low | **Lower** (cached data, sklearn only) |
| Risk of misleading conclusion | High — would just re-confirm known result | Low — tests genuinely unknown territory |
| Product decision impact | Low — doesn't inform foundation model role | **High** — determines if CBraMod is a general Foundation representation or a specialist |

---

## 8. Decision Tree Outcome

If Mission 16 is approved and executed:

```
CBraMod-200 LP vs PCA-32 LP
├── Significant (p < 0.0167, Δ ≥ 0.05): 
│   └── CBraMod has general-purpose representation value
│   └── → Consider broader Foundation Model promotion (NOT DEFAULT_PREFERRED)
│   └── → Next: multi-task evaluation (sleep, attention, ERP)
└── Not significant:
    └── CBraMod is a retrieval specialist
    └── → Opt-in server-side only (Mission 15 verdict stands)
    └── → No broader Foundation Model promotion
    └── → Consider: does CBraMod's 200-D space offer ANY classification advantage
        over 32-D V2? (If not, CBraMod's value is purely in retrieval geometry)
```

### Whether this experiment could change the Foundation Model decision

**Absolutely.** Currently:
- CBraMod's **retrieval** advantage is proven (M11, M13, M14)
- CBraMod's **classification** advantage is **unknown** (only nearest-centroid safety floor tested)

If CBraMod-200 LP significantly outperforms PCA-32 LP on MI classification, it would indicate CBraMod captures generalizable EEG structure beyond bandpower features — strengthening the case for a broader Foundation Model role.

If CBraMod-200 LP does NOT outperform PCA-32 LP (matching the retrieval results), it would confirm CBraMod is a subject-identity retrieval specialist. The Foundation Model decision would be: **opt-in only**, with CBraMod serving as a server-side specialist for retrieval/search use cases, NOT a general-purpose representation.

This is the single most decision-relevant experiment remaining.
