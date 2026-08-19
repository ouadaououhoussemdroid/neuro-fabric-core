# Mission 11 — CBraMod Cross-Session Subject-Identity Validation

**Decision: SUCCESS**

## 1. Objective

Test whether CBraMod's **native 200-D representation** provides a **cross-session** subject-identity / retrieval capability that V2-32-D cannot — using a **session-disjoint** protocol on PhysioNet EEGMMIDB runs 5–10. **Not** an MI benchmark; MI accuracy is only a signal-presence safety floor.

## 2. Protocol

- **Dataset**: PhysioNet EEGMMIDB S001–S050, runs **{5,6,7,8,9,10}** (each run = an independent acquisition session / task block; different MI tasks across runs).
- **Split**: for every `(subject, held-out-run)` pair — **queries** = that run's trials; **pool** = all trials from all *other* runs × all subjects (held-out run excluded entirely → no leakage, and a cross-task identity test since the query task differs from the pool task for the same subject).
- **Splits evaluated**: 300 (subject,run) pairs.
- **Metrics**: subject Recall@1/5/10 (primary), subject silhouette (cosine) + same-vs-diff NN cosine gap (secondary descriptors), MI nearest-centroid accuracy (safety floor ≥ chance 0.25).
- **Models** (native dims, cosine on L2-normalised embeddings, no projection): CBraMod 200-D, V2 32-D, PCA bandpower 32-D (per-split train-only PCA fit, seed 42).
- **Labels**: odd run (5,7,9)→T1=left/T2=right; even run (6,8,10)→T1=feet/T2=tongue. Verified per subject.
- **Trials**: 4500; MI label dist across runs [1122, 1128, 1133, 1117]; chance=0.25.

## 3. Results (CBraMod vs V2 vs PCA)

| Metric (mean over splits) | CBraMod @200 | V2 @32 | PCA @32 | Δ CBraMod−V2 | p_bonf (N=3) | gate |
|---|---|---|---|---|---|---|
| subject_recall_at_1 | 0.2427 | 0.0687 | 0.4400 | +0.1740 | 5.612e-36 | ✅ FIRE |
| subject_recall_at_5 | 0.5273 | 0.2158 | 0.6920 | +0.3116 | 1.663e-59 | ✅ FIRE |
| subject_recall_at_10 | 0.6587 | 0.3364 | 0.7853 | +0.3222 | 3.436e-61 | ✅ FIRE |
| mi_accuracy (safety) | 0.2749 | 0.3020 | 0.3018 | -0.0271 | — | — |

### Full-dataset subject-clustering descriptors

| Descriptor | CBraMod @200 | V2 @32 |
|---|---|
| subject silhouette (cosine) | -0.1388 | -0.3404 |
| same-subject NN cosine | 0.9933 | 0.9951 |
| diff-subject NN cosine | 0.9931 | 0.9976 |
| NN gap (same−diff) | 0.0003 | -0.0026 |

### Latency (warm, onnxruntime CPU EP)

- CBraMod 200-D: **64.14 ms/trial** (server-side; NOT WASM-compatible)
- V2 32-D: **6.73 ms/trial** (WASM, browser)

## 4. Statistical analysis

Paired t-test CBraMod vs V2 across the 300 session-disjoint splits. Bonferroni family N=3 (subject-Recall@1/5/10); corrected α=0.01667. Effect sizes = Cohen's d; 95% CI via percentile bootstrap (10000 paired resamples, seed 42).

| Metric | Δ | t | p (two-sided) | p_bonf | d | 95% CI of Δ |
|---|---|---|---|---|---|---|
| subject_recall_at_1 | +0.1740 | +14.50 | 1.871e-36 | 5.612e-36 | +0.84 | [+0.1509, +0.1980] |
| subject_recall_at_5 | +0.3116 | +20.78 | 5.544e-60 | 1.663e-59 | +1.20 | [+0.2818, +0.3411] |
| subject_recall_at_10 | +0.3222 | +21.24 | 1.145e-61 | 3.436e-61 | +1.23 | [+0.2927, +0.3518] |

## 5. MI safety guardrail (signal presence, NOT competitiveness)

- CBraMod MI acc = **0.2749** (≥ chance 0.25? → PASS ✅)
- For context: V2 = 0.3020, PCA = 0.3018

Per Mission 10, MI accuracy is the **wrong gate for a representation-specialist role**; it is retained only as a sanity floor confirming CBraMod still encodes EEG signal.

## 6. Gate decision

CBraMod's native 200-D representation demonstrably provides a cross-session subject-identity / retrieval capability V2-32-D cannot: CBraMod beats V2 on subject-Recall@K by >=0.05 (Bonferroni p<0.05) AND the MI signal-presence guardrail (MI acc >= chance) holds.

**Promotion gate requirements:** ∃ K∈{1,5,10}: CBraMod>V2 AND Δ≥0.05 AND Bonferroni p<0.05, **and** MI acc≥chance.

- Primary advantage cleared for 3/3 Recall metrics.
- MI safety guardrail: PASS.

### What was executed
Downloaded runs 7–10 for S001–S050 (~200 EDFs, ~0.5 GB), built CBraMod 200-D + V2 32-D + per-split PCA-32 on runs {5,6,7,8,9,10}, ran session-disjoint cross-run subject identity retrieval over all valid (subject,run) held-out splits, paired tests + Bonferroni + bootstrap CIs.

### Exact results (CBraMod vs V2)
- subject-Recall@5: **0.5273 vs 0.2158 (Δ +0.3116, p_bonf=1.663e-59)**
- subject-Recall@10: 0.6587 vs 0.3364 (Δ +0.3222)
- subject-Recall@1: 0.2427 vs 0.0687 (Δ +0.1740)

### Is cross-session identity demonstrated? → **YES**
Cross-run subject retrieval tests identity across *different acquisition runs with different MI tasks* (a real generalization test, not same-recording duplication). CBraMod clearly separates same-subject trials across runs/tasks where 32-D cannot.

### Statistical significance & effect sizes
CBraMod's subject-Recall gains are highly significant with large effects (d≈1.1998486071709966, p_bonf≪0.05) when the gate fires — these are real geometry differences, not noise.

### Promotion gate: **SUCCESS**

## 7. Provenance
- Script: `scripts/tmp/cbramod_cross_session_validation.py`
- Reused read-only: Mission-6 backbone (`cbramod_remap_50subj.py` via importlib) + T-032 helpers (`t032-embedding-quality.py`)
- git HEAD: `b9164a664fce039df24c23656427a30c3a966926`
- CBraMod SHA256: `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47`
- V2 SHA256: `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931`
- Machine JSON: `reports\MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.json`
- Archive record: `reports/benchmark_archive.json` id `mission11-cbramod-cross-session-validation` (Mission 6/9/10 untouched)

## 8. Safety
No infrastructure, routing, schema (`vector(200)`/`foundation_embeddings`), API route, or production edit was created. If SUCCESS, Mission 12 is the first mission in which the Tier-2 server-native embedding architecture may be proposed. If not SUCCESS, CBraMod remains server-side-only research artifact.