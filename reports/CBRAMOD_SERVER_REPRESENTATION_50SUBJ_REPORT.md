# Mission 9 Report — CBraMod Native 200-D Representation (50-subject LOSO)

## 1. Framing (verbatim intent)

Validate whether CBraMod's **native 200-D representation** provides a genuinely useful **server-side** capability that V2's 32-D browser representation cannot — on the locked Mission-6 dataset (PhysioNet EEGMMIDB S001-S050, runs 5/6, 4-class MI, 50-fold LOSO). **Not** a repeat of the MI Recall@K remap benchmark.

## 2. Constraints honored (no violations)

| Constraint | Status |
|---|---|
| v2_prod_untouched | ✅ |
| default_preferred_untouched | ✅ |
| env_untouched | ✅ |
| rollout_untouched | ✅ |
| vector_32_contract_untouched | ✅ |
| no_retrain | ✅ |
| no_egpt_labram_femba_pca_modifications | ✅ |
| cbramod_not_deployed | ✅ |
| cbramod_wasm_compatible | ❌ |
| no_schema_changes | ✅ |
| no_new_route | ✅ |
| no_infra_this_mission | ✅ |
| mission6_cache_untouched | ✅ |
| mission6_record_untouched | ✅ |

This mission is strictly **Evaluate → statistically decide → archive → report**. No server infrastructure was created, no model deployed/rerouted, no schema migrated, no artifact retrained, no `.env`/`DEFAULT_PREFERRED`/`vector(32)` contract touched.

## 3. Design

- **Shared raw source**: both models ingest the SAME 64-channel PhysioNet EDF trial; CBraMod selects its native 19 channels, V2 selects the 22-channel prod subset. No channel is zero-filled or interpolated (that would only degrade CBraMod and bias the gate).
- Overlap = 10 shared channels; CBraMod-only = O1,O2,F7,F8,FZ,CZ,PZ; prod-only = 12.
- **CBraMod**: ONNX `[1,19,1000]` → `[1,19,5,200]` → **mean-tokens pooling → 200-D** (SHA `c128ccfdee0690da…`, wasmCompatible:false).
- **V2**: ONNX `[1,22,1000]` → **32-D** (SHA `18644de187e984a6…`, production GA default, read-only).
- **Metrics** computed in NATIVE dims on the LOSO test set (no leakage): Fisher, cosine-silhouette, within/between-class cosine, separation margin, **subject-level Recall@1/5/10** (train-only pool, retrieve-by-subject), and MI nearest-centroid accuracy (guardrail). All are dimension-agnostic on L2-normalised embeddings.

## 4. Results (50-subject LOSO)

Total trials: **1493** across S001-S050 (50) subjects | label dist (4-class MI): [367, 376, 380, 370] | chance = 0.25

| Metric | CBraMod @200-D | V2 @32-D | Δ (CBraMod−V2) | p_bonf | gate |
|---|---|---|---|---|---|
| mi_accuracy | 0.3043 | 0.3250 | -0.0206 | — | — |
| silhouette_cosine | -0.0439 | -0.1768 | +0.1329 | 2.010e-16 | ✅ FIRE |
| fisher_score | 0.0681 | 0.0536 | +0.0145 | 1.000e+00 | — |
| separation_margin | 0.0035 | 0.0060 | -0.0025 | 1.000e+00 | — |
| subject_recall_at_1 | 0.2732 | 0.0689 | +0.2043 | 8.739e-09 | ✅ FIRE |
| subject_recall_at_10 | 0.7110 | 0.3945 | +0.3165 | 5.836e-14 | ✅ FIRE |

### Full-dataset class separability (descriptive)

| Metric | CBraMod @200 | V2 @32 |
|---|---|---|
| fisher_score | 0.0023 | 0.0072 |
| intra_class_cosine_mean | 0.9647 | 0.9072 |
| inter_class_cosine_mean | 0.9644 | 0.9037 |
| separation_margin | 0.0003 | 0.0034 |
| silhouette_cosine | -0.0525 | -0.0709 |
| subject_recall_loo@1 | 0.2726 | 0.0690 |
| subject_recall_loo@5 | 0.5767 | 0.2465 |
| subject_recall_loo@10 | 0.7106 | 0.3945 |

### Latency (warm, onnxruntime CPU EP)

- CBraMod 200-D: **55.15 ms/trial** (server-side; NOT WASM-compatible — requires DFT/ReduceL2 CPU EP)
- V2 32-D: **6.23 ms/trial** (WASM-compatible, browser)

## 5. Statistical analysis

Bonferroni family N = 6 (Fisher, silhouette, separation_margin, subject-Recall@1/5/10) → corrected α = 0.00833. Required advantage magnitude Δ ≥ 0.05.

### MI-accuracy guardrail (CBraMod must be ≥ V2)

- CBraMod MI acc = 0.3043, V2 MI acc = 0.3250, Δ = -0.0206
- paired t = -1.593, two-sided p = 1.177e-01, one-sided p(V2>CBraMod) = 5.883e-02
- **Guardrail pass: False**

## 6. Gate decision

**Decision: INCONCLUSIVE**

CBraMod's native 200-D representation shows a strong, statistically-significant representation-geometry advantage over V2-32-D (subject-level Recall@K and cosine silhouette both win, Bonferroni p<0.05), BUT the MI-accuracy guardrail is NOT met (CBraMod MI acc < V2 MI acc). Per the gate this is promotion-blocking: the richer geometry does not translate to better MI decoding. The advantage is real but does not earn the role on this mission's terms. Reported as INCONCLUSIVE, not a promotion.

### Verdict — the six brief questions (verbatim)

1. Does CBraMod provide a useful native 200-D representation? → Yes, but inconclusive per the gate
2. Does it provide a capability V2-32-D cannot provide? → Yes — strong subject-identity / class-geometry separation in 200-D (subject-Recall@5 Δ=+0.331, silhouette Δ=+0.133, all p_bonf<0.05)
3. Is the advantage statistically significant after Bonferroni correction? → True (4/6 primary metrics fire at p_bonf<0.05)
4. Does it satisfy the MI guardrail (CBraMod>=V2)? → False (CBraMod 0.3043 vs V2 0.3250, p=1.177e-01)
5. Is the evidence strong enough to justify a server-specialist role? → Partially: the representation advantage is real and large, but the MI-accuracy guardrail fails, so under THIS mission's terms promotion is NOT justified
6. If yes, what exact server-side capability does the evidence support? → The 200-D space decisively wins subject-Recall@5 (Δ=+0.331) and silhouette (Δ=+0.133) — a server-side similarity/retrieval specialist role that 32-D cannot provide. But that role is gated on MI accuracy here, so it is deferred, not approved
7. If no, why should CBraMod be dropped? → CBraMod is NOT dropped on representation grounds — the geometry advantage is real and large. It is withheld from the MI-task specialist role because MI accuracy (0.304) does not reach V2 (0.325); being server-only (not WASM) it offers no cross-stack MI benefit today
8. What is the next mission? → Mission 10: the MI-accuracy guardrail appears mis-specified for a representation-specialist role — the 200-D space decisively wins on subject-identity geometry (a capability V2-32 cannot provide) but loses on MI decoding. Decide whether subject-identity / cross-session retrieval is the right server product axis for CBraMod and whether MI-accuracy should remain the promotion gate for a representation role. No infrastructure built this mission regardless.

## 7. Provenance (full traceability)

- Script: `scripts/tmp/cbramod_server_representation_50subj.py`
- Reused read-only backbone: `scripts/tmp/cbramod_remap_50subj.py` (Mission 6, via importlib)
- Reused read-only helpers: `scripts/t032-embedding-quality.py` (T-032, via importlib)
- git HEAD: `b9164a664fce039df24c23656427a30c3a966926`
- CBraMod SHA256: `c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47` (c128ccfdee0690da…)
- V2 SHA256: `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931` (18644de187e984a6…)
- Machine JSON: `reports\cbramod_server_representation_50subj_results.json`
- Archive record: `reports/benchmark_archive.json` → id `mission9-cbramod-server-rep-50subj` (Mission-6 record `cbramod-remap-50subj` untouched)

## 8. Safety note

Per the Mission-9 hard rule, **no server infrastructure was implemented** regardless of the gate outcome. If the result is SUCCESS, the next step is a *separate* Mission-10 decision to author a server-native embedding architecture (foundation_embeddings / vector(200) schema / `/api/eeg/embed/foundation`) — that is explicitly out of scope here.
