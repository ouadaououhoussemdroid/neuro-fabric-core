# Mission 10 — CBraMod Product-Role Decision & Gate Re-evaluation

**Mission type:** Read-only product / architecture decision analysis (no infrastructure built).
**Inputs:** Archived Mission-9 results (`reports/cbramod_server_representation_50subj_results.json`, `reports/benchmark_archive.json` → `mission9-cbramod-server-rep-50subj`).
**Outputs:** This report + `reports/mission10_cbramod_product_role_decision.json`.
**Constraints honored:** No `foundation_embeddings`/`vector(200)` schema, no `/api/eeg/embed/foundation`, no `embedEEG()`/`DEFAULT_PREFERRED`/registry/`.env` changes, no V2/CBraMod artifact edits, no deployment, no retraining, no generic MI benchmark.

---

## Executive decision

> **CONDITIONAL — RUN ONE TARGETED VALIDATION.**

The Mission-9 evidence is **not sufficient to promote CBraMod** to a server-side specialist role today, but it is **strong enough to make the subject-identity / cross-session representation role a credible, falsifiable target** that warrants exactly one focused experiment (Section 6). CBraMod is **not dropped** (the representation advantage is large and statistically robust); it is **not promoted** (the real product task — cross-session identity — has not yet been demonstrated, and the MI guardrail failed).

Do **not** add a second embedding tier to production until the conditional validation succeeds.

---

## 1. Mission-9 evidence (authoritative, verbatim from archive)

Locked dataset: PhysioNet EEGMMIDB S001–S050, **runs 5 and 6 only**, 4-class MI, 50-fold LOSO, 1493 trials.

CBraMod native 200-D (ONNX `[1,19,1000]→[1,19,5,200]` mean-tokens; SHA `c128ccfd…`; 55 ms CPU/trial; `wasmCompatible:false`) versus V2 (ONNX `[1,22,1000]→[1,32]`; SHA `18644de1…`; 6 ms CPU/trial; WASM-compatible, browser; production GA default).

| Metric (50-fold LOSO, paired) | CBraMod | V2 | Δ (CBraMod−V2) | p_bonf (N=6) | d | gate |
|---|---|---|---|---|---|---|
| MI nearest-centroid acc (guardrail) | 0.3043 | 0.3250 | −0.0206 | — | — | ❌ CBraMod < V2 (p=0.118) |
| cosine silhouette (test set) | −0.0439 | −0.1768 | **+0.133** | 2.0e-16 | 1.81 | ✅ |
| Fisher (test set) | 0.068 | 0.054 | +0.015 | 1.000 | 0.14 | ❌ (<γ, ns) |
| separation margin | 0.0035 | 0.0060 | −0.003 | 1.000 | −0.19 | ❌ |
| subject Recall@1 | 0.2732 | 0.0689 | **+0.204** | 8.7e-09 | 1.05 | ✅ |
| subject Recall@5 | 0.5770 | 0.2463 | **+0.331** | 1.7e-12 | 1.40 | ✅ |
| subject Recall@10 | 0.7110 | 0.3945 | **+0.317** | 5.8e-14 | 1.55 | ✅ |

Full-dataset (descriptive) geometry: CBraMod intra-class cosine **0.965** vs V2 **0.907**; inter-class cosine **0.964** vs **0.904**. CBraMod folds by subject far more tightly than by MI class (intra≈inter within a subject), while V2 folds MI classes more distinctly. MI accuracy: CBraMod ≥ V2 on only **23/50** folds — the guardrail failure is robust, not a tail event.

---

## 2. Assessment of the Mission-9 MI-accuracy guardrail

### Classification: **Incorrect gate** (for a representation-specialist promotion), though not *wrong* in a narrow sense.

The MI-accuracy gate (`CBraMod MI acc ≥ V2 MI acc`) is a **role-mismatch gate**.

- Mission-9's **stated value** for CBraMod is the native 200-D representation, subject identity / similarity, offline retrieval, richer representation geometry — *not* MI classification.
- V2 is explicitly MI-oriented, browser/interactive, 32-D, low-latency (Section 9 role separation).
- Requiring CBraMod to *also* match V2 on MI accuracy forces the subject-identity specialist to compete in the MI-decoding lane — the one lane V2 was purpose-built for and CBraMod was never positioned as.

This is the textbook **"necessary-but-insufficient"** shape, but evaluated for the *wrong role*: MI accuracy is *sufficient evidence that CBraMod is a valid EEG embedding* (it encodes signal, 0.304 ≫ chance 0.25), but it is **not a necessary condition** for a server-side *subject-identity / similarity* specialist. The Mission-9 INCONCLUSIVE verdict fell from treating "MI acc ≥ V2" as a hard gate; under a subject-identity role framing, 0.304 (≫ 0.25 chance) is *acceptable* — CBraMod is a real, signal-bearing embedding, just not an MI-classifier-better-than-V2.

**Conclusion:** the MI gate was the correct guardrail *for a MI-classification role* (CBraMod fails it → correctly not a MI specialist). It is the **wrong gate for a subject-identity/retrieval role**. Re-gate against the actual product task.

---

## 3. Product-role analysis (ranked)

### ✅ D. Subject identity / cross-session representation — **plausible, conditional**
1. **What it does:** identify & retrieve a subject's future EEG recordings (e.g. longitudinal session matching, deduplication, "find all sessions for this patient" across runs/days) using CBraMod's 200-D server embedding.
2. **Why V2 can't simply replace it:** V2 subject-Recall@5 = 0.247 vs CBraMod 0.577 (Δ=+0.331, p_bonf≪0.05). V2's 32-D collapses subjects too aggressively to support high-precision subject identity retrieval. This is precisely the "capability V2-32 cannot provide."
3. **Why 200-D matters:** the richer geometry preserves subject identity at tight NN distances (intra-subj cosine 0.965) that 32-D cannot represent without catastrophic subject collision.
4. **Mission-9 support:** **YES for *within-session* identity** (strong, significant). **NOT YET for *cross-session* identity** — the metric was computed same-session (see Section 5).
5. **Still required:** a cross-run / cross-session subject-retrieval experiment (Section 6).

### ✅ C. Server-side similarity / retrieval specialist — **plausible, conditional**
1. **What it does:** server-side nearest-neighbour / similarity search over a model-namespaced 200-D vector space (offline/batch), returning "similar recordings/subjects" from a cohort store.
2. **Why V2 can't replace it:** same subject-identity gap; plus 200-D carries more separable structure for a server-side ANN index.
3. **Why 200-D matters:** dimensionality carries the subject/identity structure that 32-D saturates.
4. **Mission-9 support:** representation advantage is significant; **but a real retrieval *product* task (cross-session, recall-gated) is unproven**.
5. **Still required:** cross-session retrieval validation; latency budget check at scale.

### ✅ F. Offline foundation embedding backbone — **promising, conditional**
1. **What it does:** serve CBraMod 200-D as a richer feature backbone for offline cohort clustering, phenotyping, or as input to downstream server models (contrastive clustering, biomarker discovery).
2. **Why V2 can't replace it:** 32-D lacks the capacity for fine-grained subject/cluster structure (subject-Recall@5 0.247 → unusable for fine clustering); 200-D preserves it.
3. **Why 200-D matters:** cluster separability / downstream task headroom typically scales with embedding capacity.
4. **Mission-9 support:** indirect — geometry advantage suggests headroom; **no downstream downstream-task experiment run**.
5. **Still required:** a clustering/phenotyping downstream-task lift over V2.

### ✅ E. Offline cohort phenotyping / clustering — **promising, conditional**
Same evidence base as F; needs a clustering-grade downstream lift and subject-structure stability across sessions.

### ❌ A. Browser embedding replacement — **No**
Not WASM-compatible (DFT/ReduceL2 blocks), 9× slower (55 vs 6 ms). V2 owns the browser/latency role.

### ❌ B. MI classification specialist — **No**
CBraMod MI acc 0.304 < V2 0.325 (23/50 folds); it is not competitive at the task it was never designed for. (And V1/original already cover MI well enough; see T-031.)

### G. Research-only model — **Yes (floor)**
CBraMod is, at minimum, a valid signal-bearing research artifact (MI ≫ chance). It is certainly retained in the research/artifact registry; the question is purely whether it earns a *product* role.

### ❌ H. No useful role — **No**
The subject-identity advantage is too large and robust to dismiss.

**Ranked selected roles:** D ≫ C ≳ F ≳ E ≳ G.

---

## 4. Separating the concepts Mission 9 mixed

| Concept | Did CBraMod demonstrate it? | Evidence |
|---|---|---|
| **Representation quality (class separation)** | **Mixed / partial.** Silhouette strong (+0.133); Fisher and margin near-tie (ns). | silhouette p_bonf≪0.05; Fisher p_bonf=1.0, margin Δ<0. |
| **Subject identity** | **Yes (within-session).** | subj-Recall@5 0.577 vs 0.247; intra-cos 0.965. |
| **Cross-session retrieval** | **No — not demonstrated.** | Runs 5/6 are a single session per subject (Section 5). |
| **MI classification** | **No — loses to V2.** | 0.304 vs 0.325; CBraMod≥V2 on 23/50 folds. |
| **Interactive retrieval (<600 ms P95)** | **No.** | 55 ms/trial CPU — fine offline, hostile for interactive. |
| **Offline foundation representation** | **Suggestive, not proven.** | Richer geometry; no downstream-task lift shown. |

The central nuance: **CBraMod clusters *by subject* better than *by MI class*** (intra 0.965 ≫ inter-class separability for MI). That is why it wins subject identity but loses MI: subject variance dominates its geometry. This is a feature *for a subject-identity role*, not a bug — but it means CBraMod is **not** a general MI-improvement play.

---

## 5. Subject-level retrieval: what it proves and what it does NOT

**What it proves:** CBraMod's 200-D geometry preserves *same-session subject identity* far better than V2-32-D (subject-Recall@5 +0.331, highly significant). Nearest neighbours of a trial are overwhelmingly the same subject.

**What it does NOT prove:**

1. **Cross-session / longitudinal identity.** PhysioNet EEGMMIDB runs 5 and 6 for a given subject are recorded **in the same session** (back-to-back 4-min runs, identical electrode placement, same day). The leave-one-out-with-self-exclusion computation (the only valid form under strict LOSO — see Mission-9 methodological note) retrieves from the *same session's* other trials. So the metric is **within-session identity clustering**, not cross-day/cross-visit identification.
2. **Invariance to acquisition change.** Because same-session trials share montage, impedance, drift, and amplifier state, a portion of the high Recall is *artifactual* (near-duplicate recordings). The same-subject NN may be close simply because the recordings are near-duplicates, not because the embedding is a robust subject fingerprint.
3. **Generalization to unseen runs/tasks.** Not tested: runs 7–14 (other MI tasks) were not used.

**Bottom line:** the same-session subject-Recall is **necessary but insufficient** evidence for the product claims in Section 3 (longitudinal matching, cross-session retrieval, cohort similarity, EEG fingerprinting). It justifies *targeting* those roles; it does not *validate* them.

---

## 6. The one focused experiment to validate the role

> "Does CBraMod identify/retrieve the same subject from an **unseen acquisition run** (a separate session/context) better than V2, at server-grade precision?"

This tests **cross-run (session-disjoint) subject retrieval** — the actual product claim — rather than re-running MI.

- **Dataset:** PhysioNet EEGMMIDB S001–S050, runs **5, 6, 7, 8, 9, 10** (each run = an independent acquisition session of the same subject; same 64-channel montage, different MI task blocks → tests subject-identity invariance across task/acquisition context). Keep runs 5/6 as the primary comparison set to stay maximally consistent with Mission 9 where desirable; the full 6-run set strengthens the cross-session claim.
- **Train/test split (session-disjoint, leakage-free):** for each (subject, held-out-run) pair, the **held-out run's** trials are queries; the candidate pool is **all trials from all other runs × all subjects** (the held-out run is excluded entirely). This is subject+run-disjoint — the held-out subject appears only via its *other* runs, never its held-out run, so same-session duplication cannot inflate recall.
- **Exact task:** For each query trial, take top-K cosine neighbours in the pool; success if any shares the query's subject. Aggregate subject-Recall@1/5/10 per held-out run, then per model.
- **Metrics:** subject-Recall@1/5/10 (primary); subject silhouette (cosine, by subject) as a secondary quality gate; same-subject vs different-subject NN cosine gap.
- **Models (locked artifacts, read-only):** CBraMod 200-D (native 200-D preserved) vs V2 32-D vs PCA bandpower 32-D (baseline). Same preprocessing/pooling as Mission 9.
- **Statistical test:** paired t-test on per-(subject,run) Recall@K, CBraMod vs V2 vs PCA; **Bonferroni** over the 3 primary Recall metrics × the CBraMod-vs-V2 comparison (α=0.05/3). Report Cohen's d + percentile bootstrap 95% CI of the difference.
- **Effect-size requirement:** Δ ≥ 0.05 on subject-Recall@K (a half-decile retrieval-rank gain is product-meaningful) **and** Bonferroni p < 0.05.
- **Latency requirement:** CBraMod ≥55 ms CPU/trial is acceptable for an offline/batch server path; confirm end-to-end batch retrieval latency against an offline SLO (not the interactive <600 ms browser path).
- **Native dims:** CBraMod stays native 200-D; V2 stays 32-D; no projection to equalize dimensionality (cosine is dimension-agnostic on normalised embeddings).
- **V2 inclusion:** V2 is the *competitor* here, not a target to beat on MI — it represents the current single-tier capability.

**Success gate:** CBraMod subject-Recall@K (mean over (subject,run) folds) > V2 **and** Δ ≥ 0.05 **and** Bonferroni p < 0.05 → promotes the **server-side subject-identity / similarity specialist** role → authorises Mission-11 infrastructure (two-tier).
**Failure gate:** not met → CBraMod reverts to research-only (Role G); no two-tier architecture.

---

## 7. Clean promotion gate (for the representation-specialist role)

| Slot | Metric | Rationale |
|---|---|---|
| **Primary product metric** | Cross-run (session-disjoint) subject-Recall@1/5/10 | The actual product capability: identify subjects across unseen acquisitions. Directly tied to Roles C/D/E. |
| **Secondary quality metric** | Subject silhouette (cosine) | Confirms the geometry genuinely clusters by subject (not by session artifact), dimension-fair. |
| **Safety/guardrail metric** | CBraMod MI accuracy ≥ chance (0.25) | Not "beat V2 on MI" — just "is it a valid, signal-bearing embedding?" (0.304 ≫ 0.25). MI is secondary to the product role; competitiveness is not required for a subject-identity specialist. |
| **Statistical requirement** | Paired t-test CBraMod vs V2; Bonferroni over 3 primary Recall metrics (α=0.05/3≈0.0167) | Controls family-wise error across the three Recall thresholds that measure the same construct. |
| **Effect-size requirement** | Δ ≥ 0.05 on subject-Recall@K | A half-decile rank gain is the smallest product-meaningful retrieval improvement, not a p-value-only win. |

Why these thresholds: the primary metric *is* the product task, so it must be both statistically significant (Bonferroni, to avoid fishing) and *practically* meaningful (Δ≥0.05). The guardrail is deliberately weakened vs Mission 9 because MI decoding is **not** the product task — only signal-presence is required.

---

## 8. Promotion decision

> **CONDITIONAL — RUN ONE TARGETED VALIDATION.**

- Not **PROMOTE** (the real product task — cross-session identity — is unproven; Mission 9 proved only same-session identity).
- Not **DROP** (the within-session subject-identity advantage is large and robust — dismissing it would discard a real capability).
- **Conditional** = the evidence makes the subject-identity server role a strong, low-risk hypothesis worth one clean, product-aligned experiment (Section 6). If that experiment succeeds → promote to two-tier; if it fails → research-only.

---

## 9. V2 vs CBraMod — role separation (they coexist, they don't compete)

| Dimension | V2 (EEGConformer-prod-v2, 32-D) | CBraMod (native 200-D) |
|---|---|---|
| Deployment | Browser / WASM / interactive | Server / CPU onnxruntime / offline-batch |
| Latency | ~6 ms/trial (interactive) | ~55 ms/trial (offline) |
| Dimensionality | 32 (`vector(32)` pgvector) | 200 (`foundation_embeddings` / model-namespaced space) |
| Strength | MI classification, interactive retrieval | Subject identity, cross-session similarity, offline cohort structure |
| MI accuracy | **0.325** (better) | 0.304 (worse, but ≫ chance) |
| Subject-Recall@5 | 0.247 | **0.577** (much better) |
| Silhouette | −0.177 | **−0.044** (better separation) |

They occupy **disjoint latency/dimensionality/deployment** cells, so a single "which is better" MI-style showdown (Mission 9) was always the wrong framing. The architecture should be **complementary**: V2 for anything interactive/real-time and MI-oriented; CBraMod for offline subject-identity / cohort / similarity work that 32-D cannot represent.

---

## 10. Two-tier architecture decision

**Contingent — not yet approved.**

The evidence currently **justifies designing** the two-tier architecture but **not shipping** it:

- **Tier 1 — Interactive:** EEGConformer V2 → 32-D → browser/real-time → `vector(32)`. **(Ships; already GA.)**
- **Tier 2 — Foundation / Server:** CBraMod → native 200-D → server/offline → separate model-namespaced `vector(200)` space (`foundation_embeddings` / `/api/eeg/embed/foundation`). **(Design pending cross-session validation; do NOT build until Section 6 succeeds.)**

Two-tier is the *target architecture if validation succeeds*, and the Mission-9 within-session result is the strongest a priori reason to believe Tier 2 is worth building. But "strong within-session subject identity" ≠ "usable cross-session subject identity," and the gap between them is exactly the failure mode of every EEG fingerprint (montage drift, impedance, day-to-day physiology). Ship Tier 2 only on cross-session evidence.

---

## 12. Exact next mission

**Mission 11 — Cross-session subject-retrieval validation of the CBraMod server role.**
Implement the experiment in Section 6, on locked artifacts (read-only), producing `reports/MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.{md,json}` and appending one record to `benchmark_archive.json`. Run it **before** any two-tier schema/route work. On success → Mission 12 proposes the Tier-2 `foundation_embeddings`/`vector(200)` architecture for review (that is the first mission in which infrastructure is in scope). On failure → CBraMod moves to research-only status; no infrastructure.

---

*Generated 2026-08-13 · zcode-agent · git `b9164a66` · read-only analysis, no production/system changes.*
