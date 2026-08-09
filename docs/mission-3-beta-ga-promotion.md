# Mission 3: EEGConformer Beta → GA Promotion Readiness

**Date:** 2026-08-09  
**Status:** Locally Ready (requires staging infrastructure for full validation)  
**Preceded by:** Mission 2 — EEGConformer Canary Staging Deployment  
**Successor:** None (Mission 3 is the final rollout phase before GA)

---

## 1. Purpose and Context

Mission 2 established the EEGConformer canary mechanism (5% user routing + metrics + rollback). Mission 3 promotes EEGConformer from **beta (50% of users)** to **GA (100% of users)**, gated by performance and quality criteria.

This document defines the **exact acceptance criteria**, the **promotion procedure**, and the **rollback procedure** that must be followed when moving from beta to GA in staging or production.

Source of truth: `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md` — Phase 4 (Rollout), lines 66-71.

---

## 2. Rollout Stages (Recap)

| Stage | Cohort | Flag | Exit Criterion |
|---|---|---|---|
| Canary | 5% | `AI_EEGCONFORMER_ENABLED=canary` | < 0.5% fallback rate over 24h |
| **Beta** | 50% | `AI_EEGCONFORMER_ENABLED=beta` | **P95 latency < 600ms; no error-budget burn** |
| **GA** | 100% | `AI_EEGCONFORMER_ENABLED=ga` | One week green |
| Rollback | n/a | `= off` → `unregisterModel(...)` | < 5 min MTTR |

---

## 3. Beta Exit Criteria (Promotion Gate to GA)

Before promoting from `beta` to `ga`, ALL of the following must be satisfied:

| # | Criterion | Threshold | Source |
|---|---|---|---|
| 1 | Staging inferences | ≥ 1,000 | `docs/archived/audits/2026-06-17_eegconformer-risk-assessment.md:41` |
| 2 | Fallback rate | < 0.5% over 24h | `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md:68` |
| 3 | P95 latency | < 600ms on target devices | `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md:69` |
| 4 | Error budget | No burn (no alert-firing) | `docs/archived/roadmaps/2026-06-17_eegconformer-deployment-roadmap.md:69` |
| 5 | Recall@10 | ≥ PCA + 15pp on BCI-IV-2a holdout | `docs/archived/audits/2026-06-17_eegconformer-risk-assessment.md:43` |
| 6 | MODEL_CARD | Reviewed and signed off | `docs/archived/audits/2026-06-17_eegconformer-risk-assessment.md:44` |
| 7 | Rollback drill | Executed in staging, < 5 min MTTR | `docs/archived/audits/2026-06-17_eegconformer-risk-assessment.md:45` |

## 4. GA Exit Criteria (Stability Confirmation)

Before considering EEGConformer fully GA, the following must hold for **one week**:

| # | Criterion | Threshold |
|---|---|---|
| 1 | Fallback rate | < 0.5% daily |
| 2 | P95 latency | < 600ms |
| 3 | P50 latency | < 400ms |
| 4 | Error budget | No burn |
| 5 | No rollbacks | Zero rollbacks triggered |

---

## 5. Promotion Procedure (Beta → GA)

**Step 1: Verify beta exit criteria** (Section 3 above) using staging metrics.

**Step 2: Execute rollback drill** (Section 6 below) in staging to confirm < 5 min MTTR.

**Step 3: Flip the staging env to GA:**
```bash
# .env.staging.ga
AI_EEGCONFORMER_ENABLED=ga
```

**Step 4: Deploy to staging with GA flag.** Wait 24h, monitor metrics.

**Step 5: If stable for 24h, promote to production:**
```bash
# Production .env
AI_EEGCONFORMER_ENABLED=ga
```

**Step 6: Monitor for one week.** If all GA exit criteria hold, EEGConformer is fully GA.

---

## 6. Rollback Procedure

**To roll back from any stage (beta/GA) to PCA-only:**

```bash
# Set the stage to off in the environment
AI_EEGCONFORMER_ENABLED=off
```

**What happens automatically (per-request via `applyEEGConformerRollout()`):**

1. `getEEGConformerRolloutStage()` reads `AI_EEGCONFORMER_ENABLED=off` → returns `"off"`
2. `setRolloutStage("off")` — cohort gate now returns `false` for all users
3. `hasModel(EEGCONFORMER_ID)` → `true`, so `unregisterModel(EEGCONFORMER_ID)` is called
4. Subsequent `embedEEG()` calls: `isEEGConformerEnabledForUser()` returns `false` → `startId = chain[0]` (PCA)

**MTTR (Mean Time to Recovery):**

- The `unregisterModel()` call is a synchronous `Map.delete()` — **milliseconds**.
- The env var flip is a configuration change — **< 1 second** for the next request to pick it up.
- **Total MTTR: < 5 minutes** (well within the exit criterion).

Verified locally in `beta-deployment.test.ts` → "rollback MTTR (< 5 minutes)" test: unregister + env flip completes in < 100ms.

---

## 7. Metrics to Monitor (via `/api/public/metrics`)

| Metric | Query (Prometheus) | Alert Threshold |
|---|---|---|
| Cohort eligibility | `sum(neuro_fabric_eegconformer_cohort_checks_total{result="hit"})` | Compare hit/miss ratio to expected (5% / 50% / 100%) |
| Fallback rate | `sum(neuro_fabric_embed_fallback_total)` / `sum(neuro_fabric_model_selected_total)` | < 0.5% |
| P95 latency | `histogram_quantile(0.95, sum(rate(neuro_fabric_upload_embed_ms_bucket[5m])) by (le))` | < 600ms |
| P50 latency | `histogram_quantile(0.50, sum(rate(neuro_fabric_upload_embed_ms_bucket[5m])) by (le))` | < 400ms |
| Artifact verification | `sum(neuro_fabric_artifact_verification_total{result="fail"})` | 0 (any failure triggers alert) |
| Model selection | `sum(neuro_fabric_model_selected_total)` by `model`, `fell_back` | Track EEGConformer vs PCA usage |

---

## 8. Local Verification Status

The following have been verified locally via automated tests (no real staging infrastructure required):

| Verification | Test File | Status |
|---|---|---|
| Beta 50% cohort routing | `beta-deployment.test.ts` → "beta routing (~50% cohort)" | ✅ Verified (10,000-sample distribution: 45-55%) |
| EEGConformer registration at beta | `beta-deployment.test.ts` → "EEGConformer is registered at beta stage" | ✅ Verified via `applyEEGConformerRollout()` |
| PCA fallback when model fails | `beta-deployment.test.ts` → "beta stage fallback" tests | ✅ Verified (broken runtime → PCA) |
| Cohort/metrics recording | `beta-deployment.test.ts` → "beta metrics emission" | ✅ Verified (cohortChecks, modelSelected, artifactVerification) |
| Beta→GA promotion gate | `beta-deployment.test.ts` → "beta → GA promotion gate" | ✅ Verified (stage transitions, 100% routing at GA) |
| Rollback beta→off | `beta-deployment.test.ts` → "rollback from beta to off" | ✅ Verified (unregister + PCA fallback) |
| Rollback MTTR | `beta-deployment.test.ts` → "rollback MTTR (< 5 minutes)" | ✅ Verified (< 100ms) |
| P95 latency | `beta-deployment.test.ts` → "performance gate" | ✅ Verified (< 600ms) |
| Recall@10 SLO harness | `beta-deployment.test.ts` → "quality gate" | ✅ Verified (SLO harness validates pass/fail) |

## 9. Blocked by Staging/Production Infrastructure

The following criteria **cannot** be verified locally and require staging deployment:

| Blocked Item | Reason |
|---|---|
| ≥ 1,000 staging inferences | No staging environment with traffic |
| 24h fallback rate < 0.5% | No real user traffic in staging |
| P95 latency < 600ms on target devices | No staging devices to measure |
| Error budget monitoring | No Grafana/Prometheus dashboard wired to staging |
| Recall@10 ≥ PCA + 15pp on BCI-IV-2a | Requires real embeddings from deployed model (not synthetic test inputs) |
| One week green at GA | Requires 7 days of production/staging monitoring |

---

## 10. Summary

Mission 3 defines the **Beta → GA promotion procedure** for EEGConformer. The rollout mechanism (50% beta, 100% GA, rollback to PCA) is fully implemented in the existing codebase. All locally-verifiable criteria have been tested with synthetic data. The remaining criteria require staging infrastructure and real traffic, which is outside the scope of local implementation.
