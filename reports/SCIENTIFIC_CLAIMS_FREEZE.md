# Scientific Claims Freeze

**Date:** 2026-08-20  
**Auditor:** ZCode Autonomous Agent  
**Purpose:** Freeze all existing downstream scientific claims before rehabilitation.  
**Source:** Independent DeepTech Re-Certification Audit (`reports/DSATIVATION_DEEPTDCH_RECERTIFICATION_AUDIT_REPORT.md`)

---

## Purpose

This document freezes every downstream scientific claim currently asserted by the
Neuro-Fabric system. It does **not** delete or amend historical reports. It
classifies each claim according to the new evidence hierarchy:

| Classification | Meaning |
|----------------|---------|
| **VALID / CERTIFIED** | Independent ground truth + valid evaluation verified |
| **ENGINEERING-VALIDATED** | System works technically; scientific validity unestablished |
| **EXPERIMENTAL** | Research-stage, not production-ready |
| **PROXY / DEMONSTRATION** | Useful for engineering; not scientific evidence |
| **BLOCKED** | Requires external data or validation |

All previous metrics below are **FROZEN** and reclassified. They must not be
advertised as real-world scientific performance until independently reproduced
under new methodology.

---

## Frozen Claims

| Service       | Current Metric | Label Source         | Input Source  | Validity                | New Classification            |
|---------------|----------------|----------------------|---------------|-------------------------|-------------------------------|
| **Cognitive** | R² = 0.7348    | Proxy: θ/α band ratio | EEG features  | **INVALID**             | **PROXY / DEMONSTRATION**     |
| **Sleep Staging** | acc = 0.6718 | Proxy: band-power heuristics | EEG features | **INVALID** | **PROXY / DEMONSTRATION** |
| **Sleep Quality** | R² = 0.8193   | Proxy: linear combination of band powers | EEG features | **INVALID** | **PROXY / DEMONSTRATION** |
| **Anomaly** | AUC = 0.892     | Mismatch: Mahalanobis CV vs Ridge served | Joint-2312   | **INVALID**             | **PROXY / DEMONSTRATION** |

---

## Why Each Claim Is Invalid

### Cognitive — R² = 0.7348
- **Training script:** `scripts/train_cognitive_probe.py`
- **Label derivation:** `workload = theta_power / alpha_power` (line ~42)
- **Input features:** `bandPowerFeatures(EEG)` → same band powers
- **Circularity:** The label is a deterministic function of the same band-power
  features used as model input. The Ridge probe learns to invert a known
  mathematical function of its inputs — R² = 0.7348 is a tautology, not a
  measurement of cognitive workload understanding.
- **Split issue:** Uses `KFold(n_splits=50, shuffle=True)` — random K-fold, not
  true LOSO. Subject-level data leakage across folds is possible.
- **Dataset:** EEGMMIDB subjects 1–50. No raw EEG files in repository.

### Sleep Staging — acc = 0.6718
- **Training script:** `scripts/train_sleep_staging_probe.py`
- **Label derivation:** Band-power heuristics (δ/θ/α/β/γ patterns) — NOT
  expert PSG sleep-stage annotations
- **Input features:** `bandPowerFeatures(EEG)` — same features used to derive
  labels
- **Circularity:** Labels are a deterministic function of input features
- **Split issue:** Uses `KFold(n_splits=5)` — random K-fold, not LOSO
- **Registry issue:** `experimentId: "m39-sleep-staging-probe"` points to a
  **seed run** (status: "valid (seed)"), not the actual training run
  (m43, acc=0.6718). The served ONNX SHA matches the seed run, not the
  training run that produced the reported metric.
- **Dataset:** Sleep-EDF (loader exists at `src/lib/datasets/sleep-edf.ts`) but
  no dataset files or PSG annotations are in the repository.

### Sleep Quality — R² = 0.8193
- **Training script:** `scripts/train_sleep_quality_probe.py`
- **Label derivation:** `derive_sleep_quality_from_bandpower()` (lines 68–119)
  — explicit proxy formula using δ, θ, α, β, γ band powers
- **Input features:** `bandPowerFeatures(EEG)` — same band powers
- **Circularity:** The quality proxy score is a deterministic linear combination
  of the input features, passed through a sigmoid
- **Split issue:** Uses `KFold(n_splits=50)` — random K-fold, not LOSO
- **Registry issue:** `experimentId: "m40-sleep-quality-probe"` points to a
  **seed run** (status: "valid (seed)", R²=0.0), not the actual training run
  (m43, R²=0.8193). The served ONNX SHA matches the seed run.
- **Dataset:** Sleep-EDF (loader exists) but no dataset files in repository.

### Anomaly Detection — AUC = 0.892
- **Training script:** `scripts/train_anomaly_probe.py`
- **CV methodology:** Mahalanobis distance (AUC=0.892 on 5-fold LOSO)
- **ONNX artifact:** Ridge regression (MatMul[2312→1] + Add) — NOT Mahalanobis
  distance computation
- **Mismatch:** The docstring claims "Mahalanobis distance" but the export code
  fits Ridge on binary labels (lines 270–274)
- **Reproduced:** Ridge produces AUC ≈ 0.545 (near chance), not 0.892
- **Threshold issue:** Registry threshold `2.5` (raw Mahalanobis space) is
  divided by 10 → `0.25` and applied to Ridge output (semantically meaningless)
- **Dataset:** Synthetic artifact injection (`synthetic_amplitude_spikes +
  channel_dropout`), not real artifact annotations

---

## Preserved Artifacts (Not Deleted)

| Artifact | Location | Role in audit trail |
|----------|----------|---------------------|
| `reports/DSATIVATION_DEEPTDCH_RECERTIFICATION_AUDIT_REPORT.md` | Audit report | Adjudicating audit |
| `scripts/train_cognitive_probe.py` | Training script | Source of circular proxy |
| `scripts/train_anomaly_probe.py` | Training script | Source of Ridge/Mahalanobis mismatch |
| `scripts/train_sleep_staging_probe.py` | Training script | Source of proxy labels |
| `scripts/train_sleep_quality_probe.py` | Training script | Source of proxy labels |
| `reports/benchmark_archive.json` | Benchmark archive | Full experiment history |
| `reports/.joint_embedding_cache.npz` | Cached embeddings | Input data for training |
| `src/lib/ai/decoders/*.registry.ts` | Model registry | Frozen registry entries |

---

## Registry Reclassification

All four Tier-1 registry entries are reclassified:

```typescript
// cognitive.registry.ts
export const COGNITIVE_LINEAR_PROBE_JOINT_2312: TaskHeadDescriptor = {
  ...
  scientificStatus: "PROXY / DEMONSTRATION",
  previousMetric: { r2: 0.7348, rmse: 0.0557, mae: 0.0440, pearson_r: 0.8874 },
  previousMetricStatus: "INVALID — proxy label circularity",
  labelSource: "proxy: theta/alpha band-power ratio",
  inputSource: "bandPowerFeatures (same band powers)",
  circularity: "label = deterministic function of input features",
};

// anomaly.registry.ts
export const ANOMALY_MAHALANOBIS_PROBE_JOINT_2312: TaskHeadDescriptor = {
  ...
  scientificStatus: "PROXY / DEMONSTRATION",
  previousMetric: { auc_roc: 0.892, threshold: 2.5 },
  previousMetricStatus: "INVALID — ONNX is Ridge regression, not Mahalanobis",
  labelSource: "proxy: synthetic artifact injection",
  methodology: "CV uses Mahalanobis distance; ONNX serves Ridge regression",
  mismatch: "AUC 0.892 (Mahalanobis CV) vs AUC ≈0.545 (Ridge served)",
};

// sleep.registry.ts — STAGING
export const SLEEP_STAGING_PROBE_JOINT_2312: TaskHeadDescriptor = {
  ...
  scientificStatus: "PROXY / DEMONSTRATION",
  previousMetric: { acc_5class: 0.6718, macro_f1: 0.2908, kappa: 0.3254 },
  previousMetricStatus: "INVALID — proxy label circularity; registry→seed run",
  labelSource: "proxy: band-power heuristics",
  inputSource: "bandPowerFeatures (same band powers)",
  circularity: "label = deterministic function of input features",
};

// sleep.registry.ts — QUALITY
export const SLEEP_QUALITY_PROBE_JOINT_2312: TaskHeadDescriptor = {
  ...
  scientificStatus: "PROXY / DEMONSTRATION",
  previousMetric: { r2: 0.8193, rmse: 0.0316, mae: 0.0248, pearson_r: 0.9192 },
  previousMetricStatus: "INVALID — proxy label circularity; registry→seed run",
  labelSource: "proxy: linear combination of band powers (sigmoid)",
  inputSource: "bandPowerFeatures (same band powers)",
  circularity: "label = deterministic function of input features",
};
```

---

## Next Steps

1. **Phase 2:** Inventory all available datasets and identify ground-truth labels
2. **Phase 3:** Create dataset manifests
3. **Phase 4:** Implement true LOSO splitting
4. **Phase 5:** Remove circular supervision
5. **Phases 6–9:** Rebuild each probe with genuine labels
6. **Phase 22:** Final report — only claim what the evidence supports
