# Mission 26 — EEGPT 62→22 Remap Viability

## Status: **FAIL**

> EEGPT is dropped as a server-backbone candidate. Close the EEGPT remap thread.

---

## 1. Objective

Evaluate whether EEGPT's 2048-D ViT representation survives dropping 40/62 channels down to the 22-channel production montage. This study gates EEGPT's future role as a server-side 2048-D representation backbone.

EEGPT is the only remaining model with plausible 2048-D headroom (V2 is capped at 32-D, CBraMod at 200-D). If the remap fails, EEGPT has no production role and is dropped.

---

## 2. Final Results Table

| Gate | Metric | Threshold | EEGPT Result | Status |
| ---- | ------ | --------: | -----------: | ------ |
| A    | 62→22 cosine similarity | ≥ 0.90 | 0.9747 | **PASS** |
| B    | 22-ch MI accuracy | ≥ V2 (0.3428) | 0.2833 vs 0.3428 | **FAIL** |

---

## 3. Decision

**FAIL** — Gate B does not pass.

- Gate A passes: cosine similarity = 0.9747 (well above 0.90 threshold). The EEGPT representation is structurally preserved when 40 of 62 channels are zero-masked.
- Gate B fails: EEGPT 22-channel MI accuracy = 0.2833, below the V2 safety floor of 0.3428 (Δ = -0.0595).

EEGPT is dropped as a server-backbone candidate. Close the EEGPT remap thread. No further work on EEGPT as a server representation is warranted.

---

## 4. Artifacts

### EEGPT
- **Path**: `public/models/eegpt-encoder-int8.onnx`
- **SHA-256**: `a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36`
- **Size**: 24.94 MB
- **Quantization**: INT8
- **Input shape**: `[1, 62, 1000]` (62 channels, 250 Hz, 4-second window)
- **Output shape**: `[1, 31, 2048]` → mean-token pooled to `[1, 2048]`
- **Embedding dimension**: 2048
- **Internal architecture**: 62→19 ChanProj → 8 ViT blocks → 512-dim, 8 heads → 2048-D output

### V2 (baseline)
- **SHA-256**: `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931`
- **Baseline accuracy**: 0.3428 (50-subject LOSO, T-031)
- **Baseline std**: 0.0843

---

## 5. Datasets

- **PhysioNet EEGMMIDB 1.0.0** (S001-S010, runs 5-6)
  - 4-class MI: left hand (T1/run5), right hand (T2/run5), feet (T1/run6), tongue (T2/run6)
  - 64 channels @ 160 Hz
  - 30 trials per subject (T1/T2 events only)
- **10 subjects** used (matching T-030's original EEGPT evaluation scope)
- **300 total trials** (30 per subject × 10 subjects)

---

## 6. Channel Mapping

### 22-channel production montage (PROD_CHANNELS_22)

All 22 production channels exist within the EEGPT 62-channel layout:

| Channel | EEGPT Index | Notes |
|---------|-----------|-------|
| FP1 | 0 | Frontopolar |
| FP2 | 2 | Frontopolar |
| F5 | 8 | Frontal |
| F6 | 14 | Frontal |
| F3 | 9 | Frontal |
| F4 | 13 | Frontal |
| F1 | 10 | Frontal |
| F2 | 12 | Frontal |
| FC5 | 17 | Frontocentral |
| FC6 | 23 | Frontocentral |
| FC3 | 18 | Frontocentral |
| FC4 | 22 | Frontocentral |
| C5 | 26 | Central |
| C6 | 32 | Central |
| C3 | 27 | Central |
| C4 | 31 | Central |
| T7 | 25 | Temporal |
| T8 | 33 | Temporal |
| P7 | 43 | Parietal |
| P8 | 51 | Parietal |
| P5 | 44 | Parietal |
| P6 | 50 | Parietal |

- **Channels preserved**: 22 of 62
- **Channels zero-filled**: 40 of 62 (64.5% dropped)
- **Interpolation**: PO5/PO6 (required by EEGPT but absent from PhysioNet) interpolated from neighbors (PO7/PO3, PO4/PO8) — per the existing benchmark_tier4.py convention
- **Remap method**: Channel subset mask with zero-filling — no learned projection, no interpolation learning, no model modification

---

## 7. Preprocessing

Mirrors `benchmark_tier4.py` + `verify_eegpt.py`:

1. Channel selection: Map 64-channel PhysioNet → 62-channel EEGPT layout (PO5/PO6 interpolated)
2. Resample: 160 Hz → 250 Hz (linear interpolation)
3. Bandpass: [1.0, 40.0] Hz (FIR, firwin design) — EEGPT's training bandwidth
4. Crop/pad: 4-second window at 250 Hz = 1000 samples
5. Z-score: Per-channel normalization

**Note**: EEGPT uses [1.0, 40.0] Hz bandpass (broader than V2's [4, 38]), matching the model's training distribution.

**Note**: The existing `benchmark_tier4.py` `run_inference()` flattens EEGPT's `[1, 31, 2048]` output to 63,488-D. This M26 script uses mean-token pooling across the 31 patch-token axis to produce the correct 2048-D embedding, matching the TypeScript `applyOutputPooling("mean-tokens")` fix in `FIX_COMPLETION_REPORT.md`.

---

## 8. Gate A — Representation Preservation

### Methodology

For each EEG trial where the full 62-channel signal is available:

- **Path A (Native)**: 62-channel EEG → EEGPT → 2048-D embedding
- **Path B (Projected)**: 62-channel EEG → zero-fill 40 non-production channels → EEGPT → 2048-D embedding

Cosine similarity computed: cos(emb_native, emb_projected)

### Results

| Statistic | Value |
|-----------|-------|
| Samples | 300 |
| Mean cosine | 0.9747 |
| Median cosine | 0.9769 |
| Std | 0.0089 |
| Min | 0.9363 |
| P10 | 0.9635 |
| P25 | 0.9696 |
| P75 | 0.9808 |
| P90 | 0.9839 |
| Max | 0.9901 |
| Fraction ≥ 0.90 | 1.0000 (100%) |

### Analysis

The representation is structurally well-preserved when 40 of 62 channels are zero-masked. The mean cosine (0.9747) far exceeds the 0.90 threshold, and 100% of individual samples exceed 0.90 (minimum = 0.9363). This confirms that EEGPT's ChanProj internal layer (62→19) makes the model robust to channel dropout.

However, representation preservation alone is insufficient — the model must also produce accurate MI classifications from the 22-channel input (Gate B).

---

## 9. Gate B — MI Safety Floor

### Methodology

- 10-subject LOSO cross-validation using 22-channel EEGPT embeddings (zero-filled 62-channel input)
- Nearest-centroid classification with cosine similarity (same methodology as benchmark_tier4.py)
- V2 baseline: 0.3428 (50-subject LOSO from T-031)

### Results

| Subject | Accuracy |
|---------|----------|
| S001 | 0.5667 |
| S002 | 0.2333 |
| S003 | 0.1667 |
| S004 | 0.2333 |
| S005 | 0.2667 |
| S006 | 0.2000 |
| S007 | 0.3000 |
| S008 | 0.4333 |
| S009 | 0.2333 |
| S010 | 0.2000 |

| Statistic | EEGPT 22-ch | V2 (baseline) |
|-----------|-----------|---------------|
| Mean accuracy | 0.2833 | 0.3428 |
| Std | 0.1240 | 0.0843 |
| CI95 lower | 0.2065 | — |
| CI95 upper | 0.3602 | — |
| Delta (EEGPT - V2) | -0.0595 | — |

### Analysis

EEGPT 22-channel accuracy (0.2833) is below the V2 safety floor (0.3428) by -0.0595 (17.4% relative). The 95% CI [0.2065, 0.3602] overlaps with V2's mean but does not exceed it.

Notably, the T-030 corrected benchmark showed EEGPT at 0.3067 (10-subj, full 62 channels) — slightly above the 0.2833 we observe with 22 channels. The channel reduction drops accuracy further, and it remains below V2.

The high variance across subjects (0.1667 to 0.5667) suggests EEGPT is highly sensitive to individual subject characteristics, making the 22-channel projection unreliable as a production path.

---

## 10. Comparison Against Existing Evidence

### T-030 Corrected Results (benchmark_archive.json, 10 subjects)

| Model | Accuracy | vs PCA | p-value |
|-------|----------|--------|---------|
| PCA (bandpower) | 0.2900 | — | — |
| EEGPT (full 62-ch) | 0.3067 | +0.0167 | 0.343 |
| V2 (original) | 0.2826 | -0.0074 | — |

### M26 Results (10 subjects)

| Model | Accuracy | vs V2 (0.3428) |
|-------|----------|----------------|
| EEGPT (22-ch zero-filled) | 0.2833 | -0.0595 |
| V2 (from T-031) | 0.3428 | — |

EEGPT 22-channel accuracy (0.2833) is even lower than EEGPT's full 62-channel accuracy (0.3067), confirming that channel reduction negatively impacts performance. It also falls below V2's 50-subject accuracy (0.3428).

---

## 11. Scientific Integrity

This is an evaluation-only mission. The following constraints were honored:

| Constraint | Status |
|-----------|--------|
| No training | ✅ |
| No fine-tuning | ✅ |
| No hyperparameter optimization | ✅ |
| No model modification | ✅ |
| No ONNX modification | ✅ |
| No artifact replacement | ✅ |
| No quantization | ✅ |
| No architecture changes | ✅ |
| No production rollout | ✅ |
| No DEFAULT_PREFERRED changes | ✅ |
| No V2 changes | ✅ |
| No CBraMod changes | ✅ |
| No M25 joint-264 changes | ✅ |
| No historical benchmark rewrite | ✅ |

---

## 12. Regression Checks

| Check | Status |
|-------|--------|
| Existing V2 tests unchanged | ✅ |
| Existing CBraMod tests unchanged | ✅ |
| M25 joint-264 path unchanged | ✅ |
| No production rollout changes | ✅ |
| No artifact SHA changes | ✅ |
| No historical benchmark modifications | ✅ |

### Test results
- Joint fusion unit tests: 15/15 passed
- Joint Tier-2 E2E tests: 6/6 passed
- EEGPT/Models registration tests (tier4-registration): 27/27 passed

---

## 13. Limitations

1. **10 subjects** (not 50): Used 10 subjects to match T-030's original EEGPT evaluation scope, as full 50-subject evaluation requires ~42 minutes of EEGPT inference (830ms × 2 passes × ~1500 trials). The V2 baseline of 0.3428 is from T-031's 50-subject LOSO. A 50-subject M26 follow-up would tighten the CI but is unlikely to change the conclusion given the -0.0595 delta.

2. **10-subject V2 comparison caveat**: V2's 0.3428 was computed on 50 subjects. A 10-subject V2 baseline would likely dif
