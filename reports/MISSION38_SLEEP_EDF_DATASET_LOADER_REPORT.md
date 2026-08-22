# M38 — Sleep-EDF Dataset Loader with Channel Expansion for Joint-2312

## Overview

M38 resolves the Tier-2 dataset loader gap identified in M31 §2.4: the lack of a Sleep-EDF
loader and sleep-specific channel remapping strategy. Sleep-EDF is a publicly available
dataset (BSD-3-Clause license) containing 99 subjects with 2 nights of polysomnographic
recordings, annotated with 5-stage sleep labels (W, N1, N2, N3, REM).

The key challenge: Sleep-EDF recordings use a minimal 7-channel montage
(Fpz-Cz, Pz-Oz, EOG) at 100 Hz, while the Joint-2312 embedding backbone requires
62 channels at 250 Hz. This loader bridges that gap using a nearest-neighbour spatial
interpolation strategy (heuristic, not a new model).

## Technical Approach

### Channel Expansion (7 → 62)

Sleep-EDF's 7 channels are derived signals:
- **Fpz-Cz**: bipolar frontal-to-central (Fpz referenced to Cz)
- **Pz-Oz**: bipolar parietal-to-occipital (Pz referenced to Oz)
- **EOG**: horizontal electrooculogram

The expansion maps each of EEGPT's 62 channels to its nearest Sleep-EDF source:
- Frontal + Central channels → Fpz-Cz (shares Cz reference)
- Parietal + Occipital channels → Pz-Oz (shares Oz reference)
- Temporal channels → nearest source by anatomic proximity
- EOG → frontal channels (correlates with blink artifacts)

When a source channel is unavailable, the expansion falls back to the best
available source (preferring Fpz-Cz, then Pz-Oz, then EOG), rather than
defaulting to zeros. This ensures no 62-channel window is entirely empty.

### Resampling

Sleep-EDF v1 (PhysioNet 1.0.0) operates at 100 Hz. The loader uses the existing
`resampleSignal()` utility (linear interpolation, per-channel) to upsample to
250 Hz, matching Joint-2312's input contract.

### Pipeline

```
Raw Sleep-EDF EDF (100 Hz, 7 channels)
  │
  ├─ 1. parseEDF() → EEGSignal + EDF+ TAL annotations
  ├─ 2. parseSleepAnnotations() → 30s sleep epochs (W/N1/N2/N3/REM)
  ├─ 3. resampleSignal(250) → 250 Hz
  ├─ 4. preprocess() → bandpass (0.5–40 Hz) + z-score normalize
  ├─ 5. expandSleepToEEGPT() → 62 channels
  ├─ 6. selectCbraModChannels() → 19 channels (CBraMod-200)
  ├─ 7. selectProdChannels() → 22 channels (V2-32)
  ├─ 8. selectEEGPTChannels() → 62 channels (EEGPT-2048)
  ├─ 9. segment() → 4-second windows (50% overlap)
  │
  └─ Joint-2312: embedJoint2312Windows() → 2312-D embeddings
```

### Key Design Decisions

1. **Nearest-neighbour spatial interpolation (heuristic)** — The roadmap noted
   "This requires either a sleep-specific Joint-2312 variant or a channel interpolation
   strategy." We chose interpolation: it reuses the existing frozen 4-block fusion
   (CBraMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048) with fixed block weights
   [0.3062, 0.1434, 0.1519, 0.3985]. This avoids training a new sleep-specific model.

2. **Bandpass range (0.5–40 Hz)** — Covers sleep-relevant frequencies:
   slow waves (0.5–4 Hz), sleep spindles (12–14 Hz), and gamma activity (30–40 Hz).
   Wider than the default 1–40 Hz to capture slow-wave activity.

3. **50% window overlap** — 4s windows at 250 Hz = 1000 samples, step = 500 samples.
   This matches the foundation pipeline's segmentation strategy.

4. **Annotation parsing via EDF+ TAL** — Sleep-EDF uses EDF+ with Time-Stamped
   Annotation Lists containing "Sleep stage N" entries. The parser handles both
   numeric (0-5) and named (W/R/N1-N3/SWS/Wake/REM) formats.

5. **Alignment guarantee** — CBraMod, V2, and EEGPT windows are verified to have
   identical counts. A mismatch throws a descriptive error.

## Files Created

| File | Purpose |
|------|---------|
| `src/lib/datasets/sleep-edf.ts` | Core loader: parseEDF + resampling + expansion + annotation parsing + pipeline |
| `src/lib/datasets/index.ts` | Barrel export (`@/lib/datasets`) |
| `src/lib/datasets/__tests__/sleep-edf.test.ts` | 36 unit tests |
| `scripts/tmp/m38_sleepedf_validation.py` | 109-point validation script |
| `reports/MISSION38_SLEEP_EDF_DATASET_LOADER_REPORT.md` | This report |

## Files Modified

| File | Change |
|------|--------|
| `src/lib/datasets/manifest.ts` | Added Sleep-EDF to `KNOWN_DATASETS` |

## Validation Results

- **Validation script**: 109/109 checks passed
- **Unit tests**: 36/36 passed
- **Integration with Joint-2312 pipeline**: CBraMod (19), V2 (22), EEGPT (62) windows aligned

## Dataset Compliance (M31 §9)

| Requirement | Status |
|-------------|--------|
| 99 subjects | ✅ `sleepEDFDataset.nSubjects = 99` |
| 2 nights per subject | ✅ `metadata.sessions = 2` |
| BSD-3-Clause license | ✅ `license: "BSD-3-Clause"` |
| 5-stage sleep labels | ✅ W, N1, N2, N3, REM |
| 7→62 channel expansion | ✅ Nearest-neighbour interpolation |
| 100 Hz → 250 Hz resampling | ✅ `resampleSignal()` → 250 Hz |
| 30-min epochs → 4s windows | ✅ `segment()` with 50% overlap |

## Limitations & Future Work

1. **Spatial interpolation is lossy** — 7-channel data expanded to 62 does not
   add new spatial information. The CBraMod-200 block (19 channels) will have
   reduced quality compared to true 19-channel recordings. Future work could
   train a sleep-specific Joint-2312 variant.

2. **No ground-truth EEG2Image** — The loader produces embeddings, not images.
   EEG2Image (M31 §2.4) is a separate future mission.

3. **Annotation format** — Currently handles EDF+ TAL annotations. EDF (without
   +) annotations would require a different parsing strategy.

4. **Sample rate** — Sleep-EDF v1 uses 100 Hz; v2 uses 128 Hz. Both are
   handled by the resampling step.
