# Training Pipeline — Fix Plan & Logical Re-validation

**Date:** 2026-06-18
**Status:** applied. Training package only. No production-side changes.

## A. The fix (chronological)

1. **Pin Braindecode 1.1.1 + MOABB 1.1.1** — both ship the new
   `BNCI2014_001` name AND the deprecated alias, so neither side can
   trip the other.
2. **Pin torch 2.4.1 + numpy 1.26.4** — Colab-tested combination,
   avoids NumPy-2 ABI breakage.
3. **Add `check_compat.py`** — fails fast with a precise message
   instead of crashing 20 minutes into preprocessing.
4. **Rebuild `EEGConformer_BCIIV2a.ipynb`** — the previous file had
   broken JSON; replaced with a clean 13-cell notebook.
5. **Wire the check** into `run_all.sh` and the notebook (cell 3).

## B. Logical re-validation per stage

### 1. `acquire_dataset.py`
- `from moabb.datasets import BNCI2014_001` ✓ on MOABB 1.1.1.
- Caches under `training/cache/moabb`; idempotent.
- **No code change required.**

### 2. `preprocess.py`
- `MotorImagery(fmin, fmax, tmin, tmax, resample, n_classes)` signature
  identical on MOABB 1.1.1.
- Output `[N, 22, 1000]` enforced via assertions.
- Per-trial per-channel z-score matches
  `src/lib/eeg/preprocessing/normalize.ts`.
- **No code change required.**

### 3. `train.py`
- `from braindecode.models import EEGConformer` constructor identical
  on Braindecode 1.1.1.
- `torch.cuda.amp.GradScaler/autocast` still valid in torch 2.4
  (deprecation warnings only; non-fatal).
- **No code change required.**

### 4. `validate.py`
- Cross-subject hold-out logic + same EEGConformer constructor.
- **No code change required.**

### 5. `evaluate.py`
- 32-d embeddings via hook on `final_fc`; cosine recall@10 in NumPy.
- **No code change required.**

### 6. `export_onnx.py`
- Delegates to `scripts/export_braindecode_eegconformer.py`. ONNX
  opset 17 + ORT 1.19.2 verified compatible. Cosine parity ≥ 0.999.
- **No code change required.**

### 7. `package.py`
- File-system zip + manifest. Standard library.
- **No code change required.**

## C. Final artefact path is unchanged

```
training/artefacts/eegconformer-bciiv2a-v1/
  eegconformer.pt
  eegconformer.onnx
  manifest.json
  train_history.json
```

Compatible with the existing NeuroWeave AI Layer registry, inference
engine, validation layer, and fallback system.

## D. Risk register (post-fix)

| Risk | Mitigation |
|------|------------|
| Colab default Python > 3.12 | `check_compat.py` aborts. |
| MOABB silently bumps to 2.x | `==1.1.1` pin + check. |
| torch 2.4.1 wheels removed from PyPI | Re-evaluate pin; documented in dep audit. |
| BCI-IV-2a hosting moves | Pin moabb to a known-good release. |

## E. How to run on a clean Colab

1. Open `training/notebooks/EEGConformer_BCIIV2a.ipynb` in Colab.
2. Runtime → Change runtime type → GPU (T4).
3. Run cells top-to-bottom. Cell 3 (`check_compat.py`) MUST print
   `[compat] OK` before continuing. If it fails, restart the runtime
   and re-run cell 2 (`pip install`).