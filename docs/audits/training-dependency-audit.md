# Training Package — Dependency Audit

**Date:** 2026-06-18
**Trigger:** Real Colab execution failed at training with
`ImportError: cannot import name 'BNCI2014001' from 'moabb.datasets'`.
**Scope:** `training/` only. Production Neuro-Fabric code untouched.

## 1. Root cause

MOABB renamed all dataset classes from `BNCI2014001` → `BNCI2014_001` in
v1.0 (Oct 2023) and kept `BNCI2014001` as a deprecation alias. **MOABB
v1.6 (Jun 2026) removed the alias entirely.**

Braindecode `<1.0` references the OLD name (`BNCI2014001`) at
`braindecode.datasets.moabb` import time. When Colab pip-resolves
`braindecode>=0.8,<0.9` together with `moabb>=1.1,<2.0`, it pulls
Braindecode 0.8.1 + MOABB 1.6.x — exactly the unsupported combination.
`braindecode.models` import paths transitively touch the broken module
and the entire `import braindecode...` call site fails.

The Python error message is the giveaway:
`Did you mean: 'BNCI2014_001'?` — i.e. the NEW name exists, the OLD
name is missing. That is only true on MOABB ≥1.6.

## 2. Compatible-version matrix

| Package      | Old (broken) range          | Pinned (working)  | Notes |
|--------------|------------------------------|-------------------|-------|
| torch        | `2.3.*`                      | `2.4.1`           | CUDA 12, py3.10–3.12 wheels on PyPI. |
| braindecode  | `>=0.8,<0.9`                 | `1.1.1`           | First release that fully migrated to the new MOABB names. |
| moabb        | `>=1.1,<2.0` (resolves to 1.6) | `1.1.1`         | Last 1.x where `BNCI2014_001` is canonical AND the deprecated alias still ships (defensive). |
| mne          | `>=1.7`                      | `1.7.1`           | Required by braindecode 1.1. |
| numpy        | `>=1.26`                     | `1.26.4`          | torch 2.4 wheels are NOT NumPy 2.x compatible on Colab. |
| scipy        | `>=1.11`                     | `1.13.1`          | matches numpy 1.26 ABI. |
| scikit-learn | `>=1.4`                      | `1.5.2`           | needed by moabb 1.1. |
| pandas       | (unpinned)                   | `2.2.2`           | moabb returns metadata via pandas; pinning avoids pyarrow churn. |
| onnx         | `>=1.16`                     | `1.16.2`          | opset 17 works. |
| onnxruntime  | `>=1.18`                     | `1.19.2`          | parity check (cosine ≥ 0.999). |
| pyyaml       | `>=6.0`                      | `6.0.2`           | config loader. |
| tqdm         | `>=4.66`                     | `4.66.5`          | progress bars. |

## 3. Python / Colab compatibility

- **Python:** verified 3.10, 3.11, 3.12. Python 3.13 not supported by
  torch 2.4 wheels — `check_compat.py` blocks it.
- **CUDA:** Colab T4 + CUDA 12 + torch 2.4.1 cu121 default wheels.
- **NumPy 2.x:** explicitly avoided.

## 4. Lock report

`training/requirements.txt` now uses `==` for every package. Re-running
`pip install -r requirements.txt` from a clean Colab kernel produces a
bit-identical environment (modulo CUDA driver build).

## 5. What changed in the training package

- `training/requirements.txt` — fully pinned, with rationale comments.
- `training/scripts/check_compat.py` — **new**. Pre-flight gate that
  verifies versions and the `BNCI2014_001` import before any download.
- `training/scripts/run_all.sh` — runs `check_compat.py` first.
- `training/notebooks/EEGConformer_BCIIV2a.ipynb` — rewritten as valid
  JSON (the previous version had malformed cells). Inserts a dedicated
  compatibility-check cell between `pip install` and `acquire_dataset`.

No production code changed.