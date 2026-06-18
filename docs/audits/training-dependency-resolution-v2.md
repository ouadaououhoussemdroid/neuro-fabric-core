# Training Package — Dependency Resolution v2

**Date:** 2026-06-18
**Trigger:** Colab `pip install -r requirements.txt` failed with
`No matching distribution found for braindecode==1.1.1` — the version we
had pinned in v1 does not exist on PyPI.
**Scope:** `training/` only.

## 1. What v1 got wrong

| Pin (v1)            | Reality on PyPI                                                  |
|---------------------|------------------------------------------------------------------|
| `braindecode==1.1.1`| Not published. Available 1.x: 1.0.0, **1.1.0**, 1.2.0, …, 1.5.2. |
| `moabb==1.1.1`      | Not published. 1.x line jumps 0.4.6 → **1.4.0** → 1.4.2/1.4.3/1.5.|
| `mne==1.7.1`        | Incompatible — braindecode 1.1.0 requires `mne>=1.10`.           |
| `numpy==1.26.4`     | Incompatible — moabb 1.4.x requires `numpy>=2`.                  |
| `torch==2.4.1`      | Works, but torch 2.5.1 is the lowest minor that pairs cleanly with the torchaudio wheel braindecode pulls in. |

v1's pins were never reproducibly resolvable on a clean Python 3.11/3.12
environment.

## 2. Resolution method

1. Queried PyPI for the actual published version lists.
2. Built a clean Python 3.11 venv (Colab-equivalent interpreter).
3. Iterated pip resolution until the dependency graph closed *without*
   `--no-deps` hacks.
4. Ran the imports the training scripts actually use:
   `from moabb.datasets import BNCI2014_001` and
   `from braindecode.models import EEGConformer`.

## 3. Locked set (verified install + import)

| Package      | Version  | Why this exact pin                                       |
|--------------|----------|----------------------------------------------------------|
| torch        | 2.5.1    | Lowest minor whose torchaudio wheel loads under CUDA 12. |
| torchaudio   | 2.5.1    | Must match torch minor — mismatched wheels fail to load `libcudart`. |
| braindecode  | 1.1.0    | First published 1.x; uses the new `BNCI2014_001` MOABB name. |
| moabb        | 1.4.0    | Lowest 1.x actually on PyPI; exports `BNCI2014_001`.    |
| mne          | 1.10.1   | Required floor by braindecode 1.1.0.                     |
| numpy        | 2.0.2    | Required by moabb 1.4; supported by torch 2.5.           |
| scipy        | 1.14.1   | NumPy 2.x ABI.                                           |
| scikit-learn | 1.5.2    | Required by moabb.                                       |
| pandas       | 2.2.3    | NumPy 2.x ABI.                                           |
| onnx         | 1.17.0   | Stable export at opset 17.                               |
| onnxruntime  | 1.20.1   | CPU parity check.                                        |
| pyyaml       | 6.0.2    | Config loader.                                           |
| tqdm         | 4.66.5   | Progress bars.                                           |

## 4. Reproducibility evidence

Clean Python 3.11.14 venv, fresh pip cache:

```
$ pip install -r training/requirements.txt
…
Successfully installed braindecode-1.1.0 moabb-1.4.0 mne-1.10.1
  numpy-2.0.2 torch-2.5.1 torchaudio-2.5.1 onnx-1.17.0 onnxruntime-1.20.1 …

$ python training/scripts/check_compat.py
  ✓ torch: 2.5.1
  ✓ torchaudio: 2.5.1
  ✓ braindecode: 1.1.0
  ✓ moabb: 1.4.0
  ✓ mne: 1.10.1
  ✓ numpy: 2.0.2
  ✓ moabb.datasets.BNCI2014_001 resolves
  ✓ braindecode.models.EEGConformer resolves
[compat] OK — environment matches the pinned contract.
```

## 5. check_compat.py changes

- Reads versions via `importlib.metadata.version()` instead of module
  `__version__`. **moabb 1.4.0 ships with `__version__ == "1.3.0"`** — a
  known upstream oversight that broke the previous check.
- Strips PEP 440 local-version suffixes (`2.5.1+cu124` → `2.5.1`) before
  comparing, so Colab's CUDA-tagged torch wheels validate correctly.

## 6. Outstanding risks

- Colab occasionally upgrades its preinstalled torch ahead of our pin.
  The notebook's first cell is `pip install -r requirements.txt`, which
  forces our exact set; `check_compat.py` then aborts if anything drifts.
- numpy 2.x bytecode mismatch with cached wheels in long-lived runtimes:
  recommend "Runtime → Disconnect and delete runtime" before installing.