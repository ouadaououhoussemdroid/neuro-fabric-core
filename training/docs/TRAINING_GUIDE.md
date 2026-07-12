# EEGConformer Training Guide

Operator-facing instructions for producing the first production
`eegconformer.onnx` artefact for Neuro-Fabric.

## 1. Prerequisites

- Python 3.10–3.11.
- GPU recommended (Colab T4 / local CUDA). CPU works but is slow
  (~10× wall-clock on BCI-IV-2a).
- ~3 GB free disk for the MOABB cache.

```bash
cd training
python -m pip install -r requirements.txt
```

## 2. Configuration

Edit `configs/eegconformer-bciiv2a.yaml` only when intentionally changing
the contract. The defaults match the runtime expectations of
`registerBraindecodeEEGConformer` (22 ch · 250 Hz · 1000 samples · 32-d).

## 3. Pipeline (per stage)

| Stage      | Command                             | Output                                                   |
| ---------- | ----------------------------------- | -------------------------------------------------------- |
| Acquire    | `python scripts/acquire_dataset.py` | `cache/moabb/`                                           |
| Preprocess | `python scripts/preprocess.py`      | `cache/processed/<name>/{train,holdout}.npz`             |
| Train      | `python scripts/train.py`           | `artefacts/<name>/eegconformer.pt`, `train_history.json` |
| Validate   | `python scripts/validate.py`        | `artefacts/<name>/validation_report.json`                |
| Evaluate   | `python scripts/evaluate.py`        | `artefacts/<name>/evaluation_report.json`                |
| Export     | `python scripts/export_onnx.py`     | `artefacts/<name>/eegconformer.onnx`                     |
| Package    | `python scripts/package.py`         | `artefacts/<name>/manifest.json` (+ MODEL_CARD copy)     |

Or run the whole chain:

```bash
bash scripts/run_all.sh
```

## 4. Acceptance criteria

Before shipping the artefact:

- `validation_report.json` → `holdout_accuracy` ≥ 0.55 (4-class chance = 0.25).
- `evaluation_report.json` → `recall_at_k["10"]` ≥ 0.55 (vs ~0.30–0.40 PCA).
- `export_onnx.py` prints `PyTorch↔ORT cosine ≥ 0.999`.
- `manifest.json` includes a sha256 for `eegconformer.onnx`.

## 5. Deployment

Upload `artefacts/<name>/eegconformer.onnx` to either:

- `public/models/eegconformer.onnx` (simple, served by Vite), or
- Lovable Cloud storage with a stable hashed URL.

Then at app boot:

```ts
import { registerBraindecodeEEGConformer } from "@/lib/ai/models/registry";
registerBraindecodeEEGConformer({
  artifact: { kind: "url", url: "/models/eegconformer.onnx" },
});
```

No other code changes are required — the registry, inference engine,
validation layer, benchmark framework, and PCA fallback all remain
untouched.

## 6. Reproducibility

- Seeds are set in `_common.set_seed` from `training.seed` in the YAML.
- All dependencies are pinned in `requirements.txt`.
- Cross-subject hold-out is fixed (`dataset.holdout_subjects` in YAML).
- The exporter enforces `opset=17` and a ≥0.999 parity gate, so the same
  PyTorch checkpoint produces a bit-stable ONNX file.

## 7. Troubleshooting

| Symptom                      | Likely cause              | Fix                                                                |
| ---------------------------- | ------------------------- | ------------------------------------------------------------------ |
| MOABB download stalls        | Mirror unreachable        | Set `MNE_DATA` / `MOABB_DATA` to a pre-staged path                 |
| `channel mismatch: 25 != 22` | Wrong dataset class       | Confirm `BNCI2014_001` (BCI-IV-2a) in `_common.acquire_dataset.py` |
| Parity check < 0.999         | Wrong opset / hook target | Re-run exporter with `--opset 17`; ensure `model.fc` exists        |
| CUDA OOM                     | Batch too large           | Drop `training.batch_size` to 32                                   |
