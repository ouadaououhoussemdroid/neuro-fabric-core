# Evaluation Runbook — EEGConformer Empirical Validation

> **Phase 1C-2.** This document describes the evaluation workflow for
> producing defensible embedding-quality numbers. It is a **preparation
> step** — actually running it requires a trained checkpoint (see the
> provenance warning in `MODEL_CARD.md`).

## Prerequisites

1. A trained checkpoint at `training/artefacts/eegconformer-bciiv2a/eegconformer.pt`.
   - Currently **missing** — the shipped ONNX is random-init.
   - To produce one: `cd training && make train MODEL=eegconformer DATASET=bciiv2a`
   - Or run the Colab notebook: `training/notebooks/EEGConformer_BCIIV2a.ipynb`
2. The holdout data at `training/cache/processed/eegconformer-bciiv2a/holdout.npz`.
   - Produced by `make preprocess` (step 1 of the pipeline).
3. GPU recommended (CPU works but is ~10× slower for the 9-subject dataset).

## Workflow

### Step 1: Re-export the ONNX with the trained checkpoint

```bash
python scripts/export_braindecode_eegconformer.py \
  --architecture EEGConformer \
  --checkpoint training/artefacts/eegconformer-bciiv2a/eegconformer.pt \
  --out public/models/eegconformer.onnx \
  --channels 22 --samples 1000 --classes 4 --opset 17
```

This runs the PyTorch→ONNX export **with** the trained weights, then verifies:

- `onnx.checker.check_model` (graph validity)
- PyTorch↔ORT cosine > 0.999 (export parity)
- `onnx-simplifier` + `onnxoptimizer` (graph optimisation, T-023)

### Step 2: Re-generate the artefact manifest

The Vite plugin regenerates `public/models/manifest.json` at build time.
To regenerate manually:

```bash
node -e "
  const { generateArtefactManifest, writeArtefactManifest } =
    require('./vite-plugins/artefact-manifest.ts');
  writeArtefactManifest('public/models');
"
```

Or simply run `bun run build` (the `artefactManifestPlugin` runs automatically).

### Step 3: Run the evaluation

```bash
cd training/scripts
python evaluate.py --config ../configs/eegconformer-bciiv2a.yaml --k 10
```

This produces `training/artefacts/eegconformer-bciiv2a/evaluation_report.json`
containing:

- `cosine_analysis`: intra/inter-class cosine similarity (mean ± std)
- `cosine_analysis.separation_margin`: intra - inter (higher is better)
- `recall_at_k["10"]`: recall@10 for the trained model
- `pca_baseline.recall_at_k`: recall@10 for PCA on the same embeddings
- `beats_pca`: boolean (recall > pca_recall)

### Step 4: Run the validation (holdout accuracy)

```bash
cd training/scripts
python validate.py --config ../configs/eegconformer-bciiv2a.yaml
```

Produces `validation_report.json` with `holdout_accuracy` and per-class accuracy.

### Step 5: Interpret the results

**Acceptance criteria** (from `TRAINING_GUIDE.md` and the 06-17 risk assessment):

| Metric              | Threshold | Notes                                          |
| ------------------- | --------- | ---------------------------------------------- |
| `holdout_accuracy`  | ≥ 0.55    | Cross-subject (subject 9 holdout)              |
| `recall_at_k["10"]` | ≥ 0.55    | vs ~0.30–0.40 for PCA baseline                 |
| `beats_pca`         | true      | The learned representation must outperform PCA |
| `separation_margin` | > 0       | Intra-class cosine > inter-class cosine        |
| PyTorch↔ORT cosine  | ≥ 0.999   | Export parity (checked at export time)         |

**If the thresholds are not met:**

- Do not flip `DEFAULT_EMBEDDER_ID` to `braindecode-eegconformer-prod`.
- The PCA fallback remains the production default.
- Investigate: learning rate, epoch count, data augmentation, subject variability.

### Step 6: Wire the results into the model card

After successful evaluation, run the packaging script to fill in the model card:

```bash
cd training/scripts
python package.py --config ../configs/eegconformer-bciiv2a.yaml
```

This populates `manifest.json` with `validation_report`, `evaluation_report`,
and `train_history`, and copies everything to `artefacts/eegconformer-bciiv2a-v1/`.

## Current status

- **Artefact:** Random-init (untrained). Provenance warning in `MODEL_CARD.md`.
- **Evaluation script:** Ready (`evaluate.py` extended in T-010).
- **Validation script:** Ready (`validate.py`).
- **Holdout data:** Not generated (requires `preprocess.py` which downloads BCI-IV-2a via MOABB).
- **Blocking:** This is the single highest-leverage action in the 180-day roadmap.
