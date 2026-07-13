# Evaluation Runbook — EEGConformer Empirical Validation

## Status: COMPLETE

The EEGConformer has been trained and evaluated. Results are published below.

## Training results

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Best validation accuracy | 0.587 (epoch 49) | ≥ 0.55 | PASS |
| Holdout accuracy (subject 9) | 0.578 | ≥ 0.55 | PASS |
| Recall@10 | 0.941 | ≥ 0.55 | PASS |
| PCA baseline recall@10 | 0.943 | — | — |
| Beats PCA | false | true | NOTE |
| Separation margin | 0.230 | > 0 | PASS |
| PyTorch↔ONNX cosine | ≥ 0.999 | ≥ 0.999 | PASS |

**Per-class holdout accuracy:**

| Class | Accuracy |
|---|---|
| 0 (left hand) | 32.6% |
| 1 (right hand) | 53.5% |
| 2 (feet) | 61.1% |
| 3 (tongue) | 84.0% |

## Artefact locations

- **Trained ONNX (production):** `public/models/eegconformer.onnx` (3,236,663 bytes)
- **SHA-256:** `83fcf5dd9fee09aadabbb421e03c6d4b8c8b909ce28306549b909e5e66130921`
- **Training reports:** `training/artefacts/eegconformer-bciiv2a-v1/`
  - `train_history.json` — 80 epochs, best val_acc 0.587
  - `validation_report.json` — holdout accuracy 0.578
  - `evaluation_report.json` — recall@10 0.941, cosine analysis
  - `manifest.json` — full artefact manifest with SHA-256 hashes

## Re-evaluation workflow

To re-run the evaluation after retraining:

```bash
cd training/scripts
python evaluate.py --config ../configs/eegconformer-bciiv2a.yaml --k 10
python validate.py --config ../configs/eegconformer-bciiv2a.yaml
```

## Interpretation

The EEGConformer passes all acceptance thresholds except `beats_pca`.
The model was trained for 4-class classification (logits), not for
embedding quality (similarity search). The positive separation margin
(0.230) confirms that same-class embeddings cluster together, but the
PCA baseline matches recall@10 on this small holdout set (576 samples).

**Future work:** Fine-tune the embedding head for similarity search
to close the gap vs PCA.
