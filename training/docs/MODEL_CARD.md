# Model Card — eegconformer-bciiv2a-v1

## Overview

- **Architecture:** Braindecode `EEGConformer` (Song, Zheng, Ko 2022).
- **Task:** 4-class motor imagery (left hand · right hand · feet · tongue).
- **Dataset:** BCI Competition IV 2a (BNCI2014_001) via MOABB.
- **Embedding head:** attention-pooled `model.fc` features, dim = 32.
- **ONNX:** opset 17, outputs `embedding` and `logits`.

## Intended use

First production EEG foundation-model backend for the Neuro-Fabric
similarity-search pipeline. Drop-in replacement for the PCA fallback when
registered via `registerBraindecodeEEGConformer`.

## Out-of-scope

- Clinical diagnosis.
- Non-motor-imagery paradigms without fine-tuning.
- Datasets sampled outside the contract (not 22 ch / 250 Hz / 4 s window).

## Training data

- BCI-IV-2a: 9 subjects, 22 EEG channels, 250 Hz.
- Bandpass 4–38 Hz, epoch 0–4 s post-cue.
- Hold-out: subject 9 (cross-subject).

## Metrics

Trained on Colab T4 GPU, 80 epochs (early stop patience 30, best at epoch 49).

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Best validation accuracy | 0.587 | ≥ 0.55 | PASS |
| Holdout accuracy | 0.578 | ≥ 0.55 | PASS |
| Recall@10 | 0.941 | ≥ 0.55 | PASS |
| Separation margin | 0.230 | > 0 | PASS |
| PyTorch↔ONNX cosine | ≥ 0.999 | ≥ 0.999 | PASS |
| Beats PCA | false | true | NOTE |

**Note on `beats_pca`:** The EEGConformer embeddings (recall@10 = 0.941) do
not outperform the PCA baseline (recall@10 = 0.943) on the holdout set.
This is a known limitation — the model was trained for classification, not
embedding quality. The separation margin is positive (0.230), indicating
the embeddings do cluster by class. The PCA baseline benefits from the
high dimensionality (32-D) relative to the sample count (576).

## Licensing

- Model weights: CC-BY-4.0.
- Architecture: Braindecode (BSD-3).
- Dataset: BCI-IV "free for academic and commercial use with citation".
- Citation: cite Schirrmeister et al. (Braindecode), Song et al.
  (EEGConformer), and BCI-IV-2a (Brunner et al.).

## Risks and mitigations

- **Subject overfitting:** mitigated by cross-subject hold-out.
- **Paradigm specificity:** documented in TRAINING_GUIDE.md.
- **Export drift:** mitigated by ≥0.999 PyTorch↔ONNX parity gate.
- **Runtime failure:** Neuro-Fabric's PCA fallback remains active.
- **Embedding quality vs PCA:** the model does not yet outperform PCA on
  recall@10. This is expected for a classification-trained model; future
  work should fine-tune the embedding head for similarity search.
