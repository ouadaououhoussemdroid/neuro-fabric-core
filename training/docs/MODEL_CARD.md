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
- Datasets sampled outside the contract (≠ 22 ch / 250 Hz / 4 s window).

## Training data
- BCI-IV-2a: 9 subjects, 22 EEG channels, 250 Hz.
- Bandpass 4–38 Hz, epoch 0–4 s post-cue.
- Hold-out: subject 9 (cross-subject).

## Metrics (filled by `package.py`)
- See `manifest.json → validation_report.holdout_accuracy`.
- See `manifest.json → evaluation_report.recall_at_k["10"]`.
- See `manifest.json → train_history.best_val_acc`.

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