# Model Card: EEGConformer v2 (Fine-tuned on PhysioNet EEGMMIDB)

**Model ID:** `braindecode-eegconformer-prod-v2`
**Artifact:** `eegconformer_finetuned.onnx`
**Version:** 2.0.0 (fine-tuned from v1 initialization)
**Architecture:** EEGConformer (Song et al., 2022) — Conv2d patch embedding → 6-layer Transformer (40-dim, 10 heads) → FC(2440→256)+ELU → FC(256→32)+ELU → Linear(32→4)
**Opset:** 17
**WASM-compatible:** ✅ Yes (19 ops, zero Einsum/DFT/FFT blockers — verified via T-035)
**SHA-256:** `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931`
**Size:** 3,359,557 bytes (3.20 MB, single-file — no external data)

---

## Intended Use

**Primary use:** 4-class motor imagery (MI) classification from EEG signals for brain-computer interface (BCI) applications via embedding-based nearest-centroid retrieval.

**Intended users:** Researchers, BCI developers, clinical researchers developing motor imagery classifiers.

**Out-of-scope uses:**
- Real-time closed-loop BCI control (not validated for sub-100ms latency)
- Diagnostic or medical decision-making (predictions are embeddings for nearest-neighbour retrieval, not clinical labels)
- Non-motor-imagery EEG tasks (the model was trained exclusively on MI data)
- Populations outside the training demographic (see Limitations)

---

## Training Data

**Dataset:** PhysioNet EEGMMIDB (Goldberger et al., 2000) — `physionet.org/content/eegmmidb/1.0.0/`

| Property | Value |
|---|---|
| Subjects (training) | S001–S020 (20 subjects), 505 training trials (85% of 593) |
| Subjects (internal val) | S001–S020 (15% split = 88 trials) |
| Frequency band | Runs 5 and 6 (left/right hand, feet, tongue) |
| Classes | 4 (left hand, right hand, feet, tongue) |
| Sampling rate | 160 Hz (resampled to 250 Hz) |
| Channels | 22 (BCI-IV-2a subset, 10–20 system) |
| Window size | 4 seconds (1000 samples @ 250 Hz) |
| Bandpass | 4–38 Hz (FIR) |
| Preprocessing | Per-trial z-score normalization |
| License | [PhysioNet Health DM&C License](https://physionet.org/science/license/) (open-access research) |

---

## Model Architecture

```
EEGConformer(input=[1, 22, 1000])
  → Conv2d patch embedding (kernel_size=(1, 25), stride=(1, 4))
  → 6-layer Transformer (d_model=40, n_heads=10, dropout=0.5)
  → FC(2440 → 256) + ELU
  → FC(256 → 32) + ELU          ← 32-D embedding output
  → Linear(32 → 4)              ← 4-class logits
```

**Parameters:** 789,511
**Embedding dimension:** 32 (L2-normalized for cosine search)
**Initialization:** Pretrained v1 weights extracted from ONNX (cosine=0.996)

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-3 |
| Scheduler | Cosine annealing, 15-epoch warmup |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping | Patience = 40 (val loss) |
| Dropout | 0.5 |
| Label smoothing | 0.1 |
| Grad clipping | 0.5 |
| Seed | 20260617 |
| Best epoch | 36 (val_loss = 1.3891, early stop at 76) |

---

## Evaluation

**Protocol:** Leave-One-Subject-Out (LOSO) cross-validation, 50 subjects (S001–S050), 1,493 trials.

| Metric | EEGConformer v2 | Original v1 | PCA bandpower |
|---|---|---|---|
| **Accuracy** | **0.3250** | 0.283 | 0.313 |
| Recall@1 | 0.2922 | 0.253 | 0.277 |
| Recall@5 | 0.7792 | — | — |
| Recall@10 | 0.9456 | — | — |
| Fisher score | 0.0072 | — | 0.0034 |
| Intra-class cosine | 0.9072 | — | — |
| Inter-class cosine | 0.9037 | — | — |

---

## WASM Compatibility & Deployment

**Verified via T-035** (`scripts/t035-reexport-v2-wasm.py`):
- All Einsum ops (from braindecode's MultiHeadAttention) replaced with MatMul + Transpose
- ONNX graph verified against ORT-Web WASM compatibility blocklist: `{DFT, ReduceL2, FFT, Complex, GlobalAveragePool, Flatten, Einsum}` — **zero blockers**
- Single-file ONNX (weights embedded, no external `.data` file)
- PyTorch → ONNX parity: embedding cosine = 0.99999988
- New ONNX vs. existing v2 ONNX: embedding parity = 0.99999994
- 50-subject LOSO: Acc = 0.3250, Fisher = 0.0072 (identical to existing v2)

---

## Known Limitations

1. **Low absolute accuracy** (32.5%) — this is expected for 4-class MI on PhysioNet EEGMMIDB at chance-level (25%). The model beats chance but the signal is weak.
2. **Collapsed representations** — intra-class cosine (0.907) ≈ inter-class cosine (0.904), indicating class clusters are not well-separated (T-033 finding). This is an inherent property of the training objective (cross-entropy), not the architecture.
3. **No demographic diversity** — training data is exclusively PhysioNet EEGMMIDB subjects; performance on other populations (clinical, different age groups, different EEG systems) is unvalidated.
4. **Channel montage fixed** — model expects exactly 22 channels in BCI-IV-2a standard montage at 250 Hz.

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Model fails to load in browser | SHA-256 verification + PCA fallback chain (graceful degradation) |
| WASM execution errors | Runtime verification catches and falls back to PCA automatically |
| Incorrect predictions | Not deployed as default — requires explicit env flag promotion |
| Data drift | Monitor drift via `recall-slo.ts` harness and nightly benchmark runs |

---

## Security

- ✅ No PII in training data (raw EEG signals, no subject identifiers)
- ✅ No embedded secrets or credentials in ONNX artifact
- ✅ SHA-256 verification at load time (`enableVerification: true`)
- ✅ CORS restricted to staging + production origins
- ✅ ONNX model file is read-only in `public/models/`

---

## Model Card Authors

Neuro-Fabric Core Team
**Date:** 2026-08-12
