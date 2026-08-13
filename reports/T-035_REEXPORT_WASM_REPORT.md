# T-035: WASM-Compatible Re-Export of v2 Checkpoint

## Executive Summary

> **Verdict: ✅ ALL CHECKS PASSED**

The existing v2 checkpoint was re-exported into a genuinely WASM-compatible ONNX model. No retraining was performed. The new model:

- **Eliminates all Einsum ops** (replaced with MatMul + Transpose)
- **Uses zero WASM blockers** (19 WASM-compatible ops only)
- **Produces identical 50-subject LOSO results** to the existing v2 model (Acc = 0.3250, Fisher = 0.0072)
- **Preserves exact checkpoint weights** (embedding parity = 0.99999994)
- **Does not modify any production artifacts**

---

## 1. Problem Statement

The existing v2 ONNX model (`public/models/eegconformer_finetuned.onnx`) claims WASM compatibility but contains **Einsum** operators from braindecode's `MultiHeadAttention` implementation. Einsum is NOT supported by `onnxruntime-web`'s WASM execution provider, making the model non-deployable in browser.

**Requirement**: Re-export the same checkpoint into a genuinely WASM-compatible ONNX without Einsum ops, preserving exact weights and identical inference results.

---

## 2. Methodology

### 2.1 Einsum Removal

Braindecode's `MultiHeadAttention.forward()` uses `torch.einsum` for:
- Attention scores: `torch.einsum("bhqd, bhkd -> bhqk", queries, keys)`
- Context aggregation: `torch.einsum("bhal, bhlv -> bhav", att, values)`

The `WASMMultiHeadAttention` class replaces these with equivalent operations:

| Einsum call | Replacement |
|---|---|
| `torch.einsum("bhqd, bhkd -> bhqk", queries, keys)` | `torch.matmul(queries, keys.transpose(-2, -1))` |
| `torch.einsum("bhal, bhlv -> bhav", att, values)` | `torch.matmul(att, values)` |

PyTorch forward parity verified: embedding cosine = **1.00000000**, logits cosine = **1.00000012**.

### 2.2 ONNX Export

- Exporter: `torch.onnx.export(dynamo=False)` — legacy TorchScript exporter embeds all weights in a single file (WASM-compatible format, no external `.data` file)
- Opset: 17
- Constant folding: enabled (folds BatchNorm2d into Conv2d)

### 2.3 WASM Compatibility Verification

Project blocklist (from forensic investigation and `registry.ts`):
```
{DFT, ReduceL2, FFT, Complex, GlobalAveragePool, Flatten, Einsum}
```

---

## 3. Results

### 3.1 Artifact Checksums

| File | SHA-256 | Size |
|---|---|---|
| `training/artefacts/eegconformer-physionet-v2/eegconformer.pt` | `7ced0058...3287d9` | (checkpoint) |
| `training/artefacts/eegconformer-physionet-v2-wasm/eegconformer_finetuned.onnx` | `18644de1...08f931` | 3,359,557 bytes (3.20 MB) |

### 3.2 Numerical Parity

| Check | Method | Threshold | Result | Status |
|---|---|---|---|---|
| PyTorch parity (einsum → matmul) | Cosine sim on embeddings + logits | > 0.9999 | emb=1.00000000, logits=1.00000012 | ✅ PASS |
| PyTorch → ONNX parity | Cosine sim, 10 random inputs | > 0.999 | min emb=0.99999988, min logits=0.99999994 | ✅ PASS |
| Embedding parity (new vs existing ONNX) | Cosine sim on 1493 real embeddings | > 0.9999 | 0.99999994 | ✅ PASS |

### 3.3 WASM Compatibility

**ONNX ops used (19):**
```
Add, AveragePool, Cast, Concat, Constant, Conv, Div, Elu, Erf,
Gather, Gemm, LayerNormalization, MatMul, Mul, Reshape,
Shape, Softmax, Transpose, Unsqueeze
```

**WASM blockers found:** None ✅

**All ops in ORT-Web WASM supported set:** Yes ✅

### 3.4 Weight Preservation

| Metric | Value |
|---|---|
| Total PyTorch checkpoint params | 113 |
| ONNX initializers | 108 |
| Name-matched (biases + some weights) | 70 |
| Shape+cosine matched (attention + FFN weights) | 36 |
| Folded BN/buffer params skipped | 7 (2 Conv + 2 BN + 3 running stats) |
| Folded Conv weight reconstruction (BN folding) | cosine = 1.00000000 ✅ |
| Folded Conv bias reconstruction (BN folding) | cosine = 1.00000000 ✅ |
| Unused ONNX initializers | 2 (Conv_943, Conv_944 — folded with BN) |
| Unused MatMul initializers | 0 |
| Mismatched | 0 |

**Weight preservation is verified three ways:**
1. All 70 non-opaque PT params match ONNX by name (cosine > 0.9999)
2. All 36 opaque PT weight matrices (24 attention + 12 FFN) match ONNX inits by shape + cosine (with transposition handling for MatMul)
3. The 7 foldable BN/buffer params are verified by reconstructing the folded Conv weights from PT Conv + BN params and comparing to the ONNX Conv inits (cosine = 1.00000000)

### 3.5 50-Subject LOSO Evaluation

| Metric | Existing v2 (Einsum) | New WASM (MatMul) | Delta |
|---|---|---|---|
| Accuracy | 0.3250 | 0.3250 | +0.000000 |
| Recall@1 | 0.2922 | 0.2922 | +0.000000 |
| Recall@5 | 0.7792 | 0.7792 | +0.000000 |
| Recall@10 | 0.9456 | 0.9456 | +0.000000 |
| Fisher score | 0.0072 | 0.0072 | +0.000000 |
| Intra-class cosine | 0.9072 | 0.9072 | ~0 |
| Inter-class cosine | 0.9037 | 0.9037 | ~0 |

**Statistical test (paired t-test):** t = NaN, p = NaN, Cohen's d = 0.0 (identical results → zero variance in differences)

**Subjects:** 50 (S001–S050) | **Trials:** 1,493 | **Labels:** [367, 376, 380, 370] (4 classes)

### 3.6 Embedding Stability

| Check | Result | Status |
|---|---|---|
| Determinism (max pairwise cosine) | 1.00000000 | ✅ Deterministic |
| Amplitude ±10% (mean cosine) | 0.9928 | ✅ Stable |
| Noise (SNR=20dB, cosine sim) | 1.0000 | ✅ Robust |
| Window shift ±40ms | 0.9826 | ✅ Stable |

### 3.7 Embedding Richness

| Metric | New WASM | Existing v2 |
|---|---|---|
| Effective rank (participation ratio) | 3.27 | 3.27 |
| 90% variance at dim | 5 | 5 |
| 95% variance at dim | 8 | 8 |
| Dead dimensions | 0 | 0 |

---

## 4. Files Produced

| Path | Description |
|---|---|
| `training/artefacts/eegconformer-physionet-v2-wasm/eegconformer_finetuned.onnx` | New WASM-compatible ONNX (3.20 MB, SHA-256: `18644de1...`) |
| `training/artefacts/eegconformer-physionet-v2-wasm/t035_export_log.json` | Full export + evaluation logs |
| `scripts/t035-reexport-v2-wasm.py` | Re-export script (v2 checkpoint → WASM-compatible ONNX) |

### Production artifacts: UNCHANGED

| Path | Status |
|---|---|
| `public/models/eegconformer_finetuned.onnx` | Untouched |
| `src/lib/ai/models/registry.ts` | Untouched |
| `public/models/manifest.json` | Untouched |
| `src/lib/ai/artefacts/manifest-metadata.ts` | Untouched |

---

## 5. Verdict

> **✅ ALL CHECKS PASSED — The re-export is genuinely WASM-deployable and preserves v2 performance exactly.**

### What this means:

1. **No Einsum**: The new ONNX graph uses only 19 ops, all supported by `onnxruntime-web` WASM execution provider. No Einsum, no DFT, no unsupported ops.
2. **Weight preservation**: All 113 checkpoint params are accounted for — 70 by direct name matching, 36 by shape+cosine greedy matching, and 7 folded BN/Conv params verified by reconstruction. The embedding parity of 0.99999994 is the ultimate proof.
3. **Identical performance**: The new model produces identical LOSO results (Acc = 0.3250, Fisher = 0.0072) on the same 50-subject protocol with 1,493 trials.
4. **Single-file format**: Weights are embedded in the ONNX file (no external `.data` file needed).
5. **No production changes**: The existing production model, registry, manifest, and all deployment configurations remain untouched.

### Recommendation:

The re-exported model at `training/artefacts/eegconformer-physionet-v2-wasm/eegconformer_finetuned.onnx` is ready for the next step — deployment to `public/models/` and manifest/registry updates — pending explicit promotion authorization. Mission 5 has NOT been started.