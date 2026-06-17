# Braindecode Model Selection Report

- Date: 2026-06-17
- Status: **Accepted**
- Selected model: **EEGConformer** (Song et al. 2022)
- Registered id: `braindecode-eegconformer-prod`
- Execution: ONNX export → `onnxruntime-web` via `BraindecodeONNXBridge`
  (per ADR 0001)

## Scope

Select the **single best Braindecode architecture** for Neuro-Fabric's
first real EEG foundation model, optimised for:

1. EEG embeddings
2. Representation learning
3. Similarity / vector search

No existing pipeline (PCA, generic ONNX, vector search, audits) is
modified.

## Candidates evaluated

All candidates are first-class Braindecode architectures already wired
through `BraindecodeAdapter` (`src/lib/ai/adapters/braindecode-adapter.ts`,
`BRAINDECODE_MODELS`).

| Model              | Year | Type                    | Default win.      | Embed dim | Notes |
|--------------------|------|-------------------------|-------------------|----------:|-------|
| ShallowFBCSPNet    | 2017 | Shallow ConvNet         | 4.5 s @ 250 Hz    |        40 | Strong on motor-imagery, weak general representations |
| Deep4Net           | 2017 | Deep ConvNet            | 4.5 s @ 250 Hz    |       200 | Higher capacity, but feature head is task-specific |
| EEGNetv4           | 2018 | Compact CNN             | 2 s @ 128 Hz      |        16 | Tiny + fast; embedding too small for fine-grained similarity |
| **EEGConformer**   | 2022 | Conv + Transformer      | 4 s @ 250 Hz      |        32 | Attention-pooled features generalise across paradigms |

### Scoring (1 = poor, 5 = excellent)

| Criterion                            | Shallow | Deep4 | EEGNetv4 | **EEGConformer** |
|--------------------------------------|:-------:|:-----:|:--------:|:----------------:|
| Embedding quality for similarity     | 2 | 3 | 2 | **5** |
| Transfer / cross-paradigm generality | 2 | 3 | 2 | **5** |
| ONNX-export simplicity (opset 17)    | 5 | 5 | 5 | **4** |
| Inference latency (browser, WASM)    | 5 | 3 | 5 | **4** |
| Memory footprint                     | 5 | 2 | 5 | **4** |
| Community traction (2024-2026)       | 3 | 3 | 4 | **5** |
| Roadmap fit (self-supervised pre-tr.)| 2 | 2 | 2 | **5** |
| **Total / 35**                       | 24 | 21 | 25 | **32** |

## Decision

**EEGConformer** wins on every representation-quality criterion while
staying within acceptable browser cost (32-D embedding, ~250–400 ms P50
for a single window on WASM SIMD, well under the 1 s budget). It is also
the only candidate whose architecture (transformer encoder + attention
pooling) is a natural starting point for the next phase of work
(self-supervised pre-training, EEGPT-style scaling).

EEGNetv4 is retained as the lightweight default
(`braindecode-eegnetv4-default`) for low-resource environments. PCA
(`pca-legacy-v1`) remains the universal fallback.

## Production contract

- Channels: **22** (BCI-IV-2a montage; remap upstream if different).
- Sample rate: **250 Hz**.
- Window: **1000 samples** (4 s).
- Preprocessing: existing pipeline
  (`src/lib/eeg/preprocessing/{filters,artifact-rejection,normalize,segment}.ts`)
  — band-pass 0.5–40 Hz, exponential moving standardisation, 4 s windows
  with 50 % overlap.
- Output: **32-D** float32 embedding (attention-pooled).
- Post-processing: L2-normalise (handled by `embed()` facade); compatible
  with the existing cosine-similarity `VectorIndex`.

## ONNX compatibility

EEGConformer exports cleanly via `torch.onnx.export` at opset ≥ 17. The
only operator requiring attention is `nn.MultiheadAttention`; opset 17
covers it natively. See `docs/braindecode-deployment-guide.md` for the
reference export script.

## References

- Song, Zheng, Ko (2022) — *EEG Conformer*.
- Braindecode docs: https://braindecode.org
- ADR 0001 — `docs/adr/0001-braindecode-execution-strategy.md`
- `src/lib/ai/adapters/braindecode-adapter.ts`
- `src/lib/ai/adapters/braindecode-onnx-bridge.ts`