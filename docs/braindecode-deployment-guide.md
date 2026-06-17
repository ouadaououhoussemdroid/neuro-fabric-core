# Braindecode EEGConformer — Deployment Guide

End-to-end workflow for taking a trained Braindecode **EEGConformer** model
from a Python research environment to a production artefact registered in
the Neuro-Fabric AI Foundation Layer.

This guide does **not** modify any existing pipeline. PCA remains the
default fallback; generic ONNX models keep working; the existing EEG
preprocessing chain is reused verbatim.

## 0. Prerequisites

- Python 3.10+, `torch>=2.1`, `braindecode>=0.8`, `onnx>=1.15`.
- A trained EEGConformer checkpoint (or use the reference recipe below).

## 1. Artifact preparation workflow

The reference export script lives at
`scripts/export_braindecode_eegconformer.py`. It:

1. Instantiates EEGConformer with the production contract
   (22 channels, 1000 samples, 32-D embedding head).
2. Loads weights from a `.pt` checkpoint.
3. Wraps the model so it exposes a named `"embedding"` output
   (penultimate attention-pooled features) alongside the optional
   `"logits"` head.
4. Calls `torch.onnx.export(..., opset_version=17, dynamic_axes={"input": {0: "batch"}})`.
5. Validates the resulting graph with `onnx.checker.check_model`.
6. Smoke-tests the exported file with `onnxruntime` against the original
   PyTorch outputs (cosine similarity ≥ 0.999).

The output is a single `eegconformer.onnx` file (~7 MB).

## 2. Hosting the artefact

Place `eegconformer.onnx` on any static origin reachable by the browser
(Lovable Cloud Storage bucket, CDN, signed URL). No server-side runtime
is required — inference runs in the user's browser via
`onnxruntime-web`.

## 3. Deployment workflow (TypeScript)

Register the artefact once at boot, before the first call to `embedEEG()`:

```ts
import { registerBraindecodeEEGConformer } from "@/lib/ai/models/registry";

registerBraindecodeEEGConformer({
  artifact: { kind: "url", url: "/models/eegconformer.onnx" },
  // Optional overrides:
  // channels: 22, sampleRate: 250, windowSamples: 1000,
  // embeddingDim: 32, embeddingOutputName: "embedding",
});
```

`embedEEG()` will now route:

```
EEG → preprocessing → EEGConformer (ONNX)
         ↓ on failure
       generic ONNX (if registered)
         ↓
       PCA legacy
```

No call sites change. Existing callers using `embed()` keep working.

## 4. Validation

The standard validation layer (`src/lib/ai/validation`) rejects NaN /
Inf / zero / wrong-dim vectors. The `embed()` facade L2-normalises so the
result plugs straight into the cosine `VectorIndex`. To pin the
contract:

```ts
await embedEEG(input, {
  preferredModelId: "braindecode-eegconformer-prod",
  expectedDim: 32,
});
```

## 5. Benchmarking

```ts
import { benchmarkAll } from "@/lib/ai/benchmark";

await benchmarkAll(
  ["pca-legacy-v1", "braindecode-eegconformer-prod"],
  input,
  20,
);
```

Numbers and methodology: `docs/audits/2026-06-17_braindecode-benchmark.md`.

## 6. Rollback

Call `unregisterModel("braindecode-eegconformer-prod")`. The platform
immediately reverts to its previous fallback chain (generic ONNX → PCA).
No data migration required because vectors are tagged with `modelId`
in `NeuralVectorIndex`.