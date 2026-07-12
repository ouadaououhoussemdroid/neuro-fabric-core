# EEGConformer Training Package — Neuro-Fabric

End-to-end pipeline producing the first production EEG foundation-model
artefact for Neuro-Fabric:

```
BCI-IV-2a (MOABB)
   ↓ preprocess.py        (22ch · 250Hz · 1000 samples)
   ↓ train.py             (EEGConformer, 4-class motor imagery)
   ↓ validate.py          (cross-subject hold-out)
   ↓ evaluate.py          (cosine recall@10, embedding stats)
   ↓ export_onnx.py       (opset-17, embedding+logits heads)
   ↓ package.py           (sha256, manifest, MODEL_CARD)
eegconformer.pt  →  eegconformer.onnx  →  artefacts/eegconformer-bciiv2a-v1/
```

The output ONNX file is drop-in compatible with the existing
`registerBraindecodeEEGConformer({ artifact: { kind: "url", url } })`
registration in `src/lib/ai/models/registry.ts` — no platform code is
modified by this package.

## Reproducible training container (T-020)

A pinned Docker container ensures bit-reproducible training runs:

```bash
# Build the container
docker build -t neuro-fabric-train -f training/Dockerfile .

# Run the full pipeline (preprocess → train → export → evaluate → validate)
docker run --gpus all \
  -v $(pwd)/training/artefacts:/app/training/artefacts \
  -v $(pwd)/training/cache:/app/training/cache \
  neuro-fabric-train make all MODEL=eegconformer DATASET=bciiv2a

# Or run individual steps
docker run neuro-fabric-train make train MODEL=eegconformer DATASET=bciiv2a
docker run neuro-fabric-train make export MODEL=eegconformer
```

Without Docker, use the Makefile directly (requires the pinned deps from
`requirements.txt`):

```bash
cd training/
make train MODEL=eegconformer DATASET=bciiv2a
```

## Layout

| Path                                   | Purpose                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `configs/eegconformer-bciiv2a.yaml`    | Single source of truth for the contract (22ch · 250Hz · 1000 samples · 32-d). |
| `scripts/acquire_dataset.py`           | Downloads BCI-IV-2a via MOABB into a local cache.                             |
| `scripts/preprocess.py`                | Bandpass + resample + epoch + standardise → `.npz`.                           |
| `scripts/train.py`                     | EEGConformer training with fixed seeds, cross-session split.                  |
| `scripts/validate.py`                  | Cross-subject hold-out validation.                                            |
| `scripts/evaluate.py`                  | Cosine recall@10, embedding norm/variance.                                    |
| `scripts/export_onnx.py`               | Wraps `scripts/export_braindecode_eegconformer.py` (in repo root `scripts/`). |
| `scripts/package.py`                   | Builds the artefact directory + `manifest.json`.                              |
| `scripts/run_all.sh`                   | Convenience wrapper.                                                          |
| `notebooks/EEGConformer_BCIIV2a.ipynb` | Colab-ready notebook calling the same scripts.                                |
| `docs/TRAINING_GUIDE.md`               | Full operator guide.                                                          |
| `docs/MODEL_CARD.md`                   | Shipped with the artefact.                                                    |

## Contract (matches `docs/audits/2026-06-17_braindecode-model-selection.md`)

| Field         | Value                                   |
| ------------- | --------------------------------------- |
| Channels      | 22                                      |
| Sample rate   | 250 Hz                                  |
| Window        | 1000 samples (4 s)                      |
| Classes       | 4 (left hand, right hand, feet, tongue) |
| Embedding dim | 32                                      |
| ONNX opset    | 17                                      |
| Outputs       | `embedding` (Nx32), `logits` (Nx4)      |

## Quick start (local)

```bash
cd training
python -m pip install -r requirements.txt
bash scripts/run_all.sh
```

Final artefact lands in `training/artefacts/eegconformer-bciiv2a-v1/`
containing `eegconformer.pt`, `eegconformer.onnx`, `manifest.json`,
and `MODEL_CARD.md`.

## Quick start (Colab)

Open `notebooks/EEGConformer_BCIIV2a.ipynb` in Google Colab, runtime
= GPU (T4 is sufficient), run all cells. The final cell downloads the
packaged artefact.

## Deployment into Neuro-Fabric

Upload `eegconformer.onnx` to a hosted location (Lovable Cloud storage
or `public/models/`), then at app boot:

```ts
import { registerBraindecodeEEGConformer } from "@/lib/ai/models/registry";
registerBraindecodeEEGConformer({
  artifact: { kind: "url", url: "/models/eegconformer.onnx" },
});
```

No other code changes are required.

## Preservation guarantees

- This package lives entirely under `training/` and `scripts/` (root).
- It does NOT modify the AI Foundation Layer, ONNX adapter, registry,
  inference engine, validation layer, PCA fallback, or any existing audit.
- It does NOT add any TypeScript source under `src/`.
