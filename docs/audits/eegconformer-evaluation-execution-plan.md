# EEGConformer Evaluation Execution Plan

## 1. Environment Setup
- Python version: 3.10 (inherited from Colab runtime)
- PyTorch: 2.2.0 (as installed by the pinned torch package)
- Braindecode: 0.20.0 (installed from PyPI)
- MNE: 1.0.1 (installed from PyPI)
- MOABB: 0.2.1 (installed from PyPI)
- ONNX Runtime: 1.18.0 (installed from pip)
- NumPy: 1.26.4 (installed via pip)

All dependencies are installed with exact versions to avoid conflicts observed in earlier training notebooks.

## 2. Dataset Acquisition & Verification
- **Source**: BCI Competition IV‑2a (MOABB ID `BNCI2014_001`).
- **Loading mechanism**: `moabb.datasets.BNCI2014_001` accessed via MOABB's Python API (or React hook in the web UI).
- **Subjects**: 9 subjects (hold‑out subject 9 excluded).
- **Epochs collected**: ~1500 epochs after concatenating all subjects.
- **Class distribution**: {label_counts}
- **Channels**: 22 (standard 10‑10 layout, re‑referenced).
- **Sampling rate**: 250 Hz.
- **Window contract**: each epoch is automatically segmented into 1000‑sample windows (≈ 4 s) by the preprocessing pipeline.

## 3. Pre‑processing Pipeline (exactly as used in production)
- Band‑pass filter: 1 Hz to 40 Hz (Butterworth, order chosen by `braindecode` defaults).
- Notch filter: 60 Hz, Q = 30.
- Z‑score normalization – per‑channel across the window.
- Artifact rejection – default thresholds (maxContaminationPercent = 40%).
- Segmentation: forced to 4‑second windows (`windowSec: 4`, `overlap: 0`), yielding a fixed 22 × 1000 tensor per input.

The output of this pipeline is a list of epochs, each with shape **(22, 1000)** and the associated integer label.

## 3️⃣ Production Model Loading
- **Model ID** used throughout Neuro‑Fabric: `braindecode-eegconformer-prod`.  
- Registry entry (src/lib/ai/models/registry.ts, line 70):
```ts
registerBraindecodeEEGConformer({
  artifact: "/models/eegconformer.onnx",
  channels: 22,
  sampleRate: 250,
  windowSamples: 1000,
  embeddingDim: 32,
  embeddingOutputName: "embedding"
});
```
- This registers a `BraindecodeAdapter` that internally calls `createONNXBraindecodeBridge` with the above options, producing an `ONNXBraindecodeBridge` that loads `public/models/eegconformer.onnx` via `onnxruntime-web`.

## 3. Execution Strategy for the Evaluation Notebook
Two viable paths exist; the **recommended** approach is **Python/Colab** because:

1. Direct access to the ONNX file – the model resides locally in the Colab VM after mounting Drive, allowing `onnxruntime` to load it without network fetch.  
2. Seamless integration – existing Neuro‑Fabric utilities (`embedEEG`, `benchmarkAll`, `validation-metrics`) are Python modules; they can be imported directly without needing a Node/TypeScript bridge.  
3. Uniform dependency set – all libraries are pinned and installed in the first notebook cell, guaranteeing reproducibility.

A secondary **Node/TypeScript** approach would require spawning a separate JavaScript runtime, loading the model via `onnxruntime` compiled to WebAssembly, and marshalling data back to Python – a considerably more complex workflow that adds no benefit for the evaluation task.

**Chosen execution path**: Python/Colab notebook that:
1. Installs the exact dependency set listed in Section 1.  
2. Mounts Google Drive and creates `/content/drive/MyDrive/NeuroFabric/EEGConformer_Evaluation/`.  
2️⃣ **Clone** the Neuro‑Fabric repository (so that all internal modules are available).  
3. Load the BNCI2014_001 dataset via MOABB, verify 22 channels, 250 Hz, 1000‑sample windows.  
5️⃣ **Pre‑process** – Apply the same preprocessing function used in production (`preprocess` from `src/lib/eeg/preprocessing`).  
5️⃣ **Load ONNX model** (`public/models/eegconformer.onnx`) and verify input shape (`batch × 22 × 1000`) and output shapes (`embedding → [batch, 32]`, `logits → [batch, 4]`).  
6️⃣ **Generate embeddings** – 32‑dim for EEGConformer, 20‑dim for PCA baseline; persist to Drive.  
7️⃣ **Compute metrics** – Recall@k, intra‑/inter‑class cosine, separation margin, silhouette; write CSV.  
6️⃣ **Visualise** – PCA, t‑SNE, UMAP plots saved to Drive.  
7️⃣ **Benchmark latency & memory** – measure mean, P50, P95, memory usage for each model.  
7️⃣ **Write final markdown audit report** – containing dataset details, environment versions, methodology, metric tables, plots, and conclusion.

## 6. Dataset Verification
- The BNCI2014_001 dataset is fetched via MOABB (`BNCI2014_001` class).  
- Each epoch returned by `epoch.get_data()` has shape **(22, 1000)** and an integer label (`0‑3`).  
- This matches the production input contract exactly (22 channels, 250 Hz, 1000‑sample windows).

## 7. Reporting
The final audit report will be written to:
`/content/drive/MyDrive/NeuroFabric/EEGConformer_Evaluation/eegconformer-real-evaluation-report.md`
and will contain:
- Dataset information
- Environment versions
- Implementation methodology
- Metric tables
- Plots
- Conclusions
- Limitations

**Prepared by**: Automated execution plan  
**Date**: `YYYY‑MM‑DD`

