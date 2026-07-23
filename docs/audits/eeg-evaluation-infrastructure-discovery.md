# EEG Evaluation Infrastructure Discovery

## 1. Dataset Source(s)

| Dataset | MOABB ID | Description | Sample Rate | Channels | Window |
|---------|----------|-------------|-------------|----------|--------|
| **BCI Competition IV 2a** | `BNCI2014_001` | 9 subjects, 2 sessions each, 4-class motor imagery (left/right hand, feet, tongue) | 250 Hz | 22 (standard 10‑10 layout, re‑referenced) | 1000 samples (≈ 4 s) per window |
| Additional BCI‑related corpora (for future expansion) | `BNCI2014_004`, `Cho2017`, `PhysionetMI` | Motor imagery / various tasks | 250 Hz – 512 Hz | 3 – 64 channels | Varies |

*The primary benchmark used throughout the project is **BNCI2014_001** (BCI‑IV‑2a).*

### How the dataset is loaded
- Loading is orchestrated in **`src/hooks/use-moabb.ts`**.
- The hook uses **MOABB** (via Pyodide) to instantiate the appropriate `moabb.datasets` class.
- After loading, epochs are serialized to JSON and returned as `MOABBEpoch[]` objects.
- Each epoch payload contains:
  - `subject`, `session`, `label`
  - `data` (list of channel time‑series)
  - `sfreq` (250 Hz)
  - `ch_names` (auto‑generated)
  - `n_channels` (22 for BNCI2014_001)

## 2. Pre‑processing Chain (src/lib/eeg/preprocessing)

The preprocessing pipeline is **pure TypeScript** and is shared by all downstream evaluation and inference paths.

| Step | Module | Default Parameters | Purpose |
|------|--------|--------------------|---------|
| **Band‑pass filter** | `filters.ts` (`bandpass`) | `low: 1 Hz`, `high: 40 Hz` | Removes DC and high‑frequency noise |
| **Notch filter** | `filters.ts` (`notch`) | `fc: 60 Hz`, `q: 30` | Suppresses power‑line interference |
| **Z‑score normalization** | `normalize.ts` (`zscore`) | `normalize: true` (default) | Centers each channel to zero mean, unit variance |
| **Segmentation** | `segment.ts` (`segment`) | `windowSec: 2`, `overlap: 0.5` (default) | Splits continuous recordings into 2‑second windows with 50 % overlap. For the BCI‑IV‑2a benchmark the configuration is overridden to `window_samples: 1000` (4 s) to match the 250 Hz contract. |
| **Artifact rejection** | `artifact-rejection.ts` | `enabled: true`, `maxContaminationPercent: 40` | Detects and discards windows with excessive amplitude deviations. |

The **entry point** is `preprocess(input, options)` in `src/lib/eeg/preprocessing/index.ts`. It returns a processed signal, segmented windows, and a detailed report.

## 3. Evaluation / Benchmark Infrastructure

| Concern | File / Module | Key Functions |
|---------|----------------|----------------|
| **Embedding‑quality metrics** | `src/lib/ai/benchmark/validation-metrics.ts` | `cosineMatrix`, `recallAtK`, `intraInterClassCosine`, `pcaBaselineRecall`, `validateEmbeddings` – compute intra/inter cosine stats, recall@k, PCA baseline recall, separation margin, etc. |
| **Similarity search utilities** | `src/lib/vector-search/*` | `VectorIndex`, `NeuralVectorIndex`, `runRecallSLO` – handle storage, nearest‑neighbor search, and SLO‑level recall verification. |
| **Benchmark orchestration** | `src/lib/ai/benchmark/index.ts` | `benchmarkAll`, `benchmarkAdapter` – run a set of models over a fixed input and report latency, dimension, fallback flag, etc. |
| **Evaluation dataset handling** | `src/hooks/use-moabb.ts` (loading) → `src/lib/ai/benchmark/validation-metrics.ts` (metric calculation) – the loaded epochs are turned into a flat `number[][]` matrix where each row is a 32‑dim (or 20‑dim for PCA) embedding. |
| **Validation metrics testing** | `src/lib/ai/benchmark/__tests__/benchmark-production.test.ts` (future) – currently houses the benchmark execution logic. |

### Expected Input Shape for Embedding Evaluation
- **Number of samples**: Variable (typically a few hundred–few thousand epochs after hold‑out split).
- **Embedding dimension**: 
  - **PCA baseline** → 20‑dim (legacy)  
  - **EEGConformer** → 32‑dim (production)
- Each row corresponds to one epoch’s embedding vector; an accompanying `labels[]` array holds the integer class label (0‑3) for that epoch.

## 4. Files and Paths Summary

| Category | Path | Purpose |
|----------|------|---------|
| **Dataset configuration** | `training/configs/eegconformer-bciiv2a.yaml` | Defines channels, sample_rate, window_samples, dataset loader (`moabb`), hold‑out subjects, and export settings. |
| **Hook for dataset loading** | `src/hooks/use-moabb.ts` | React hook that loads MOABB datasets in the browser, provides progress & error states. |
| **Pre‑processing** | `src/lib/eeg/preprocessing/index.ts` | Consolidated `preprocess` function that runs filtering, normalization, segmentation, artifact rejection. |
| **Embedding generation** | `src/lib/ai/inference/embed-eeg.ts` | Calls `embedEEG` with `preferredModelId: "braindecode-eegconformer-prod"`; returns `EmbedResult` with `embedding` (32‑dim) and `modelId`. |
| **Metric calculation** | `src/lib/ai/benchmark/validation-metrics.ts` | Takes `emb` (matrix) + `labels` and outputs `EmbeddingValidationReport` with recall, cosine stats, PCA baseline, etc. |
| **Vector storage & search** | `src/lib/vector-search/*` (e.g., `neural-index.ts`) | Persists embeddings in a `VectorIndex` (Supabase pgvector) and provides `search` / `nearest` operations. |
| **Benchmark orchestration** | `src/lib/ai/benchmark/index.ts` | `benchmarkAll` used to obtain latency, embedding‑dim, fallback stats. |
| **Documentation** | `docs/audits/` | Series of audit markdown files (e.g., `eeg-evaluation-infrastructure-discovery.md`). |

## 5. Recommended Next Execution Path

1. **Load the benchmark dataset** – invoke `useMOABB().loadDataset('BNCI2014_001', subjects=[1,2,3,4,4,5,6,7,8,9])` (or the hold‑out subset defined in the YAML).  
2. **Pre‑process** – call `preprocess` with defaults; verify that the resulting windows are 22‑channel, 250 Hz, 1000‑sample length.  
3. **Generate embeddings** – call `embedEEG` (or directly `decodeCognitiveState`) for each window using `preferredModelId: "braindecode-eegconformer-prod"`; collect the resulting 32‑dim vectors and labels.  
4. **Run validation metrics** – feed the matrix + label vector into `validateEmbeddings` (or `intraInterClassCosine` / `recallAtK`).  
5. **Persist embeddings** – optionally store them in the `VectorIndex` for later similarity retrieval.  
6. **Benchmark latency** – use `benchmarkAll(['pca-legacy-v1','braindecode-eegconformer-prod'], input, 5)` to obtain mean latency, P50, P95, and fallback flags.  

All of the above can be scripted in a temporary Node/TypeScript runner (e.g., `benchmark_runner.ts`) and executed with `npx ts-node`. The output will be a JSON object containing per‑model latency, dimension, fallback status, and any NANs/Inf checks.

---

*This discovery document captures the current state of the EEG data pipeline, preprocessing steps, and evaluation tooling. It deliberately avoids adding any new evaluation code – the next phase will instantiate the above workflow to compute the required quality metrics and compare PCA‑legacy versus EEGConformer embeddings.*