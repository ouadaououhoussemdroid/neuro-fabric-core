# NeuroWeave Backend Architecture

End-to-end EEG processing pipeline. All stages are implemented as pure
TypeScript modules under `src/lib/` and run inside TanStack Start server
functions / server routes (Cloudflare Worker runtime).

```
Upload  ->  Preprocessing  ->  Embedding  ->  Similarity Search   ->  Decoder
EDF/CSV/    bandpass            band-power      VectorIndex             attention
NPY         notch               + PCA / AE      cosine top-k            workload
            z-score             (linear)                                arousal
            segment
```

## Modules

| Path                                  | Responsibility                                      |
| ------------------------------------- | --------------------------------------------------- |
| `src/lib/eeg/types.ts`                | Shared `EEGSignal`, `EEGWindow`, report types       |
| `src/lib/eeg/parsers/`                | EDF (16-bit LE), CSV, NumPy `.npy` v1/v2 readers    |
| `src/lib/eeg/preprocessing/filters`   | Biquad zero-phase bandpass + notch (filtfilt)       |
| `src/lib/eeg/preprocessing/normalize` | Per-channel z-score, demean                         |
| `src/lib/eeg/preprocessing/segment`   | Fixed-length overlapping window slicer              |
| `src/lib/eeg/preprocessing/index.ts`  | `preprocess()` orchestrator + timing report         |
| `src/lib/eeg/loaders/physionet.ts`    | PhysioNet `eegmmidb` v1.0.0 HTTPS loader            |
| `src/lib/eeg/loaders/bci-competition` | BCI Competition IV 2a loader (mirror-based)         |
| `src/lib/eeg/loaders/tuh.ts`          | TUH EEG loader scaffold (mirror + index required)   |
| `src/lib/embeddings/features.ts`      | Per-channel band-power features via DFT (Hann)      |
| `src/lib/embeddings/pca.ts`           | Power-iteration PCA with deflation                  |
| `src/lib/embeddings/autoencoder.ts`   | Linear AE scaffold (encoder = top-k PCA basis)      |
| `src/lib/embeddings/index.ts`         | `embedSignal()` - windows to latent vector          |
| `src/lib/vector-search/cosine.ts`     | Cosine + L2 distance                                |
| `src/lib/vector-search/index.ts`      | `VectorIndex` brute-force top-k                     |
| `src/lib/decoder/features.ts`         | Whole-signal normalised band stats                  |
| `src/lib/decoder/index.ts`            | Baseline spectral decoders (atten/workload/arousal) |
| `src/lib/synthetic/index.ts`          | 1/f pink-noise + band-bump EEG synth                |
| `src/lib/logging/index.ts`            | Structured JSON logger + `startTimer()`             |

## Data Flow

1. **Ingest** - `POST /api/eeg/upload` parses the uploaded file into an
   `EEGSignal { channels, data[C][N], sampleRate }`.
2. **Preprocess** - `preprocess()` applies:
   - Bandpass 1-40 Hz (zero-phase biquad, default).
   - Notch 50 or 60 Hz (Q ~ 30).
   - Per-channel z-score.
   - Segmentation into 2 s / 50 % overlap windows.
     Each step records its duration in `report.steps[]`.
3. **Embed** - `embedSignal()` extracts 5-band power features per channel per
   window (Hann-windowed DFT, capped at 512 points), mean-pools across
   windows, and projects through a linear autoencoder fit on the
   per-window feature matrix. When too few windows exist for a stable PCA
   fit, the raw mean-pooled band-power vector is returned. The vector is a
   real numeric embedding, never a mock.
4. **Search** - `VectorIndex.search(query, k)` ranks indexed vectors by
   cosine similarity. Used by the embeddings explorer "Top matches" panel
   and by similarity APIs.
5. **Decode** - `decodeCognitiveState()` returns probabilities for
   attention, workload, and arousal, derived only from real spectral
   ratios of the input signal:
   - attention proportional to beta / (alpha + theta) (Pope-index style)
   - workload proportional to theta / alpha
   - arousal proportional to beta + gamma (already normalised to total)

## Synthetic Data

`generateSyntheticEEG()` produces multi-channel data with Voss-McCartney
pink (1/f) noise plus per-band sinusoids whose amplitudes are configurable.
Output is a normal `EEGSignal` and feeds the same preprocessing ->
embedding -> decoder pipeline as real recordings.

## Logging

Every stage timed via `startTimer()` emits a structured JSON line, e.g.:

    {"level":"info","msg":"timing.eeg.upload.embed","ts":"...","model":"linear-ae","dim":64,"durationMs":42.31}

## Datasets

- **PhysioNet eegmmidb** - fully working over HTTPS. `physionet.list()`
  enumerates the 109 x 14 EDF files.
- **BCI Competition IV 2a** - loader takes a `mirrorBase` because the
  original distribution lacks a stable anonymous HTTPS endpoint.
- **TUH EEG** - architecture only. Requires the operator to stage an HTTPS
  mirror and provide a `TuhIndexEntry[]`; otherwise `list()` returns `[]`.

## Non-goals (current MVP)

- No GPU inference path; the autoencoder is linear and runs on CPU.
- No persistent vector store; `VectorIndex` is in-memory.
- No ICA / regression-based artefact removal (filters only).
- No learned cognitive-state classifier; baseline ratios are exposed under
  a stable API so a trained model can be swapped in later.
