# Neuro-Fabric Core

Open-source neurotechnology platform for EEG acquisition, neural inference,
and cognitive decoding. Runs real Braindecode EEGConformer embeddings via
ONNX Runtime Web in the browser, with a pgvector-backed similarity search
and a concept-graph provenance layer.

## Quick start

```bash
# Install dependencies (bun is the authoritative package manager)
bun install

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your Supabase URL, keys, and CRON_SECRET

# Start the dev server
bun run dev
```

## Architecture

```
EEG input (file / BrainFlow / LSL)
   ↓ parse (EDF/BDF/CSV/NPY) + preprocess (IIR filtfilt, FFT band-power)
   ↓ embed (Braindecode EEGConformer → ONNX → PCA fallback)
   ↓ persist (pgvector vector(32) + ivfflat + concept-graph ltree)
   ↓ decode (trained logistic regression → heuristic spectral fallback)
Cognitive state report + embedding + saliency map
```

### Key modules

| Module              | Path                         | Purpose                                                   |
| ------------------- | ---------------------------- | --------------------------------------------------------- |
| EEG parsers         | `src/lib/eeg/parsers/`       | EDF/EDF+/BDF/CSV/NPY decoding                             |
| Preprocessing       | `src/lib/eeg/preprocessing/` | IIR biquad filters, FFT band-power, segmentation          |
| Acquisition         | `src/lib/eeg/acquisition.ts` | Hardware-agnostic `AcquisitionSource` interface           |
| AI Foundation Layer | `src/lib/ai/`                | Adapter pattern: ONNX, PCA, Braindecode, EEGPT            |
| Embeddings          | `src/lib/embeddings/`        | PCA, autoencoder, FFT features, subject aggregation       |
| Vector search       | `src/lib/vector-search/`     | pgvector `NeuralVectorIndex` + recall@10 SLO              |
| Decoder             | `src/lib/decoder/`           | Cognitive state (attention, workload, arousal)            |
| Concept graph       | `src/lib/graph/`             | Subject → session → window → embedding provenance         |
| Training            | `training/`                  | PyTorch + MOABB + MLflow pipeline (Dockerfile + Makefile) |

## Scripts

```bash
bun run dev          # Start dev server
bun run build        # Production build
bun run test         # Run test suite
bun run test:coverage # Run tests with coverage report
bun run lint         # ESLint
bun run typecheck    # TypeScript compiler check
bun run format       # Prettier
```

## Training

The EEGConformer model is trained via the reproducible pipeline in `training/`:

```bash
cd training/
make train MODEL=eegconformer DATASET=bciiv2a

# Or via Docker:
docker build -t neuro-fabric-train -f training/Dockerfile .
docker run neuro-fabric-train make train MODEL=eegconformer DATASET=bciiv2a
```

See `training/README.md` and `training/docs/TRAINING_GUIDE.md` for details.

## Documentation

- **Current state:** `docs/audits/2026-06-19_project_state_audit.md`
- **Execution plan:** `docs/roadmaps/2026-06-19_open_source_execution_blueprint.md`
- **Architecture:** `docs/architecture.md`, `docs/ai-layer-architecture.md`
- **ADRs:** `docs/adr/0001-braindecode-execution-strategy.md`, `docs/adr/0002-eeg-embedding-storage-contract.md`
- **Training guide:** `training/docs/TRAINING_GUIDE.md`

Documents marked `⚠️ Historical document` are retained as baselines for
traceability but may contain outdated claims. Always cross-reference with
the current project-state audit linked above.

## License

See individual component licenses:

- Weights: CC-BY-4.0 (see `training/docs/MODEL_CARD.md`)
- Architecture: BSD-3-Clause (Braindecode)
- Datasets: per-dataset (see `src/lib/datasets/manifest.ts`)
