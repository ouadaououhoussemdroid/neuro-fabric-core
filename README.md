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

## Security Architecture

| Layer | Implementation | Notes |
|-------|---------------|-------|
| Auth | HttpOnly, Secure, SameSite=Strict cookies | In-memory session sync via `/api/auth/sync` |
| Rate limiting | PostgreSQL `check_rate_limit` (cross-isolate) | 20 req/60s for API, 5 req/60s for auth |
| WebSocket auth | Bearer token or `?token=` query param | Verified via Supabase JWT |
| Artifact integrity | SHA-256 verification on model load | Checked at `onnxruntime` session creation |
| API security | JWT Bearer auth on all Tier-1 routes | `request-auth.ts` validates tokens server-side |
| SQL injection | Row Level Security (RLS) | All tables enforce `auth.uid()` policies |

### Tier-1 Services

| Service | Model | Metric | Status |
|---------|-------|--------|--------|
| Subject Identity | Joint-2312 (2312-D) | Recall@10 ≥ 0.7856 | ✅ Production |
| Cognitive State | Linear probe (2312→1) | R²=0.7348 | ✅ Production |
| Anomaly Detection | Mahalanobis probe (2312→1) | AUC=0.892 | ✅ Production |
| Sleep Staging | 5-class probe (2312→5) | Acc=0.6718 | ✅ Production |
| Sleep Quality | Regression probe (2312→1) | R²=0.8193 | ✅ Production |

### Production Readiness

Run the readiness gate before deploying:

```bash
./scripts/check_production_readiness.sh
```

This validates model artifacts, manifest integrity, migration ordering, CI workflow,
and security configuration. All checks must pass for deployment.

## Documentation

- **Technical audit & debt:** `reports/AUDIT_2026.md`, `reports/TECHNICAL_DEBT.md`
- **Roadmap:** `reports/ROADMAP_NEXT_PHASE.md`
- **Model inventory:** `reports/MODEL_INVENTORY.md`
- **ADRs:** `docs/adr/0001-braindecode-execution-strategy.md`, `docs/adr/0002-eeg-embedding-storage-contract.md`
- **Training guide:** `training/docs/TRAINING_GUIDE.md`
- **Contributing:** [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- **Historical docs:** `docs/archived/` (superseded Phase 1 documents retained for traceability)

## License

See individual component licenses:

- Weights: CC-BY-4.0 (see `training/docs/MODEL_CARD.md`)
- Architecture: BSD-3-Clause (Braindecode)
- Datasets: per-dataset (see `src/lib/datasets/manifest.ts`)
