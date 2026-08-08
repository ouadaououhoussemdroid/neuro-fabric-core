# Contributing to Neuro-Fabric Core

Thank you for your interest in contributing to Neuro-Fabric Core! This document
outlines the workflow and conventions for contributors.

## Prerequisites

- **Node.js** >= 20
- **Package manager:** `bun` is the authoritative package manager. `npm` works
  as a fallback but all CI runs through `bun`.
- **Python** >= 3.12 (for training pipeline and ONNX export scripts)
- **Supabase CLI** (for local database migrations)

## Getting Started

```bash
# 1. Clone and install
git clone https://github.com/ouadaououhoussemdroid/neuro-fabric-core.git
cd neuro-fabric-core
bun install

# 2. Configure environment
cp .env.example .env
# Edit .env with your Supabase URL, keys, and a generated CRON_SECRET.

# 3. Start the dev server
bun run dev
```

The dev server exposes:
- Main app: `http://localhost:5173`
- API routes: `http://localhost:5173/api/*`
- Metrics: `http://localhost:5173/api/public/metrics` (requires `CRON_SECRET`)

## Development Workflow

### Branch naming

```
main                    # production-ready, protected
feat/T-xxx-short-desc   # new features
fix/T-xxx-short-desc    # bug fixes
docs/T-xxx-short-desc   # documentation
```

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(eeg): add Sleep-EDF dataset loader
fix(metrics): resolve double-counting on empty labels
```

### Before opening a PR

All of these must pass locally:

```bash
bun run typecheck       # tsc --noEmit
bun run lint            # eslint .
bun run test            # vitest run
bun run test:coverage   # verify coverage doesn't drop below thresholds
```

## Project Structure

```
src/
├── lib/                    # Core library modules
│   ├── eeg/                # EEG parsing, preprocessing, acquisition
│   ├── ai/                 # AI foundation layer (adapters, models, embeddings)
│   ├── embeddings/         # Feature extraction, PCA, autoencoder
│   ├── vector-search/      # pgvector NeuralVectorIndex + recall SLO
│   ├── decoder/            # Cognitive state decoder
│   ├── graph/              # Concept graph (provenance layer)
│   ├── metrics/            # Prometheus-style metrics primitives
│   └── datasets/           # Dataset manifest + loaders
├── integrations/           # Supabase, external service integrations
├── routes/                 # TanStack Router route handlers
└── logging/                # Structured logging
```

## Coding Conventions

### TypeScript
- Target `ES2022`, module `Node16` — use native `import`/`export`.
- Type-only imports use `import type` to avoid bundling overhead.
- Prefer `const` over `let`; avoid `any` (use `unknown` + narrowing).
- Exported functions should have JSDoc comments explaining parameters
  and return types.

### Testing
- Tests live alongside source in `__tests__/` sub-directories.
- Use `vitest` — `describe`, `it`, `expect` from `vitest`.
- Mock external I/O (network, filesystem, DB) with `vi.fn()` / `vi.mock()`.
- Aim for **80%+ line coverage** on `src/lib/` modules.

### Metrics
- All new endpoints or significant operations should emit metrics via
  `src/lib/metrics/index.ts`.
- Use existing Counter/Gauge/Histogram primitives — don't create ad-hoc ones.
- Prometheus text format is exposed at `/api/public/metrics`.

### Documentation
- Update the relevant section in this guide if you change the contribution
  workflow itself.
- For significant architectural changes, add an ADR in `docs/adr/`.

## Database Migrations

Migrations live in `supabase/migrations/`. Apply locally with:

```bash
supabase db push
```

When adding a new table:
1. Enable RLS (`ALTER TABLE ... ENABLE ROW LEVEL SECURITY`)
2. Add policies for `authenticated` and `service_role`
3. Add relevant indexes (e.g. `ivfflat` for vector columns)

## Model Artefacts

ONNX models are served from `public/models/`. Update `public/models/manifest.json`
with the new model ID and SHA-256 hash when adding or updating a model.

## Code of Conduct

Be respectful and constructive. We are building infrastructure for
neurotechnology research — precision and reproducibility matter.
