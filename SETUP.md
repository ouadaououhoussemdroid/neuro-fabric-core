# 🛠️ Local Development Environment Setup

**Last Updated:** 2026-06-19  
**Status:** Ready for Bootstrap

---

## Quick Start (5 minutes)

If you have `git`, `Node.js 18+`, `Python 3.11+`, and `bun` already installed:

```bash
# Clone the repository
git clone https://github.com/ouadaououhoussemdroid/neuro-fabric-core.git
cd neuro-fabric-core

# Run the bootstrap script
bash scripts/bootstrap.sh

# Or manually:
bun install
python -m venv venv
source venv/bin/activate  # Linux/Mac: or venv\Scripts\activate on Windows
pip install -r training/requirements.txt

# Start development server
bun run dev
```

Open: http://localhost:5173/

---

## Prerequisites

### Required
- **Git** (2.30+) — version control
- **Node.js** (18+) — JavaScript runtime
- **Bun** (1.0+) — package manager & runtime
- **Python** (3.11+) — ML/training pipeline

### Optional
- **VS Code** — editor (with Prettier + ESLint extensions)
- **CUDA 12.1** — GPU acceleration for training (optional)

---

## Installation Steps

### 1. Install Bun

```bash
# macOS / Linux
curl -fsSL https://bun.sh/install | bash

# Windows (PowerShell)
powershell -c "irm https://bun.sh/install.ps1 | iex"

# Verify
bun --version
```

### 2. Clone Repository

```bash
git clone https://github.com/ouadaououhoussemdroid/neuro-fabric-core.git
cd neuro-fabric-core
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r training/requirements.txt
```

**Verify Python setup:**
```bash
python training/setup_env.py
```

Expected output:
```
✅ Python 3.11.x detected
✅ torch 2.5.1 installed
✅ braindecode 1.1.0 installed
✅ moabb 1.4.0 installed
✅ All dependencies verified
```

### 4. Install Node Dependencies

```bash
bun install
```

This installs ~200 packages including:
- React 19
- TanStack Start
- ONNX Runtime Web
- Tailwind CSS
- Radix UI

### 5. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

No changes needed — the variables are public dev credentials.

---

## Verification

### Test 1: Python Environment

```bash
python << 'EOF'
import torch
import braindecode
import moabb
print(f"PyTorch: {torch.__version__}")
print(f"Braindecode: {braindecode.__version__}")
print(f"MOABB: {moabb.__version__}")
print("✅ All Python dependencies working")
EOF
```

### Test 2: Bun & Node

```bash
bun run dev
```

Expected output:
```
➜  Local:   http://localhost:5173/
➜  press h to show help
```

Press `Ctrl+C` to stop.

### Test 3: TypeScript Compilation

```bash
bun run build
```

Should complete without errors.

---

## Project Structure

```
neuro-fabric-core/
├── src/                        # TypeScript/React frontend + backend
│   ├── lib/
│   │   ├── eeg/               # EEG parsers & preprocessing
│   │   ├── embeddings/        # PCA, feature extraction
│   │   ├── decoder/           # Cognitive metrics
│   │   ├── ai/                # ONNX model registry
│   │   └── vector-search/     # Similarity search
│   ├── routes/                # File-based routing (TanStack Start)
│   ├── components/            # React component library
│   └── integrations/          # External service integrations
│
├── training/                   # Python ML pipeline
│   ├── scripts/
│   │   ├── acquire_dataset.py      # Download BCI-IV-2a
│   │   ├── preprocess.py           # EEG processing
│   │   ├── train.py                # Model training
│   │   ├── validate.py             # Cross-subject validation
│   │   ├── evaluate.py             # Performance metrics
│   │   └── export_onnx.py          # ONNX export
│   ├── configs/
│   │   └── eegconformer-bciiv2a.yaml
│   ├── notebooks/
│   │   └── EEGConformer_BCIIV2a.ipynb
│   └── requirements.txt
│
├── docs/                       # Documentation
│   ├── REALITY_CHECK.md        # Honest audit of current state
│   ├── BLUEPRINT_PREPARATION.md # 4-phase product roadmap
│   ├── architecture.md
│   └── roadmaps/
│       └── 2026-06-19_open_source_execution_blueprint.md
│
├── scripts/
│   └── bootstrap.sh            # Automated setup
│
├── .env.example                # Template for environment variables
├── package.json                # Node dependencies
├── bunfig.toml                 # Bun configuration
├── vite.config.ts              # Vite build configuration
├── tsconfig.json               # TypeScript configuration
└── eslint.config.js            # Linting rules
```

---

## Common Commands

### Development

```bash
# Start dev server (frontend + backend)
bun run dev

# Type-check only (no build)
bun run build --mode development

# Lint code
bun run lint

# Format code
bun run format
```

### Training (Python)

```bash
cd training

# Full pipeline (1-2 hours)
bash scripts/run_all.sh

# Or step-by-step:
python scripts/acquire_dataset.py    # ~30 min (first run)
python scripts/preprocess.py          # ~10 min
python scripts/train.py               # ~30 min (GPU recommended)
python scripts/validate.py            # ~5 min
python scripts/evaluate.py            # ~2 min
python scripts/export_onnx.py         # ~2 min
```

Output: `training/artefacts/eegconformer-bciiv2a-v1/eegconformer.onnx`

---

## Troubleshooting

### ❌ `bun: command not found`

```bash
# Add Bun to PATH
export PATH="$HOME/.bun/bin:$PATH"

# Or restart terminal
```

### ❌ `venv: command not found`

```bash
# Try with python3
python3 -m venv venv

# Or use conda
conda create -n neuro python=3.11
conda activate neuro
```

### ❌ PyTorch/CUDA errors

```bash
# Reinstall torch without cache
pip install --no-cache-dir torch==2.5.1 torchvision torchaudio

# CPU-only (slower but works everywhere)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### ❌ `Port 5173 already in use`

```bash
# Use a different port
bun run dev --port 3000
```

### ❌ Permission denied on `bootstrap.sh`

```bash
chmod +x scripts/bootstrap.sh
bash scripts/bootstrap.sh
```

---

## Next Steps

After successful setup, see the **[Roadmap](docs/roadmaps/2026-06-19_open_source_execution_blueprint.md)** for tasks:

1. **T-010** — Validate EEGConformer on BCI-IV-2a
2. **T-011** — Migrate vector index to pgvector
3. **T-025** — Train cognitive decoder v0

---

## Need Help?

- Check **[docs/](docs/)** for architecture & design docs
- See **[REALITY_CHECK.md](docs/REALITY_CHECK.md)** for what works vs. what's missing
- Open an issue on GitHub with logs

---
