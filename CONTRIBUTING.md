# Contributing to Neuro-Fabric Core

## Development Workflow

### 1. Set Up Local Environment

See [SETUP.md](SETUP.md) for detailed instructions.

Quick version:
```bash
bash scripts/bootstrap.sh
bun run dev
```

### 2. Create a Feature Branch

Branch naming convention:
- `feature/T-###-description` — new feature (T-### from roadmap)
- `fix/issue-description` — bug fix
- `docs/topic` — documentation
- `refactor/component-name` — refactoring

Example:
```bash
git checkout -b feature/T-010-eegconformer-validation
```

### 3. Make Changes

#### TypeScript/React Code
```bash
# Development server auto-reloads on save
bun run dev

# Type-check
bun run build

# Lint
bun run lint

# Format
bun run format
```

#### Python Code (Training)
```bash
cd training

# Individual scripts
python scripts/train.py

# Or full pipeline
bash scripts/run_all.sh
```

### 4. Commit Changes

Use clear, atomic commits:

```bash
git add .
git commit -m "feat: T-010 validate EEGConformer on BCI-IV-2a

- Run model on holdout test set
- Report recall@10 vs PCA baseline
- Add validation notebook

Closes: #42"
```

### 5. Push & Create Pull Request

```bash
git push origin feature/T-010-eegconformer-validation
```

Then:
1. Open PR on GitHub
2. Link to related task/issue
3. Describe what changed and why
4. Wait for CI checks to pass

---

## Code Style

### TypeScript
- Use `bun run format` (Prettier)
- Use `bun run lint` (ESLint)
- Strict mode enabled (`tsconfig.json`)
- Path imports use `@/` alias

### Python
- Follow PEP 8
- Type hints for functions
- Docstrings for modules/classes
- Black formatting (future: add to CI)

---

## Testing

### Frontend
```bash
# Run type-check
bun run build

# Run tests (when added)
bun run test
```

### Training
```bash
cd training

# Validate environment
python setup_env.py

# Test individual script
python scripts/preprocess.py --test
```

---

## Documentation

Every feature should include:
1. **Code comments** — why, not what
2. **README update** — if new user-facing feature
3. **Docstring** — for exported functions/classes
4. **Example usage** — in docstring or separate `.md`

---

## The Roadmap (T-001 to T-028)

See [docs/roadmaps/2026-06-19_open_source_execution_blueprint.md](docs/roadmaps/2026-06-19_open_source_execution_blueprint.md)

Current focus (30-day sprint):
- **T-010** — EEGConformer validation
- **T-011** — pgvector migration
- **T-025** — Cognitive decoder v0

---

## Review Checklist

Before submitting PR, ensure:
- [ ] Code compiles (`bun run build`)
- [ ] Linting passes (`bun run lint`)
- [ ] Formatting applied (`bun run format`)
- [ ] Tests pass (if applicable)
- [ ] Commit message is clear
- [ ] Branch is up-to-date with `main`
- [ ] No secrets in code or `.env` (check `.gitignore`)

---

## Questions?

- Check [docs/](docs/)
- See [REALITY_CHECK.md](docs/REALITY_CHECK.md) for architecture decisions
- Open an issue on GitHub

---
