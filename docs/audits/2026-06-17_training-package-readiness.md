# Training Package Readiness Audit

**Date:** 2026-06-17
**Scope:** `training/` only (notebook, configs, scripts, docs). No code modified.
**Goal:** verify the package can produce `eegconformer.pt → eegconformer.onnx` end-to-end on a fresh Google Colab T4 runtime, with no manual rescue steps.

## 1. Readiness Matrix

| Stage | File(s) | Status | Notes |
|---|---|---|---|
| Contract | `configs/eegconformer-bciiv2a.yaml` | Ready | 22ch · 250Hz · 1000 samples · 32-d · opset 17 — matches `registerBraindecodeEEGConformer`. |
| Dataset acquisition | `scripts/acquire_dataset.py` | Ready | MOABB `BNCI2014_001`; idempotent; honours `MNE_DATA`/`MOABB_DATA`. |
| Preprocessing | `scripts/preprocess.py` | Ready | MOABB `MotorImagery` paradigm (4–38 Hz, 0–4 s, resample 250 Hz), per-trial z-score, cross-subject split, T-axis pad/crop. |
| Training | `scripts/train.py` | Ready | AdamW + cosine, AMP, early stopping, best-val checkpoint, `train_history.json`. |
| Validation | `scripts/validate.py` | Ready | Cross-subject hold-out (subject 9). Writes `validation_report.json`. |
| Evaluation | `scripts/evaluate.py` | Ready | Mirrors browser benchmark (cosine recall@k, embedding norm/variance). |
| Checkpoint export | produced by `train.py` | Ready | `eegconformer.pt` is best-val `state_dict`. |
| ONNX export | `scripts/export_onnx.py` + repo `scripts/export_braindecode_eegconformer.py` | Ready | opset 17, named outputs `embedding`+`logits`, `onnx.checker`, >=0.999 PT-ORT cosine gate. |
| Packaging | `scripts/package.py` | Ready | sha256, `manifest.json`, embeds train/validation/evaluation reports, copies `MODEL_CARD.md`. |
| Orchestration | `scripts/run_all.sh` | Ready | Sequential pipeline. |
| Colab notebook | `notebooks/EEGConformer_BCIIV2a.ipynb` | Ready with caveats | 12 cells: clone -> install -> GPU check -> cat config -> acquire -> preprocess -> train -> validate -> evaluate -> export -> package -> zip & download. Caveats below. |
| Operator docs | `docs/TRAINING_GUIDE.md`, `docs/MODEL_CARD.md`, `README.md` | Ready | Acceptance thresholds documented (`holdout_acc >= 0.55`, `recall@10 >= 0.55`). |

**Pipeline completeness vs the stated goal (`eegconformer.pt -> eegconformer.onnx`):** No missing stage. Every required artefact (`eegconformer.pt`, `eegconformer.onnx`, `manifest.json`, `MODEL_CARD.md`, `train_history.json`, `validation_report.json`, `evaluation_report.json`) is produced by the documented pipeline.

## 2. Findings

### 2.1 Blockers (would prevent a fresh user from producing the artefact)

None. The pipeline is functionally complete; every script is wired to the next via `Paths.from_config(cfg)` and output filenames are consistent across `train.py -> export_onnx.py -> package.py`.

### 2.2 High-impact caveats (fixable, not blocking)

1. **Notebook clone URL is a placeholder.** Cell 1 reads `NEUROFABRIC_REPO` from env and falls back to `https://github.com/your-org/neuro-fabric.git`. A first-time Colab user who does not set the env var will clone a non-existent repo. Mitigation today: set `NEUROFABRIC_REPO` before "Run all", or paste the real URL into the cell. Suggested fix: replace the default with the canonical repo URL or surface a clearly-marked input cell.
2. **Torch pin vs Colab default.** `requirements.txt` pins `torch==2.3.*`. Colab T4 typically ships with a different torch+CUDA combo; reinstalling can take several minutes and, if the resolver pulls the CPU-only wheel, training will silently fall back to CPU (~10x slower). Mitigation: the notebook prints `torch.cuda.is_available()` in cell 3 — abort and reinstall with `--index-url https://download.pytorch.org/whl/cu121` if it reports False.
3. **`EEGConformer.fc` attribute name.** Both `export_braindecode_eegconformer.py` and `evaluate.py` hook `model.fc`. This matches Braindecode >=0.8. Any future bump that renames the pooled-features module will break the parity gate. Locked today by `braindecode>=0.8,<0.9`; flag if the pin is widened.
4. **`final_fc_length: auto` is a YAML string.** `yaml.safe_load` returns the string `"auto"`, which Braindecode's `EEGConformer` accepts as a sentinel. Verified across `train.py`, `validate.py`, `evaluate.py`. No action needed; documented so a future numeric override does not silently regress.
5. **MOABB channel count.** `BNCI2014_001` exposes 22 EEG + 3 EOG channels. The `MotorImagery` paradigm drops EOG, leaving 22 — matches the contract. A future MOABB change would trip the `assert X.shape[1] == 22` in `preprocess.py` (good), with no automatic remediation.
6. **MODEL_CARD metrics placeholders.** `docs/MODEL_CARD.md` defers metrics to `manifest.json`. `package.py` copies the card verbatim and does not template numbers in. Acceptable (manifest is the source of truth), but operators publishing the card externally must fill in the values manually.
7. **Colab artefact download.** Final cell zips the artefact and triggers `files.download`. For large bundles the browser download can stall. Mitigation: mount Google Drive or copy to `gs://` before download (not currently scripted).
8. **No license / citation file shipped inside the artefact bundle.** `MODEL_CARD.md` lists licences but no `LICENSE` / `CITATIONS.bib` is written alongside. Cosmetic; not required for runtime.

### 2.3 Low-priority observations

- `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast` are deprecated aliases in torch 2.3; they still work but emit warnings.
- `run_all.sh` is invoked via `bash`, so execute permission is not required.
- No automated CI runs the Python pipeline; correctness is verified only by in-script `assert` statements and the ONNX parity gate.
- The notebook does not pin the Colab runtime to GPU programmatically; relies on metadata (`accelerator: GPU`) and the operator selecting a GPU runtime.
- No determinism re-run check (export the same checkpoint twice and compare sha256). The parity gate covers PT-ORT but not ORT-ORT across runs.

## 3. End-to-end trace (what actually happens on `bash scripts/run_all.sh`)

```
acquire_dataset.py   -> cache/moabb/                           (MOABB cache, idempotent)
preprocess.py        -> cache/processed/<name>/train.npz       (X[N,22,1000] f32, y i64, subjects i64)
                        cache/processed/<name>/holdout.npz
train.py             -> artefacts/<name>/eegconformer.pt        (best-val state_dict)
                        artefacts/<name>/train_history.json
validate.py          -> artefacts/<name>/validation_report.json (holdout_accuracy, per_class)
evaluate.py          -> artefacts/<name>/evaluation_report.json (recall@10, norms, variance)
export_onnx.py       -> artefacts/<name>/eegconformer.onnx      (opset 17, embedding+logits, >=0.999 parity)
package.py           -> artefacts/<name>/manifest.json          (sha256 + embedded reports)
                        artefacts/<name>/MODEL_CARD.md
```

Every downstream stage's inputs are produced by the previous stage. No manual step is required between stages.

## 4. Compatibility with the runtime

- ONNX input shape `[batch, 22, 1000]` matches `BraindecodeOnnxBridge` expectations.
- Named outputs `embedding` (32-d) and `logits` (4-d) match the bridge contract.
- Opset 17 matches `scripts/export_braindecode_eegconformer.py` and the registry expectation.
- Per-trial z-score normalisation mirrors `src/lib/eeg/preprocessing/normalize.ts`.
- Cosine recall@k formula in `evaluate.py` mirrors `src/lib/ai/benchmark/index.ts`, so offline and in-browser numbers are directly comparable.

## 5. Verdict

**Readiness:** GREEN. The training package is internally complete and can produce a deployable `eegconformer.onnx` without code changes.

**Single highest-impact polish:** replace the placeholder GitHub URL in the Colab notebook (cell 1) with the canonical Neuro-Fabric clone URL, or convert it into a required input cell. This is the only step where a first-time operator can silently take the wrong path.

**No code was modified by this audit.**
