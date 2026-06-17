#!/usr/bin/env bash
# Full pipeline: acquire → preprocess → train → validate → evaluate → export → package
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

python acquire_dataset.py
python preprocess.py
python train.py
python validate.py
python evaluate.py
python export_onnx.py
python package.py

echo "[run_all] artefact ready under training/artefacts/"