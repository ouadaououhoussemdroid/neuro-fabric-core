# T-020 — Reproducible training Makefile.
# Usage:
#   make train MODEL=eegconformer DATASET=bciiv2a
#   make preprocess MODEL=eegconformer DATASET=bciiv2a
#   make export MODEL=eegconformer
#   make evaluate MODEL=eegconformer
#   make all MODEL=eegconformer DATASET=bciiv2a

PYTHON ?= python
MODEL ?= eegconformer
DATASET ?= bciiv2a
CONFIG ?= training/configs/eegconformer-bciiv2a.yaml

.PHONY: train preprocess export evaluate validate all clean

preprocess:
	$(PYTHON) training/scripts/preprocess.py --config $(CONFIG)

train: preprocess
	$(PYTHON) training/scripts/train.py --config $(CONFIG)

export:
	$(PYTHON) training/scripts/export_onnx.py --config $(CONFIG)

evaluate:
	$(PYTHON) training/scripts/evaluate.py --config $(CONFIG)

validate:
	$(PYTHON) training/scripts/validate.py --config $(CONFIG)

all: train export evaluate validate

clean:
	rm -rf training/cache/processed/* training/artefacts/*
