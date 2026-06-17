"""Thin wrapper around the repo-root exporter.

Delegates to scripts/export_braindecode_eegconformer.py so that the
PyTorch↔ONNX parity contract lives in exactly one place.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from _common import REPO_ROOT, Paths, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    exporter = REPO_ROOT / "scripts" / "export_braindecode_eegconformer.py"
    ckpt = paths.artefacts / "eegconformer.pt"
    out = paths.artefacts / "eegconformer.onnx"
    cmd = [
        sys.executable, str(exporter),
        "--checkpoint", str(ckpt),
        "--out", str(out),
        "--channels", str(cfg["signal"]["channels"]),
        "--samples", str(cfg["signal"]["window_samples"]),
        "--classes", str(cfg["model"]["n_classes"]),
        "--opset", str(cfg["export"]["opset"]),
    ]
    print("[export_onnx]", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[export_onnx] wrote → {out}")


if __name__ == "__main__":
    main()