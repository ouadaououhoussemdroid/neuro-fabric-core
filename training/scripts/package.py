"""Bundle the artefact directory with a manifest + MODEL_CARD.

Output:
    artefacts/<name>/
        eegconformer.pt
        eegconformer.onnx
        manifest.json
        MODEL_CARD.md
        train_history.json
        validation_report.json
        evaluation_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _common import TRAINING_ROOT, Paths, load_config


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    onnx_path = paths.artefacts / "eegconformer.onnx"
    pt_path = paths.artefacts / "eegconformer.pt"
    if not onnx_path.exists() or not pt_path.exists():
        raise SystemExit("[package] missing eegconformer.pt or eegconformer.onnx — run train + export first")

    manifest = {
        "name": cfg["name"],
        "architecture": cfg["architecture"],
        "task": cfg["task"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "channels": cfg["signal"]["channels"],
            "sample_rate_hz": cfg["signal"]["sample_rate_hz"],
            "window_samples": cfg["signal"]["window_samples"],
            "embedding_dim": cfg["model"]["embedding_dim"],
            "n_classes": cfg["model"]["n_classes"],
            "onnx_opset": cfg["export"]["opset"],
            "outputs": ["embedding", "logits"],
        },
        "files": {
            "checkpoint": {
                "path": pt_path.name,
                "bytes": pt_path.stat().st_size,
                "sha256": _sha256(pt_path),
            },
            "onnx": {
                "path": onnx_path.name,
                "bytes": onnx_path.stat().st_size,
                "sha256": _sha256(onnx_path),
            },
        },
    }
    for opt in ("train_history.json", "validation_report.json", "evaluation_report.json"):
        p = paths.artefacts / opt
        if p.exists():
            with p.open() as f:
                manifest[opt.replace(".json", "")] = json.load(f)

    (paths.artefacts / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Copy MODEL_CARD template if not already present.
    card_src = TRAINING_ROOT / "docs" / "MODEL_CARD.md"
    card_dst = paths.artefacts / "MODEL_CARD.md"
    if card_src.exists() and not card_dst.exists():
        shutil.copyfile(card_src, card_dst)
    print(f"[package] artefact ready → {paths.artefacts}")
    print(f"[package] onnx sha256={manifest['files']['onnx']['sha256']}")


if __name__ == "__main__":
    main()