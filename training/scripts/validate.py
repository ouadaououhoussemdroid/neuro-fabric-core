"""Cross-subject hold-out validation for the trained EEGConformer."""
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import Paths, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    import torch
    from braindecode.models import EEGConformer

    data = np.load(paths.processed / "holdout.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]); y = torch.from_numpy(data["y"])

    sig = cfg["signal"]; m = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGConformer(
        n_outputs=m["n_classes"],
        n_chans=sig["channels"],
        n_times=sig["window_samples"],
        final_fc_length=m["final_fc_length"],
    ).to(device)
    state = torch.load(paths.artefacts / "eegconformer.pt", map_location=device)
    model.load_state_dict(state); model.eval()

    with torch.no_grad():
        logits = model(X.to(device))
        pred = logits.argmax(-1).cpu().numpy()
    acc = float((pred == y.numpy()).mean())
    per_class = {int(c): float((pred[y.numpy() == c] == c).mean()) for c in np.unique(y.numpy())}
    report = {"holdout_accuracy": acc, "per_class_accuracy": per_class, "n": int(len(y))}
    out = paths.artefacts / "validation_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[validate] holdout_acc={acc:.4f}  n={len(y)}  → {out}")


if __name__ == "__main__":
    main()