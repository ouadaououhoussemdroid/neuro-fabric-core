"""Train EEGConformer on preprocessed BCI-IV-2a.

Saves the best-validation checkpoint to artefacts/<name>/eegconformer.pt.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _common import Paths, load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    paths = Paths.from_config(cfg)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from braindecode.models import EEGConformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    data = np.load(paths.processed / "train.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"])
    y = torch.from_numpy(data["y"])

    # Subject-stratified internal val split (10%) for early stopping.
    rng = np.random.default_rng(cfg["training"]["seed"])
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(1, int(0.1 * len(idx)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_loader = DataLoader(
        TensorDataset(X[tr_idx], y[tr_idx]),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[val_idx], y[val_idx]),
        batch_size=cfg["training"]["batch_size"],
    )

    sig = cfg["signal"]; m = cfg["model"]
    model = EEGConformer(
        n_outputs=m["n_classes"],
        n_chans=sig["channels"],
        n_times=sig["window_samples"],
        final_fc_length=m["final_fc_length"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"] and device.type == "cuda")

    best_val = math.inf
    best_acc = 0.0
    patience = cfg["training"]["early_stopping_patience"]
    stale = 0
    out_pt = paths.artefacts / "eegconformer.pt"
    history = []

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl += loss_fn(logits, yb).item() * len(yb)
                vc += (logits.argmax(-1) == yb).sum().item()
                vn += len(yb)
        vl /= max(1, vn); va = vc / max(1, vn)
        history.append({"epoch": epoch, "val_loss": vl, "val_acc": va})
        print(f"[train] epoch={epoch:03d}  val_loss={vl:.4f}  val_acc={va:.4f}")

        if vl < best_val - 1e-4:
            best_val = vl; best_acc = va; stale = 0
            torch.save(model.state_dict(), out_pt)
        else:
            stale += 1
            if stale >= patience:
                print(f"[train] early stop @ epoch={epoch} (patience={patience})")
                break

    with (paths.artefacts / "train_history.json").open("w") as f:
        json.dump({"history": history, "best_val_loss": best_val, "best_val_acc": best_acc}, f, indent=2)
    print(f"[train] best val_loss={best_val:.4f} val_acc={best_acc:.4f}  ckpt={out_pt}")


if __name__ == "__main__":
    main()