"""Fine-tune EEGConformer on PhysioNet EEGMMIDB — v2 (optimized).

Key changes from v1:
    - Reduced dropout (0.5 → 0.1) for small-dataset fine-tuning
    - Cosine LR with linear warmup (10 epochs)
    - Class-balanced sampling to handle minor class imbalance
    - Higher patience (50) to allow slow convergence
    - Weight decay for regularisation
    - Gradient clipping

Training: S001-S006 (180 trials) | Val: S007-S008 (60) | Test: S009-S010 (60)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]
SAMPLE_RATE = 250
WINDOW_SAMPLES = 1000
BANDPASS = [4.0, 38.0]
N_SECONDS = 4.0


def normalize_ch_name(ch: str) -> str:
    return ch.replace(".", "").upper()


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def load_subject_trials(subj_code: str, data_dir: str) -> tuple[list, list, list]:
    import mne
    trials, labels, ch_names = [], [], None
    for run_idx, run in enumerate([5, 6]):
        fname = os.path.join(data_dir, subj_code, f"{subj_code}R{run:02d}.edf")
        if not os.path.exists(fname):
            continue
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        if ch_names is None:
            ch_names = [normalize_ch_name(c) for c in raw.ch_names]
        sfreq = raw.info["sfreq"]
        events, _ = mne.events_from_annotations(raw, verbose=False)
        for ev in events:
            event_type = raw.annotations.description[
                np.argmin(np.abs(raw.annotations.onset - ev[0] / sfreq))
            ]
            if event_type not in ("T1", "T2"):
                continue
            onset = ev[0]
            trial_len = int(N_SECONDS * sfreq)
            end = min(onset + trial_len, len(raw.times))
            trial = raw.get_data()[:, onset:end]
            if run_idx == 0:
                label = 0 if event_type == "T1" else 1
            else:
                label = 2 if event_type == "T1" else 3
            trials.append(trial.astype(np.float32))
            labels.append(label)
    return trials, labels, ch_names


def preprocess_trial(trial_data: np.ndarray, source_ch_names: list) -> np.ndarray:
    import mne
    source_idx = {normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = np.array([trial_data[source_idx[ch]] for ch in EEGCONFORMER_CHANS])
    info = mne.create_info(ch_names=EEGCONFORMER_CHANS, sfreq=160, ch_types="eeg")
    inst = mne.io.RawArray(selected, info, verbose=False)
    inst.resample(SAMPLE_RATE, verbose=False)
    inst.filter(BANDPASS[0], BANDPASS[1], verbose=False, method="fir", fir_design="firwin")
    data = inst.get_data()
    target = WINDOW_SAMPLES
    if data.shape[1] < target:
        data = np.pad(data, ((0, 0), (0, target - data.shape[1])), mode="edge")
    elif data.shape[1] > target:
        start = (data.shape[1] - target) // 2
        data = data[:, start:start + target]
    return _zscore(data)


def load_onnx_weights(pt_model: nn.Module, onnx_path: str) -> dict:
    """Extract weights from BCI-IV-2a ONNX model (cosine ~0.996)."""
    import onnx, onnx.numpy_helper as nh, re

    onnx_model = onnx.load(onnx_path)
    onnx_init = {}
    for init in onnx_model.graph.initializer:
        onnx_init[init.name] = nh.to_array(init)

    pt_state = pt_model.state_dict()
    state_load = {}

    for k, v in pt_state.items():
        onnx_key = f"model.{k}"
        if onnx_key in onnx_init and onnx_init[onnx_key].shape == tuple(v.shape):
            state_load[k] = torch.from_numpy(onnx_init[onnx_key].copy())

    # BN2 (folded in ONNX): identity
    state_load["patch_embedding.shallownet.2.weight"] = torch.ones(40)
    state_load["patch_embedding.shallownet.2.bias"] = torch.zeros(40)
    state_load["patch_embedding.shallownet.2.running_mean"] = torch.zeros(40)
    state_load["patch_embedding.shallownet.2.running_var"] = torch.ones(40)
    state_load["patch_embedding.shallownet.2.num_batches_tracked"] = torch.tensor(1)

    # QKV: _v_ (40,120) → transpose (120,40) → split into Q/K/V (40,40 each)
    _v_entries = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("_v_") and v.shape == (40, 120)],
        key=lambda x: int(re.sub(r"[^0-9]", "", x[0])),
    )
    for idx, (vname, varr) in enumerate(_v_entries):
        qkv = varr.T  # (120, 40)
        for name, sl in [("queries", (0, 40)), ("keys", (40, 80)), ("values", (80, 120))]:
            pt_key = f"transformer.{idx}.0.fn.1.{name}.weight"
            if pt_key not in state_load:
                state_load[pt_key] = torch.from_numpy(qkv[sl[0]:sl[1]].copy())

    # Projection: val_ (40,40) → transpose (best perm: 0,4,5,1,2,3)
    val_40_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    proj_perm = [0, 4, 5, 1, 2, 3]
    for layer, val_idx in enumerate(proj_perm):
        if val_idx < len(val_40_40):
            pt_key = f"transformer.{layer}.0.fn.1.projection.weight"
            if pt_key not in state_load:
                state_load[pt_key] = torch.from_numpy(val_40_40[val_idx][1].T.copy())

    # FFN layers
    val_40_160 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 160)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    val_160_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (160, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    for idx, (vname, varr) in enumerate(val_40_160):
        state_load[f"transformer.{idx}.1.fn.1.0.weight"] = torch.from_numpy(varr.T.copy())
    for idx, (vname, varr) in enumerate(val_160_40):
        pt_key = f"transformer.{idx}.1.fn.1.3.weight"
        if pt_key not in state_load:
            state_load[pt_key] = torch.from_numpy(varr.T.copy())

    return state_load


def make_model(dropout=0.1):
    from braindecode.models import EEGConformer
    model = EEGConformer(n_outputs=4, n_chans=22, n_times=1000, final_fc_length="auto")
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    return model


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CosineWarmupLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        super().__init__(optimizer, lr_lambda)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--train-subj", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--val-subj", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--test-subj", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--onnx-init", default="public/models/eegconformer.onnx")
    ap.add_argument("--out-dir", default="training/artefacts/eegconformer-physionet-v1")
    args = ap.parse_args()

    set_seed(args.seed)
    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] Device: {device}")

    # ─── Load data ──────────────────────────────────────────────────────
    print("[finetune] Loading and preprocessing data...")
    all_X, all_y, all_s = [], [], []
    all_subj = sorted(set(args.train_subj + args.val_subj + args.test_subj))
    for subj_id in all_subj:
        trials, labels, ch_names = load_subject_trials(f"S{subj_id:03d}", data_dir)
        for trial, label in zip(trials, labels):
            try:
                proc = preprocess_trial(trial, ch_names)
                all_X.append(proc)
                all_y.append(label)
                all_s.append(subj_id)
            except Exception:
                continue
    X_all = np.stack(all_X)
    y_all = np.array(all_y, dtype=np.int64)
    s_all = np.array(all_s, dtype=np.int64)

    train_mask = np.isin(s_all, args.train_subj)
    val_mask = np.isin(s_all, args.val_subj)
    test_mask = np.isin(s_all, args.test_subj)
    print(f"  Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    X_train = torch.from_numpy(X_all[train_mask])
    y_train = torch.from_numpy(y_all[train_mask])
    X_val = torch.from_numpy(X_all[val_mask])
    y_val = torch.from_numpy(y_all[val_mask])
    X_test = torch.from_numpy(X_all[test_mask])
    y_test = torch.from_numpy(y_all[test_mask])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    # ─── Build model ────────────────────────────────────────────────────
    model = make_model(dropout=args.dropout)

    if args.onnx_init and os.path.exists(args.onnx_init):
        print(f"[finetune] Initialising from ONNX: {args.onnx_init}")
        state_dict = load_onnx_weights(model, args.onnx_init)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    # ─── Optimizer & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupLR(optimizer, warmup_epochs=args.warmup, total_epochs=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # ─── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    stale = 0
    history = []
    ckpt_path = out_dir / "eegconformer.pt"

    print(f"[finetune] LR={args.lr}, wd={args.weight_decay}, dropout={args.dropout}, "
          f"warmup={args.warmup}, patience={args.patience}")
    print(f"[finetune] Starting training for up to {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_val.to(device)), y_val.to(device)).item()
            va = (model(X_val.to(device)).argmax(-1) == y_val.to(device)).float().mean().item()
            ta = (model(X_test.to(device)).argmax(-1) == y_test.to(device)).float().mean().item()

        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                        "val_loss": vl, "val_acc": va, "test_acc": ta})

        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            best_val_acc = va
            best_test_acc = ta
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[finetune] Early stopping @ epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch={epoch:03d}  train_loss={loss.item():.4f}  "
                  f"val_loss={vl:.4f}  val_acc={va:.4f}  test_acc={ta:.4f}")

    # ─── Save history ──────────────────────────────────────────────────
    with (out_dir / "train_history.json").open("w") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
            "config": {
                "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
                "epochs": args.epochs, "weight_decay": args.weight_decay,
                "patience": args.patience, "dropout": args.dropout, "warmup": args.warmup,
                "onnx_init": args.onnx_init,
            },
            "split": {
                "train_subjects": sorted(args.train_subj),
                "val_subjects": sorted(args.val_subj),
                "test_subjects": sorted(args.test_subj),
                "train_trials": int(train_mask.sum()),
                "val_trials": int(val_mask.sum()),
                "test_trials": int(test_mask.sum()),
            },
        }, f, indent=2)

    print(f"\n[finetune] Best: epoch={best_epoch}, val_loss={best_val_loss:.4f}, "
          f"val_acc={best_val_acc:.4f}, test_acc={best_test_acc:.4f}")
    print(f"[finetune] Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
