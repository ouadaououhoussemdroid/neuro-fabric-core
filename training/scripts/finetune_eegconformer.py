"""Fine-tune EEGConformer on PhysioNet EEGMMIDB motor imagery data.

Trains the production EEGConformer architecture (22 ch, 250 Hz, 1000 samples,
4-class MI) on PhysioNet EEGMMIDB runs 5-6 using a subject-stratified split:

    Train: S001-S006 (6 subjects, ~180 trials)
    Val:   S007-S008 (2 subjects, ~60 trials)
    Test:  S009-S010 (2 subjects, ~60 trials)  — held-out, never seen during training

The model is initialised from the BCI-IV-2a pretrained checkpoint (extracted
from the existing ONNX weights when the .pt is unavailable) and fine-tuned with
AdamW + cosine LR + early stopping on validation loss.

Critical rules (Mission 4):
    - No data leakage: train/val/test split is by subject (no overlap).
    - No changes to production defaults: same architecture, same preprocessing.
    - Deterministic evaluation: fixed seed, deterministic PyTorch settings.

Usage:
    python training/scripts/finetune_eegconformer.py \
        --config training/configs/eegconformer-bciiv2a.yaml \
        --data-dir /tmp/eegmmidb \
        --epochs 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

# 22-channel BCI-IV-2a 10-20 subset
EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]

SAMPLE_RATE = 250
WINDOW_SAMPLES = 1000
BANDPASS = [4.0, 38.0]
N_SECONDS = 4.0
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]


def normalize_ch_name(ch: str) -> str:
    return ch.replace(".", "").upper()


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def load_subject_trials(subj_code: str, data_dir: str) -> tuple[list, list, list]:
    """Load EDF for a single subject, return (trials, labels, ch_names)."""
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
                label = 0 if event_type == "T1" else 1  # Run 5: left(0), right(1)
            else:
                label = 2 if event_type == "T1" else 3    # Run 6: feet(2), tongue(3)
            trials.append(trial.astype(np.float32))
            labels.append(label)

    return trials, labels, ch_names


def preprocess_trial(trial_data: np.ndarray, source_ch_names: list) -> np.ndarray:
    """Resample to 250 Hz, bandpass 4-38 Hz, select 22 channels, z-score.

    Input:  [n_ch, n_samples] at 160 Hz
    Output: [22, 1000] float32
    """
    import mne

    source_idx = {normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = np.array([trial_data[source_idx[ch]] for ch in EEGCONFORMER_CHANS])  # [22, T]

    info = mne.create_info(ch_names=EEGCONFORMER_CHANS, sfreq=160, ch_types="eeg")
    inst = mne.io.RawArray(selected, info, verbose=False)
    inst.resample(SAMPLE_RATE, verbose=False)
    inst.filter(BANDPASS[0], BANDPASS[1], verbose=False, method="fir", fir_design="firwin")

    data = inst.get_data()  # [22, T']

    target = WINDOW_SAMPLES
    if data.shape[1] < target:
        pad = target - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="edge")
    elif data.shape[1] > target:
        start = (data.shape[1] - target) // 2
        data = data[:, start:start + target]

    return _zscore(data)


def load_onnx_weights(pt_model: torch.nn.Module, onnx_path: str) -> dict:
    """Extract weights from the existing BCI-IV-2a ONNX model.

    The ONNX model stores some weights in folded form (BatchNorm folded into
    Conv, combined QKV projection). We reconstruct the PyTorch state dict:

    - Directly matched initializer names (72 keys)
    - QKV: _v_ entries (40,120) → transpose to (120,40), split into Q/K/V rows
    - Projection: val_ (40,40) entries → transpose
    - FFN layers: val_ (40,160)→fn.1.0.weight.T, val_ (160,40)→fn.1.3.weight.T
    - Projection layer permutation: layer {i} uses a permuted val_ entry

    Returns a state_dict dict mapping PyTorch keys → tensors.
    """
    import onnx
    import onnx.numpy_helper as nh

    onnx_model = onnx.load(onnx_path)
    onnx_init = {}
    for init in onnx_model.graph.initializer:
        onnx_init[init.name] = nh.to_array(init)

    pt_state = pt_model.state_dict()
    state_load = {}

    # 1. Directly matched weights (strip 'model.' prefix)
    for k, v in pt_state.items():
        onnx_key = f"model.{k}"
        if onnx_key in onnx_init and onnx_init[onnx_key].shape == tuple(v.shape):
            state_load[k] = torch.from_numpy(onnx_init[onnx_key].copy())

    # 2. BN2 (BatchNorm2d after depthwise conv) — folded in ONNX, set to identity
    state_load["patch_embedding.shallownet.2.weight"] = torch.ones(40)
    state_load["patch_embedding.shallownet.2.bias"] = torch.zeros(40)
    state_load["patch_embedding.shallownet.2.running_mean"] = torch.zeros(40)
    state_load["patch_embedding.shallownet.2.running_var"] = torch.ones(40)
    state_load["patch_embedding.shallownet.2.num_batches_tracked"] = torch.tensor(1)

    # 3. QKV weights from _v_ (40,120) entries
    _v_entries = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("_v_") and v.shape == (40, 120)],
        key=lambda x: int(re.sub(r"[^0-9]", "", x[0])),
    )
    for idx, (vname, varr) in enumerate(_v_entries):
        qkv = varr.T  # (120, 40) — PyTorch combined Linear(40, 120) weight
        for name, sl in [("queries", (0, 40)), ("keys", (40, 80)), ("values", (80, 120))]:
            pt_key = f"transformer.{idx}.0.fn.1.{name}.weight"
            if pt_key not in state_load:
                state_load[pt_key] = torch.from_numpy(qkv[sl[0]:sl[1]].copy())

    # 4. Projection weights from val_ (40,40) entries
    # Best permutation found via exhaustive search: (0, 4, 5, 1, 2, 3)
    val_40_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    proj_perm = [0, 4, 5, 1, 2, 3]
    for layer, val_idx in enumerate(proj_perm):
        if val_idx < len(val_40_40):
            vname, varr = val_40_40[val_idx]
            pt_key = f"transformer.{layer}.0.fn.1.projection.weight"
            if pt_key not in state_load:
                state_load[pt_key] = torch.from_numpy(varr.T.copy())

    # 5. FFN first layer from val_ (40,160) entries
    val_40_160 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (40, 160)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    for idx, (vname, varr) in enumerate(val_40_160):
        pt_key = f"transformer.{idx}.1.fn.1.0.weight"
        if pt_key not in state_load:
            state_load[pt_key] = torch.from_numpy(varr.T.copy())

    # 6. FFN second layer from val_ (160,40) entries
    val_160_40 = sorted(
        [(k, v) for k, v in onnx_init.items() if k.startswith("val_") and v.shape == (160, 40)],
        key=lambda x: int(x[0].split("_")[1]),
    )
    for idx, (vname, varr) in enumerate(val_160_40):
        pt_key = f"transformer.{idx}.1.fn.1.3.weight"
        if pt_key not in state_load:
            state_load[pt_key] = torch.from_numpy(varr.T.copy())

    # 7. BN params in patch_embedding (shallower.1 — depthwise conv BN)
    # ONNX has shallower.1.weight and shallower.1.bias but not running stats
    # Set running stats to identity
    if "patch_embedding.shallownet.1.weight" not in state_load:
        state_load["patch_embedding.shallownet.1.weight"] = torch.ones(40)
    if "patch_embedding.shallownet.1.bias" not in state_load:
        state_load["patch_embedding.shallownet.1.bias"] = torch.zeros(40)
    # Running stats for the conv BN — set to identity
    state_load["patch_embedding.shallownet.1.running_mean"] = torch.zeros(40)
    state_load["patch_embedding.shallownet.1.running_var"] = torch.ones(40)
    state_load["patch_embedding.shallownet.1.num_batches_tracked"] = torch.tensor(1)
    # Projection BatchNorm (projection.0 is Conv2d(40,40,(1,1)), projection.1 would be BN
    # But in the PyTorch model, projection might not have BN. Let's check:
    # projection.0.weight/bias are Conv2d — already loaded from ONNX
    # There might not be a projection BN in the PyTorch model

    return state_load


def set_deterministic(seed: int) -> None:
    """Set all RNGs for deterministic training (matches _common.set_seed)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Config YAML (defaults to eegconformer-bciiv2a.yaml)")
    ap.add_argument("--data-dir", default=None,
                    help="Path to PhysioNet EEGMMIDB directory")
    ap.add_argument("--train-subj", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="Training subjects (default: 1-6)")
    ap.add_argument("--val-subj", type=int, nargs="+", default=[7, 8],
                    help="Validation subjects (default: 7-8)")
    ap.add_argument("--test-subj", type=int, nargs="+", default=[9, 10],
                    help="Test subjects (default: 9-10)")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Max training epochs")
    ap.add_argument("--lr", type=float, default=6.25e-4,
                    help="Learning rate")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Weight decay")
    ap.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience")
    ap.add_argument("--amp", action="store_true", default=True,
                    help="Use mixed precision (auto-disabled on CPU)")
    ap.add_argument("--seed", type=int, default=20260617,
                    help="Random seed")
    ap.add_argument("--onnx-init", default=None,
                    help="Path to pre-trained ONNX for weight initialisation")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for checkpoint and history")
    args = ap.parse_args()

    set_deterministic(args.seed)

    # Resolve paths
    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    if not os.path.exists(data_dir):
        print(f"[finetune] ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else \
        repo_root / "training" / "artefacts" / "eegconformer-physionet-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_subj = set(args.train_subj) | set(args.val_subj) | set(args.test_subj)

    # ─── Load and preprocess data ──────────────────────────────────────────
    print("[finetune] Loading and preprocessing PhysioNet EEGMMIDB data...")
    all_X, all_y, all_subj_ids = [], [], []
    for subj_id in sorted(all_subj):
        subj_code = f"S{subj_id:03d}"
        trials, labels, ch_names = load_subject_trials(subj_code, data_dir)
        if len(trials) == 0:
            print(f"  {subj_code}: SKIP (no trials)")
            continue

        processed = []
        valid_labels = []
        for trial, label in zip(trials, labels):
            try:
                proc = preprocess_trial(trial, ch_names)
                processed.append(proc)
                valid_labels.append(label)
            except Exception as e:
                print(f"  {subj_code}: WARN preprocessing error: {e}")
                continue

        X = np.stack(processed)
        y = np.array(valid_labels, dtype=np.int64)
        s = np.full(len(valid_labels), subj_id, dtype=np.int64)
        all_X.append(X)
        all_y.append(y)
        all_subj_ids.append(s)
        print(f"  {subj_code}: {len(valid_labels)} trials OK")

    X_all = np.concatenate(all_X)
    y_all = np.array(np.concatenate(all_y))
    s_all = np.array(np.concatenate(all_subj_ids))

    train_mask = np.isin(s_all, args.train_subj)
    val_mask = np.isin(s_all, args.val_subj)
    test_mask = np.isin(s_all, args.test_subj)

    print(f"\n[finetune] Dataset split:")
    print(f"  Train: {train_mask.sum()} trials, subjects={sorted(args.train_subj)}")
    print(f"  Val:   {val_mask.sum()} trials, subjects={sorted(args.val_subj)}")
    print(f"  Test:  {test_mask.sum()} trials, subjects={sorted(args.test_subj)}")

    # Convert to tensors
    X_train = torch.from_numpy(X_all[train_mask])
    y_train = torch.from_numpy(y_all[train_mask])
    X_val = torch.from_numpy(X_all[val_mask])
    y_val = torch.from_numpy(y_all[val_mask])
    X_test = torch.from_numpy(X_all[test_mask])
    y_test = torch.from_numpy(y_all[test_mask])

    # ─── Build model ──────────────────────────────────────────────────────
    from braindecode.models import EEGConformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[finetune] Device: {device}")

    model = EEGConformer(
        n_outputs=4,
        n_chans=22,
        n_times=1000,
        final_fc_length="auto",
    ).to(device)

    # Initialise from ONNX if available
    if args.onnx_init and os.path.exists(args.onnx_init):
        print(f"[finetune] Initialising from ONNX: {args.onnx_init}")
        state_dict = load_onnx_weights(model, args.onnx_init)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {result.missing_keys}")
        # Verify weight loading quality
        model.eval()
        # Ensure repo root is on path for 'scripts' package imports
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.export_braindecode_eegconformer import EEGConformerExportWrapper
        wrapper = EEGConformerExportWrapper(model)
        torch.manual_seed(42)
        dummy = torch.randn(1, 22, 1000, device=device)
        with torch.no_grad():
            pt_emb, _ = wrapper(dummy)

        import onnxruntime as ort
        ort_sess = ort.InferenceSession(args.onnx_init, providers=["CPUExecutionProvider"])
        ort_emb = torch.from_numpy(ort_sess.run(None, {"input": dummy.cpu().numpy()})[0])
        cos = torch.nn.functional.cosine_similarity(
            pt_emb.flatten().unsqueeze(0).cpu(),
            ort_emb.flatten().unsqueeze(0),
        ).item()
        print(f"  Weight init quality: cosine={cos:.6f} (target >0.999)")
    else:
        print("[finetune] No ONNX init provided, training from scratch")

    # ─── Training setup ──────────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    best_val_loss = math.inf
    best_val_acc = 0.0
    best_test_acc = 0.0
    stale = 0
    history = []
    ckpt_path = out_dir / "eegconformer.pt"

    print(f"\n[finetune] Starting training for up to {args.epochs} epochs...")
    print(f"  LR={args.lr}, batch_size={args.batch_size}, patience={args.patience}")

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validation
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += loss_fn(logits, yb).item() * len(yb)
                vcorrect += (logits.argmax(-1) == yb).sum().item()
                vtotal += len(yb)
        vloss /= max(1, vtotal)
        vacc = vcorrect / max(1, vtotal)

        # Test (for monitoring, not for model selection)
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                test_loss += loss_fn(logits, yb).item() * len(yb)
                test_correct += (logits.argmax(-1) == yb).sum().item()
                test_total += len(yb)
        test_loss /= max(1, test_total)
        test_acc = test_correct / max(1, test_total)

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_loss": vloss, "val_acc": vacc,
            "test_loss": test_loss, "test_acc": test_acc,
        })

        print(f"  epoch={epoch:03d}  train_loss={loss.item():.4f}  "
              f"val_loss={vloss:.4f}  val_acc={vacc:.4f}  "
              f"test_acc={test_acc:.4f}")

        # Early stopping on val loss
        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_val_acc = vacc
            best_test_acc = test_acc
            stale = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[finetune] Early stopping @ epoch {epoch} (patience={args.patience})")
                break

    # ─── Save training history ──────────────────────────────────────────
    with (out_dir / "train_history.json").open("w") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "config": {
                "seed": args.seed,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
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

    print(f"\n[finetune] Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    print(f"[finetune] Test acc (at best epoch) = {best_test_acc:.4f}")
    print(f"[finetune] Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
