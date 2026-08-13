"""Fine-tune EEGConformer on PhysioNet EEGMMIDB motor imagery data (v2).

Same architecture, preprocessing, hyperparameters, and evaluation methodology
as the 20-subject run, but trained on 40 subjects (S001-S040) with strict
held-out test on S041-S050.

Usage:
    python training/scripts/finetune_eegconformer_v2.py \
        --data-dir $TMP/eegmmidb \
        --out-dir training/artefacts/eegconformer-physionet-v2
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

# ─── Fix braindecode/moabb import (version incompatibility) ─────────────────────
_moabb_datasets_fixed = False
def _fix_moabb():
    global _moabb_datasets_fixed
    if _moabb_datasets_fixed:
        return
    try:
        import moabb.datasets as mds
        if not hasattr(mds, "BNCI2014001"):
            mds.BNCI2014001 = mds.BNCI2014_001
        if not hasattr(mds, "HGD"):
            mds.HGD = mds.PhysionetMI
        _moabb_datasets_fixed = True
    except Exception:
        pass

_fix_moabb()

# ─── Constants (identical to original) ──────────────────────────────────────────

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

    trials, labels, ch_names = [], [], []
    for run_idx, run in enumerate([5, 6]):
        fname = os.path.join(data_dir, subj_code, f"{subj_code}R{run:02d}.edf")
        if not os.path.exists(fname):
            continue
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        if ch_names == []:
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
    """Resample to 250 Hz, bandpass 4-38 Hz, select 22 channels, z-score.

    Input:  [n_ch, n_samples] at 160 Hz
    Output: [22, 1000] float32
    """
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
        pad = target - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="edge")
    elif data.shape[1] > target:
        start = (data.shape[1] - target) // 2
        data = data[:, start:start + target]

    return _zscore(data)


# ─── ONNX weight loading (identical to original) ─────────────────────────────────

def load_onnx_weights(pt_model: torch.nn.Module, onnx_path: str) -> dict:
    """Extract weights from the existing BCI-IV-2a ONNX model."""
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

    # 2. BatchNorm2d (shallownet.2) — folded in ONNX, set to identity
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

    # 7. Note: shallownet.0 and shallownet.1 are Conv2d layers (loaded in step 1).
    # BatchNorm2d is at shallownet.2 (set to identity in step 2).
    # Conv2d weights are directly matched from ONNX — no override needed.

    return state_load


def set_deterministic(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """Linear warmup + cosine annealing."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--train-subj", type=int, nargs="+", default=list(range(1, 41)),
                    help="Training subjects (default: 1-40)")
    ap.add_argument("--val-subj", type=int, nargs="+", default=None,
                    help="Validation subjects (default: internal 15% split of train)")
    ap.add_argument("--test-subj", type=int, nargs="+", default=list(range(41, 51)),
                    help="Test subjects (default: 41-50)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=15, help="Warmup epochs")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--onnx-init", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    set_deterministic(args.seed)

    # Resolve paths
    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    if not os.path.exists(data_dir):
        print(f"[finetune] ERROR: data dir not found: {data_dir}")
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else \
        repo_root / "training" / "artefacts" / "eegconformer-physionet-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_subj = sorted(args.train_subj)
    test_subj = sorted(args.test_subj)
    all_subj = set(train_subj) | set(test_subj)
    if args.val_subj:
        val_subj = sorted(args.val_subj)
        all_subj |= set(val_subj)
    else:
        val_subj = None

    # ─── Load and preprocess data ──────────────────────────────────────────
    cache_path = os.path.join(os.path.dirname(data_dir), "eegmmidb_cached.npz")
    print(f"[finetune] Loading data: train={train_subj}, test={test_subj}")
    if os.path.exists(cache_path):
        print(f"[finetune] Loading from cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        X_all = cache["X_all"]
        y_all = cache["y_all"]
        s_all = cache["s_all"]
        source_ch_names = list(cache["source_ch_names"])
        print(f"  Cache loaded: {len(X_all)} trials, {len(set(s_all))} subjects")
    else:
        print(f"[finetune] Cache not found, loading EDF files...")
        subj_data = {}
        for subj_id in sorted(all_subj):
            subj_code = f"S{subj_id:03d}"
            trials, labels, ch_names = load_subject_trials(subj_code, data_dir)
            if len(trials) == 0:
                print(f"  {subj_code}: SKIP (no trials)")
                continue
            processed, valid_labels = [], []
            for trial, label in zip(trials, labels):
                try:
                    proc = preprocess_trial(trial, ch_names)
                    processed.append(proc)
                    valid_labels.append(label)
                except Exception as e:
                    print(f"  {subj_code}: WARN preprocessing error: {e}")
                    continue
            if len(processed) == 0:
                continue
            subj_data[subj_id] = {
                "X": np.stack(processed),
                "y": np.array(valid_labels, dtype=np.int64),
            }
            print(f"  {subj_code}: {len(valid_labels)} trials OK")
        Xs = [subj_data[s]["X"] for s in sorted(subj_data)]
        ys = [subj_data[s]["y"] for s in sorted(subj_data)]
        ss = [np.full(len(subj_data[s]["y"]), s, dtype=np.int64) for s in sorted(subj_data)]
        X_all = np.concatenate(Xs)
        y_all = np.concatenate(ys)
        s_all = np.concatenate(ss)
        source_ch_names = [c for c in EEGCONFORMER_CHANS]

    # Build train/val/test splits from cached arrays
    available_subj = sorted(set(s_all.tolist()))
    if val_subj:
        train_ids = [s for s in available_subj if s in train_subj]
        val_ids = [s for s in available_subj if s in val_subj]
        test_ids = [s for s in available_subj if s in test_subj]
    else:
        # Internal 85/15 val split within train subjects
        train_ids = sorted([s for s in available_subj if s in train_subj])
        rng = np.random.RandomState(args.seed)
        n_val = max(1, len(train_ids) // 7)  # ~15%
        val_ids = train_ids[:n_val]
        train_ids = train_ids[n_val:]
        test_ids = sorted([s for s in available_subj if s in test_subj])

    print(f"\n[finetune] Split: train={len(train_ids)} subjects, val={len(val_ids)} subjects, test={len(test_ids)} subjects")

    def build_split(ids):
        if not ids:
            return None
        mask = np.isin(s_all, ids)
        return X_all[mask], y_all[mask]

    X_train, y_train = build_split(train_ids)
    X_val, y_val = build_split(val_ids)
    X_test, y_test = build_split(test_ids)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Total samples: {X_train.shape[0] + X_val.shape[0]}")

    # ─── Build model ──────────────────────────────────────────────
    from braindecode.models import EEGConformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[finetune] Device: {device}")

    model = EEGConformer(
        n_outputs=4,
        n_chans=22,
        n_times=1000,
        final_fc_length="auto",
        drop_prob=args.dropout,
        att_drop_prob=args.dropout,
    ).to(device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Initialise from ONNX
    onnx_path = args.onnx_init or str(repo_root / "public" / "models" / "eegconformer.onnx")
    if os.path.exists(onnx_path):
        print(f"[finetune] Initialising from ONNX: {onnx_path}")
        state_dict = load_onnx_weights(model, onnx_path)
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {result.missing_keys}")
        # Verify weight loading quality
        model.eval()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from scripts.export_braindecode_eegconformer import EEGConformerExportWrapper
        wrapper = EEGConformerExportWrapper(model)
        torch.manual_seed(42)
        dummy = torch.randn(1, 22, 1000, device=device)
        with torch.no_grad():
            pt_emb, _ = wrapper(dummy)
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_emb = torch.from_numpy(ort_sess.run(None, {"input": dummy.cpu().numpy()})[0])
        cos = torch.nn.functional.cosine_similarity(
            pt_emb.flatten().unsqueeze(0).cpu(),
            ort_emb.flatten().unsqueeze(0),
        ).item()
        print(f"  Weight init quality: cosine={cos:.6f} (target >0.999)")

    # ─── Training setup ──────────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test) if X_test is not None else None
    y_test_t = torch.from_numpy(y_test) if X_test is not None else None

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # We'll use manual warmup + cosine LR
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # Calculate total optimizer steps for warmup + cosine
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    best_val_loss = math.inf
    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0
    stale = 0
    history = []
    ckpt_path = out_dir / "eegconformer.pt"

    print(f"\n[finetune] Starting training for up to {args.epochs} epochs...")
    print(f"  LR={args.lr}, WD={args.weight_decay}, batch={args.batch_size}, "
          f"patience={args.patience}, warmup={args.warmup}, "
          f"label_smoothing={args.label_smoothing}, grad_clip={args.grad_clip}, "
          f"dropout={args.dropout}")
    print(f"  steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    global_step = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Manual LR scheduling with warmup + cosine
            global_step += 1
            lr_now = get_warmup_cosine_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

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

        # Test (monitoring only)
        test_acc = 0.0
        if y_test_t is not None:
            model.eval()
            test_loader = DataLoader(
                TensorDataset(X_test_t, y_test_t),
                batch_size=args.batch_size, shuffle=False,
            )
            tloss, tcorrect, ttotal = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    tloss += loss_fn(logits, yb).item() * len(yb)
                    tcorrect += (logits.argmax(-1) == yb).sum().item()
                    ttotal += len(yb)
            test_acc = tcorrect / max(1, ttotal)

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_loss": vloss, "val_acc": vacc,
            "test_acc": test_acc, "lr": lr_now,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep={epoch:03d}: train_loss={loss.item():.4f} "
                  f"val_loss={vloss:.4f} val_acc={vacc:.4f} test_acc={test_acc:.4f} lr={lr_now:.7f}")

        # Early stopping on val loss
        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_val_acc = vacc
            best_epoch = epoch
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
            "best_epoch": best_epoch,
            "config": {
                "seed": args.seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "warmup": args.warmup,
                "label_smoothing": args.label_smoothing,
                "grad_clip": args.grad_clip,
                "dropout": args.dropout,
                "onnx_init": onnx_path,
            },
            "split": {
                "train_subjects": train_ids,
                "val_subjects": val_ids,
                "test_subjects": test_ids,
                "train_trials": int(X_train.shape[0]),
                "val_trials": int(X_val.shape[0]),
                "test_trials": int(X_test.shape[0]) if X_test is not None else 0,
            },
        }, f, indent=2)

    print(f"\n[finetune] Best val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")
    print(f"[finetune] Test acc (at best epoch) = {best_test_acc:.4f}")
    print(f"[finetune] Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    main()
