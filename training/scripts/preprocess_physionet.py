"""Preprocess PhysioNet EEGMMIDB to the Neuro-Fabric EEGConformer contract.

Output: training/cache/processed/<name>/{train,val,test}.npz with
    X: float32 [N, 22, 1000]
    y: int64   [N]
    subjects: int64 [N]

Pipeline:
    1. Load EDF files for runs 5-6 (4-class motor imagery)
    2. Select 22-channel BCI-IV-2a subset (all present in PhysioNet 64-ch)
    3. Resample 160 → 250 Hz, bandpass 4-38 Hz
    4. 4-second windows
    5. Per-trial z-score per channel (matches runtime normalization contract
       in src/lib/eeg/preprocessing/normalize.ts and the T-030/T-031 benchmark)
    6. Subject-stratified split: S001-S006 train, S007-S008 val, S009-S010 test
       (no subject overlap → no data leakage)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 22-channel BCI-IV-2a 10-20 subset — all present in PhysioNet EEGMMIDB
EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]

SAMPLE_RATE = 250
WINDOW_SAMPLES = 1000  # 4 s
BANDPASS = [4.0, 38.0]
N_SECONDS = 4.0


def normalize_ch_name(ch: str) -> str:
    return ch.replace(".", "").upper()


def _zscore(x: np.ndarray) -> np.ndarray:
    """Per-trial per-channel z-score: [22, 1000] → z-scored."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def load_subject_trials(subj_code: str, data_dir: str) -> tuple[list, list, list]:
    """Load EDF for a single subject, return (trials, labels, ch_names).

    Each trial is [22, n_samples] at 160 Hz.
    Run 5: T1=left(0), T2=right(1); Run 6: T1=feet(2), T2=tongue(3).
    """
    import mne

    trials = []
    labels = []
    ch_names = None

    for run_idx, run in enumerate([5, 6]):
        fname = os.path.join(data_dir, subj_code, f"{subj_code}R{run:02d}.edf")
        if not os.path.exists(fname):
            continue

        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        if ch_names is None:
            ch_names = [normalize_ch_name(c) for c in raw.ch_names]
        sfreq = raw.info["sfreq"]  # 160 Hz

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
            trial = raw.get_data()[:, start:end]

            label = 0 if event_type == "T1" else 1 if run_idx == 0 else (2 if event_type == "T1" else 3)
            trials.append(trial.astype(np.float32))
            labels.append(label)

    return trials, labels, ch_names


def preprocess_trial(trial_data: np.ndarray, source_ch_names: list) -> np.ndarray:
    """Resample to 250 Hz, bandpass 4-38 Hz, select 22 channels, z-score.

    Input:  trial_data [64, n_samples] at 160 Hz
    Output: [22, 1000] float32
    """
    import mne

    # Channel selection — all 22 BCI-IV-2a channels exist in PhysioNet
    source_idx = {normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = np.array([trial_data[source_idx[ch]] for ch in EEGCONFORMER_CHANS])  # [22, T]

    info = mne.create_info(ch_names=EEGCONFORMER_CHANS, sfreq=160, ch_types="eeg")
    inst = mne.io.RawArray(selected, info, verbose=False)

    inst.resample(SAMPLE_RATE, verbose=False)
    inst.filter(BANDPASS[0], BANDPASS[1], verbose=False, method="fir", fir_design="firwin")

    data = inst.get_data()  # [22, T']

    # Center-pad or crop to 1000 samples
    target = WINDOW_SAMPLES
    if data.shape[1] < target:
        pad = target - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="edge")
    elif data.shape[1] > target:
        start = (data.shape[1] - target) // 2
        data = data[:, start:start + target]

    return _zscore(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Path to PhysioNet EEGMMIDB directory (e.g. /tmp/eegmmidb)")
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                    help="Subject IDs (1-10)")
    ap.add_argument("--train-subj", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    help="Subjects for training set")
    ap.add_argument("--val-subj", type=int, nargs="+", default=[7, 8],
                    help="Subjects for validation set")
    ap.add_argument("--test-subj", type=int, nargs="+", default=[9, 10],
                    help="Subjects for held-out test set")
    ap.add_argument("--out", default=None,
                    help="Output directory for .npz files")
    args = ap.parse_args()

    # Resolve data directory
    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    if not os.path.exists(data_dir):
        print(f"[preprocess] ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # Resolve output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        # Default: training/cache/processed/eegconformer-physionet-v1/
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = repo_root / "training" / "cache" / "processed" / "eegconformer-physionet-v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_subj = set(args.train_subj)
    val_subj = set(args.val_subj)
    test_subj = set(args.test_subj)
    all_subj = train_subj | val_subj | test_subj

    print(f"[preprocess] Data dir: {data_dir}")
    print(f"[preprocess] Train subjects: {sorted(train_subj)}")
    print(f"[preprocess] Val subjects: {sorted(val_subj)}")
    print(f"[preprocess] Test subjects: {sorted(test_subj)}")
    print(f"[preprocess] Channels: {len(EEGCONFORMER_CHANS)}")

    # Load and preprocess all trials
    all_X, all_y, all_subj_ids = [], [], []
    for subj_id in sorted(all_subj):
        subj_code = f"S{subj_id:03d}"
        print(f"  Loading {subj_code}...", end=" ", flush=True)
        trials, labels, ch_names = load_subject_trials(subj_code, data_dir)
        if len(trials) == 0:
            print(f"  SKIP (no trials)")
            continue

        # Preprocess each trial
        processed = []
        valid_labels = []
        for trial, label in zip(trials, labels):
            try:
                proc = preprocess_trial(trial, ch_names)
                processed.append(proc)
                valid_labels.append(label)
            except Exception as e:
                print(f"\n  WARN: {subj_code} trial error: {e}")
                continue

        if len(processed) == 0:
            print(f"  SKIP (all trials failed)")
            continue

        X = np.stack(processed)  # [N, 22, 1000]
        y = np.array(valid_labels, dtype=np.int64)
        subj_arr = np.full(len(valid_labels), subj_id, dtype=np.int64)

        all_X.append(X)
        all_y.append(y)
        all_subj_ids.append(subj_arr)
        print(f"{len(valid_labels)} trials, {y.shape[0]} OK")

    if not all_X:
        print("[preprocess] ERROR: no data loaded")
        sys.exit(1)

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    s_all = np.concatenate(all_subj_ids)

    # Split by subject (no overlap → no leakage)
    splits = {"train": train_subj, "val": val_subj, "test": test_subj}
    label_counts = {c: int((y_all == c).sum()) for c in range(4)}
    print(f"\n[preprocess] Total: {len(y_all)} trials, class dist: {label_counts}")

    classes = [f"class_{i}" for i in range(4)]
    for split_name, split_subjs in splits.items():
        mask = np.isin(s_all, list(split_subjs))
        n = mask.sum()
        print(f"  {split_name}: {n} trials, subjects={sorted(split_subjs)}")
        np.savez_compressed(
            out_dir / f"{split_name}.npz",
            X=X_all[mask],
            y=y_all[mask],
            subjects=s_all[mask],
            classes=np.array(classes),
        )

    print(f"\n[preprocess] Wrote → {out_dir}")


if __name__ == "__main__":
    main()
