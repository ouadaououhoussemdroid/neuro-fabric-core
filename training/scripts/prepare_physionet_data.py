"""Pre-process and cache PhysioNet EEGMMIDB data for fast training/benchmarking.

Loads all EDF files, applies preprocessing (22-channel selection, resample to
250 Hz, bandpass 4-38 Hz, z-score), and saves results to a .npz cache file.
Also caches raw trials for PCA baseline feature computation.
"""
import os, warnings
import numpy as np
import mne
warnings.filterwarnings("ignore")

TMP = os.environ.get("TMP", "/tmp")
DATA_DIR = os.path.join(TMP, "eegmmidb")
CACHE_PATH = os.path.join(TMP, "eegmmidb_cached.npz")

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


def normalize_ch_name(ch):
    return ch.replace(".", "").upper()


def _zscore(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def load_subject(subj_code, data_dir):
    """Load raw EDF trials for a single subject."""
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


def preprocess_trial(trial_data, source_ch_names):
    """Resample to 250 Hz, bandpass 4-38 Hz, select 22 channels, z-score.

    Input:  [n_ch, n_samples] at 160 Hz
    Output: [22, 1000] float32
    """
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


def main():
    print(f"Loading data from: {DATA_DIR}")
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("S") and os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Found {len(subjects)} subjects")

    all_X, all_raw_X, all_y, all_subj_ids = [], [], [], []
    source_ch_names = None

    for subj_code in subjects:
        subj_id = int(subj_code[1:])
        trials, labels, ch_names = load_subject(subj_code, DATA_DIR)
        if len(trials) == 0:
            print(f"  {subj_code}: SKIP (no trials)")
            continue
        if source_ch_names is None:
            source_ch_names = ch_names

        processed, raw_kept, valid_labels = [], [], []
        for i, (trial, label) in enumerate(zip(trials, labels)):
            try:
                proc = preprocess_trial(trial, source_ch_names)
                processed.append(proc)
                raw_kept.append(trial)
                valid_labels.append(label)
            except Exception as e:
                print(f"  {subj_code}: WARN: {e}")
                continue

        if len(processed) == 0:
            continue

        all_X.append(np.stack(processed))
        all_raw_X.append(np.stack(raw_kept))
        all_y.append(np.array(valid_labels, dtype=np.int64))
        all_subj_ids.append(np.full(len(valid_labels), subj_id, dtype=np.int64))
        print(f"  {subj_code}: {len(valid_labels)} trials OK")

    X_all = np.concatenate(all_X).astype(np.float32)
    raw_all = np.concatenate(all_raw_X).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int64)
    s_all = np.concatenate(all_subj_ids).astype(np.int64)

    print(f"\nTotal: {len(X_all)} trials, {len(set(s_all))} subjects")
    print(f"X shape: {X_all.shape}, raw shape: {raw_all.shape}, y shape: {y_all.shape}")
    print(f"Class distribution: {np.bincount(y_all)}")
    per_subj = {int(sid): int(np.sum(s_all == sid)) for sid in sorted(set(s_all))}
    print(f"Per-subject trial counts: {per_subj}")

    np.savez_compressed(CACHE_PATH,
        X_all=X_all, raw_X_all=raw_all, y_all=y_all, s_all=s_all,
        source_ch_names=np.array(source_ch_names))
    print(f"\nCached to: {CACHE_PATH}")
    print(f"Cache size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
