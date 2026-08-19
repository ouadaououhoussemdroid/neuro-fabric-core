#!/usr/bin/env python3
"""
Batch-wise EEGPT inference for subjects 11-50.
Each invocation processes a batch of subjects, appending to the 50-subj cache.
Designed to fit within 10-minute Bash tool timeout (~500 trials at 870ms/trial ≈ 7.3min).
"""
import sys, os, time, json
import numpy as np
import onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(REPO, "reports")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m26_retrieval_reassessment import (
    sha256_file, l2_normalize, preprocess_eegpt_trial, build_22ch_input, run_eegpt,
    EEGPT_SHA256, EEGPT_MODEL_PATH, CACHE_PATH,
)

EEGPT_CACHE_50 = os.path.join(REPORTS, ".m26_eegpt_50subj_cache.npz")
PROGRESS_FILE = os.path.join(REPORTS, ".m26_50subj_progress.json")

N_BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BATCH_SIZE = 5  # subjects per batch (~450 trials, ~7 min)

SUBJECTS = list(range(1, 51))
N_SUBJECTS = 50
RUNS = [5, 6, 7, 8, 9, 10]


def load_physionet_batch(subject_ids):
    """Load PhysioNet trials for a batch of subjects."""
    import mne
    import importlib.util
    m6_spec = importlib.util.spec_from_file_location("m6", os.path.join(os.path.dirname(__file__), "cbramod_remap_50subj.py"))
    m6 = importlib.util.module_from_spec(m6_spec)
    m6_spec.loader.exec_module(m6)
    t032 = m6.t032
    DATA_DIR = getattr(m6, "DATA_DIR")

    out = {}
    for subj_id in subject_ids:
        scode = f"S{subj_id:03d}"
        trials = []
        for run in RUNS:
            fname = os.path.join(DATA_DIR, scode, f"{scode}R{run:02d}.edf")
            if not os.path.exists(fname):
                print(f"  WARN: {fname} not found")
                continue
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            ch = [t032.normalize_ch_name(c) for c in raw.ch_names]
            sfreq = raw.info["sfreq"]
            events, _ = mne.events_from_annotations(raw, verbose=False)
            is_odd = (run % 2 == 1)
            for ev in events:
                idx = np.argmin(np.abs(raw.annotations.onset - ev[0] / sfreq))
                et = raw.annotations.description[idx]
                if et not in ("T1", "T2"):
                    continue
                onset = ev[0]
                tlen = int(4.0 * sfreq)
                start = int(onset)
                end = min(start + tlen, len(raw.times))
                trial = raw.get_data()[:, start:end].astype(np.float32)
                label = (0 if et == "T1" else 1) if is_odd else (2 if et == "T1" else 3)
                trials.append({
                    "data": trial, "ch_names": ch, "sfreq": float(sfreq),
                    "subject": subj_id, "run": run, "mi_label": label
                })
        if trials:
            out[subj_id] = trials
        print(f"  {scode}: {len(trials)} trials", flush=True)
    return out


def init_cache():
    """Initialize 50-subj cache from 10-subj cache if it doesn't exist."""
    if os.path.exists(EEGPT_CACHE_50):
        return
    cache_10 = os.path.join(REPORTS, ".m26_eegpt_retrieval_cache.npz")
    if os.path.exists(cache_10):
        import shutil
        shutil.copy2(cache_10, EEGPT_CACHE_50)
        print(f"  Initialized 50-subj cache from 10-subj cache", flush=True)


def get_progress():
    """Read progress file or return 0."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_batches": 0, "n_batch": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def main():
    print(f"M26 Batch Inference: Batch {N_BATCH}", flush=True)
    print(f"  Batch size: {BATCH_SIZE} subjects", flush=True)

    # Verify EEGPT artifact
    actual_sha = sha256_file(EEGPT_MODEL_PATH)
    assert actual_sha == EEGPT_SHA256, f"SHA mismatch: {actual_sha}"
    print(f"  EEGPT verified: {actual_sha[:16]}...", flush=True)

    # Initialize cache
    init_cache()

    # Load cache for subject/run alignment
    z = np.load(CACHE_PATH, allow_pickle=True)
    subj_full = np.asarray(z["subj_ids"], int)
    run_full = np.asarray(z["run_ids"], int)

    # Determine which subjects to process in this batch
    all_subjects = SUBJECTS
    done_subjects = []
    if os.path.exists(EEGPT_CACHE_50):
        c = np.load(EEGPT_CACHE_50, allow_pickle=True)
        done_subjects = sorted(set(c["eegpt_subj"].tolist()))
    pending_subjects = [s for s in all_subjects if s not in done_subjects]
    batch_subjects = pending_subjects[:BATCH_SIZE]

    if not batch_subjects:
        print(f"  All subjects cached! ({len(done_subjects)} subjects)", flush=True)
        progress = get_progress()
        progress["completed_batches"] = len(done_subjects) // BATCH_SIZE
        progress["n_batch"] = N_BATCH
        progress["all_done"] = True
        save_progress(progress)
        # Trigger full evaluation
        print("  All subjects cached. Running retrieval evaluation...", flush=True)
        return "EVAL_READY"

    print(f"  Done subjects: {done_subjects}", flush=True)
    print(f"  Batch subjects: {batch_subjects}", flush=True)

    # Load PhysioNet data for this batch
    print(f"  Loading PhysioNet data...", flush=True)
    t0 = time.perf_counter()
    subjects_data = load_physionet_batch(batch_subjects)
    load_time = time.perf_counter() - t0
    print(f"  Data loaded in {load_time:.1f}s", flush=True)

    # Run EEGPT inference
    print(f"  Running EEGPT inference...", flush=True)
    sess = ort.InferenceSession(EEGPT_MODEL_PATH, providers=["CPUExecutionProvider"])

    new_embs = []
    new_subj = []
    new_runs = []
    new_labels = []

    total = sum(len(subjects_data[s]) for s in subjects_data)
    t0 = time.perf_counter()
    trial_idx = 0
    for subj_id in sorted(subjects_data.keys()):
        for tr in subjects_data[subj_id]:
            data_62ch = preprocess_eegpt_trial(tr["data"], tr["ch_names"])
            data_22ch = build_22ch_input(data_62ch)
            emb = run_eegpt(sess, data_22ch)
            emb = l2_normalize(emb)
            new_embs.append(emb)
            new_subj.append(subj_id)
            new_runs.append(tr["run"])
            new_labels.append(tr["mi_label"])
            trial_idx += 1
            if trial_idx % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = trial_idx / elapsed
                remaining = (total - trial_idx) / rate if rate > 0 else 0
                print(f"  {trial_idx}/{total} ({elapsed:.0f}s, {remaining:.0f}s remaining)", flush=True)

    infer_time = time.perf_counter() - t0
    print(f"  Inference: {total} trials in {infer_time:.1f}s ({infer_time/total*1000:.0f}ms/trial)", flush=True)

    # Append to cache
    if os.path.exists(EEGPT_CACHE_50):
        c = np.load(EEGPT_CACHE_50, allow_pickle=True)
        eegpt_embs = np.concatenate([c["eegpt_embs"], np.array(new_embs, dtype=np.float32)], axis=0)
        eegpt_subj = np.concatenate([c["eegpt_subj"], np.array(new_subj, dtype=int)], axis=0)
        eegpt_runs = np.concatenate([c["eegpt_runs"], np.array(new_runs, dtype=int)], axis=0)
        eegpt_labels = np.concatenate([c["eegpt_labels"], np.array(new_labels, dtype=int)], axis=0)
        total_time = float(c.get("inference_time_sec", 0)) + infer_time
    else:
        eegpt_embs = np.array(new_embs, dtype=np.float32)
        eegpt_subj = np.array(new_subj, dtype=int)
        eegpt_runs = np.array(new_runs, dtype=int)
        eegpt_labels = np.array(new_labels, dtype=int)
        total_time = infer_time

    np.savez_compressed(EEGPT_CACHE_50,
                        eegpt_embs=eegpt_embs,
                        eegpt_subj=eegpt_subj,
                        eegpt_runs=eegpt_runs,
                        eegpt_labels=eegpt_labels,
                        inference_time_sec=total_time,
                        eegpt_sha256=EEGPT_SHA256)

    done_count = len(set(eegpt_subj.tolist()))
    print(f"  Cache saved: {eegpt_embs.shape}, subjects={done_count}/{N_SUBJECTS}", flush=True)

    # Save progress
    progress = get_progress()
    progress["completed_batches"] = done_count // BATCH_SIZE
    progress["n_batch"] = N_BATCH
    progress["done_count"] = done_count
    save_progress(progress)

    if done_count >= N_SUBJECTS:
        progress["all_done"] = True
        save_progress(progress)
        return "EVAL_READY"

    return "CONTINUE"


if __name__ == "__main__":
    result = main()
    if result == "EVAL_READY":
        print("\n✓ ALL 50 SUBJECTS CACHED. Ready for evaluation.", flush=True)
    else:
        print(f"\n→ Batch {N_BATCH} complete. Run next batch.", flush=True)
