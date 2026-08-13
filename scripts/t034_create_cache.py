"""Pre-compute T-034 data cache to avoid repeated EDF loading across experiment runs."""
import importlib.util
import os
import sys

import numpy as np

# Import functions from the training script
spec = importlib.util.spec_from_file_location(
    "t034_train",
    os.path.join(os.path.dirname(__file__), "t034-train-representation.py"),
)
t034 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t034)

load_subject_trials = t034.load_subject_trials
preprocess_trial = t034.preprocess_trial

data_dir = os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
cache_path = os.path.join(os.path.dirname(data_dir), "eegmmidb_t034_cached.npz")

all_subj = list(range(6, 51))  # S006-S050
print(f"[cache] Loading {len(all_subj)} subjects...")

subj_data = {}
source_ch_names = None
for subj_id in all_subj:
    subj_code = f"S{subj_id:03d}"
    trials, labels, ch_names = load_subject_trials(subj_code, data_dir)
    if source_ch_names is None and ch_names:
        source_ch_names = ch_names
    if len(trials) == 0:
        print(f"  {subj_code}: SKIP (no trials)")
        continue
    processed, valid_labels = [], []
    for trial, label in zip(trials, labels):
        try:
            proc = preprocess_trial(trial, source_ch_names if source_ch_names else ch_names)
            processed.append(proc)
            valid_labels.append(label)
        except Exception as e:
            print(f"  {subj_code}: WARN preprocessing error: {e}")
            continue
    if len(processed) > 0:
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
print(f"  Total: {len(X_all)} trials, {len(set(s_all))} subjects")

# Use a cache key that matches S006-S050
cache_key = f"subjects_{'_'.join(str(s) for s in sorted(set(s_all.tolist())))}"
save_dict = {
    "X_all": X_all,
    "y_all": y_all,
    "s_all": s_all,
    "source_ch_names": np.array(source_ch_names, dtype=object),
    "cache_key_str": cache_key,
    f"X_{cache_key}": X_all,
    f"y_{cache_key}": y_all,
    f"s_{cache_key}": s_all,
}
np.savez(cache_path, **save_dict)
print(f"[cache] Saved: {cache_path} ({os.path.getsize(cache_path)} bytes, key={cache_key})")
