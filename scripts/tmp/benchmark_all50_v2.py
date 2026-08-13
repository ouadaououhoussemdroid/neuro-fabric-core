#!/usr/bin/env python3
"""Benchmark v2 model (trained on S001-S040) on ALL 50 subjects for stats."""
import os, sys, json, numpy as np, warnings
import onnxruntime as ort
warnings.filterwarnings("ignore")

import moabb.datasets as mds
if not hasattr(mds, "BNCI2014001"):
    mds.BNCI2014001 = mds.BNCI2014_001
if not hasattr(mds, "HGD"):
    mds.HGD = mds.PhysionetMI

TMP = os.environ.get("TMP", "/tmp")
CACHE_PATH = os.path.join(TMP, "eegmmidb_cached.npz")
REPORT_DIR = "reports"

EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]

CHANCE_LEVEL = 0.25
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
PCA_SEEDS = [0x2026_0711, 0x2026_0712, 0x2026_0713]
ALL_SUBJECTS = list(range(1, 51))


def normalize_ch_name(ch):
    return ch.replace(".", "").upper()


def compute_bandpower_features(trial_data, source_ch_names, target_channels):
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    sfreq = 160
    source_idx = {normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = []
    for tc in target_channels:
        if tc in source_idx:
            selected.append(trial_data[source_idx[tc]])
        else:
            selected.append(np.mean(trial_data, axis=0))
    selected = np.array(selected)
    target_sr = 250
    if sfreq != target_sr:
        n = selected.shape[1]
        n_samples = int(n * target_sr / sfreq)
        t_old = np.linspace(0, 1, n, endpoint=False)
        t_new = np.linspace(0, 1, n_samples, endpoint=False)
        selected = np.stack([np.interp(t_new, t_old, ch) for ch in selected])
        sfreq = target_sr
        n = n_samples
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mags = np.abs(np.fft.rfft(selected * window[None, :], axis=1)) / n
    features = []
    for c in range(selected.shape[0]):
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            power = np.sum(mags[c][mask] ** 2)
            features.append(power)
    return np.array(features, dtype=np.float32)


def nearest_centroid_accuracy(train_embs, train_labels, test_embs, test_labels):
    train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    test_norm = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)
    classes = sorted(set(train_labels))
    centroids = []
    for c in classes:
        mask = np.array(train_labels) == c
        centroid = train_norm[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.array(centroids)
    sims = test_norm @ centroids.T
    preds = np.array(classes)[np.argmax(sims, axis=1)]
    correct = (preds == np.array(test_labels)).sum()
    return correct / len(test_labels)


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0, float(np.mean(values)), float(np.mean(values))
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
    alpha = 0.05
    return float(np.mean(values)), float(np.std(values, ddof=1)), \
        float(np.percentile(means, alpha/2*100)), float(np.percentile(means, (1-alpha/2)*100))


def permutation_test(a, b, n_perm=N_PERMUTATIONS, seed=42):
    rng = np.random.RandomState(seed)
    diffs = np.array(a) - np.array(b)
    observed = np.mean(diffs)
    n = len(diffs)
    count = sum(1 for _ in range(n_perm) if abs(np.mean(diffs * rng.choice([-1,1],size=n))) >= abs(observed))
    return (count + 1) / (n_perm + 1)


def paired_t_test(a, b):
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = np.array(a) - np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = float(diff.mean() / (pooled_std + 1e-8)) if pooled_std > 0 else 0.0
    return float(t_stat), float(p_value), d


def load_onnx(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    return {"session": sess, "input_name": inp.name, "output_name": out.name}


def run_inf(sess_info, inputs):
    sess, inp_name, out_name = sess_info["session"], sess_info["input_name"], sess_info["output_name"]
    results = []
    for inp in inputs:
        out = sess.run([out_name], {inp_name: inp})[0]
        results.append(out.flatten().astype(np.float32))
    return np.array(results)


def main():
    print("Loading cache...")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_all = cache["X_all"]
    raw_all = cache["raw_X_all"]
    y_all = cache["y_all"]
    s_all = cache["s_all"]
    source_ch_names = list(cache["source_ch_names"])

    mask = np.isin(s_all, ALL_SUBJECTS)
    X_subj = X_all[mask]
    raw_subj = raw_all[mask]
    y_subj = y_all[mask]
    s_subj = s_all[mask]

    results = {}

    # PCA baseline
    print("\n=== PCA Bandpower ===")
    from sklearn.decomposition import PCA
    feats_all, labels_all, subj_all = [], [], []
    for subj_id in ALL_SUBJECTS:
        m = s_subj == subj_id
        for trial, label in zip(raw_subj[m], y_subj[m]):
            feats = compute_bandpower_features(trial, source_ch_names, EEGCONFORMER_CHANS)
            feats_all.append(feats)
            labels_all.append(label)
            subj_all.append(subj_id)
    feats_all = np.array(feats_all)
    labels_all = np.array(labels_all)
    subj_all = np.array(subj_all)

    fold_accs = []
    for subj_id in ALL_SUBJECTS:
        test_mask = subj_all == subj_id
        train_mask = ~test_mask
        train_f = feats_all[train_mask]
        test_f = feats_all[test_mask]
        train_l = labels_all[train_mask].tolist()
        test_l = labels_all[test_mask].tolist()
        seed_accs = []
        for seed in PCA_SEEDS:
            fm = train_f.mean(axis=0)
            fs = train_f.std(axis=0) + 1e-8
            train_n = (train_f - fm) / fs
            test_n = (test_f - fm) / fs
            k = min(32, train_n.shape[1], train_n.shape[0] - 1)
            pca = PCA(n_components=k, random_state=seed)
            train_p = pca.fit_transform(train_n)
            test_p = pca.transform(test_n)
            train_p = train_p / (np.linalg.norm(train_p, axis=1, keepdims=True) + 1e-8)
            test_p = test_p / (np.linalg.norm(test_p, axis=1, keepdims=True) + 1e-8)
            acc = nearest_centroid_accuracy(train_p, train_l, test_p, test_l)
            seed_accs.append(acc)
        fold_accs.append(np.mean(seed_accs))

    pca_mean, pca_std, pca_lo, pca_hi = bootstrap_ci(fold_accs)
    print(f"PCA: {pca_mean:.4f} ± {pca_std:.4f} (CI: {pca_lo:.4f}-{pca_hi:.4f})")
    results["PCA_Bandpower"] = {"mean_accuracy": pca_mean, "std": pca_std, "ci95": [pca_lo, pca_hi],
                                 "per_subject": [float(a) for a in fold_accs]}

    # Original EEGConformer
    print("\n=== EEGConformer (original) ===")
    sess = load_onnx("public/models/eegconformer.onnx")
    inputs = [X_subj[i][None, :, :].astype(np.float32) for i in range(len(y_subj))]
    embs = run_inf(sess, inputs)

    fold_accs_orig = []
    for subj_id in ALL_SUBJECTS:
        test_mask = s_subj == subj_id
        train_mask = ~test_mask
        acc = nearest_centroid_accuracy(embs[train_mask], y_subj[train_mask].tolist(),
                                         embs[test_mask], y_subj[test_mask].tolist())
        fold_accs_orig.append(acc)

    orig_mean, orig_std, orig_lo, orig_hi = bootstrap_ci(fold_accs_orig)
    print(f"Original: {orig_mean:.4f} ± {orig_std:.4f} (CI: {orig_lo:.4f}-{orig_hi:.4f})")
    results["EEGConformer"] = {"mean_accuracy": orig_mean, "std": orig_std, "ci95": [orig_lo, orig_hi],
                                "per_subject": [float(a) for a in fold_accs_orig]}

    # Fine-tuned EEGConformer (v2, trained on S001-S040)
    print("\n=== EEGConformer-FT (v2) ===")
    sess2 = load_onnx("training/artefacts/eegconformer-physionet-v2/eegconformer_finetuned.onnx")
    embs2 = run_inf(sess2, inputs)

    fold_accs_ft = []
    for subj_id in ALL_SUBJECTS:
        test_mask = s_subj == subj_id
        train_mask = ~test_mask
        acc = nearest_centroid_accuracy(embs2[train_mask], y_subj[train_mask].tolist(),
                                         embs2[test_mask], y_subj[test_mask].tolist())
        fold_accs_ft.append(acc)

    ft_mean, ft_std, ft_lo, ft_hi = bootstrap_ci(fold_accs_ft)
    print(f"FT (v2): {ft_mean:.4f} ± {ft_std:.4f} (CI: {ft_lo:.4f}-{ft_hi:.4f})")
    results["EEGConformer-FT"] = {"mean_accuracy": ft_mean, "std": ft_std, "ci95": [ft_lo, ft_hi],
                                   "per_subject": [float(a) for a in fold_accs_ft]}

    # Statistics
    print("\n=== Statistics ===")
    # FT vs Original (all 50 subjects)
    t1, p1, d1 = paired_t_test(fold_accs_ft, fold_accs_orig)
    perm1 = permutation_test(fold_accs_ft, fold_accs_orig)
    print(f"FT vs Original: Δ={np.mean(np.array(fold_accs_ft)-np.array(fold_accs_orig)):+.4f}, "
          f"p(t)={p1:.4f}, p(perm)={perm1:.4f}, d={d1:.3f}")

    # FT vs PCA
    t2, p2, d2 = paired_t_test(fold_accs_ft, fold_accs)
    perm2 = permutation_test(fold_accs_ft, fold_accs)
    print(f"FT vs PCA: Δ={np.mean(np.array(fold_accs_ft)-np.array(fold_accs)):+.4f}, "
          f"p(t)={p2:.4f}, p(perm)={perm2:.4f}, d={d2:.3f}")

    # Original vs PCA
    t3, p3, d3 = paired_t_test(fold_accs_orig, fold_accs)
    perm3 = permutation_test(fold_accs_orig, fold_accs)
    print(f"Original vs PCA: Δ={np.mean(np.array(fold_accs_orig)-np.array(fold_accs)):+.4f}, "
          f"p(t)={p3:.4f}, p(perm)={perm3:.4f}, d={d3:.3f}")

    # Save results
    output = {
        "subjects": "all 50 (S001-S050)",
        "note": "v2 model trained on S001-S040, benchmarked on all 50 subjects with LOSO. "
                "Training subjects (S001-S040) may have slightly inflated embedding quality.",
        "PCA": results["PCA_Bandpower"],
        "EEGConformer": results["EEGConformer"],
        "EEGConformer-FT": results["EEGConformer-FT"],
        "stats_ft_vs_original": {"delta": float(np.mean(np.array(fold_accs_ft)-np.array(fold_accs_orig))),
                                  "t": t1, "p_t": p1, "p_perm": perm1, "d": d1},
        "stats_ft_vs_pca": {"delta": float(np.mean(np.array(fold_accs_ft)-np.array(fold_accs))),
                             "t": t2, "p_t": p2, "p_perm": perm2, "d": d2},
        "stats_original_vs_pca": {"delta": float(np.mean(np.array(fold_accs_orig)-np.array(fold_accs))),
                                   "t": t3, "p_t": p3, "p_perm": perm3, "d": d3},
    }

    with open(os.path.join(REPORT_DIR, "t031_all50_v2_model.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: reports/t031_all50_v2_model.json")


if __name__ == "__main__":
    main()
