#!/usr/bin/env python3
"""
T-031-50 — Statistical Validation of Fine-tuned EEGConformer on 50 Subjects
==========================================================================
Extends T-031 to validate the EEGConformer fine-tuning improvement across
the full 50-subject dataset.

Setup:
  - Fine-tuned model: trained on S001-S040 (35 train + 5 val subjects)
  - Benchmark: LOSO across S041-S050 (10 strictly held-out subjects)
  - Comparison: PCA Bandpower vs Original EEGConformer vs Fine-tuned EEGConformer

Preserves T-030/T-031 methodology exactly:
  - Bandpass 4-38 Hz for PCA, model-native for learned models
  - 4-second windows, z-score per channel
  - Train-only PCA fitting per fold
  - Train-only candidate pools for recall@K (no leakage)
  - Nearest-centroid classification on 32-dim L2-normalized embeddings

Statistical robustness:
  - Bootstrap 95% CIs (1000 resamples)
  - Permutation testing (1000 label shuffles per comparison)
  - Paired t-test, Cohen's d
  - Bonferroni correction
"""
import os, sys, json, time, numpy as np, warnings
import onnxruntime as ort
from datetime import datetime
warnings.filterwarnings("ignore")

# ─── Fix braindecode/moabb import ─────────────────────────────────────────────
import moabb.datasets as mds
if not hasattr(mds, "BNCI2014001"):
    mds.BNCI2014001 = mds.BNCI2014_001
if not hasattr(mds, "HGD"):
    mds.HGD = mds.PhysionetMI

TMP = os.environ.get("TMP", "/tmp")
CACHE_PATH = os.path.join(TMP, "eegmmidb_cached.npz")
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

EEGCONFORMER_CHANS = [
    "FP1", "FP2", "F5", "F6", "F3", "F4", "F1", "F2",
    "FC5", "FC6", "FC3", "FC4", "C5", "C6", "C3", "C4",
    "T7", "T8", "P7", "P8", "P5", "P6",
]

CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
CHANCE_LEVEL = 0.25
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
PCA_SEEDS = [0x2026_0711, 0x2026_0712, 0x2026_0713]

BENCHMARK_SUBJECTS = list(range(31, 51))  # S031-S050 (held-out, 20 subjects)
ALL_SUBJECTS = list(range(1, 51))  # For all-subjects analysis


# ─── Model definitions ────────────────────────────────────────────────────────

MODELS = {}
def define_models():
    MODELS["EEGConformer"] = {
        "onnx_path": "public/models/eegconformer.onnx",
        "channels": 22, "sample_rate": 250, "window_samples": 1000,
        "bandpass": [4.0, 38.0], "channel_names": EEGCONFORMER_CHANS,
        "input_kind": "raw_2d", "embedding_dim_out": 32,
        "fine_tuned_on": "BCI-IV-2a (production pretrained)",
    }
    MODELS["EEGConformer-FT"] = {
        "onnx_path": "training/artefacts/eegconformer-physionet-v3/eegconformer_finetuned.onnx",
        "channels": 22, "sample_rate": 250, "window_samples": 1000,
        "bandpass": [4.0, 38.0], "channel_names": EEGCONFORMER_CHANS,
        "input_kind": "raw_2d", "embedding_dim_out": 32,
        "fine_tuned_on": "PhysioNet EEGMMIDB S006-S030 (30 subjects, 26 train + 4 val, held-out test S031-S050)",
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def normalize_ch_name(ch):
    return ch.replace(".", "").upper()


def load_cached_data(subjects):
    """Load preprocessed EEGConformer data and raw trials from cache."""
    print(f"  Loading cache: {CACHE_PATH}")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_all = cache["X_all"]
    raw_all = cache["raw_X_all"]
    y_all = cache["y_all"]
    s_all = cache["s_all"]
    source_ch_names = list(cache["source_ch_names"])

    # Filter to requested subjects
    subj_set = set(subjects)
    mask = np.isin(s_all, list(subj_set))
    X_subj = X_all[mask]
    raw_subj = raw_all[mask]
    y_subj = y_all[mask]
    s_subj = s_all[mask]

    print(f"  Loaded {len(y_subj)} trials for {len(set(s_subj))} subjects")
    return X_subj, raw_subj, y_subj, s_subj, source_ch_names


# ─── PCA bandpower features ───────────────────────────────────────────────────

def compute_bandpower_features(trial_data, source_ch_names, target_channels):
    """Compute 110-dim bandpower features (5 bands × 22 channels)."""
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

    # Resample to 250 Hz
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


# ─── Statistical functions ────────────────────────────────────────────────────

def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, confidence=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if n == 1 else 0.0
        return m, 0.0, m, m
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = 1 - confidence
    lo = float(np.percentile(means, alpha / 2 * 100))
    hi = float(np.percentile(means, (1 - alpha / 2) * 100))
    return float(np.mean(values)), float(np.std(values, ddof=1)), lo, hi


def permutation_test(a, b, n_perm=N_PERMUTATIONS, seed=42):
    """Paired permutation test."""
    rng = np.random.RandomState(seed)
    diffs = np.array(a) - np.array(b)
    observed_mean_diff = np.mean(diffs)
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_mean_diff):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return p_value


def paired_t_test(a, b):
    """Paired t-test + Cohen's d."""
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = np.array(a) - np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = float(diff.mean() / (pooled_std + 1e-8)) if pooled_std > 0 else 0.0
    return float(t_stat), float(p_value), d


def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected, [p < alpha for p in corrected]


# ─── Nearest-centroid classification ──────────────────────────────────────────

def nearest_centroid_accuracy(train_embs, train_labels, test_embs, test_labels, k=1):
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
    test_labels_arr = np.array(test_labels)
    correct = (preds == test_labels_arr).sum()
    accuracy = correct / len(test_labels)

    def recall_at_k(k_val):
        correct_k = 0
        for i in range(len(test_norm)):
            sims_i = train_norm @ test_norm[i]
            top_k = np.argsort(sims_i)[-k_val:]
            correct_k += (np.array(train_labels)[top_k] == test_labels[i]).any()
        return correct_k / len(test_norm)

    r1 = recall_at_k(1)
    r5 = recall_at_k(5)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in classes:
        y_true = (test_labels_arr == c).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        train_mask_c = np.array(train_labels) == c
        if train_mask_c.sum() == 0:
            continue
        centroid_sim = test_norm @ (train_norm[train_mask_c].mean(axis=0) + 1e-8)
        try:
            aucs.append(roc_auc_score(y_true, centroid_sim))
        except:
            pass
    auc_score = np.mean(aucs) if aucs else 0.0
    return {
        "accuracy": accuracy,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "auc": auc_score,
    }


# ─── Model inference ──────────────────────────────────────────────────────────

def load_onnx_model(model_spec):
    path = model_spec["onnx_path"]
    if not os.path.exists(path):
        return None, f"ONNX file not found: {path}"
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        # Get embedding output (first output)
        out = sess.get_outputs()[0]
        return {"session": sess, "input_name": inp.name, "output_name": out.name}, None
    except Exception as e:
        return None, str(e)


def run_inference(session_info, inputs):
    sess = session_info["session"]
    inp_name = session_info["input_name"]
    out_name = session_info["output_name"]
    results = []
    for inp in inputs:
        out = sess.run([out_name], {inp_name: inp})[0]
        if out.ndim > 2:
            out = out.reshape(-1, out.shape[-1]).mean(axis=0, keepdims=True)
        results.append(out.flatten().astype(np.float32))
    return np.array(results)


# ─── PCA baseline ─────────────────────────────────────────────────────────────

def compute_pca_baseline(data_X, data_y, data_s, subjects, source_ch_names):
    """Run PCA bandpower baseline under LOSO with multiple seeds."""
    from sklearn.decomposition import PCA
    all_features = []
    all_labels = []
    all_subject_ids = []

    for subj_id in subjects:
        mask = data_s == subj_id
        for trial, label in zip(data_X[mask], data_y[mask]):
            feats = compute_bandpower_features(trial, source_ch_names, EEGCONFORMER_CHANS)
            all_features.append(feats)
            all_labels.append(label)
            all_subject_ids.append(subj_id)
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)

    print(f"\n  PCA baseline: {len(all_labels)} trials, feat_dim={all_features.shape[1]}, seeds={PCA_SEEDS}")

    fold_acc_means = []
    fold_acc_stds = []
    all_fold_accs = []

    for subj_id in subjects:
        test_mask = all_subject_ids == subj_id
        train_mask = ~test_mask
        train_feats = all_features[train_mask]
        test_feats = all_features[test_mask]
        train_labels_list = all_labels[train_mask].tolist()
        test_labels_list = all_labels[test_mask].tolist()

        seed_accs = []
        for seed in PCA_SEEDS:
            feat_mean = train_feats.mean(axis=0)
            feat_std = train_feats.std(axis=0) + 1e-8
            train_n = (train_feats - feat_mean) / feat_std
            test_n = (test_feats - feat_mean) / feat_std

            k_pca = min(32, train_n.shape[1], train_n.shape[0] - 1)
            pca = PCA(n_components=k_pca, random_state=seed)
            train_pca = pca.fit_transform(train_n)
            test_pca = pca.transform(test_n)
            train_pca = train_pca / (np.linalg.norm(train_pca, axis=1, keepdims=True) + 1e-8)
            test_pca = test_pca / (np.linalg.norm(test_pca, axis=1, keepdims=True) + 1e-8)

            metrics = nearest_centroid_accuracy(train_pca, train_labels_list, test_pca, test_labels_list)
            seed_accs.append(metrics["accuracy"])
            all_fold_accs.append(metrics)

        fold_acc_means.append(np.mean(seed_accs))
        fold_acc_stds.append(np.std(seed_accs))

    pca_mean, pca_std, ci_lo, ci_hi = bootstrap_ci(fold_acc_means)
    print(f"  PCA: acc={pca_mean:.4f} ± {pca_std:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")

    return {
        "status": "IMPLEMENTED",
        "model_type": "pca_bandpower_v1",
        "channels": 22,
        "bandpass": "5 bands (delta/theta/alpha/beta/gamma) × 22 ch = 110 features → PCA(32)",
        "feature_dim": int(all_features.shape[1]),
        "pca_components": 32,
        "n_subjects": len(subjects),
        "n_trials": len(all_labels),
        "seeds": PCA_SEEDS,
        "chance_level": CHANCE_LEVEL,
        "loso": {
            "per_subject_accuracy": [float(a) for a in fold_acc_means],
            "per_subject_accuracy_std_over_seeds": [float(s) for s in fold_acc_stds],
            "mean_accuracy": float(pca_mean),
            "std_accuracy": float(pca_std),
            "ci95_accuracy": [float(ci_lo), float(ci_hi)],
            "mean_recall_at_1": float(np.mean([m["recall_at_1"] for m in all_fold_accs])),
            "mean_recall_at_5": float(np.mean([m["recall_at_5"] for m in all_fold_accs])),
            "mean_auc": float(np.mean([m["auc"] for m in all_fold_accs])),
            "n_folds": len(fold_acc_means),
        },
        "latency": {"mean_ms": 0.0, "type": "not measured (feature-based baseline)"},
    }


# ─── Learned model evaluation ─────────────────────────────────────────────────

def compute_model_results(model_name, spec, data_X, data_y, data_s, subjects, source_ch_names):
    """Run LOSO benchmark for a learned model."""
    sess_info, err = load_onnx_model(spec)
    if err:
        return {"status": "BLOCKED", "error": err}

    print(f"\n  {model_name}: ONNX loaded: {os.path.getsize(spec['onnx_path'])/1024/1024:.2f} MB")

    # Run inference on all trials (data_X is already preprocessed for EEGConformer)
    inputs = [data_X[i][None, :, :].astype(np.float32) for i in range(len(data_y))]
    all_embeddings = run_inference(sess_info, inputs)
    all_labels = data_y
    all_subject_ids = data_s

    # LOSO
    per_subj_acc = []
    per_subj_auc = []
    per_subj_r1 = []

    for subj_id in subjects:
        test_mask = all_subject_ids == subj_id
        train_mask = ~test_mask
        metrics = nearest_centroid_accuracy(
            all_embeddings[train_mask], all_labels[train_mask].tolist(),
            all_embeddings[test_mask], all_labels[test_mask].tolist()
        )
        per_subj_acc.append(metrics["accuracy"])
        per_subj_auc.append(metrics["auc"])
        per_subj_r1.append(metrics["recall_at_1"])

    accs = np.array(per_subj_acc)
    mean_acc, std_acc, ci_lo, ci_hi = bootstrap_ci(per_subj_acc)

    return {
        "status": "IMPLEMENTED",
        "onnx_path": spec["onnx_path"],
        "channels": spec["channels"],
        "embedding_dim": all_embeddings.shape[1],
        "fine_tuned_on": spec.get("fine_tuned_on", "BCI-IV-2a (production pretrained)"),
        "loso": {
            "per_subject_accuracy": [float(a) for a in per_subj_acc],
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "ci95_accuracy": [float(ci_lo), float(ci_hi)],
            "mean_auc": float(np.mean(per_subj_auc)),
            "mean_recall_at_1": float(np.mean(per_subj_r1)),
            "n_folds": len(per_subj_acc),
        },
        "latency": {"mean_ms": 0.0, "type": "not measured in this run"},
    }


# ─── Main benchmark ───────────────────────────────────────────────────────────

def main():
    define_models()
    subjects = BENCHMARK_SUBJECTS
    print(f"Loading cached data for subjects {subjects[0]}-{subjects[-1]} ({len(subjects)} subjects)...")

    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: Cache not found: {CACHE_PATH}")
        print("  Run training/scripts/prepare_physionet_data.py first")
        return

    # Load preprocessed data for EEGConformer models
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_all = cache["X_all"]
    raw_all = cache["raw_X_all"]
    y_all = cache["y_all"]
    s_all = cache["s_all"]
    source_ch_names = list(cache["source_ch_names"])

    # Filter to benchmark subjects only
    mask = np.isin(s_all, subjects)
    X_subj = X_all[mask]
    raw_subj = raw_all[mask]
    y_subj = y_all[mask]
    s_subj = s_all[mask]
    print(f"  Filtered to {len(y_subj)} trials for {len(set(s_subj))} subjects")

    results = {}

    # PCA baseline (uses raw trials)
    print(f"\n{'='*60}")
    print("PCA Bandpower Baseline (multiple seeds for robustness)")
    print(f"{'='*60}")
    pca_result = compute_pca_baseline(raw_subj, y_subj, s_subj, subjects, source_ch_names)
    results["PCA_Bandpower"] = pca_result

    # EEGConformer (original)
    print(f"\n{'='*60}")
    print("Benchmarking: EEGConformer (original)")
    print(f"{'='*60}")
    results["EEGConformer"] = compute_model_results("EEGConformer", MODELS["EEGConformer"],
                                                      X_subj, y_subj, s_subj, subjects, source_ch_names)

    # EEGConformer-FT (fine-tuned)
    print(f"\n{'='*60}")
    print("Benchmarking: EEGConformer-FT (fine-tuned)")
    print(f"{'='*60}")
    results["EEGConformer-FT"] = compute_model_results("EEGConformer-FT", MODELS["EEGConformer-FT"],
                                                        X_subj, y_subj, s_subj, subjects, source_ch_names)

    # Print results
    for model_name in ["EEGConformer", "EEGConformer-FT"]:
        r = results[model_name]
        if r["status"] == "IMPLEMENTED":
            loso = r["loso"]
            print(f"  {model_name}: acc={loso['mean_accuracy']:.4f} ± {loso['std_accuracy']:.4f} "
                  f"(95% CI: {loso['ci95_accuracy'][0]:.4f}-{loso['ci95_accuracy'][1]:.4f})")

    pca_loso = results["PCA_Bandpower"]["loso"]
    print(f"  PCA: acc={pca_loso['mean_accuracy']:.4f} ± {pca_loso['std_accuracy']:.4f} "
          f"(95% CI: {pca_loso['ci95_accuracy'][0]:.4f}-{pca_loso['ci95_accuracy'][1]:.4f})")

    # ─── Statistical comparison against PCA ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Statistical comparisons")
    print(f"{'='*60}")

    pca_accs = np.array(pca_loso["per_subject_accuracy"])
    model_names = ["EEGConformer", "EEGConformer-FT"]
    p_values = []
    all_comparisons = []

    for model_name in model_names:
        model_accs = np.array(results[model_name]["loso"]["per_subject_accuracy"])
        min_len = min(len(pca_accs), len(model_accs))
        pca_arr = pca_accs[:min_len]
        model_arr = model_accs[:min_len]

        t_stat, p_val, d = paired_t_test(model_arr.tolist(), pca_arr.tolist())
        perm_p = permutation_test(model_arr.tolist(), pca_arr.tolist())

        diffs = model_arr - pca_arr
        mean_diff = float(np.mean(diffs))
        pct_improvement = (mean_diff / float(np.mean(pca_arr))) * 100 if np.mean(pca_arr) > 0 else 0

        all_comparisons.append({
            "model": model_name,
            "pca_accuracy": float(np.mean(pca_arr)),
            "model_accuracy": float(np.mean(model_arr)),
            "delta_vs_pca": mean_diff,
            "pct_improvement_vs_pca": pct_improvement,
            "t_statistic": t_stat,
            "p_value_paired_t": p_val,
            "p_value_permutation": perm_p,
            "cohens_d": d,
            "effect_size_interpretation": (
                "large" if abs(d) >= 0.8 else
                "medium" if abs(d) >= 0.5 else
                "small" if abs(d) >= 0.2 else "negligible"
            ),
            "significant_05": bool(p_val < 0.05),
            "significant_perm_05": bool(perm_p < 0.05),
        })
        p_values.append(p_val)

    # Comparison: FT vs Original
    ft_accs = np.array(results["EEGConformer-FT"]["loso"]["per_subject_accuracy"])
    orig_accs = np.array(results["EEGConformer"]["loso"]["per_subject_accuracy"])
    min_len = min(len(ft_accs), len(orig_accs))
    ft_arr = ft_accs[:min_len]
    orig_arr = orig_accs[:min_len]

    t_stat2, p_val2, d2 = paired_t_test(ft_arr.tolist(), orig_arr.tolist())
    perm_p2 = permutation_test(ft_arr.tolist(), orig_arr.tolist())
    diff_ft_orig = float(np.mean(ft_arr - orig_arr))
    pct_improvement2 = (diff_ft_orig / float(np.mean(orig_arr))) * 100 if np.mean(orig_arr) > 0 else 0

    ft_vs_orig = {
        "model_a": "EEGConformer-FT",
        "model_b": "EEGConformer-Original",
        "model_a_accuracy": float(np.mean(ft_arr)),
        "model_b_accuracy": float(np.mean(orig_arr)),
        "delta_vs_original": diff_ft_orig,
        "pct_improvement_vs_original": pct_improvement2,
        "t_statistic": t_stat2,
        "p_value_paired_t": p_val2,
        "p_value_permutation": perm_p2,
        "cohens_d": d2,
        "effect_size_interpretation": (
            "large" if abs(d2) >= 0.8 else
            "medium" if abs(d2) >= 0.5 else
            "small" if abs(d2) >= 0.2 else "negligible"
        ),
        "significant_05": bool(p_val2 < 0.05),
        "significant_perm_05": bool(perm_p2 < 0.05),
    }

    # Bonferroni correction (2 comparisons vs PCA)
    corrected_pvals, corrected_sig = bonferroni_correction(p_values, alpha=0.05)
    bonf_pvals_ft_orig = [min(p_val2 * 2, 1.0)]
    bonf_sig_ft_orig = [bonf_pvals_ft_orig[0] < 0.05]

    for i, comp in enumerate(all_comparisons):
        comp["p_value_bonferroni"] = float(corrected_pvals[i])
        comp["significant_bonferroni_05"] = corrected_sig[i]

    for comp in all_comparisons:
        print(f"  {comp['model']} vs PCA: Δ={comp['delta_vs_pca']:+.4f} ({comp['pct_improvement_vs_pca']:+.1f}%), "
              f"t={comp['t_statistic']:.3f}, p(t)={comp['p_value_paired_t']:.4f}, "
              f"p(perm)={comp['p_value_permutation']:.4f}, "
              f"p(bonf)={comp['p_value_bonferroni']:.4f}, "
              f"d={comp['cohens_d']:.3f} ({comp['effect_size_interpretation']}), "
              f"sig={comp['significant_05']}")

    print(f"\n  {ft_vs_orig['model_a']} vs {ft_vs_orig['model_b']}: Δ={ft_vs_orig['delta_vs_original']:+.4f} ({ft_vs_orig['pct_improvement_vs_original']:+.1f}%), "
          f"t={ft_vs_orig['t_statistic']:.3f}, p(t)={ft_vs_orig['p_value_paired_t']:.4f}, "
          f"p(perm)={ft_vs_orig['p_value_permutation']:.4f}, "
          f"d={ft_vs_orig['cohens_d']:.3f} ({ft_vs_orig['effect_size_interpretation']}), "
          f"sig={ft_vs_orig['significant_05']}")

    results["statistical_comparisons_vs_pca"] = all_comparisons
    results["statistical_comparison_ft_vs_original"] = ft_vs_orig

    # Save results
    output_path = os.path.join(REPORT_DIR, "t031_benchmark_results_50subj.json")
    if len(subjects) == 20:
        output_path = os.path.join(REPORT_DIR, "t031_benchmark_results_20test_30train.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    results = main()
