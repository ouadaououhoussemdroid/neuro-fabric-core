#!/usr/bin/env python3
"""
T-034: Representation Learning Experiment — Evaluation Script

Evaluates T-034 trained models using the exact same corrected protocol as T-032/T-033:
- LOSO cross-validation on 50 subjects (S001-S050)
- Train-only candidate pools for retrieval (no self-retrieval)
- Per-fold PCA refit (no leakage)
- Corrected label mapping (Run 5 T1=left/0 T2=right/1, Run 6 T1=feet/2 T2=tongue/3)
- 4-38 Hz bandpass, 22 channels, 250 Hz, 1000 samples

Imports the corrected evaluation functions from scripts/t033-embedding-dimension-ablation.py
via importlib to guarantee protocol consistency. Does NOT create a second methodology.

Models evaluated:
    1. PCA bandpower baseline
    2. EEGConformer v1 (public/models/eegconformer.onnx)
    3. EEGConformer v2 (production, public/models/eegconformer_finetuned.onnx)
    4. T-034 baseline (CE + label_smoothing)
    5. T-034 aug (CE + label_smoothing + augmentation)
    6. T-034 contrastive (CE + SupCon)
    7. T-034 aug_contrastive (CE + augmentation + SupCon)
    (+ optional λ-ablation configs if provided)

Usage:
    python scripts/t034-evaluate.py \\
        --t034-dir training/artefacts/eegconformer-t034-baseline \\
        --t034-dir training/artefacts/eegconformer-t034-aug \\
        --t034-dir training/artefacts/eegconformer-t034-contrastive \\
        --t034-dir training/artefacts/eegconformer-t034-aug_contrastive
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

# ─── Import corrected T-033 evaluation functions ─────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
T033_SCRIPT = REPO_ROOT / "scripts" / "t033-embedding-dimension-ablation.py"

def _load_t033():
    """Load the T-033 module via importlib (handles hyphenated filename)."""
    spec = importlib.util.spec_from_file_location("t033", str(T033_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_T033 = None

def t033():
    global _T033
    if _T033 is None:
        _T033 = _load_t033()
    return _T033


# ─── Model configuration ────────────────────────────────────────────────────

V1_ONNX = str(REPO_ROOT / "public" / "models" / "eegconformer.onnx")
V2_ONNX = str(REPO_ROOT / "public" / "models" / "eegconformer_finetuned.onnx")
V3_ONNX = str(REPO_ROOT / "training" / "artefacts" / "eegconformer-physionet-v3" / "eegconformer_finetuned.onnx")


def discover_t034_models(t034_dirs):
    """Discover T-034 ONNX models from output directories.

    Each directory should contain eegconformer.onnx (the exported model).
    """
    models = {}
    for d in t034_dirs:
        d = Path(d)
        onnx_path = d / "eegconformer.onnx"
        if onnx_path.exists():
            config_name = d.name  # e.g. "eegconformer-t034-baseline"
            # Extract the experiment name: "eegconformer-t034-baseline" -> "t034_baseline"
            exp_name = config_name.replace("eegconformer-t034-", "")
            model_key = f"t034_{exp_name}"
            models[model_key] = {
                "onnx_path": str(onnx_path),
                "description": f"T-034 {exp_name.replace('_', ' ')}",
                "embedding_dim": 32,
                "t034_config": exp_name,
            }
            # Also load train history for collapse monitoring data
            hist_path = d / "train_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    hist = json.load(f)
                models[model_key]["train_history"] = hist
        else:
            print(f"  WARNING: {onnx_path} not found, skipping")
    return models


# ─── Statistical comparisons ───────────────────────────────────────────────

def paired_t_test(a, b):
    """Paired t-test returning t-statistic, p-value, and Cohen's d."""
    a = np.array(a)
    b = np.array(b)
    t_stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    cohen_d = diff.mean() / (diff.std() + 1e-9)
    return float(t_stat), float(p_value), float(cohen_d)


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction. Returns adjusted alpha and significance."""
    n = len(p_values)
    adjusted_alpha = alpha / n
    return adjusted_alpha, n


def effect_size_label(d):
    """Label Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ─── Main evaluation ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t034-dir", type=str, action="append", default=[],
                    help="T-034 training output directories (each containing eegconformer.onnx)")
    ap.add_argument("--data-dir", default=None,
                    help="Path to PhysioNet EEGMMIDB data")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--skip-v1-v3", action="store_true",
                    help="Skip v1 and v3 (faster, if only PCA/v2/T-034 needed)")
    args = ap.parse_args()

    t = t033()

    data_dir = args.data_dir or os.path.join(os.environ.get("TMP", "/tmp"), "eegmmidb")
    report_dir = Path(args.report_dir) if args.report_dir else REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("T-034: Representation Learning Evaluation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Data: {data_dir}")
    print(f"Subjects: S001-S050 (50 subjects, LOSO)")
    print(f"Protocol: T-032/T-033 corrected (train-only pools, per-fold PCA, no leakage)")

    # ─── Step 1: Load and preprocess data ────────────────────────────
    print("\nStep 1: Loading PhysioNet EEGMMIDB data...")
    subjects_data = t.load_physionet_subjects(list(range(1, 51)), runs=[5, 6])
    print(f"Loaded {len(subjects_data)} subjects")

    all_windows = []
    all_labels = []
    all_subject_ids = []
    for subj_id in sorted(subjects_data.keys()):
        sd = subjects_data[subj_id]
        for i, trial in enumerate(sd["trials"]):
            win = t.preprocess_for_eegconformer(trial, sd["ch_names"])
            all_windows.append(win)
            all_labels.append(sd["labels"][i])
            all_subject_ids.append(subj_id)

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    print(f"Total trials: {len(all_windows)}")
    print(f"Label distribution: {np.bincount(all_labels, minlength=4)}")

    # ─── Step 2: Extract embeddings from all models ─────────────────────
    print("\nStep 2: Extracting embeddings from models...")

    model_results = {}

    # PCA bandpower baseline
    pca_features = np.array([t.bandpower_features(w) for w in all_windows])
    model_results["pca_bandpower"] = {
        "description": "PCA bandpower baseline (110 features → PCA(32))",
        "embedding_dim": 32,
        "type": "pca_bandpower",
        "features": pca_features,
    }
    print(f"  PCA bandpower: {pca_features.shape}")

    # v1 (if available)
    if os.path.exists(V1_ONNX) and not args.skip_v1_v3:
        v1_embs, _ = t.batched_onnx_inference(V1_ONNX, all_windows, intermediate_tensor=None)
        model_results["eegconformer_v1"] = {
            "description": "EEGConformer v1 (original)",
            "embedding_dim": 32,
            "type": "onnx",
            "onnx_path": V1_ONNX,
            "embeddings": v1_embs,
        }
        print(f"  v1 32-D: {v1_embs.shape}")

    # v2 (production baseline)
    if os.path.exists(V2_ONNX):
        v2_embs, _ = t.batched_onnx_inference(V2_ONNX, all_windows, intermediate_tensor=None)
        model_results["eegconformer_v2"] = {
            "description": "EEGConformer v2 (production baseline)",
            "embedding_dim": 32,
            "type": "onnx",
            "onnx_path": V2_ONNX,
            "embeddings": v2_embs,
        }
        print(f"  v2 32-D: {v2_embs.shape}")
    else:
        print(f"  WARNING: v2 ONNX not found at {V2_ONNX}")

    # v3 (if available)
    if os.path.exists(V3_ONNX) and not args.skip_v1_v3:
        v3_embs, _ = t.batched_onnx_inference(V3_ONNX, all_windows, intermediate_tensor=None)
        model_results["eegconformer_v3"] = {
            "description": "EEGConformer v3",
            "embedding_dim": 32,
            "type": "onnx",
            "onnx_path": V3_ONNX,
            "embeddings": v3_embs,
        }
        print(f"  v3 32-D: {v3_embs.shape}")

    # T-034 models
    t034_models = discover_t034_models(args.t034_dir)
    for model_key, model_info in t034_models.items():
        embs, _ = t.batched_onnx_inference(
            model_info["onnx_path"], all_windows, intermediate_tensor=None
        )
        model_info["embeddings"] = embs
        model_info["type"] = "onnx"
        model_info["embedding_dim"] = 32
        model_results[model_key] = model_info
        print(f"  {model_key} 32-D: {embs.shape}")

    # ─── Step 3: Run LOSO evaluation on each model ────────────────────────
    print("\nStep 3: Running LOSO cross-validation (50 folds)...")

    results = {
        "experiment_id": "T-034",
        "experiment_name": "EEGConformer Representation Learning Experiment",
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Evaluates whether better training objectives (EEG augmentation + supervised "
            "contrastive loss) produce a better 32-D EEGConformer embedding. Uses the "
            "exact same corrected protocol as T-032/T-033."
        ),
        "data": {
            "dataset": "PhysioNet EEGMMIDB",
            "subjects": "S001-S050 (50 subjects)",
            "n_subjects": 50,
            "n_trials": len(all_labels),
            "label_distribution": np.bincount(all_labels, minlength=4).tolist(),
            "class_names": t.CLASS_NAMES,
        },
        "protocol": "LOSO (50 folds), train-only candidate pools, per-fold PCA refit, no leakage",
        "preprocessing": "160→250 Hz resample, 22-ch BCI-IV-2a subset, bandpass 4-38 Hz, z-score per channel, 1000 samples (4s)",
        "models": {},
        "results": {},
        "statistical_comparisons": [],
        "train_history_comparison": {},
    }

    per_fold_results = {}  # for paired tests

    for model_name in ["pca_bandpower", "eegconformer_v1", "eegconformer_v2", "eegconformer_v3"] + \
                       [k for k in t034_models.keys()]:
        if model_name not in model_results:
            continue

        info = model_results[model_name]
        print(f"\n  Evaluating {model_name}...")

        if info["type"] == "pca_bandpower":
            # PCA bandpower: use LOSO with per-fold PCA refit
            loso = t.run_loso(
                info["features"], all_labels, all_subject_ids,
                needs_pca=True, pca_dim=32, is_features=True
            )
        else:
            loso = t.run_loso(
                info["embeddings"], all_labels, all_subject_ids,
                needs_pca=False
            )

        # Class separability (on full dataset)
        cs = t.class_separability(info.get("embeddings") if info.get("embeddings") is not None else info["features"], all_labels)

        # Embedding richness
        er = t.embedding_richness(info.get("embeddings") if info.get("embeddings") is not None else info["features"])

        # Stability (only for ONNX models with onnx_path, not PCA)
        if info["type"] == "onnx" and info.get("onnx_path"):
            import onnxruntime as ort
            sess = ort.InferenceSession(info["onnx_path"], providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            session_info = {"session": sess, "input_name": inp.name, "output_name": "embedding"}
            np.random.seed(42)
            test_indices = np.random.choice(len(all_windows), 15, replace=False)
            stability_results = []
            for idx in test_indices:
                w = all_windows[idx]
                s = t.embedding_stability_onnx(session_info, w)
                stability_results.append(s)
            stability = {
                "n_windows": 15,
                "determinism": {"mean_max_pairwise_cosine": float(np.mean([s["determinism"]["max_pairwise_cosine"] for s in stability_results]))},
                "amplitude_scaling": {"mean_cosine": float(np.mean([s["amplitude_scaling"]["mean_cosine"] for s in stability_results]))},
                "noise_robustness": {"cosine_similarity": float(np.mean([s["noise_robustness"]["cosine_similarity"] for s in stability_results]))},
                "window_boundary_shift": {"mean_cosine": float(np.nanmean([s["window_boundary_shift"]["mean_cosine"] for s in stability_results]))},
            }
        else:
            stability = {"note": "stability not computed for PCA bandpower"}

        model_results[model_name]["loso"] = loso
        model_results[model_name]["class_separability"] = cs
        model_results[model_name]["embedding_richness"] = er
        model_results[model_name]["embedding_stability"] = stability

        results["results"][model_name] = {
            "loso": loso,
            "class_separability": cs,
            "embedding_richness": er,
            "embedding_stability": stability,
        }

        per_fold_results[model_name] = {
            "accuracy": np.array(loso["loso"]["per_fold_accuracy"]),
            "f1": np.array(loso["loso"]["per_fold_f1"]),
            "r1": np.array(loso["loso"]["per_fold_r1"]),
        }

        # Store train history comparison for T-034 models
        if "train_history" in info:
            hist = info["train_history"]
            results["train_history_comparison"][model_name] = {
                "best_epoch": hist.get("best_epoch"),
                "best_val_loss": hist.get("best_val_loss"),
                "best_val_acc": hist.get("best_val_acc"),
                "best_test_acc": hist.get("best_test_acc"),
                "config": hist.get("config", {}),
                "best_emb_stats": hist.get("best_emb_stats", {}),
            }

        # Print summary
        r = loso["loso"]
        print(f"    Acc={r['mean_accuracy']:.4f}±{r['std_accuracy']:.4f}  "
              f"R@1={r['recall_at_1']['mean']:.4f}  "
              f"R@10={r['recall_at_10']['mean']:.4f}  "
              f"Fisher={cs['fisher_score']:.4f}  "
              f"Intra={cs['intra_class_cosine_mean']:.4f}  "
              f"Inter={cs['inter_class_cosine_mean']:.4f}")

    # ─── Step 4: Statistical comparisons ────────────────────────────────
    print("\nStep 4: Statistical comparisons (v2 as baseline)...")

    baseline = "eegconformer_v2"
    n_comparisons = 0
    comparison_pairs = []

    for model_name in per_fold_results:
        if model_name == baseline:
            continue
        # Accuracy comparison
        t_stat, p_val, cohen_d = paired_t_test(
            per_fold_results[baseline]["accuracy"],
            per_fold_results[model_name]["accuracy"]
        )
        comparison_pairs.append({
            "comparison": f"{baseline} vs {model_name}",
            "metric": "loso_accuracy",
            "mean_baseline": float(per_fold_results[baseline]["accuracy"].mean()),
            "mean_other": float(per_fold_results[model_name]["accuracy"].mean()),
            "delta": float(per_fold_results[model_name]["accuracy"].mean() -
                          per_fold_results[baseline]["accuracy"].mean()),
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohen_d,
            "effect_size": effect_size_label(cohen_d),
        })
        n_comparisons += 1

        # Recall@1 comparison
        t_stat_r1, p_val_r1, cohen_d_r1 = paired_t_test(
            per_fold_results[baseline]["r1"],
            per_fold_results[model_name]["r1"]
        )
        comparison_pairs.append({
            "comparison": f"{baseline} vs {model_name}",
            "metric": "recall_at_1",
            "mean_baseline": float(per_fold_results[baseline]["r1"].mean()),
            "mean_other": float(per_fold_results[model_name]["r1"].mean()),
            "delta": float(per_fold_results[model_name]["r1"].mean() -
                          per_fold_results[baseline]["r1"].mean()),
            "t_statistic": t_stat_r1,
            "p_value": p_val_r1,
            "cohens_d": cohen_d_r1,
            "effect_size": effect_size_label(cohen_d_r1),
        })
        n_comparisons += 1

    adjusted_alpha, n_tests = bonferroni_correction(
        [c["p_value"] for c in comparison_pairs]
    )

    for comp in comparison_pairs:
        comp["bonferroni_significant"] = comp["p_value"] < adjusted_alpha
        comp["adjusted_alpha"] = adjusted_alpha

    results["statistical_comparisons"] = comparison_pairs
    results["bonferroni_correction"] = {
        "n_tests": n_tests,
        "adjusted_alpha": adjusted_alpha,
        "description": f"Bonferroni correction across {n_tests} pairwise comparisons",
    }

    print(f"\n  {n_tests} comparisons, adjusted α = {adjusted_alpha:.6f}")
    for comp in comparison_pairs:
        sig = "✅" if comp["bonferroni_significant"] else "❌"
        print(f"  {comp['comparison']} ({comp['metric']}): "
              f"Δ={comp['delta']:+.4f}, p={comp['p_value']:.4e}, "
              f"d={comp['cohens_d']:.3f} ({comp['effect_size']}) {sig}")

    # ─── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Embedding quality comparison")
    print("=" * 70)
    print(f"{'Model':<28} {'Acc':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'Fisher':>8} {'Intra':>8} {'Inter':>8}")
    print("-" * 70)

    for model_name in ["pca_bandpower", "eegconformer_v1", "eegconformer_v2", "eegconformer_v3"] + \
                       [k for k in sorted(t034_models.keys())]:
        if model_name not in model_results:
            continue
        if "loso" not in model_results[model_name]:
            continue
        r = model_results[model_name]["loso"]["loso"]
        cs = model_results[model_name]["class_separability"]
        print(f"{model_name:<28} {r['mean_accuracy']:>10.4f} {r['recall_at_1']['mean']:>8.4f} "
              f"{r['recall_at_5']['mean']:>8.4f} {r['recall_at_10']['mean']:>8.4f} "
              f"{cs['fisher_score']:>8.4f} {cs['intra_class_cosine_mean']:>8.4f} "
              f"{cs['inter_class_cosine_mean']:>8.4f}")

    # ─── Training history comparison ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training history comparison (T-034 models)")
    print("=" * 70)
    print(f"{'Model':<28} {'Exp':<20} {'BestEpoch':>10} {'BestValAcc':>12} {'EffRank':>8} {'Intra':>8} {'Inter':>8}")
    print("-" * 70)

    for model_name in sorted(t034_models.keys()):
        hist = results["train_history_comparison"].get(model_name)
        if hist:
            emb_stats = hist.get("best_emb_stats", {})
            print(f"{model_name:<28} {hist.get('config', {}).get('experiment', ''):<20} "
                  f"{hist.get('best_epoch', '?'):>10} {hist.get('best_val_acc', 0):>12.4f} "
                  f"{emb_stats.get('effective_rank', '?'):>8.2f} "
                  f"{emb_stats.get('intra_class_cosine', '?'):>8.3f} "
                  f"{emb_stats.get('inter_class_cosine', '?'):>8.3f}")

    # ─── Save results ───────────────────────────────────────────────────
    output_path = report_dir / "t034_representation_learning_results.json"
    # Build serializable results (exclude raw embeddings)
    for model_name in results["results"]:
        if "embeddings" in model_results.get(model_name, {}):
            del model_results[model_name]["embeddings"]
        if "features" in model_results.get(model_name, {}):
            del model_results[model_name]["features"]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
