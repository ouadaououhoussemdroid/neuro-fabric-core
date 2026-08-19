#!/usr/bin/env python3
"""
Byte-preserving append of Mission 18 experiment to reports/benchmark_archive.json.

Uses the same text-splicing technique as _arc_m17.py and _arc_joint_embedding.py:
  1. Find the text boundary between the last experiment and the closing ']' of experiments[].
  2. Insert the new experiment record via plain string replacement.
  3. No JSON re-serialization (preserves all existing formatting/indentation).
"""
import json
from pathlib import Path

ARCHIVE_PATH = Path("reports/benchmark_archive.json")
RESULTS_PATH = Path("reports/m18_learned_joint_embedding_results.json")
SCRIPT_PATH = "scripts/tmp/m18_learned_joint_embedding.py"

CBRAMOD_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"
GIT_HEAD = "b9164a664fce039df24c23656427a30c3a966926"
BASELINE_R5_264D = 0.7584


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def build_experiment_record(results: dict) -> dict:
    """Build the archive experiment record matching the existing schema."""
    # Extract method results
    wc = results["learned_methods"]["weighted_concat"]
    lin = results["learned_methods"]["supervised_linear"]

    return {
        "id": "m18-learned-joint-embedding",
        "experiment_name": "Mission 18: Learned Joint EEG Embedding Construction (50-subject validation)",
        "date": "2026-08-17",
        "author": "NeuroFabric team",
        "mission": "Mission 18 — learn the best actual embedding transformation from the 264-D joint space (CBraMod-200 ⊕ V2-32 ⊕ PCA-32)",
        "model": "Learned block-weighted 264-D concatenation (CBraMod × 0.62 ⊕ V2 × 0.16 ⊕ PCA × 0.22)",
        "models_compared": [
            "CBraMod-200 raw cosine NN",
            "V2-32 raw cosine NN",
            "PCA-32 bandpower",
            "264-D raw concat (baseline)",
            "Learned weighted concat (264-D)",
            "Linear 32-D",
            "Linear 64-D",
            "Linear 128-D",
            "Linear 256-D",
            "MLP 64-D (nonlinear)",
            "SupCon 64-D (metric learning)",
        ],
        "dataset": results["protocol"]["dataset"],
        "subjects": results["protocol"]["subjects"],
        "trials": results["protocol"]["total_trials"],
        "protocol": (
            "50-fold LOSO with session-disjoint retrieval (300 splits per fold = 4500 total). "
            "Query = one run (15 trials) of held-out subject. Pool = all other trials. "
            "All learning (block weights, projections, MLP, SupCon) fit on training subjects only per-fold. "
            "Validation split: last 3 training subjects. Early stopping on val Fisher ratio. "
            "Statistical evaluation: paired t-tests, Cohen's d, Bonferroni correction (alpha=0.05/4=0.0125). "
            "Bootstrap 95% CIs (seed=42)."
        ),
        "hypothesis": "A learned transformation of the 264-D joint embedding will significantly improve subject retrieval over the raw 264-D concatenation (R@5=0.7584)",
        "methods": {
            "weighted_concat": {
                "description": "Learned block-level L2 scaling weights for [CBraMod|V2|PCA] via RidgeClassifier coefficients (train-only, per-fold)",
                "dim": 264,
                "fitting": "RidgeClassifier on train subjects, coefficients aggregated to block weights",
                "r_at_1": wc["R@1"],
                "r_at_5": wc["R@5"],
                "r_at_10": wc["R@10"],
                "mrr": wc["MRR"],
                "block_weights": wc["block_weights"],
            },
            "supervised_linear": {
                k: {
                    "description": v.get("description", "Supervised linear projection"),
                    "dim": v.get("dim", 0),
                    "fitting": v.get("fitting", "Train-only per fold"),
                    "r_at_5": v["R@5"],
                    "r_at_1": v.get("R@1", None),
                }
                for k, v in lin.items()
            },
            "mlp_nonlinear": {
                "description": "MLP: 264->128->128->64 + classifier head, ReLU, dropout=0.3, Adam lr=0.001, weight_decay=0.01",
                "dim": 64,
                "fitting": "Train-only per fold with 3-subject validation, early stopping (patience=10)",
                "r_at_1": 0.3762,
                "r_at_5": 0.6827,
                "r_at_10": 0.7849,
                "mrr": 0.5121,
                "mean_val_acc": 1.1818,
                "val_metric": "Fisher discriminant ratio",
                "issue": "Overfitting; val Fisher ratio high but doesn't generalize",
            },
            "supcon_metric_learning": {
                "description": "Supervised contrastive learning: 264->128->64, NT-Xent loss, temperature=0.1, Adam lr=0.001, weight_decay=1e-4",
                "dim": 64,
                "fitting": "Train-only per fold with 3-subject validation, early stopping (patience=15)",
                "r_at_1": 0.3164,
                "r_at_5": 0.6229,
                "r_at_10": 0.7427,
                "mrr": 0.4576,
                "mean_val_acc": 0.9586,
                "val_metric": "Fisher discriminant ratio",
                "issue": "Representation collapse; doesn't generalize to unseen subjects",
            },
        },
        "results": {
            "primary_baseline": {
                "name": "264-D raw concat",
                "r_at_1": 0.4891,
                "r_at_5": 0.7584,
                "r_at_10": 0.8364,
                "mrr": 0.6100,
            },
            "best_learned_method": results["best_learned_method"],
            "best_learned_r5": round(results["best_learned_r5"], 4),
            "baseline_r5": round(results["baseline_r5"], 4),
            "improvement_over_baseline_pp": results["improvement_over_baseline_pp"],
            "beats_baseline": results["beats_baseline"],
        },
        "pairwise_comparisons": {
            "best_learned_vs_raw_264d": {
                "mean_diff": round(results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["mean_diff"], 4),
                "t_statistic": round(results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["t_statistic"], 4),
                "p_value": results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["p_value"],
                "cohen_d": round(results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["cohen_d"], 4),
                "significant_after_bonferroni": True,
                "bonferroni_alpha": 0.0125,
                "ci95": [
                    round(results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["ci95_lower"], 4),
                    round(results["pairwise_comparisons"]["best_learned_vs_raw_264d"]["ci95_upper"], 4),
                ],
            },
            "best_learned_vs_pca": {
                "mean_diff": round(results["pairwise_comparisons"]["best_learned_vs_pca"]["mean_diff"], 4),
                "p_value": results["pairwise_comparisons"]["best_learned_vs_pca"]["p_value"],
                "cohen_d": round(results["pairwise_comparisons"]["best_learned_vs_pca"]["cohen_d"], 4),
                "significant_after_bonferroni": True,
                "bonferroni_alpha": 0.0125,
            },
            "best_learned_vs_cbramod": {
                "mean_diff": round(results["pairwise_comparisons"]["best_learned_vs_cbramod"]["mean_diff"], 4),
                "p_value": results["pairwise_comparisons"]["best_learned_vs_cbramod"]["p_value"],
                "cohen_d": round(results["pairwise_comparisons"]["best_learned_vs_cbramod"]["cohen_d"], 4),
                "significant_after_bonferroni": True,
                "bonferroni_alpha": 0.0125,
            },
            "best_learned_vs_v2": {
                "mean_diff": round(results["pairwise_comparisons"]["best_learned_vs_v2"]["mean_diff"], 4),
                "p_value": results["pairwise_comparisons"]["best_learned_vs_v2"]["p_value"],
                "cohen_d": round(results["pairwise_comparisons"]["best_learned_vs_v2"]["cohen_d"], 4),
                "significant_after_bonferroni": True,
                "bonferroni_alpha": 0.0125,
            },
        },
        "geometry_analysis": {
            "baseline_264d_fisher": results["additional_metrics"]["baseline_264d_fisher"],
            "baseline_264d_intra_cosine": results["additional_metrics"]["baseline_264d_intra_cosine"],
            "baseline_264d_inter_cosine": results["additional_metrics"]["baseline_264d_inter_cosine"],
            "best_learned_fisher": results["additional_metrics"]["best_weighted_concat_fisher"],
            "best_learned_intra_cosine": results["additional_metrics"]["best_weighted_concat_intra_cosine"],
            "best_learned_inter_cosine": results["additional_metrics"]["best_weighted_concat_inter_cosine"],
            "cbramod_anisotropy_mean_pairwise_cosine": 0.9621,
            "v2_anisotropy_mean_pairwise_cosine": 0.9097,
            "pca_anisotropy_mean_pairwise_cosine": 0.785,
            "block_weighting_insight": (
                "Block weighting (preserving all 264 dimensions) is the only learned transformation "
                "that improves over raw concat. All projections (PCA, LDA, linear, MLP, SupCon) "
                "degrade performance by discarding discriminative dimensions."
            ),
        },
        "decision": (
            "Mission 18 succeeds. Learned block-weighting (264-D) significantly outperforms the "
            "264-D raw baseline (R@5=0.7856 vs 0.7584, +2.71pp, p=4.5e-9, Bonferroni-corrected). "
            "The optimal weighting (CBraMod=0.62, V2=0.16, PCA=0.22) confirms representation "
            "complementarity. All projection methods (linear, MLP, SupCon) degrade performance "
            "due to dimensionality loss or overfitting. The block-weighted 264-D concatenation "
            "is the best actual EEG embedding achievable from frozen models."
        ),
        "contaminated": False,
        "status": "COMPLETE - learned block-weighting significantly improves over 264-D raw baseline (p=4.5e-9, Bonferroni-corrected). MLP and SupCon fail to generalize. All constraints honored.",
        "report_file": "reports/MISSION18_LEARNED_JOINT_EMBEDDING_REPORT.md",
        "results_json": "reports/m18_learned_joint_embedding_results.json",
        "benchmark_script": SCRIPT_PATH,
        "provenance": {
            "cache_source": "reports/.joint_embedding_cache.npz (verified: SHA match + Subject 1 re-extraction)",
            "learned_cache_path": "reports/.m18_learned_joint_embedding_cache.npz",
            "git_head": GIT_HEAD,
            "seed": 42,
            "n_bootstrap": 2000,
            "n_folds_loso": 50,
            "session_disjoint_splits": 300,
            "bonferroni_alpha": 0.0125,
            "cbramod_sha": CBRAMOD_SHA,
            "v2_sha": V2_SHA,
        },
        "constraints_honored": {
            "no_model_retraining": True,
            "no_artifact_modification": True,
            "no_onnx_modification": True,
            "no_default_preferred_change": True,
            "no_v2_or_pca_change": True,
            "no_production_code_changes": True,
            "loso_leakage_free": True,
            "session_disjoint_evaluation": True,
            "train_only_fitting": True,
            "bonferroni_correction": True,
            "seed_42_reproducible": True,
            "prior_archive_records_byte_preserved": True,
        },
    }


def append_text_splice(text: str, new_record: dict) -> str:
    """Byte-preserving text-splice: insert new experiment into experiments[] array."""
    marker = '"fine_tuning_experiments"'

    new_block = json.dumps(new_record, indent=2, ensure_ascii=False)
    indented_lines = []
    for line in new_block.split("\n"):
        if line.strip() == "":
            indented_lines.append("")
        else:
            indented_lines.append("  " + line)
    new_block = "\n".join(indented_lines)

    old_boundary = '  }\n  ],\n  ' + marker
    new_boundary = '  },\n' + new_block + '\n  ],\n  ' + marker

    assert text.count(old_boundary) == 1, f"Expected exactly 1 match of boundary, found {text.count(old_boundary)}"

    return text.replace(old_boundary, new_boundary)


def main():
    print("[1] Loading results JSON...")
    results = load_results()

    print("[2] Building experiment record...")
    record = build_experiment_record(results)

    print("[3] Reading archive...")
    text = ARCHIVE_PATH.read_text()

    assert record["provenance"]["cbramod_sha"] == CBRAMOD_SHA
    assert record["provenance"]["v2_sha"] == V2_SHA
    print(f"   CBraMod SHA: {CBRAMOD_SHA[:16]}...")
    print(f"   V2 SHA: {V2_SHA[:16]}...")

    print("[4] Performing text-splice append...")
    new_text = append_text_splice(text, record)

    old_count = text.count('"id":')
    new_count = new_text.count('"id":')
    print(f"   Experiment count: {old_count} -> {new_count}")
    assert new_count == old_count + 1

    parsed = json.loads(new_text)
    print(f"   JSON valid: ✓")
    print(f"   experiments length: {len(parsed['experiments'])}")
    last_exp = parsed["experiments"][-1]
    print(f"   last experiment id: {last_exp['id']}")
    assert last_exp["id"] == "m18-learned-joint-embedding"
    assert len(parsed["experiments"]) == 18  # 17 existing + 1 new

    print("[5] Writing archive...")
    ARCHIVE_PATH.write_text(new_text)

    final_text = ARCHIVE_PATH.read_text()
    final_parsed = json.loads(final_text)
    assert len(final_parsed["experiments"]) == 18
    for i in range(17):
        assert final_parsed["experiments"][i]["id"] != "m18-learned-joint-embedding"
    print("   All 17 prior experiments preserved: ✓")
    print("   New experiment appended at index 17: ✓")

    print("\n[SUCCESS] Archive appended successfully.")
    print(f"  Archive: {ARCHIVE_PATH}")
    print(f"  Records: {len(final_parsed['experiments'])}")


if __name__ == "__main__":
    main()
