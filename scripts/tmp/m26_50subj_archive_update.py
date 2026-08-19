#!/usr/bin/env python3
"""Quick archive update for M26 50-subject results."""
import json, os

SRC_TMP = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(SRC_TMP)
REPO = os.path.dirname(SCRIPTS)
REPORTS = os.path.join(REPO, "reports")
ARCHIVE_PATH = os.path.join(REPORTS, "benchmark_archive.json")
RESULTS_PATH = os.path.join(REPORTS, "m26_eegpt_50subj_retrieval_results.json")
REPORT_PATH = os.path.join(REPORTS, "MISSION26_EEGPT_50SUBJ_RETRIEVAL_REPORT.md")

# Load results
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

rr = results["retrieval_results"]
sc = results["statistical_comparisons"]

eegpt_r5 = rr["eegpt_2048"]["recall_at_5"]["mean"]
joint_r5 = rr["joint_264"]["recall_at_5"]["mean"]
eegpt_mrr = rr["eegpt_2048"]["mrr"]["mean"]
joint_mrr = rr["joint_264"]["mrr"]["mean"]
pca_r5 = rr["pca_32"]["recall_at_5"]["mean"]
cb_r5 = rr["cbramod_200"]["recall_at_5"]["mean"]
v2_r5 = rr["v2_32"]["recall_at_5"]["mean"]

p_joint = sc["eegpt_vs_joint_264_r5"]["p_value"]
p_cb = sc["eegpt_vs_cbramod_200_r5"]["p_value"]
p_v2 = sc["eegpt_vs_v2_32_r5"]["p_value"]
p_pca = sc["eegpt_vs_pca_32_r5"]["p_value"]

joint_sig = sc["eegpt_vs_joint_264_r5"]["significant_after_bonferroni"]
cb_sig = sc["eegpt_vs_cbramod_200_r5"]["significant_after_bonferroni"]
v2_sig = sc["eegpt_vs_v2_32_r5"]["significant_after_bonferroni"]
pca_sig = sc["eegpt_vs_pca_32_r5"]["significant_after_bonferroni"]

actual_sha = results["eegpt_inference"]["sha256"]
cb_sha = results["cache_alignment"]["cbramod_sha256"]
v2_sha = results["cache_alignment"]["v2_sha256"]

git_head = os.popen("git rev-parse HEAD").read().strip()

eegpt_better_than_joint = (eegpt_r5 >= joint_r5) and not joint_sig
eegpt_better_than_cb = cb_sig
eegpt_better_than_v2 = v2_sig
decision = "EXTEND" if (eegpt_better_than_joint and eegpt_better_than_cb and eegpt_better_than_v2) else "INCONCLUSIVE"

with open(ARCHIVE_PATH, "r") as f:
    arch = json.load(f)

record = {
    "id": "m26-eegpt-50subj-retrieval",
    "experiment_name": "M26 Extended: EEGPT 50-Subject Session-Disjoint Retrieval Evaluation",
    "date": "2026-08-13",
    "author": "zcode-agent",
    "mission": "Extended M26: EEGPT-2048 on the 50-subject session-disjoint retrieval protocol (M13/M18), definitive evaluation for server-backbone decision.",
    "model": "onnx-eegpt (EEGPT ViT, INT8-quantised, 2048-D)",
    "model_version": "pretrained (no fine-tuning, no modification)",
    "dataset": "PhysioNet EEGMMIDB 1.0.0 (S001-S050, runs 5-10)",
    "subjects": 50,
    "protocol": "50-subject LOSO session-disjoint retrieval: 300 splits, query=15 trials from (subject, held-out-run), pool=all other trials. Metrics: R@1/R@5/R@10/MRR (cosine, L2-normalized). Bonferroni alpha=0.0125 (4 comparisons).",
    "results": {
        "r5_eegpt_2048": float(eegpt_r5),
        "r5_joint_264": float(joint_r5),
        "r5_pca_32": float(pca_r5),
        "r5_cbramod_200": float(cb_r5),
        "r5_v2_32": float(v2_r5),
        "mrr_eegpt_2048": float(eegpt_mrr),
        "mrr_joint_264": float(joint_mrr),
        "eegpt_vs_joint_r5_p": float(p_joint),
        "eegpt_vs_joint_r5_sig": bool(joint_sig),
        "eegpt_vs_cbramod_r5_p": float(p_cb),
        "eegpt_vs_cbramod_r5_sig": bool(cb_sig),
        "eegpt_vs_v2_r5_p": float(p_v2),
        "eegpt_vs_v2_r5_sig": bool(v2_sig),
        "eegpt_vs_pca_r5_p": float(p_pca),
        "eegpt_vs_pca_r5_sig": bool(pca_sig),
        "n_splits": 300,
    },
    "decision": decision,
    "contaminated": False,
    "status": "COMPLETED",
    "report_file": REPORT_PATH,
    "benchmark_script": "scripts/tmp/m26_eegpt_50subj_retrieval.py",
    "source_json": RESULTS_PATH,
    "git_head": git_head,
    "constraints_honored": results["constraints_honored"],
    "provenance": {
        "eegpt_artifact_sha256": actual_sha,
        "cbramod_artifact_sha256": cb_sha,
        "v2_artifact_sha256": v2_sha,
        "joint_block_weights": [0.62, 0.16, 0.22],
        "eegpt_embedding_cache": "reports/.m26_eegpt_50subj_cache.npz",
        "reassessment_10subj_link": "reports/MISSION26_RETRIEVAL_REASSESSMENT.md",
    },
}

arch["experiments"] = [e for e in arch["experiments"] if e.get("id") != record["id"]]
arch["experiments"].append(record)

new_artifacts = [
    {"type": "report", "path": REPORT_PATH,
     "description": "M26 50-subject EEGPT retrieval evaluation report"},
    {"type": "script", "path": "scripts/tmp/m26_eegpt_50subj_retrieval.py",
     "description": "M26 50-subject EEGPT retrieval evaluation script (imports from m26_50subj_eval_only.py)"},
    {"type": "script", "path": "scripts/tmp/m26_50subj_eval_only.py",
     "description": "M26 50-subject EEGPT evaluation runner (uses cached embeddings)"},
    {"type": "json", "path": RESULTS_PATH,
     "description": "M26 50-subject EEGPT retrieval results (R@K/MRR + statistics)"},
    {"type": "cache", "path": "reports/.m26_eegpt_50subj_cache.npz",
     "description": "EEGPT-2048 embeddings cache (4500 trials, 2048-D, L2-normalized)"},
]
existing = {(a.get("type"), a.get("path")) for a in arch.get("preserved_artifacts", [])}
if "preserved_artifacts" not in arch:
    arch["preserved_artifacts"] = []
for a in new_artifacts:
    key = (a["type"], a["path"])
    if key not in existing:
        arch["preserved_artifacts"].append(a)

with open(ARCHIVE_PATH, "w") as f:
    json.dump(arch, f, indent=2)

print(f"Archive updated: {len(arch['experiments'])} experiments, {len(arch['preserved_artifacts'])} artifacts")
print()
print("=== FINAL 50-SUBJECT RESULTS ===")
print(f"EEGPT-2048 R@5: {eegpt_r5:.4f}")
print(f"Joint-264  R@5: {joint_r5:.4f}")
print(f"PCA-32     R@5: {pca_r5:.4f}")
print(f"CBraMod-200 R@5: {cb_r5:.4f}")
print(f"V2-32      R@5: {v2_r5:.4f}")
print()
print(f"EEGPT vs Joint p: {p_joint:.2e} -> non-inferior: {eegpt_better_than_joint}")
print(f"EEGPT vs PCA p:   {p_pca:.2e} -> sig: {pca_sig}")
print(f"EEGPT vs CB p:    {p_cb:.2e} -> sig: {cb_sig}")
print(f"EEGPT vs V2 p:    {p_v2:.2e} -> sig: {v2_sig}")
print()
print(f"Decision: {decision}")
print(f"Verdict: EEGPT-2048 justified as server representation candidate: {eegpt_better_than_joint and eegpt_better_than_cb and eegpt_better_than_v2}")
