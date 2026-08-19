#!/usr/bin/env python3
"""Append the Mission 6 CBraMod remap experiment to reports/benchmark_archive.json.

Idempotent: removes any prior experiment id 'cbramod-remap-50subj' before appending,
and any prior bug id 'MISSION5-1' / preserved entries before re-adding."""
import json, os, hashlib, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "reports" / "benchmark_archive.json"
RESULTS = REPO / "reports" / "cbramod_remap_50subj_results.json"

with open(ARCHIVE, "r") as f:
    arch = json.load(f)
with open(RESULTS, "r") as f:
    res = json.load(f)

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def cbramod_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


# ── Compose the experiment record from the results JSON ────────────────────────
record = {
    "id": "cbramod-remap-50subj",
    "experiment_name": "CBraMod 19->22 channel remap study + 50-subject LOSO validation",
    "date": "2026-08-14",
    "author": "zcode-agent",
    "mission": "Mission 6 (Next Model Mission) — CBraMod 19->22 remap + 50-subj validation vs V2 + PCA",
    "model": "CBraMod (onnx-cbramod), EEGConformer v2 (braindecode-eegconformer-prod-v2), PCA Bandpower",
    "model_version": "CBraMod v1 (native 19-ch; 200-D mean-tokens); EEGConformer v2 FP32 canonical (GA); PCA bandpower (32-D)",
    "dataset": "PhysioNet EEGMMIDB 1.0.0 (S001-S050, runs 5-6, 4-class MI: left/right_hand, feet/tongue)",
    "subjects": 50,
    "trials": res["data"]["n_trials"],
    "protocol": res["preprocessing"]["protocol"],
    "preprocessing": {
        "channels_cbramod": res["preprocessing"]["channels_cbramod"],
        "channels_v2_prod": res["preprocessing"]["channels_v2_prod"],
        "shared_channels": res["remap_design"]["shared_channels"],
        "cbramod_only_channels": res["remap_design"]["cbramod_only_channels"],
        "prod_only_channels": res["remap_design"]["prod_only_channels"],
        "zero_filled_channels": res["remap_design"]["zero_filled_channels"],
        "interpolated_channels": res["remap_design"]["interpolated_channels"],
        "sample_rate_hz": res["preprocessing"]["sample_rate_hz"],
        "window_samples": res["preprocessing"]["window_samples"],
        "bandpass_hz": res["preprocessing"]["bandpass_hz"],
        "normalization": res["preprocessing"]["normalization"],
        "resampling": res["preprocessing"]["resampling"],
        "pca_features": res["preprocessing"]["pca_features"],
        "pca_components": res["preprocessing"]["pca_components"],
        "pca_fit": res["preprocessing"]["pca_fit"],
        "fairness_note": res["remap_design"]["fairness_note"],
        "assumption": res["remap_design"]["assumption"],
    },
    "remapping_design": res["remap_design"],
    "artifacts": res["artifacts"],
    "results": res["results"],
    "class_separability": res["class_separability"],
    "latency_ms": res["latency_ms"],
    "statistical_comparisons": res["statistical_comparisons"],
    "bonferroni": res["bonferroni"],
    "decision": res["decision"],
    "contaminated": False,
    "status": "valid (negative result — CBraMod does not beat V2 or PCA; Bonferroni-corrected, p>0.05)",
    "report_file": "reports/CBRAMOD_REMAP_50SUBJ_REPORT.md",
    "benchmark_script": "scripts/tmp/cbramod_remap_50subj.py",
    "source_json": "reports/cbramod_remap_50subj_results.json",
    "git_head": git_head,
    "constraint_compliance": res["constraints"],
    "provenance": {
        "script": "scripts/tmp/cbramod_remap_50subj.py",
        "script_sha256": cbramod_sha(REPO / "scripts" / "tmp" / "cbramod_remap_50subj.py"),
        "results_json": "reports/cbramod_remap_50subj_results.json",
        "cbramod_artifact_sha256": res["artifacts"]["cbramod"]["sha256"],
        "v2_artifact_sha256": res["artifacts"]["eegconformer_v2"]["sha256"],
        "v2_rollout_stage_at_run": "ga (AI_EEGCONFORMER_ENABLED=ga, 100% cohort; 24h soak skipped per resource constraints; Mission 5)",
        "cbramod_deployment_status": "NOT deployed (wasmCompatible=false; DFT/ReduceL2 blockers); server-side specialist role gated on this study",
    },
}

# ── Append experiment (replace any prior same-id to keep idempotent) ────────────
arch["experiments"] = [e for e in arch["experiments"] if e.get("id") != record["id"]]
arch["experiments"].append(record)

# ── Bugs & corrections: promote_ga.sh [3/6] manifest-traversal bug ─────────────
bug = {
    "bug_id": "MISSION5-1",
    "bug_name": "promote_ga.sh gate [3/6] manifest traversal iterates a JSON object as an array",
    "affected_files": ["scripts/promote_ga.sh"],
    "description": "Gate [3/6] locates the v2 entry via `next(e.get('sha256',...) for e in m.get('models',[]) if e.get('registryId')=='eegconformer-prod-v2')`, but public/models/manifest.json stores `models` as a keyed JSON OBJECT (not an array). Iterating a dict yields string keys, so `e.get(...)` raises AttributeError: 'str' object has no attribute 'get'. The gate was never reached (the script blocked earlier at [2/6]); SHA provenance for [3/6] was instead verified manually by iterating models.values().",
    "fix": "Iterate m.get('models', {}).values() instead of m.get('models', []) so each entry is a dict.",
    "impact": "A future 'clean' GA promotion through promote_ga.sh would crash at [3/6] SHA verification instead of completing, even though the manifest SHA (18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931) is correct.",
    "status": "flagged (not fixed — latent; do not modify Mission 5 promotion machinery)",
    "severity": "latent / medium (would block future scripted GA promotions at the SHA gate)",
}
arch["bugs_and_corrections"] = [b for b in arch["bugs_and_corrections"] if b.get("bug_id") != bug["bug_id"]]
arch["bugs_and_corrections"].append(bug)

# ── Preserved artifacts: register the 3 new files ─────────────────────────────
new_artifacts = [
    {"type": "report", "path": "reports/CBRAMOD_REMAP_50SUBJ_REPORT.md",
     "description": "Mission 6 human-readable report: CBraMod 19->22 remap + 50-subj LOSO (negative result)"},
    {"type": "json", "path": "reports/cbramod_remap_50subj_results.json",
     "description": "Mission 6 full machine-readable results (metrics, CIs, pairwise stats, provenance)"},
    {"type": "script", "path": "scripts/tmp/cbramod_remap_50subj.py",
     "description": "Mission 6 eval script: CBraMod@19 + V2@22 + PCA, reuses T-032 helpers, clean 50-fold LOSO, Bonferroni, latency"},
]
existing = {(a.get("type"), a.get("path")) for a in arch["preserved_artifacts"]}
for a in new_artifacts:
    if (a["type"], a["path"]) not in existing:
        arch["preserved_artifacts"].append(a)

with open(ARCHIVE, "w") as f:
    json.dump(arch, f, indent=2)
print(f"Appended experiment 'cbramod-remap-50subj' -> {ARCHIVE}")
print(f"  experiments[] count: {len(arch['experiments'])}")
print(f"  bugs_and_corrections count: {len(arch['bugs_and_corrections'])}")
print(f"  preserved_artifacts count: {len(arch['preserved_artifacts'])}")
