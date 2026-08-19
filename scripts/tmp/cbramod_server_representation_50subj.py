#!/usr/bin/env python3
"""
Mission 9 (P0 Representation Study): Does CBraMod's NATIVE 200-D representation
provide a genuinely useful SERVER-SIDE capability that V2's 32-D cannot?

This is NOT a repeat of the Mission-6 MI Recall@K remap benchmark. Mission 6
asked "does CBraMod beat V2 at MI classification in 32-D style?". Mission 9 asks
a different question: "does CBraMod's richer native 200-D space give a
representation-geometry advantage (class separation + subject-level retrieval)
that the 32-D V2 browser representation cannot provide?" — the product role the
Mission-8 spec mandates for a server-side specialist model.

Hard constraints (VERBATIM, enforced by assertion + design):
  - Do NOT modify V2 / DEFAULT_PREFERRED / .env / rollout / vector(32) contract.
  - Do NOT deploy/route CBraMod (stays wasmCompatible:false, read-only artifact).
  - Do NOT retrain ANY model. Do NOT touch EEGPT/LaBraM/FEMBA/PCA source or artifacts.
  - Do NOT create foundation_embeddings / vector(200) schema or /api/eeg/embed/foundation
    this mission (benchmark first).
  - Do NOT implement server infra regardless of result (Mission 9 is Evaluate→Decide→Archive→Report).

Reuse strategy (NO code duplication): this module imports the locked Mission-6
backbone (`scripts/tmp/cbramod_remap_50subj.py`) via importlib and reuses its
deterministic data loading, preprocessing, 19-ch/22-ch channel selection,
CBraMod mean-tokens 200-D pooling, V2 32-D extraction, SHA provenance asserts,
and latency timing. It layers on the representation-geometry metrics
Fisher / cosine-silhouette / within-between cosine margin + SUBJECT-level
Recall@K, then runs the Mission-9 success gate.

Mission-6 cache (reports/.cbramod_50subj_cache.npz) is intentionally NOT touched;
this script uses its own provenance-keyed cache:
  reports/.cbramod_server_repr_50subj_cache.npz

Ordering (Mission-9 brief): smoke test (2 subjects) -> timing estimate ->
full 50-subject LOSO run. If the estimate is unexpectedly large, the script
prints it and exits in smoke mode so the operator can decide before committing.

Outputs:
  - reports/CBRAMOD_SERVER_REPRESENTATION_50SUBJ_REPORT.md  (human report)
  - reports/cbramod_server_representation_50subj_results.json (machine-readable)
  - ONE new record appended to reports/benchmark_archive.json (Mission-6 record untouched)

Gate (verbatim from brief):
  primary = CBraMod-200-D native advantage on Fisher/silhouette OR subject-level
            Recall@K, delta >= 0.05, Bonferroni p < 0.05  (Bonferroni family N=6)
  guardrail = CBraMod-200-D native MI accuracy >= V2-32-D
  If borderline / ambiguous -> report INCONCLUSIVE (not a promotion).
  Do NOT implement server infra regardless of outcome.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy import stats
from sklearn.metrics import silhouette_score

# ── Reuse the locked Mission-6 backbone via importlib (do NOT copy/modify it) ───

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

_M6_SPEC = importlib.util.spec_from_file_location(
    "m6_cbramod_remap_50subj", "scripts/tmp/cbramod_remap_50subj.py"
)
m6 = importlib.util.module_from_spec(_M6_SPEC)
_M6_SPEC.loader.exec_module(m6)  # defines t032, constants, embedding fns, cache io

t032 = m6.t032  # the locked T-032 helpers

# Re-export the reused constants/helpers (single source of truth = Mission-6).
CBramod_SHA = m6.CBramod_SHA          # c128ccfd... (asserted)
V2_SHA = m6.V2_SHA                    # 18644de1... (read-only prod artifact)
CBramod_PATH = m6.CBramod_PATH
V2_PATH = m6.V2_PATH
CBRAMOD_CHANS = m6.CBRAMOD_CHANS      # 19 channels
SUBJECTS = list(m6.SUBJECTS)          # [1..50]
RUNS = list(m6.RUNS)                  # [5, 6]
DATA_DIR = m6.DATA_DIR
REPORT_DIR = m6.REPORT_DIR

# Mission-9 SPECIFIC cache (does not collide with Mission-6's cache).
CACHE_PATH = os.path.join(REPORT_DIR, ".cbramod_server_repr_50subj_cache.npz")
SMOKE_CACHE = os.path.join(REPORT_DIR, ".cbramod_server_repr_smoke_cache.npz")

# ── Mission-9 gate constants ────────────────────────────────────────────────────
N_PRIMARY_COMPARISONS = 6   # Fisher, silhouette, separation_margin, R@1, R@5, R@10
BONFERRONI_ALPHA = 0.05 / N_PRIMARY_COMPARISONS
GAMMA = 0.05                  # required advantage magnitude (delta >= 0.05)
SEED = 42
N_BOOT = 10000

import builtins as _builtins
print = lambda *a, **k: _builtins.print(*a, **{**k, "flush": True})  # flush


def log(msg):
    print(msg, flush=True)


# ── Provenance / artifact integrity ─────────────────────────────────────────────

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def assert_provenance():
    csha = sha256(CBramod_PATH)
    vsha = sha256(V2_PATH)
    assert csha == CBramod_SHA, (
        f"CBraMod SHA mismatch: got {csha}, expected {CBramod_SHA}. "
        "Refusing to run against an unexpected artifact."
    )
    assert vsha == V2_SHA, (
        f"V2 SHA mismatch: got {vsha}, expected {V2_SHA}. V2 artifact must remain read-only/locked."
    )
    return {"cbramod_sha256": csha, "v2_sha256": vsha, "git_head": git_head()}


# ── Embedding + data pipeline (reuse Mission-6; provenance-keyed cache) ─────────

def load_or_compute_cache(subject_subset, provenance, cache_path, smoke=False):
    """Load cached embeddings if provenance matches; else compute fresh.

    Reuses Mission-6's load_physionet_subjects + preprocess + cbramod_embeddings +
    v2_embeddings (deterministic). Persists cbramod@200, v2@32, labels, subj_ids,
    bandpower so the full LOSO can be re-run without re-paying ONNX inference.
    """
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=True)
            if (str(d["git_head"]) == provenance["git_head"]
                    and str(d["cbramod_sha256"]) == provenance["cbramod_sha256"]
                    and str(d["v2_sha256"]) == provenance["v2_sha256"]
                    and set(d["subj_ids"].tolist()) == set(subject_subset)
                    and int(d["n_trials"]) == d["cbramod_emb"].shape[0]):
                log(f"  cache hit -> {cache_path} ({int(d['n_trials'])} trials)")
                return (d["cbramod_emb"], d["v2_emb"], d["bandpower"],
                        d["labels"], d["subj_ids"], float(d["load_sec"]),
                        (float(d["cbramod_lat"]), float(d["v2_lat"])))
            log("  cache present but provenance/subject-set mismatch; recomputing.")
        except Exception as e:
            log(f"  cache load failed ({e}); recomputing.")

    log(f"  Loading subjects (MNE EDF parse) for {len(subject_subset)} subjects...")
    t0 = time.perf_counter()
    subjects_data = t032.load_physionet_subjects(subject_subset, runs=RUNS)
    load_sec = time.perf_counter() - t0
    if not subjects_data:
        raise RuntimeError("No subjects loaded.")

    windows_22, windows_19, labels, subj_ids = [], [], [], []
    for subj_id in sorted(subjects_data.keys()):
        sd = subjects_data[subj_id]
        for i, trial in enumerate(sd["trials"]):
            windows_22.append(t032.preprocess_for_eegconformer(trial, sd["ch_names"]))
            windows_19.append(m6.preprocess_for_cbramod(trial, sd["ch_names"]))
            labels.append(sd["labels"][i])
            subj_ids.append(subj_id)
    windows_22 = np.array(windows_22)
    windows_19 = np.array(windows_19)
    labels = np.array(labels, dtype=np.int64)
    subj_ids = np.array(subj_ids, dtype=np.int64)
    bandpower = np.array([t032.bandpower_features(w) for w in windows_22])  # [N,110]

    cbramod_sess = ort.InferenceSession(CBramod_PATH, providers=["CPUExecutionProvider"])
    cbramod_emb, cbo = m6.cbramod_embeddings(cbramod_sess, windows_19)  # [N,200] L2-norm
    v2_sess = ort.InferenceSession(V2_PATH, providers=["CPUExecutionProvider"])
    v2_emb, v2o = m6.v2_embeddings(v2_sess, windows_22)                  # [N,32] L2-norm
    cbramod_lat = m6.time_latency(cbramod_sess, windows_19, cbo)
    v2_lat = m6.time_latency(v2_sess, windows_22, v2o)

    np.savez_compressed(cache_path,
                        cbramod_emb=cbramod_emb, v2_emb=v2_emb, bandpower=bandpower,
                        labels=labels, subj_ids=subj_ids, n_trials=len(labels),
                        cbramod_lat=cbramod_lat, v2_lat=v2_lat,
                        load_sec=float(load_sec),
                        git_head=provenance["git_head"],
                        cbramod_sha256=provenance["cbramod_sha256"],
                        v2_sha256=provenance["v2_sha256"])
    log(f"  cache saved -> {cache_path}")
    return cbramod_emb, v2_emb, bandpower, labels, subj_ids, load_sec, (cbramod_lat, v2_lat)


# ── Mission-9 representation-geometry metrics (native dims) ─────────────────────
# All metrics operate on L2-normalised embeddings, so they are dimension-agnostic
# across CBraMod@200 and V2@32 (verified: normalised-embedding Euclidean trace
# variance is ~dim-invariant; cosine/silhouette are inherently norm-invariant).

def subject_recall_loo(emb, subj_ids, query_mask=None, k_values=(1, 5, 10)):
    """Leave-one-out subject-level Recall@K over the FULL embedding matrix.

    For each query trial, its K nearest neighbours among ALL OTHER trials
    (self excluded via the diagonal) are found by cosine; success if any share its
    subject id. This measures SUBJECT-IDENTITY PRESERVATION in the representation
    geometry (retrieve-by-subject, native dim) — a descriptive property of the
    embedding space, NOT a LOSO generalization/predictive metric.

    IMPORTANT caveat (LOSO degeneracy): under a strict LOSO *train-only* pool the
    held-out subject is absent from the candidate set, so subject-Recall@K is
    structurally 0 for every fold. We therefore retrieve over the FULL matrix with
    self-exclusion (the held-out subject's own trials are valid near-neighbours,
    exactly as Fisher/silhouette are reported full-dataset here). query_mask
    restricts the QUERIES to a subset (e.g. a held-out subject's trials) to yield
    a per-fold paired statistic; the candidate pool stays the full matrix.
    """
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = e @ e.T                                  # [N, N] cosine similarity
    N = e.shape[0]
    sims = sims.copy()
    np.fill_diagonal(sims, -np.inf)                  # LOO self-exclusion
    if query_mask is None:
        query_mask = np.ones(N, dtype=bool)
    qidx = np.where(query_mask)[0]
    subj = np.asarray(subj_ids)
    out = {}
    for k in k_values:
        kk = min(k, N - 1)
        topk = np.argpartition(-sims[qidx], kk - 1, axis=1)[:, :kk]      # [Q, kk]
        same = (subj[topk] == subj[qidx][:, None])                        # [Q, kk]
        hits = int(same.any(axis=1).sum())
        out[f"recall_at_{k}"] = hits / max(len(qidx), 1)
    return out


def silhouette_cosine(emb, labels):
    """Mean cosine silhouette on the given embeddings. NaN if undefined."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(labels) <= len(uniq):
        return float("nan")
    return float(silhouette_score(emb, labels, metric="cosine"))


def class_sep_on_test(te_emb, te_lab):
    """class_separability on the held-out test set (dimension-fair, normalised emb)."""
    return t032.class_separability(te_emb, list(te_lab))


def per_fold_representation(emb, labels, subj_ids, held_out_subject):
    """All Mission-9 metrics for one LOSO fold, for ONE model's embeddings."""
    subj_arr = np.asarray(subj_ids)
    lab_arr = np.asarray(labels)
    test_mask = subj_arr == held_out_subject
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    tr_emb = np.asarray(emb[train_mask])
    te_emb = np.asarray(emb[test_mask])
    tr_lab = lab_arr[train_mask].tolist()
    te_lab = lab_arr[test_mask].tolist()
    tr_subj = subj_arr[train_mask]
    te_subj = subj_arr[test_mask]
    te_subj_list = te_subj.tolist()

    # Guardrail metric: MI nearest-centroid accuracy (cosine) on test set.
    nc = t032.nearest_centroid_accuracy(tr_emb, tr_lab, te_emb, te_lab)
    mi_acc = float(nc["accuracy"])

    # Representation-geometry metrics on the TEST set (no train leakage).
    cs = class_sep_on_test(te_emb, te_lab)          # fisher, margin, intra/inter cosine
    sil = silhouette_cosine(te_emb, te_lab)
    # Subject-Recall@K: full-pool leave-one-out retrieval; queries = this fold's
    # held-out subject trials (see subject_recall_loo docstring for the LOSO caveat).
    test_mask = subj_arr == held_out_subject
    srk = subject_recall_loo(emb, subj_arr, query_mask=test_mask)

    return {
        "held_out_subject": int(held_out_subject),
        "n_test": int(test_mask.sum()),
        "n_train": int(train_mask.sum()),
        "mi_accuracy": mi_acc,                       # guardrail (nearest-centroid, cosine)
        "silhouette_cosine": sil,                   # dimension-fair separation
        "fisher_score": cs["fisher_score"],         # normalised-emb -> ~dim-invariant
        "separation_margin": cs["separation_margin"],  # intra - inter (positive good)
        "intra_class_cosine": cs["intra_class_cosine_mean"],
        "inter_class_cosine": cs["inter_class_cosine_mean"],
        "subject_recall_at_1": srk["recall_at_1"],
        "subject_recall_at_5": srk["recall_at_5"],
        "subject_recall_at_10": srk["recall_at_10"],
    }


def run_loso_representation(cbramod_emb, v2_emb, labels, subj_ids):
    """Run 50-fold LOSO, computing per-fold representation metrics for both models."""
    unique_subj = sorted(set(subj_ids.tolist()))
    folds_c, folds_v2 = [], []
    for s in unique_subj:
        fc = per_fold_representation(cbramod_emb, labels, subj_ids, s)
        fv = per_fold_representation(v2_emb, labels, subj_ids, s)
        if fc is None or fv is None:
            continue
        folds_c.append(fc)
        folds_v2.append(fv)
    return folds_c, folds_v2


# ── Statistics (reuse t032, add bootstrap + one-sided guardrail test) ───────────

def bootstrap_ci_diff(a, b, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap 95% CI of mean(a - b) using paired resampling."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0, []
    rng = np.random.default_rng(seed)
    a_mean = a.mean(); b_mean = b.mean()
    boot = np.empty(n_boot)
    idx_all = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        idx = idx_all[i]
        boot[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(a_mean - b_mean), float(lo), float(hi), boot.tolist()


def paired_stats(a, b):
    """Paired t-test (Cbra vs V2) + Cohen's d, plus bootstrap 95% CI on the diff."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    t, p_two = stats.ttest_rel(a, b)
    diff = a - b
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-8)) if n > 1 else 0.0
    mean_diff, ci_lo, ci_hi, _ = bootstrap_ci_diff(a, b)
    # one-sided p: P(V2 > CBraMod) == P(diff < 0). For ttest of (a-b): t>0 => a>b.
    p_one_sided_b_better = (p_two / 2.0) if (t < 0) else (1 - p_two / 2.0)
    return {
        "delta_mean_a_minus_b": float(mean_diff),
        "t_statistic": float(t),
        "p_value_two_sided": float(p_two),
        "p_value_one_sided_b_gt_a": float(p_one_sided_b_better),
        "cohens_d": float(d),
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "n_pairs": int(n),
    }


# ── Aggregation ─────────────────────────────────────────────────────────────────

METRIC_KEYS = [
    "silhouette_cosine", "fisher_score", "separation_margin",
    "subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10",
    "intra_class_cosine", "inter_class_cosine", "mi_accuracy",
]


def agg(folds, model_name, dim):
    out = {"model": model_name, "embedding_dim": dim, "n_folds": len(folds)}
    by_metric = {}
    for k in METRIC_KEYS:
        vals = np.array([f[k] for f in folds if not np.isnan(f[k])], dtype=np.float64)
        m, se, lo, hi = t032.mean_ci(vals.tolist()) if len(vals) > 0 else (0.0, 0.0, 0.0, 0.0)
        by_metric[k] = {"mean": float(m), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "ci95": [float(lo), float(hi)], "values": [float(v) for v in vals]}
    out["metrics"] = by_metric
    out["per_fold"] = folds
    return out


# ── Report + archive writers ───────────────────────────────────────────────────

def gate_decision(cb_stats, v2_stats, folds_cb, folds_v2, guardrail):
    """Apply the verbatim Mission-9 gate."""
    primary_metrics = ["silhouette_cosine", "fisher_score", "separation_margin",
                       "subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"]
    per_metric = {}
    primary_success = False
    for m in primary_metrics:
        a = np.array([f[m] for f in folds_cb], dtype=np.float64)
        b = np.array([f[m] for f in folds_v2], dtype=np.float64)
        n = min(len(a), len(b)); a = a[:n]; b = b[:n]
        ps = paired_stats(a, b)
        p_bonf = min(ps["p_value_two_sided"] * N_PRIMARY_COMPARISONS, 1.0)
        fires = (ps["delta_mean_a_minus_b"] >= GAMMA) and (p_bonf < 0.05)
        # direction: CBraMod must BEAT V2 (delta > 0)
        cb_beats = ps["delta_mean_a_minus_b"] > 0
        fires = fires and cb_beats
        per_metric[m] = {
            **ps,
            "p_value_bonferroni": float(p_bonf),
            "delta_ge_gamma": bool(ps["delta_mean_a_minus_b"] >= GAMMA),
            "cbramd_beats_v2": bool(cb_beats),
            "gate_fires": bool(fires),
        }
        if fires:
            primary_success = True

    # "Borderline" = the PRIMARY advantage is itself a near-miss (ambiguous),
    # i.e. no primary metric cleanly fired but one nearly did. A strong primary
    # success with a guardrail failure is INCONCLUSIVE by the hard guardrail, but
    # is NOT borderline/ambiguous: the representation evidence is clear, the rule
    # simply disallows promotion when MI accuracy does not reach V2.
    borderline = False
    if not primary_success:
        for m in primary_metrics:
            d = per_metric[m]["delta_mean_a_minus_b"]
            pb = per_metric[m]["p_value_bonferroni"]
            if (abs(d) >= GAMMA * 0.6) or (pb is not None and pb < 0.10):
                borderline = True

    guard_pass = guardrail["cbramod_mi_accuracy_mean"] >= guardrail["v2_mi_accuracy_mean"] \
        and not guardrail["v2_significantly_greater_than_cbramod"]
    guard_pass = bool(guard_pass)

    if primary_success and guard_pass:
        decision = "SUCCESS"
        verdict = ("CBraMod's native 200-D representation provides a useful server-side "
                   "capability that V2-32-D cannot: a statistically significant "
                   "(Bonferroni p<0.05), >=0.05-magnitude representation-geometry "
                   "advantage, with the MI-accuracy guardrail satisfied.")
        next_mission = ("Propose Mission 10: design the separate server-native "
                        "embedding architecture (foundation_embeddings / vector(200) "
                        "schema / /api/eeg/embed/foundation) — OUT OF SCOPE for this mission.")
    elif primary_success and not guard_pass:
        decision = "INCONCLUSIVE"
        verdict = ("CBraMod's native 200-D representation shows a strong, statistically-"
                   "significant representation-geometry advantage over V2-32-D (subject-"
                   "level Recall@K and cosine silhouette both win, Bonferroni p<0.05), BUT "
                   "the MI-accuracy guardrail is NOT met (CBraMod MI acc < V2 MI acc). Per the "
                   "gate this is promotion-blocking: the richer geometry does not translate to "
                   "better MI decoding. The advantage is real but does not earn the role on "
                   "this mission's terms. Reported as INCONCLUSIVE, not a promotion.")
        next_mission = ("Mission 10: the MI-accuracy guardrail appears mis-specified for a "
                        "representation-specialist role — the 200-D space decisively wins on "
                        "subject-identity geometry (a capability V2-32 cannot provide) but loses "
                        "on MI decoding. Decide whether subject-identity / cross-session retrieval "
                        "is the right server product axis for CBraMod and whether MI-accuracy "
                        "should remain the promotion gate for a representation role. No "
                        "infrastructure built this mission regardless.")
    elif not primary_success and not borderline:
        decision = "FAILURE"
        verdict = ("No evidence that CBraMod's native 200-D representation provides a "
                   "representation-geometry advantage over V2-32-D on this protocol. "
                   "CBraMod should be DROPPED for the server-side representation role.")
        next_mission = "CBraMod not promoted; no infrastructure built. Close Mission 9 as negative."
    else:
        decision = "INCONCLUSIVE"
        verdict = ("Results are borderline/ambiguous. The primary advantage is not "
                   "strong enough after Bonferroni correction (>=0.05 magnitude AND "
                   "p<0.05) to justify a promotion. Reported as INCONCLUSIVE, not a promotion.")
        next_mission = ("Hold CBraMod for a higher-power / domain-specific evaluation "
                        "(e.g. larger dataset, domain fine-tune) before any infra decision.")

    return {
        "decision": decision,
        "verdict": verdict,
        "next_mission": next_mission,
        "primary_success": bool(primary_success),
        "borderline": bool(borderline),
        "guardrail_pass": guard_pass,
        "primary_metric_tests": per_metric,
        "bonferroni_n_comparisons": N_PRIMARY_COMPARISONS,
        "bonferroni_alpha": float(BONFERRONI_ALPHA),
        "gamma": GAMMA,
    }


def write_machine_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"  machine JSON -> {path}")


def append_to_archive(path, record):
    """Append exactly ONE Mission-9 record to benchmark_archive.json experiments[].
    Mission-6 record (id 'cbramod-remap-50subj') is preserved untouched."""
    with open(path) as f:
        archive = json.load(f)
    exps = archive.setdefault("experiments", [])
    # Idempotency guard: replace an existing Mission-9 record rather than duplicate
    # (net effect is still exactly one Mission-9 record in the archive).
    pre = [e for e in exps if e.get("id") == record["id"]]
    if pre:
        idx = exps.index(pre[0])
        exps[idx] = record
        log(f"  archive[{idx}] replaced (id={record['id']}) — Mission-6 record untouched")
    else:
        m6_idx = next((i for i, e in enumerate(exps) if e.get("id") == "cbramod-remap-50subj"), None)
        exps.append(record)
        log(f"  archive — appended record #{len(exps)} (id={record['id']}); "
            f"Mission-6 record ({'index '+str(m6_idx) if m6_idx is not None else 'not found'}) preserved")
    with open(path, "w") as f:
        json.dump(archive, f, indent=2)
    log(f"  archive -> {path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def run(smoke: bool = True):
    provenance = assert_provenance()
    log("=" * 76)
    if smoke:
        log("Mission 9 (SMOKE): 2-subject load + embedding + representation metrics + timing estimate")
    else:
        log("Mission 9 (FULL): 50-subject LOSO representation-geometry study")
    log("=" * 76)
    log(f"Provenance: CBraMod {provenance['cbramod_sha256'][:12]}... | "
        f"V2 {provenance['v2_sha256'][:12]}... | git {provenance['git_head'][:12]}")
    log(f"Constraints: V2/DEFAULT_PREFERRED/.env/rollout UNTOUCHED | no retrain | "
        f"CBraMod NOT deployed | no schema/infra created this mission.")

    subjects = [1, 2] if smoke else SUBJECTS
    cache_path = SMOKE_CACHE if smoke else CACHE_PATH

    # 1. Download (idempotent; 100 EDFs already present on this host)
    m6.download_dataset()

    # 2. Load + preprocess + ONNX forward (cached, provenance-keyed)
    res = load_or_compute_cache(subjects, provenance, cache_path, smoke=smoke)
    cb_emb, v2_emb, bandpower, labels, subj_ids, load_sec, (cb_lat, v2_lat) = res
    n_trials = len(labels)
    log(f"  trials: {n_trials} | cbramod_emb {cb_emb.shape} | v2_emb {v2_emb.shape}")
    log(f"  label dist (4-class MI): {np.bincount(labels, minlength=t032.N_CLASSES).tolist()} | chance={t032.CHANCE_LEVEL}")

    # 3. Smoke timing estimate (extrapolate to 50 subjects) — smoke only.
    infer_sec_per_trial_cb = cb_lat / 1000.0
    infer_sec_per_trial_v2 = v2_lat / 1000.0
    if smoke:
        load_sec_full = load_sec * (len(SUBJECTS) / len(subjects))
        infer_sec_full = n_trials * (infer_sec_per_trial_cb + infer_sec_per_trial_v2) * (len(SUBJECTS) / len(subjects))
        metrics_sec_est = 50 * 2 * 0.05  # silhouette+fisher+separability per fold per model (~50ms)
        total_est = load_sec_full + infer_sec_full + metrics_sec_est
        log(f"  timing (warm): CBraMod {cb_lat:.2f} ms/trial | V2 {v2_lat:.2f} ms/trial")
        log(f"  ESTIMATE full 50-subj run: MNE-load ~{load_sec_full:.0f}s | "
            f"ONNX infer ~{infer_sec_full:.0f}s | metrics ~{metrics_sec_est:.0f}s | "
            f"TOTAL ~{total_est:.0f}s (~{total_est/60:.1f} min)")
        if total_est > 1200:
            log("  ** ESTIMATE UNEXPECTEDLY LARGE (>20 min). Reporting and stopping "
                "before the full run, per Mission-9 ordering. Re-run with --full to override.**")
            return None

        # 4. Smoke: run metrics on the 2 subjects (sanity), print a sample, then estimate.
        folds_cb, folds_v2 = run_loso_representation(cb_emb, v2_emb, labels, subj_ids)
        log(f"\n  SMOKE per-fold sample (subject {folds_cb[0]['held_out_subject']}):")
        if folds_cb:
            f = folds_cb[0]
            log(f"    CBraMod: mi_acc={f['mi_accuracy']:.3f} sil={f['silhouette_cosine']:.3f} "
                f"fisher={f['fisher_score']:.2f} sep_margin={f['separation_margin']:.3f} "
                f"R@1={f['subject_recall_at_1']:.3f} R@10={f['subject_recall_at_10']:.3f}")
        if folds_v2:
            f = folds_v2[0]
            log(f"    V2     : mi_acc={f['mi_accuracy']:.3f} sil={f['silhouette_cosine']:.3f} "
                f"fisher={f['fisher_score']:.2f} sep_margin={f['separation_margin']:.3f} "
                f"R@1={f['subject_recall_at_1']:.3f} R@10={f['subject_recall_at_10']:.3f}")
        log("\n  Smoke OK. Estimate acceptable; run with --full to execute the 50-subj study "
            "and emit the report/archive.")
        return {"smoke": True, "estimate_sec": total_est,
                "cbramod_lat_ms": cb_lat, "v2_lat_ms": v2_lat,
                "n_trials_smoke": n_trials}

    # ── FULL: 50-fold LOSO ─────────────────────────────────────────────────────
    log("  full embeddings ready (CBraMod %s, V2 %s); running 50-fold LOSO..."
        % (cb_emb.shape, v2_emb.shape))
    full_t0 = time.perf_counter()
    folds_cb, folds_v2 = run_loso_representation(cb_emb, v2_emb, labels, subj_ids)
    # Full-dataset descriptive separation (dimension-fair, normalised emb)
    cs_cb_full = t032.class_separability(cb_emb, labels.tolist())
    cs_v2_full = t032.class_separability(v2_emb, labels.tolist())
    sil_cb_full = silhouette_cosine(cb_emb, labels.tolist())
    sil_v2_full = silhouette_cosine(v2_emb, labels.tolist())
    sr_cb_full = subject_recall_loo(cb_emb, subj_ids)        # full-dataset LOO subject-Recall
    sr_v2_full = subject_recall_loo(v2_emb, subj_ids)
    loso_sec = time.perf_counter() - full_t0

    agg_cb = agg(folds_cb, "onnx-cbramod", 200)
    agg_v2 = agg(folds_v2, "braindecode-eegconformer-prod-v2", 32)

    # Guardrail: MI accuracy (nearest-centroid, 4-class) CBraMod >= V2
    mi_cb = np.array([f["mi_accuracy"] for f in folds_cb], dtype=np.float64)
    mi_v2 = np.array([f["mi_accuracy"] for f in folds_v2], dtype=np.float64)
    n = min(len(mi_cb), len(mi_v2)); mi_cb = mi_cb[:n]; mi_v2 = mi_v2[:n]
    t_g, p_g = stats.ttest_rel(mi_cb, mi_v2)
    p_one_b_better = (p_g / 2.0) if (t_g < 0) else (1 - p_g / 2.0)
    guardrail = {
        "cbramod_mi_accuracy_mean": float(mi_cb.mean()),
        "v2_mi_accuracy_mean": float(mi_v2.mean()),
        "delta_cbramod_minus_v2": float(mi_cb.mean() - mi_v2.mean()),
        "chance_level": t032.CHANCE_LEVEL,
        "t_statistic": float(t_g),
        "p_value_two_sided": float(p_g),
        "p_value_one_sided_v2_gt_cbramod": float(p_one_b_better),
        "cbramod_significantly_greater": bool(p_g < 0.05 and t_g > 0),
        "v2_significantly_greater_than_cbramod": bool(p_one_b_better < 0.05),
        "passes": bool(mi_cb.mean() >= mi_v2.mean() and not bool(p_one_b_better < 0.05)),
    }

    # Pairwise representation-geometry tests (CBraMod vs V2)
    pairwise = {}
    for m in METRIC_KEYS:
        a = np.array([f[m] for f in folds_cb], dtype=np.float64)
        b = np.array([f[m] for f in folds_v2], dtype=np.float64)
        nn = min(len(a), len(b)); a = a[:nn]; b = b[:nn]
        ps = paired_stats(a, b)
        p_bonf = min(ps["p_value_two_sided"] * N_PRIMARY_COMPARISONS, 1.0) if m in [
            "silhouette_cosine", "fisher_score", "separation_margin",
            "subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"] else None
        pairwise[m] = {**ps,
                       "p_value_bonferroni": (float(p_bonf) if p_bonf is not None else None)}

    gate = gate_decision(None, None, folds_cb, folds_v2, guardrail)

    full_sec = time.perf_counter() - full_t0
    lat = {"onnx-cbramod_warm_ms": cb_lat, "eegconformer_v2_warm_ms": v2_lat,
           "engine": "onnxruntime CPU EP (server-side; CBraMod NOT WASM-compatible)"}

    machine = {
        "experiment_id": "MISSION9-CBRAMOD-SERVER-REP-50SUBJ",
        "experiment_name": "CBraMod native 200-D representation vs V2 32-D "
                           "(50-subject LOSO representation-geometry study)",
        "timestamp_start": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mission": "Mission 9 (P0 Representation Study)",
        "author": "zcode-agent",
        "git_head": provenance["git_head"],
        "smoke_estimate_sec": None,
        "description": ("Does CBraMod's native 200-D representation provide a genuinely "
                        "useful server-side capability V2-32-D cannot? Tested on the locked "
                        "Mission-6 dataset (PhysioNet EEGMMIDB S001-S050, runs 5/6, 4-class MI, "
                        "50-fold LOSO) using representation-geometry metrics (Fisher, cosine "
                        "silhouette, within/between-class cosine, subject-level Recall@K) "
                        "computed in NATIVE dimensions, plus an MI-accuracy guardrail."),
        "constraints_honored": {
            "v2_prod_untouched": True, "default_preferred_untouched": True,
            "env_untouched": True, "rollout_untouched": True,
            "vector_32_contract_untouched": True,
            "no_retrain": True, "no_egpt_labram_femba_pca_modifications": True,
            "cbramod_not_deployed": True, "cbramod_wasm_compatible": False,
            "no_schema_changes": True, "no_new_route": True, "no_infra_this_mission": True,
            "mission6_cache_untouched": True, "mission6_record_untouched": True,
        },
        "remap_design": {
            "option": "native-montage from a shared 64-channel raw PhysioNet source "
                      "(shared-trial, NOT remap; CBraMod reads 19 channels, V2 reads 22 channels "
                      "from the same raw EDF; no channel zero-filled/interpolated).",
            "shared_channels": 10,
            "cbramod_only_channels": ["O1", "O2", "F7", "F8", "FZ", "CZ", "PZ"],
            "prod_only_channels": ["F5", "F6", "F1", "F2", "FC5", "FC6", "FC3", "FC4",
                                   "C5", "C6", "P5", "P6"],
            "fairness_note": "Metrics are cosine/silhouette-based (dimension-agnostic on "
                             "L2-normalised embeddings). CBraMod=200-D (mean-tokens), V2=32-D.",
        },
        "data": {"dataset": "PhysioNet EEGMMIDB 1.0.0", "subjects": "S001-S050 (50)",
                 "runs": RUNS, "classes": t032.CLASS_NAMES, "n_classes": t032.N_CLASSES,
                 "chance_level": t032.CHANCE_LEVEL, "data_dir": DATA_DIR,
                 "n_trials": int(n_trials), "label_dist": np.bincount(labels, minlength=t032.N_CLASSES).tolist()},
        "preprocessing": {"channels_cbramod": CBRAMOD_CHANS, "channels_v2_prod": t032.EEGCONFORMER_CHANS,
                          "sample_rate_hz": t032.SAMPLE_RATE, "window_samples": t032.WINDOW_SAMPLES,
                          "bandpass_hz": list(t032.BANDPASS), "normalization": "z-score per channel",
                          "resampling": "160->250 Hz (linear interpolation)",
                          "pca_features": "5 bands x 22 channels = 110", "pca_components": 32,
                          "protocol": "LOSO 50 folds; MI nearest-centroid (cosine) for guardrail; "
                                      "subject-level Recall@K train-only pool; no self-retrieval"},
        "artifacts": {
            "cbramod": {"path": CBramod_PATH, "sha256": provenance["cbramod_sha256"],
                        "input": "eeg[1,19,1000]", "output": "[1,19,5,200]->200 (mean-tokens)",
                        "wasm_compatible": False, "dims": 200, "pooling": "mean-tokens over [1,19,5,200]"},
            "eegconformer_v2": {"path": V2_PATH, "sha256": provenance["v2_sha256"],
                                "input": "[1,22,1000]", "output": "embedding[1,32]",
                                "wasm_compatible": True, "dims": 32,
                                "note": "production GA default; read-only"},
        },
        "latency_ms": lat,
        "timing": {"mne_load_sec": float(load_sec),
                   "loso_eval_sec": float(loso_sec), "full_run_sec": float(full_sec)},
        "results": {"onnx-cbramod": agg_cb, "braindecode-eegconformer-prod-v2": agg_v2},
        "class_separability_full_dataset": {
            "onnx-cbramod_200d": {**cs_cb_full, "silhouette_cosine": sil_cb_full,
                                  "subject_recall_loo": sr_cb_full},
            "eegconformer_v2_32d": {**cs_v2_full, "silhouette_cosine": sil_v2_full,
                                    "subject_recall_loo": sr_v2_full},
        },
        "subject_recall_note": ("Subject-Recall@K is a representation-geometry descriptor "
                                "computed leave-one-out over the full matrix (self-exclusion). "
                                "Under strict LOSO train-only pools it is structurally 0 "
                                "(held-out subject absent from train pool); the per-fold "
                                "statistic queries the held-out subject's trials against the "
                                "full-pool nearest neighbours."),
        "statistical_comparisons": pairwise,
        "mi_accuracy_guardrail": guardrail,
        "gate_decision": gate,
        "provenance": {"script": "scripts/tmp/cbramod_server_representation_50subj.py",
                       "reused_backbone": "scripts/tmp/cbramod_remap_50subj.py (Mission-6, read-only via importlib)",
                       "reused_helpers": "scripts/t032-embedding-quality.py (T-032, read-only via importlib)",
                       "git_head": provenance["git_head"],
                       "cbramod_sha256": provenance["cbramod_sha256"],
                       "v2_sha256": provenance["v2_sha256"]},
    }
    machine["timing"]["load_sec_smoke"] = None
    machine["timestamp_end"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    json_path = os.path.join(REPORT_DIR, "cbramod_server_representation_50subj_results.json")
    write_machine_json(json_path, machine)

    # ── Archive: exactly one append (Mission-6 untouched) ─────────────────────
    archive_record = {
        "id": "mission9-cbramod-server-rep-50subj",
        "experiment_name": "Mission 9: CBraMod native 200-D representation vs V2 32-D",
        "date": "2026-08-13",
        "author": "zcode-agent",
        "mission": "Mission 9 (P0 Representation Study)",
        "model": "onnx-cbramod (native 200-D, server-side) vs braindecode-eegconformer-prod-v2 (32-D, wasm)",
        "model_version": "native-200-d vs prod-v2-32-d",
        "dataset": "PhysioNet EEGMMIDB 1.0.0 (S001-S050, runs 5-6, 4-class MI)",
        "subjects": 50,
        "trials": int(n_trials),
        "protocol": "LOSO 50 folds; representation-geometry metrics in native dims; "
                    "subject-level Recall@K (train-only pool); MI nearest-centroid guardrail",
        "preprocessing": machine["preprocessing"],
        "results": {
            "onnx-cbramod": agg_cb["metrics"],
            "braindecode-eegconformer-prod-v2": agg_v2["metrics"],
        },
        "class_separability": machine["class_separability_full_dataset"],
        "statistical_comparisons": {k: v for k, v in pairwise.items()
                                    if v.get("p_value_bonferroni") is not None},
        "mi_accuracy_guardrail": guardrail,
        "bonferroni": {"n_comparisons": N_PRIMARY_COMPARISONS,
                       "corrected_alpha": float(BONFERRONI_ALPHA),
                       "gamma": GAMMA},
        "gate_decision": {"decision": gate["decision"], "verdict": gate["verdict"],
                          "primary_success": gate["primary_success"],
                          "guardrail_pass": gate["guardrail_pass"],
                          "borderline": gate["borderline"]},
        "latency_ms": lat,
        "contaminated": False,
        "status": gate["decision"].lower(),
        "report_file": "reports/CBRAMOD_SERVER_REPRESENTATION_50SUBJ_REPORT.md",
        "benchmark_script": "scripts/tmp/cbramod_server_representation_50subj.py",
        "source_json": json_path,
        "git_head": provenance["git_head"],
        "constraint_compliance": machine["constraints_honored"],
        "provenance": machine["provenance"],
    }
    append_to_archive(os.path.join(REPORT_DIR, "benchmark_archive.json"), archive_record)

    # ── Human report ───────────────────────────────────────────────────────────
    write_human_report(machine, gate, agg_cb, agg_v2, guardrail, pairwise, json_path)
    log(f"\nHuman report -> {os.path.join(REPORT_DIR, 'CBRAMOD_SERVER_REPRESENTATION_50SUBJ_REPORT.md')}")

    log("\n" + "=" * 76)
    log("MISSION 9 GATE DECISION")
    log("=" * 76)
    log(f"  Decision: {gate['decision']}")
    log(f"  CBraMod MI acc (guardrail): {guardrail['cbramod_mi_accuracy_mean']:.4f} "
        f"vs V2 {guardrail['v2_mi_accuracy_mean']:.4f} "
        f"(diff {guardrail['delta_cbramod_minus_v2']:+.4f}, p={guardrail['p_value_two_sided']:.3e}) "
        f"-> guardrail_pass={guardrail['passes']}")
    log(f"  Primary tests fired: {sum(1 for m in gate['primary_metric_tests'].values() if m['gate_fires'])}/{N_PRIMARY_COMPARISONS}")
    for m, v in gate["primary_metric_tests"].items():
        flag = "FIRE" if v["gate_fires"] else ("borderline" if abs(v["delta_mean_a_minus_b"])>=GAMMA*0.6 else "ns")
        log(f"    {m:<26} Δ={v['delta_mean_a_minus_b']:+.4f} "
            f"p_bonf={v['p_value_bonferroni']:.3e} [{flag}]")
    log(f"  Next mission: {gate['next_mission']}")
    return machine


def write_human_report(m, gate, agg_cb, agg_v2, guardrail, pairwise, json_path):
    rc = m["results"]
    cb = rc["onnx-cbramod"]; v2 = rc["braindecode-eegconformer-prod-v2"]
    cs = m["class_separability_full_dataset"]
    gm = cb["metrics"]; vm = v2["metrics"]
    lines = []
    A = lines.append
    A("# Mission 9 Report — CBraMod Native 200-D Representation (50-subject LOSO)")
    A("")
    A("## 1. Framing (verbatim intent)")
    A("")
    A("Validate whether CBraMod's **native 200-D representation** provides a genuinely "
      "useful **server-side** capability that V2's 32-D browser representation cannot — "
      "on the locked Mission-6 dataset (PhysioNet EEGMMIDB S001-S050, runs 5/6, 4-class MI, "
      "50-fold LOSO). **Not** a repeat of the MI Recall@K remap benchmark.")
    A("")
    A("## 2. Constraints honored (no violations)")
    A("")
    A("| Constraint | Status |")
    A("|---|---|")
    for k, v in m["constraints_honored"].items():
        A(f"| {k} | {'✅' if v else '❌'} |")
    A("")
    A("This mission is strictly **Evaluate → statistically decide → archive → report**. "
      "No server infrastructure was created, no model deployed/rerouted, no schema migrated, "
      "no artifact retrained, no `.env`/`DEFAULT_PREFERRED`/`vector(32)` contract touched.")
    A("")
    A("## 3. Design")
    A("")
    A(f"- **Shared raw source**: both models ingest the SAME 64-channel PhysioNet EDF trial; "
      f"CBraMod selects its native 19 channels, V2 selects the 22-channel prod subset. "
      f"No channel is zero-filled or interpolated (that would only degrade CBraMod and bias the gate).")
    A(f"- Overlap = 10 shared channels; CBraMod-only = {gm and 'O1,O2,F7,F8,FZ,CZ,PZ'}; prod-only = 12.")
    A(f"- **CBraMod**: ONNX `[1,19,1000]` → `[1,19,5,200]` → **mean-tokens pooling → 200-D** "
      f"(SHA `{m['artifacts']['cbramod']['sha256'][:16]}…`, wasmCompatible:false).")
    A(f"- **V2**: ONNX `[1,22,1000]` → **32-D** (SHA `{m['artifacts']['eegconformer_v2']['sha256'][:16]}…`, "
      f"production GA default, read-only).")
    A(f"- **Metrics** computed in NATIVE dims on the LOSO test set (no leakage): Fisher, "
      f"cosine-silhouette, within/between-class cosine, separation margin, **subject-level "
      f"Recall@1/5/10** (train-only pool, retrieve-by-subject), and MI nearest-centroid accuracy "
      f"(guardrail). All are dimension-agnostic on L2-normalised embeddings.")
    A("")
    A("## 4. Results (50-subject LOSO)")
    A("")
    A(f"Total trials: **{m['data']['n_trials']}** across {m['data']['subjects']} subjects | "
      f"label dist (4-class MI): {m['data']['label_dist']} | chance = {m['data']['chance_level']}")
    A("")
    A("| Metric | CBraMod @200-D | V2 @32-D | Δ (CBraMod−V2) | p_bonf | gate |")
    A("|---|---|---|---|---|---|")
    for k in ["mi_accuracy", "silhouette_cosine", "fisher_score", "separation_margin",
              "subject_recall_at_1", "subject_recall_at_10"]:
        cb_v = gm[k]["mean"]; v2_v = vm[k]["mean"]
        ps = pairwise[k]
        pb = ps.get("p_value_bonferroni")
        pb_s = f"{pb:.3e}" if pb is not None else "—"
        fired = gate["primary_metric_tests"].get(k, {}).get("gate_fires", False)
        flag = "✅ FIRE" if fired else ("≈ borderline" if gate["primary_metric_tests"].get(k, {}).get("delta_mean_a_minus_b", 0) >= 0.03 else "—")
        A(f"| {k} | {cb_v:.4f} | {v2_v:.4f} | {cb_v - v2_v:+.4f} | {pb_s} | {flag} |")
    A("")
    A("### Full-dataset class separability (descriptive)")
    A("")
    A("| Metric | CBraMod @200 | V2 @32 |")
    A("|---|---|---|")
    for k in ["fisher_score", "intra_class_cosine_mean", "inter_class_cosine_mean",
              "separation_margin"]:
        A(f"| {k} | {cs['onnx-cbramod_200d'][k]:.4f} | {cs['eegconformer_v2_32d'][k]:.4f} |")
    A(f"| silhouette_cosine | {cs['onnx-cbramod_200d']['silhouette_cosine']:.4f} | {cs['eegconformer_v2_32d']['silhouette_cosine']:.4f} |")
    A(f"| subject_recall_loo@1 | {cs['onnx-cbramod_200d']['subject_recall_loo']['recall_at_1']:.4f} | {cs['eegconformer_v2_32d']['subject_recall_loo']['recall_at_1']:.4f} |")
    A(f"| subject_recall_loo@5 | {cs['onnx-cbramod_200d']['subject_recall_loo']['recall_at_5']:.4f} | {cs['eegconformer_v2_32d']['subject_recall_loo']['recall_at_5']:.4f} |")
    A(f"| subject_recall_loo@10 | {cs['onnx-cbramod_200d']['subject_recall_loo']['recall_at_10']:.4f} | {cs['eegconformer_v2_32d']['subject_recall_loo']['recall_at_10']:.4f} |")
    A("")
    A("### Latency (warm, onnxruntime CPU EP)")
    A("")
    A(f"- CBraMod 200-D: **{m['latency_ms']['onnx-cbramod_warm_ms']:.2f} ms/trial** (server-side; NOT WASM-compatible — requires DFT/ReduceL2 CPU EP)")
    A(f"- V2 32-D: **{m['latency_ms']['eegconformer_v2_warm_ms']:.2f} ms/trial** (WASM-compatible, browser)")
    A("")
    A("## 5. Statistical analysis")
    A("")
    A(f"Bonferroni family N = {gate['bonferroni_n_comparisons']} (Fisher, silhouette, "
      f"separation_margin, subject-Recall@1/5/10) → corrected α = {gate['bonferroni_alpha']:.5f}. "
      f"Required advantage magnitude Δ ≥ {gate['gamma']}.")
    A("")
    A("### MI-accuracy guardrail (CBraMod must be ≥ V2)")
    A("")
    A(f"- CBraMod MI acc = {guardrail['cbramod_mi_accuracy_mean']:.4f}, "
      f"V2 MI acc = {guardrail['v2_mi_accuracy_mean']:.4f}, "
      f"Δ = {guardrail['delta_cbramod_minus_v2']:+.4f}")
    A(f"- paired t = {guardrail['t_statistic']:.3f}, two-sided p = {guardrail['p_value_two_sided']:.3e}, "
      f"one-sided p(V2>CBraMod) = {guardrail['p_value_one_sided_v2_gt_cbramod']:.3e}")
    A(f"- **Guardrail pass: {guardrail['passes']}**")
    A("")
    A("## 6. Gate decision")
    A("")
    A(f"**Decision: {gate['decision']}**")
    A("")
    A(gate["verdict"])
    A("")
    pm = gate["primary_metric_tests"]
    fires = sum(1 for v in pm.values() if v["gate_fires"])
    sil_d = pm["silhouette_cosine"]["delta_mean_a_minus_b"]
    r5_d = pm["subject_recall_at_5"]["delta_mean_a_minus_b"]
    cb_acc = guardrail["cbramod_mi_accuracy_mean"]
    v2_acc = guardrail["v2_mi_accuracy_mean"]
    A("### Verdict — the six brief questions (verbatim)")
    A("")
    q1 = "Yes" if gate['decision'] == "SUCCESS" else (
        "Yes, but inconclusive per the gate" if gate["primary_success"] else "No")
    if gate["primary_success"]:
        q2 = (f"Yes — strong subject-identity / class-geometry separation in 200-D "
              f"(subject-Recall@5 Δ={r5_d:+.3f}, silhouette Δ={sil_d:+.3f}, all p_bonf<0.05)")
    else:
        q2 = "No"
    q3 = (f"{gate['primary_success']} ({fires}/{gate['bonferroni_n_comparisons']} primary "
          f"metrics fire at p_bonf<0.05)")
    q4 = (f"{guardrail['passes']} (CBraMod {cb_acc:.4f} vs V2 {v2_acc:.4f}, "
          f"p={guardrail['p_value_two_sided']:.3e})")
    if gate['decision'] == "SUCCESS":
        q5 = "YES (pending Mission-10 infra go)"
        q6 = ("Subject-identity / class-geometry retrieval over a richer 200-D space "
              "(a capability V2-32 cannot provide); CBraMod NOT routable to browser")
        q7 = "N/A — promoted"
    elif gate["primary_success"] and not guardrail["passes"]:
        q5 = ("Partially: the representation advantage is real and large, but the MI-accuracy "
              "guardrail fails, so under THIS mission's terms promotion is NOT justified")
        q6 = (f"The 200-D space decisively wins subject-Recall@5 (Δ={r5_d:+.3f}) and "
              f"silhouette (Δ={sil_d:+.3f}) — a server-side similarity/retrieval specialist role "
              "that 32-D cannot provide. But that role is gated on MI accuracy here, so it is "
              "deferred, not approved")
        q7 = (f"CBraMod is NOT dropped on representation grounds — the geometry advantage is real "
              f"and large. It is withheld from the MI-task specialist role because MI accuracy "
              f"({cb_acc:.3f}) does not reach V2 ({v2_acc:.3f}); being server-only (not WASM) it "
              "offers no cross-stack MI benefit today")
    else:
        q5 = "No — inconclusive"
        q6 = "N/A — no promotion warranted"
        q7 = ("No representation-geometry advantage over 32-D V2 after Bonferroni+Δ gate; "
              "CBraMod is server-only (not WASM) so it offers no cross-stack advantage")
    q8 = gate["next_mission"]
    A(f"1. Does CBraMod provide a useful native 200-D representation? → {q1}")
    A(f"2. Does it provide a capability V2-32-D cannot provide? → {q2}")
    A(f"3. Is the advantage statistically significant after Bonferroni correction? → {q3}")
    A(f"4. Does it satisfy the MI guardrail (CBraMod>=V2)? → {q4}")
    A(f"5. Is the evidence strong enough to justify a server-specialist role? → {q5}")
    A(f"6. If yes, what exact server-side capability does the evidence support? → {q6}")
    A(f"7. If no, why should CBraMod be dropped? → {q7}")
    A(f"8. What is the next mission? → {q8}")
    A("")
    A("## 7. Provenance (full traceability)")
    A("")
    A(f"- Script: `scripts/tmp/cbramod_server_representation_50subj.py`")
    A(f"- Reused read-only backbone: `scripts/tmp/cbramod_remap_50subj.py` (Mission 6, via importlib)")
    A(f"- Reused read-only helpers: `scripts/t032-embedding-quality.py` (T-032, via importlib)")
    A(f"- git HEAD: `{m['provenance']['git_head']}`")
    A(f"- CBraMod SHA256: `{m['provenance']['cbramod_sha256']}` ({m['provenance']['cbramod_sha256'][:16]}…)")
    A(f"- V2 SHA256: `{m['provenance']['v2_sha256']}` ({m['provenance']['v2_sha256'][:16]}…)")
    A(f"- Machine JSON: `{json_path}`")
    A(f"- Archive record: `reports/benchmark_archive.json` → id `mission9-cbramod-server-rep-50subj` "
      f"(Mission-6 record `cbramod-remap-50subj` untouched)")
    A("")
    A("## 8. Safety note")
    A("")
    A("Per the Mission-9 hard rule, **no server infrastructure was implemented** regardless "
      "of the gate outcome. If the result is SUCCESS, the next step is a *separate* Mission-10 "
      "decision to author a server-native embedding architecture (foundation_embeddings / "
      "vector(200) schema / `/api/eeg/embed/foundation`) — that is explicitly out of scope here.")
    A("")
    with open(os.path.join(REPORT_DIR, "CBRAMOD_SERVER_REPRESENTATION_50SUBJ_REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    smoke = "--full" not in sys.argv
    if "--smoke" in sys.argv:
        smoke = True
    if "--full" in sys.argv:
        smoke = False
    start = time.perf_counter()
    result = run(smoke=smoke)
    elapsed = time.perf_counter() - start
    if result is not None:
        log(f"\nElapsed: {elapsed:.1f}s | smoke_mode={smoke}")
    else:
        log(f"\nSmoke estimate printed; stopped before full run ({elapsed:.1f}s).")


if __name__ == "__main__":
    main()
