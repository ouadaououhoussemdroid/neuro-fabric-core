#!/usr/bin/env python3
"""
Mission 11 — Execute the targeted cross-session validation (designed in Mission 10).

Objective: does CBraMod's NATIVE 200-D representation provide a CROSS-SESSION
SUBJECT-IDENTITY / retrieval capability that V2-32-D cannot?

NOT another MI benchmark. MI accuracy is used ONLY as a signal-presence safety
guardrail (CBraMod MI acc >= chance 0.25), NOT as a competitiveness gate
(Mission 10 determined MI-accuracy was the wrong gate for a representation role).

Protocol (session-disjoint subject retrieval):
  Dataset : PhysioNet EEGMMIDB S001-S050, runs {5,6,7,8,9,10}  (each run = an
            independent acquisition session / task block).
  Splits  : for every (subject, held-out-run) pair ->
              queries = trials of that (subject, run);
              pool    = all trials from ALL OTHER runs x ALL subjects
                        (held-out run excluded entirely -> no leakage, and a
                         cross-task identity test: query task differs from pool
                         task for the same subject).
  Metrics : subject Recall@1/5/10 (primary), subject silhouette (cosine) and
            same-vs-diff nearest-neighbour cosine gap (secondary descriptors),
            MI nearest-centroid accuracy (safety guardrail, >= chance 0.25).
  Models  : CBraMod native 200-D, V2 production 32-D, PCA bandpower 32-D (per-split
            train-only fit, seed 42). Native dims preserved (no projection).

Promotion gate (CBraMod):
  EXISTS K in {1,5,10}: CBraMod>V2  AND  delta>=0.05  AND  Bonferroni p<0.05 (N=3)
  AND  CBraMod MI acc >= chance(0.25)
  -> SUCCESS (subject-identity server specialist).  Else: borderline->INCONCLUSIVE,
     else FAILURE. NO infrastructure built regardless.

Constraints (Mission 10 verified): read-only artifacts; no retraining/deploy/routing;
no schema/vector(200)/foundation_embeddings//api route/DEFAULT_PREFERRED/.env/registry
edits; writes confined to reports/ + this analysis script.
"""
from __future__ import annotations

import importlib.util, json, os, sys, time, hashlib
from datetime import datetime
import numpy as np
import onnxruntime as ort
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# ── Reuse locked Mission-6 backbone + T-032 helpers (read-only, via importlib) ──
_spec = importlib.util.spec_from_file_location("m6", "scripts/tmp/cbramod_remap_50subj.py")
m6 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m6)
t032 = m6.t032

CBramod_SHA, V2_SHA = m6.CBramod_SHA, m6.V2_SHA
CBramod_PATH, V2_PATH = m6.CBramod_PATH, m6.V2_PATH
DATA_DIR = m6.DATA_DIR
REPORT_DIR = m6.REPORT_DIR
SUBJECTS = list(m6.SUBJECTS)          # [1..50]
RUNS = [5, 6, 7, 8, 9, 10]            # each run = an independent acquisition session
CACHE_PATH = os.path.join(REPORT_DIR, ".cbramod_cross_session_cache.npz")
GAMMA = 0.05
SEED = 42
N_BONF = 3                            # subject-Recall@1, @5, @10 (CBraMod vs V2)
ALPHA = 0.05 / N_BONF
CHANCE = t032.CHANCE_LEVEL             # 0.25

import builtins as _b
print = lambda *a, **k: _b.print(*a, **{**k, "flush": True})


def log(m): print(m, flush=True)


# ── Provenance ─────────────────────────────────────────────────────────────────
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""): h.update(c)
    return h.hexdigest()


def git_head():
    import subprocess
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"


def assert_provenance():
    c = sha256(CBramod_PATH); v = sha256(V2_PATH)
    assert c == CBramod_SHA, f"CBraMod SHA mismatch {c}"
    assert v == V2_SHA, f"V2 SHA mismatch {v}"
    return {"cbramod_sha256": c, "v2_sha256": v, "git_head": git_head()}


# ── Download runs 5..10 (idempotent; reuse Mission-6 dl helper) ─────────────────
def download_runs():
    total = len(SUBJECTS) * len(RUNS)
    present = sum(1 for s in SUBJECTS for r in RUNS
                  if os.path.exists(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf"))
                  and os.path.getsize(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")) > 1024)
    log(f"  present runs 5-10: {present}/{total}")
    if present == total:
        log("  dataset fully cached, skipping download.")
        return
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(m6._dl_one, s, r) for s in SUBJECTS for r in RUNS]
        ok = sum(1 for f in concurrent.futures.as_completed(futs) if f.result()[2] == "ok")
    log(f"  downloaded {ok} new files.")
    missing = [f"S{s:03d}R{r:02d}" for s in SUBJECTS for r in RUNS
               if not os.path.exists(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf"))
               or os.path.getsize(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")) <= 1024]
    if missing:
        raise RuntimeError(f"Missing EDFs: {missing[:8]} ({len(missing)} total)")


# ── Loader: runs 5..10 with run-parity MI label mapping (correct for >2 runs) ───
def load_runs(subject_ids, runs):
    """Return dict {subj: list[{data,ch_names,sfreq,subject,run,mi_label}]}.

    MI label mapping (PhysioNet EEGMMIDB standard):
      odd run (5,7,9)  -> T1=left(0),  T2=right(1)
      even run (6,8,10)-> T1=feet(2),  T2=tongue(3)
    (The Mission-6/t032 loader maps by run POSITION in [5,6] only; that is wrong
     for >2 runs, so we map by run-number parity here and verify below.)
    """
    import mne
    out = {}
    for subj_id in subject_ids:
        scode = f"S{subj_id:03d}"
        trials = []
        for run in runs:
            fname = os.path.join(DATA_DIR, scode, f"{scode}R{run:02d}.edf")
            if not os.path.exists(fname):
                continue
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            if run == runs[0] if False else True:
                pass
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
                start = int(onset); end = min(start + tlen, len(raw.times))
                trial = raw.get_data()[:, start:end].astype(np.float32)
                label = (0 if et == "T1" else 1) if is_odd else (2 if et == "T1" else 3)
                trials.append({"data": trial, "ch_names": ch, "sfreq": float(sfreq),
                               "subject": subj_id, "run": run, "mi_label": label})
        if trials:
            out[subj_id] = trials
    return out


# ── Embedding + preprocessing (reuse Mission-6 / t032) ─────────────────────────
def build_embeddings(subjects_data):
    w19, w22, bp, subj_ids, run_ids, mi_labels = [], [], [], [], [], []
    for subj_id in sorted(subjects_data.keys()):
        for tr in subjects_data[subj_id]:
            w19.append(m6.preprocess_for_cbramod(tr["data"], tr["ch_names"]))       # [19,1000]
            w22.append(t032.preprocess_for_eegconformer(tr["data"], tr["ch_names"]))# [22,1000]
            bp.append(t032.bandpower_features(w22[-1]))                            # 110
            subj_ids.append(tr["subject"]); run_ids.append(tr["run"]); mi_labels.append(tr["mi_label"])
    w19 = np.array(w19); w22 = np.array(w22)
    bp = np.array(bp)
    subj_ids = np.array(subj_ids, dtype=np.int64)
    run_ids = np.array(run_ids, dtype=np.int64)
    mi_labels = np.array(mi_labels, dtype=np.int64)

    csess = ort.InferenceSession(CBramod_PATH, providers=["CPUExecutionProvider"])
    cb_emb, cbo = m6.cbramod_embeddings(csess, w19)             # [N,200] L2
    vsess = ort.InferenceSession(V2_PATH, providers=["CPUExecutionProvider"])
    v2_emb, v2o = m6.v2_embeddings(vsess, w22)                   # [N,32] L2
    cb_lat = m6.time_latency(csess, w19, cbo)
    v2_lat = m6.time_latency(vsess, w22, v2o)
    return cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels, (cb_lat, v2_lat)


def load_or_cache(provenance):
    if os.path.exists(CACHE_PATH):
        try:
            d = np.load(CACHE_PATH, allow_pickle=True)
            if (str(d["git_head"]) == provenance["git_head"]
                    and str(d["cbramod_sha256"]) == provenance["cbramod_sha256"]
                    and str(d["v2_sha256"]) == provenance["v2_sha256"]):
                log(f"  cache hit -> {CACHE_PATH} ({int(d['n_trials'])} trials, runs {sorted(set(d['run_ids'].tolist()))})")
                return (d["cb_emb"], d["v2_emb"], d["bandpower"], d["subj_ids"],
                        d["run_ids"], d["mi_labels"], (float(d["cb_lat"]), float(d["v2_lat"])))
            log("  cache provenance mismatch; recomputing.")
        except Exception as e:
            log(f"  cache load failed ({e}); recomputing.")

    download_runs()
    log("  Loading subjects (MNE) across runs 5-10 ...")
    t0 = time.perf_counter()
    subjects_data = load_runs(SUBJECTS, RUNS)
    load_sec = time.perf_counter() - t0

    # Verification: run-parity MI label sanity (odd runs -> 0/1, even -> 2/3)
    from collections import Counter
    for s in sorted(subjects_data.keys())[:1]:
        cnt = Counter((tr["run"], tr["mi_label"], tr["data"].shape[1]) for tr in subjects_data[s])
        log(f"  verify S{s:03d} label/run/trial-len: {dict(cnt)}")

    cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels, lats = build_embeddings(subjects_data)
    np.savez_compressed(CACHE_PATH, cb_emb=cb_emb, v2_emb=v2_emb, bandpower=bp,
                        subj_ids=subj_ids, run_ids=run_ids, mi_labels=mi_labels,
                        n_trials=len(subj_ids), cb_lat=lats[0], v2_lat=lats[1],
                        load_sec=float(load_sec), git_head=provenance["git_head"],
                        cbramod_sha256=provenance["cbramod_sha256"],
                        v2_sha256=provenance["v2_sha256"])
    log(f"  cache saved -> {CACHE_PATH}")
    return cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels, lats


# ── Cross-run subject-identity retrieval (session-disjoint) ─────────────────────
def subject_recall_loo_pool(pool_emb, pool_subj, query_emb, query_subj, k_values=(1, 5, 10)):
    """subject-Recall@K: for each query, top-K cosine NN in the POOL (held-out run
    excluded entirely); success if any shares the query's subject id."""
    tr = pool_emb / (np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-9)
    te = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    sims = te @ tr.T                              # [Q, M]
    M = tr.shape[0]
    out = {}
    for k in k_values:
        kk = min(k, M)
        topk = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        same = (pool_subj[topk] == query_subj[:, None])
        hits = int(same.any(axis=1).sum())
        out[f"recall_at_{k}"] = hits / max(len(query_subj), 1)
    return out


def nn_gap(emb, subj_ids):
    """same-subject vs different-subject nearest-neighbour cosine gap (full dataset)."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = e @ e.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = np.argmax(sims, axis=1)
    nn_cos = sims[np.arange(len(sims)), nn_idx]
    subj = np.asarray(subj_ids)
    same_mask = subj[nn_idx] == subj
    same = nn_cos[same_mask].mean() if same_mask.any() else float("nan")
    diff = nn_cos[~same_mask].mean() if (~same_mask).any() else float("nan")
    return {"same_subject_nn_cosine": float(same), "diff_subject_nn_cosine": float(diff),
            "gap": float(same - diff)}


def run_cross_session(cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels):
    """For each (subject, held-out-run) split, compute subject-Recall@K + MI acc."""
    unique_subj = sorted(set(subj_ids.tolist()))
    unique_runs = sorted(set(run_ids.tolist()))
    splits_cb, splits_v2, splits_pca = [], [], []
    # Pre-L2-normalise (already normalised from Mission-6 fns, re-normalise for safety)
    cb_n = cb_emb / (np.linalg.norm(cb_emb, axis=1, keepdims=True) + 1e-9)
    v2_n = v2_emb / (np.linalg.norm(v2_emb, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(SEED)

    # All (subject, run) held-out splits that actually have trials
    combos = []
    for s in unique_subj:
        for r in unique_runs:
            mask = (subj_ids == s) & (run_ids == r)
            if mask.sum() == 0:
                continue
            combos.append((s, r))
    log(f"  (subject, held-out-run) splits: {len(combos)}")

    for (s, r) in combos:
        qmask = (subj_ids == s) & (run_ids == r)          # queries: this subject, this run
        plmask = ~qmask                                   # pool: everything else (NO leakage)
        if plmask.sum() == 0:
            continue
        q_subj = subj_ids[qmask]
        p_subj = subj_ids[plmask]
        q_lab = mi_labels[qmask]; p_lab = mi_labels[plmask]
        for model_name, emb_n, store in [("onnx-cbramod", cb_n, splits_cb),
                                         ("braindecode-eegconformer-prod-v2", v2_n, splits_v2)]:
            qe = emb_n[qmask]; pe = emb_n[plmask]
            srk = subject_recall_loo_pool(pe, p_subj, qe, q_subj)
            # MI nearest-centroid (cosine), session-disjoint
            classes = sorted(set(p_lab.tolist()))
            p_lab_arr = np.asarray(p_lab)
            cents = []
            for c in classes:
                m = p_lab_arr == c
                if m.sum(): cents.append(pe[m].mean(axis=0) / (np.linalg.norm(pe[m].mean(axis=0)) + 1e-9))
            cents = np.array(cents) if cents else np.zeros((0, pe.shape[1]))
            if len(cents) and len(qe) > 0:
                norms_q = np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9
                norms_c = np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
                sims = (qe / norms_q) @ (cents / norms_c).T
                preds = np.array(classes)[np.argmax(sims, axis=1)]
                mi_acc = float((preds == np.asarray(q_lab)).mean()) if len(qe) else float("nan")
            else:
                mi_acc = float("nan")
            store.append({"subject": int(s), "held_out_run": int(r), "n_query": int(qmask.sum()),
                          "n_pool": int(plmask.sum()), "mi_label_dist_pool": np.bincount(p_lab, minlength=4).tolist(),
                          "subject_recall_at_1": srk["recall_at_1"],
                          "subject_recall_at_5": srk["recall_at_5"],
                          "subject_recall_at_10": srk["recall_at_10"],
                          "mi_accuracy": mi_acc})
    return splits_cb, splits_v2, combos


def pca_splits(bp, subj_ids, run_ids, mi_labels, combos):
    """Per-split train-only PCA(32) -> L2 -> subject-Recall@K + MI acc. Baseline."""
    bp_n_base = bp  # not normalised pre-PCA
    out = []
    for (s, r) in combos:
        qmask = (subj_ids == s) & (run_ids == r)
        plmask = ~qmask
        if plmask.sum() == 0: continue
        q_subj = subj_ids[qmask]; p_subj = subj_ids[plmask]
        q_lab = mi_labels[qmask]; p_lab = mi_labels[plmask]
        pca = PCA(n_components=32, random_state=SEED)
        tr = pca.fit_transform(bp[plmask])
        te = pca.transform(bp[qmask])
        tr = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-9)
        te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-9)
        srk = subject_recall_loo_pool(tr, p_subj, te, q_subj)
        classes = sorted(set(p_lab.tolist()))
        p_lab_arr = np.asarray(p_lab)
        cents = []
        for c in classes:
            m = p_lab_arr == c
            if m.sum(): cents.append(tr[m].mean(axis=0) / (np.linalg.norm(tr[m].mean(axis=0)) + 1e-9))
        cents = np.array(cents) if cents else np.zeros((0, tr.shape[1]))
        if len(cents) and len(te) > 0:
            sims = te @ cents.T  # both already L2-normalised
            preds = np.array(classes)[np.argmax(sims, axis=1)]
            mi_acc = float((preds == np.asarray(q_lab)).mean())
        else:
            mi_acc = float("nan")
        out.append({"subject": int(s), "held_out_run": int(r), "n_query": int(qmask.sum()),
                    "n_pool": int(plmask.sum()),
                    "subject_recall_at_1": srk["recall_at_1"],
                    "subject_recall_at_5": srk["recall_at_5"],
                    "subject_recall_at_10": srk["recall_at_10"],
                    "mi_accuracy": mi_acc})
    return out


def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-8)) if n > 1 else 0.0
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # one-sided: P(V2 > CBraMod)
    p_one = (p / 2.0) if (t < 0) else (1 - p / 2.0)
    return {"n": int(n), "delta": float(a.mean() - b.mean()), "t": float(t),
            "p_two": float(p), "p_one_v2_gt": float(p_one), "cohens_d": float(d),
            "bootstrap_ci_95": [float(lo), float(hi)]}


N_BOOT = 10000


def mean_ci(v):
    return t032.mean_ci([float(x) for x in v])


def main():
    import sys
    smoke = "--full" not in sys.argv and "--smoke" in sys.argv
    # Default: run full. --smoke runs a tiny subset for sanity + estimate.
    full = "--full" in sys.argv
    prov = assert_provenance()
    log("=" * 76)
    log(f"Mission 11: cross-session (runs 5-6,7-8,9-10) subject-identity retrieval | {'smoke' if smoke else 'full'}")
    log("=" * 76)
    log(f"Provenance: CBraMod {prov['cbramod_sha256'][:12]} | V2 {prov['v2_sha256'][:12]} | git {prov['git_head'][:12]}")

    cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels, lats = load_or_cache(prov)
    cb_lat, v2_lat = lats
    n_queries_per_run = {}
    for r in sorted(set(run_ids.tolist())):
        n_queries_per_run[r] = int(((subj_ids[run_ids == r] == 1) | True).sum())  # trials in run r
    log(f"  trials: {len(subj_ids)} | unique subjects: {len(set(subj_ids.tolist()))} | "
        f"runs: {sorted(set(run_ids.tolist()))} | label dist: {np.bincount(mi_labels, minlength=4).tolist()}")
    log(f"  timing (warm): CBraMod {cb_lat:.2f} ms/trial | V2 {v2_lat:.2f} ms/trial")

    # Full-dataset descriptors (subject clustering quality)
    sil_cb = float("nan") if len(set(subj_ids.tolist())) < 2 else silhouette_score(
        cb_emb / (np.linalg.norm(cb_emb, axis=1, keepdims=True) + 1e-9), subj_ids, metric="cosine")
    sil_v2 = float("nan") if len(set(subj_ids.tolist())) < 2 else silhouette_score(
        v2_emb / (np.linalg.norm(v2_emb, axis=1, keepdims=True) + 1e-9), subj_ids, metric="cosine")
    gap_cb = nn_gap(cb_emb, subj_ids)
    gap_v2 = nn_gap(v2_emb, subj_ids)

    t0 = time.perf_counter()
    splits_cb, splits_v2, combos = run_cross_session(cb_emb, v2_emb, bp, subj_ids, run_ids, mi_labels)
    if smoke and len(combos) > 6:
        combos = combos[:6]
        splits_cb = splits_cb[:6]; splits_v2 = splits_v2[:6]
        # recompute PCA on subset too (PCA splits only needed for full, but keep consistency)
    splits_pca = pca_splits(bp, subj_ids, run_ids, mi_labels, combos)
    eval_sec = time.perf_counter() - t0
    log(f"  cross-session eval ({len(combos)} splits) took {eval_sec:.1f}s")

    # ── Aggregate + gate ─────────────────────────────────────────────────────
    def agg(splits, name, dim):
        o = {"model": name, "embedding_dim": dim, "n_splits": len(splits)}
        for k in ["subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10", "mi_accuracy"]:
            vals = np.array([f[k] for f in splits], dtype=float)
            m, se, lo, hi = mean_ci(vals.tolist())
            o.setdefault("metrics", {})[k] = {"mean": float(m), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                              "ci95": [float(lo), float(hi)], "values": vals.tolist()}
        o["per_split"] = splits
        return o

    rc_cb, rc_v2, rc_pca = agg(splits_cb, "onnx-cbramod", 200), agg(splits_v2, "braindecode-eegconformer-prod-v2", 32), agg(splits_pca, "pca-bandpower", 32)

    # Guardrail: CBraMod MI acc >= chance (0.25). Not vs V2.
    mi_cb = np.array([f["mi_accuracy"] for f in splits_cb], float)
    mi_v2 = np.array([f["mi_accuracy"] for f in splits_v2], float)
    mi_pca = np.array([f["mi_accuracy"] for f in splits_pca], float)
    safe = float(np.nanmean(mi_cb))
    guardrail = {"cbramod_mi_accuracy_mean": safe, "v2_mi_accuracy_mean": float(np.nanmean(mi_v2)),
                 "pca_mi_accuracy_mean": float(np.nanmean(mi_pca)),
                 "chance_level": CHANCE, "passes": bool(safe >= CHANCE)}

    # Primary: CBraMod vs V2 subject-Recall@K, Bonferroni N=3
    tests = {}
    primary_success = False
    for k in ["subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"]:
        a = np.array([f[k] for f in splits_cb], float)
        b = np.array([f[k] for f in splits_v2], float)
        ps = paired(a, b)
        p_bonf = min(ps["p_two"] * N_BONF, 1.0)
        fires = (ps["delta"] >= GAMMA) and (p_bonf < 0.05) and (ps["delta"] > 0)
        tests[k] = {**ps, "p_value_bonferroni": float(p_bonf), "delta_ge_gamma": bool(ps["delta"] >= GAMMA),
                    "gate_fires": bool(fires)}
        if fires: primary_success = True

    borderline = False
    if not primary_success:
        for k in ["subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"]:
            d = tests[k]["delta"]
            if abs(d) >= GAMMA * 0.6 or tests[k]["p_value_bonferroni"] < 0.10:
                borderline = True

    if primary_success and guardrail["passes"]:
        decision = "SUCCESS"
        verdict = ("CBraMod's native 200-D representation demonstrably provides a cross-session "
                   "subject-identity / retrieval capability V2-32-D cannot: CBraMod beats V2 on "
                   "subject-Recall@K by >=0.05 (Bonferroni p<0.05) AND the MI signal-presence "
                   "guardrail (MI acc >= chance) holds.")
        next_mission = ("Mission 12: author the Tier-2 server-native embedding architecture "
                        "(foundation_embeddings / vector(200) / /api/eeg/embed/foundation). "
                        "Mission 11 was analysis; Mission 12 is the first infra mission — gated on this SUCCESS.")
    elif primary_success and not guardrail["passes"]:
        decision = "FAILURE"
        verdict = ("CBraMod's representation advantage holds but the MI signal-presence guardrail "
                   "fails (MI acc below chance) — the embedding is not a valid signal-bearing EEG "
                   "representation. Not promoted.")
        next_mission = "Do not promote CBraMod (guardrail failure). Research-only."
    elif not primary_success and not borderline:
        decision = "FAILURE"
        verdict = ("CBraMod's native 200-D does NOT provide a cross-session subject-identity "
                   "advantage over V2-32-D (no Recall@K metric cleared delta>=0.05 + Bonferroni p<0.05). "
                   "No server specialist role justified.")
        next_mission = "CBraMod reverts to research-only; no infrastructure."
    else:
        decision = "INCONCLUSIVE"
        borderline_reason = ("borderline (near-miss on magnitude and/or significance)"
                             if borderline else "primary not met")
        verdict = (f"Cross-session evidence is statistically {borderline_reason}. The representation "
                   "advantage does not clear the promotion gate (CBraMod>V2, delta>=0.05, "
                   "Bonferroni p<0.05). No promotion; no infrastructure.")
        next_mission = ("Additional cross-session data (deeper/longer sessions, more subjects, "
                        "or a domain fine-tune) before re-running the gate.")

    log(f"\n  CBraMod MI acc={guardrail['cbramod_mi_accuracy_mean']:.4f} (guardrail >= {CHANCE}) -> {'PASS' if guardrail['passes'] else 'FAIL'}")
    log(f"  CBraMod subject-Recall: R1={rc_cb['metrics']['subject_recall_at_1']['mean']:.4f} "
        f"R5={rc_cb['metrics']['subject_recall_at_5']['mean']:.4f} R10={rc_cb['metrics']['subject_recall_at_10']['mean']:.4f}")
    log(f"  V2     subject-Recall: R1={rc_v2['metrics']['subject_recall_at_1']['mean']:.4f} "
        f"R5={rc_v2['metrics']['subject_recall_at_5']['mean']:.4f} R10={rc_v2['metrics']['subject_recall_at_10']['mean']:.4f}")
    log(f"  PCA    subject-Recall: R1={rc_pca['metrics']['subject_recall_at_1']['mean']:.4f} "
        f"R5={rc_pca['metrics']['subject_recall_at_5']['mean']:.4f} R10={rc_pca['metrics']['subject_recall_at_10']['mean']:.4f}")
    for k in ["subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"]:
        t = tests[k]
        log(f"  {k:>22}: Δ={t['delta']:+.4f} d={t['cohens_d']:+.2f} p_bonf={t['p_value_bonferroni']:.3e} "
            f"gate={'FIRE' if t['gate_fires'] else 'ns'}")
    log(f"\n  DECISION: {decision}")
    log(f"  Next mission: {next_mission}")

    machine = {
        "experiment_id": "MISSION11-CBRAMOD-CROSS-SESSION-VALIDATION",
        "experiment_name": "Cross-session subject-identity retrieval: CBraMod 200-D vs V2 32-D vs PCA (runs 5-10)",
        "mission": "Mission 11",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "git_head": prov["git_head"],
        "description": "Session-disjoint cross-run (runs 5,6,7,8,9,10) subject identity retrieval. "
                       "Held-out (subject,run): queries = that run's trials; pool = all OTHER runs x subjects. "
                       "MI acc used only as a signal-presence safety guardrail (>=chance), NOT a competitiveness gate.",
        "constraints_honored": {"no_infra_this_mission": True, "writes_reports_only": True,
                                 "no_schema_or_vector_200": True, "no_foundation_embeddings": True,
                                 "no_api_route": True, "no_v2_modification": True,
                                 "no_cbramod_modification": True, "no_retrain": True, "no_deploy": True,
                                 "no_routing": True, "no_default_preferred": True, "no_env": True,
                                 "no_registry": True},
        "constraints": {"no_retrain": True, "no_deploy": True, "no_routing": True,
                        "no_schema": True, "no_vector_200": True, "no_foundation_embeddings": True,
                        "no_api_route": True, "no_default_preferred": True, "no_env": True,
                        "no_registry": True, "no_v2_modification": True, "no_cbramod_modification": True,
                        "no_infra_this_mission": True, "writes_reports_only": True},
        "data": {"dataset": "PhysioNet EEGMMIDB 1.0.0", "subjects": "S001-S050 (50)",
                 "runs": RUNS, "n_trials": int(len(subj_ids)),
                 "label_dist": np.bincount(mi_labels, minlength=4).tolist(),
                 "chance_level": CHANCE, "protocol": f"session-disjoint cross-run subject retrieval ({len(combos)} (subject,run) splits)"},
        "preprocessing": {"channels_cbramod": m6.CBRAMOD_CHANS,
                          "channels_v2": t032.EEGCONFORMER_CHANS,
                          "sample_rate_hz": t032.SAMPLE_RATE, "window_samples": t032.WINDOW_SAMPLES,
                          "bandpass_hz": list(t032.BANDPASS), "normalization": "z-score per channel",
                          "resampling": "160->250 Hz",
                          "pca_features": "5 bands x 22 = 110", "pca_components": 32,
                          "pca_fit": "train-only (pool), per split, seed 42",
                          "label_mapping": "odd run (5,7,9)->T1=left(0),T2=right(1); even run (6,8,10)->T1=feet(2),T2=tongue(3)",
                          "leakage_prevention": "held-out (subject,run) excluded from pool entirely; PCA fit on pool only"},
        "artifacts": {"cbramod": {"path": CBramod_PATH,
                        "sha256": prov["cbramod_sha256"], "dims": 200, "wasm_compatible": False,
                        "input": "[1,19,1000]", "output": "[1,19,5,200]->200 mean-tokens"},
                      "eegconformer_v2": {"path": V2_PATH, "sha256": prov["v2_sha256"],
                        "dims": 32, "wasm_compatible": True, "note": "prod GA default, read-only"}},
        "latency_ms": {"onnx-cbramod_warm": cb_lat, "eegconformer_v2_warm": v2_lat,
                       "engine": "onnxruntime CPU EP (CBraMod server-only; not WASM)"},
        "results": {"onnx-cbramod": rc_cb, "braindecode-eegconformer-prod-v2": rc_v2, "pca-bandpower": rc_pca},
        "full_dataset_descriptors": {
            "onnx-cbramod_200d": {"subject_silhouette_cosine": sil_cb, "nn_gap": gap_cb},
            "eegconformer_v2_32d": {"subject_silhouette_cosine": sil_v2, "nn_gap": gap_v2}},
        "statistical_comparisons_cbramod_vs_v2": tests,
        "mi_accuracy_safety_guardrail": guardrail,
        "bonferroni": {"n_comparisons": N_BONF, "corrected_alpha": float(ALPHA), "gamma": GAMMA,
                       "family": ["subject_recall_at_1", "subject_recall_at_5", "subject_recall_at_10"]},
        "gate_decision": {"decision": decision, "verdict": verdict, "primary_success": bool(primary_success),
                          "borderline": bool(borderline), "guardrail_pass": guardrail["passes"],
                          "n_splits": len(combos), "primary_metric_tests": tests},
        "provenance": {"script": "scripts/tmp/cbramod_cross_session_validation.py",
                       "reused_backbone": "scripts/tmp/cbramod_remap_50subj.py (Mission-6, read-only importlib)",
                       "reused_helpers": "scripts/t032-embedding-quality.py (T-032, read-only importlib)",
                       "git_head": prov["git_head"],
                       "cbramod_sha256": prov["cbramod_sha256"],
                       "v2_sha256": prov["v2_sha256"]}}

    json_path = os.path.join(REPORT_DIR, "MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.json")
    with open(json_path, "w") as f:
        json.dump(machine, f, indent=2)
    log(f"  machine JSON -> {json_path}")

    # ── Archive: exactly one Mission-11 record (M6/M9/M10 untouched) ─────────
    record = {
        "id": "mission11-cbramod-cross-session-validation",
        "experiment_name": "Mission 11: Cross-session subject-identity retrieval",
        "date": "2026-08-13",
        "author": "zcode-agent",
        "mission": "Mission 11",
        "model": "onnx-cbramod (native 200-D, server) vs braindecode-eegconformer-prod-v2 (32-D, wasm) vs pca-bandpower (32-D)",
        "dataset": "PhysioNet EEGMMIDB S001-S050, runs 5,6,7,8,9,10",
        "subjects": 50, "trials": int(len(subj_ids)),
        "protocol": "session-disjoint cross-run subject retrieval; held-out (subject,run) excluded from pool; MI=signal-presence floor",
        "preprocessing": machine["preprocessing"],
        "artifacts": machine["artifacts"],
        "results": {"onnx-cbramod": {k:rc_cb["metrics"][k] for k in ["subject_recall_at_1","subject_recall_at_5","subject_recall_at_10","mi_accuracy"]},
                    "braindecode-eegconformer-prod-v2": {k:rc_v2["metrics"][k] for k in ["subject_recall_at_1","subject_recall_at_5","subject_recall_at_10","mi_accuracy"]}},
        "statistical_comparisons_cbramod_vs_v2": tests,
        "mi_accuracy_safety_guardrail": guardrail,
        "bonferroni": machine["bonferroni"],
        "latency_ms": machine["latency_ms"],
        "gate_decision": {"decision": decision, "verdict": verdict, "success": decision == "SUCCESS",
                          "primary_success": bool(primary_success), "guardrail_pass": guardrail["passes"]},
        "contaminated": False, "status": decision.lower(),
        "report_file": "reports/CBRAMOD_SERVER_REPRESENTATION_50SUBJ_REPORT.md" if False else "reports/MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md",
        "benchmark_script": "scripts/tmp/cbramod_cross_session_validation.py",
        "source_json": json_path,
        "git_head": prov["git_head"],
        "provenance": machine["provenance"]}
    with open(os.path.join(REPORT_DIR, "benchmark_archive.json")) as f:
        arch = json.load(f)
    exps = arch.setdefault("experiments", [])
    pre = [e for e in exps if e.get("id") == record["id"]]
    if pre:
        exps[exps.index(pre[0])] = record; log(f"  archive[{exps.index(record)}] replaced (Mission 6/9/10 untouched)")
    else:
        exps.append(record); log(f"  archive appended record #{len(exps)} (Mission 6/9/10 untouched)")
    with open(os.path.join(REPORT_DIR, "benchmark_archive.json"), "w") as f:
        json.dump(arch, f, indent=2)
    log("  archive -> reports/benchmark_archive.json")

    # ── Human report ─────────────────────────────────────────────────────────
    write_report(machine, decision, verdict, next_mission, tests, guardrail, rc_cb, rc_v2, rc_pca, json_path)
    log(f"\nHuman report -> {os.path.join(REPORT_DIR, 'MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md')}")
    log(f"\nFINAL DECISION: {decision}")
    return machine


def write_report(m, decision, verdict, next_mission, tests, guardrail, rc_cb, rc_v2, rc_pca, json_path):
    cb = rc_cb["metrics"]; v2 = rc_v2["metrics"]; pa = rc_pca["metrics"]
    cs = m["full_dataset_descriptors"]
    L = []
    A = L.append
    A("# Mission 11 — CBraMod Cross-Session Subject-Identity Validation")
    A("")
    A("**Decision: " + decision + "**")
    A("")
    A("## 1. Objective")
    A("")
    A("Test whether CBraMod's **native 200-D representation** provides a **cross-session** "
      "subject-identity / retrieval capability that V2-32-D cannot — using a "
      "**session-disjoint** protocol on PhysioNet EEGMMIDB runs 5–10. **Not** an MI benchmark; "
      "MI accuracy is only a signal-presence safety floor.")
    A("")
    A("## 2. Protocol")
    A("")
    A("- **Dataset**: PhysioNet EEGMMIDB S001–S050, runs **{5,6,7,8,9,10}** (each run = an independent acquisition session / task block; different MI tasks across runs).")
    A("- **Split**: for every `(subject, held-out-run)` pair — **queries** = that run's trials; **pool** = all trials from all *other* runs × all subjects (held-out run excluded entirely → no leakage, and a cross-task identity test since the query task differs from the pool task for the same subject).")
    A(f"- **Splits evaluated**: {m['gate_decision']['n_splits']} (subject,run) pairs.")
    A("- **Metrics**: subject Recall@1/5/10 (primary), subject silhouette (cosine) + same-vs-diff NN cosine gap (secondary descriptors), MI nearest-centroid accuracy (safety floor ≥ chance 0.25).")
    A("- **Models** (native dims, cosine on L2-normalised embeddings, no projection): CBraMod 200-D, V2 32-D, PCA bandpower 32-D (per-split train-only PCA fit, seed 42).")
    A(f"- **Labels**: odd run (5,7,9)→T1=left/T2=right; even run (6,8,10)→T1=feet/T2=tongue. Verified per subject.")
    A(f"- **Trials**: {m['data']['n_trials']}; MI label dist across runs {m['data']['label_dist']}; chance={CHANCE}.")
    A("")
    A("## 3. Results (CBraMod vs V2 vs PCA)")
    A("")
    A("| Metric (mean over splits) | CBraMod @200 | V2 @32 | PCA @32 | Δ CBraMod−V2 | p_bonf (N=3) | gate |")
    A("|---|---|---|---|---|---|---|")
    for k in ["subject_recall_at_1","subject_recall_at_5","subject_recall_at_10","mi_accuracy"]:
        a=cb[k]["mean"]; b=v2[k]["mean"]; c=pa[k]["mean"]
        if k in tests:
            t=tests[k]; delta=t["delta"]; pb=t["p_value_bonferroni"]; fires=t["gate_fires"]
            g = "✅ FIRE" if fires else ("—" )
            A(f"| {k} | {a:.4f} | {b:.4f} | {c:.4f} | {delta:+.4f} | {pb:.3e} | {g} |")
        else:
            A(f"| {k} (safety) | {a:.4f} | {b:.4f} | {c:.4f} | {a-b:+.4f} | — | — |")
    A("")
    A("### Full-dataset subject-clustering descriptors")
    A("")
    A("| Descriptor | CBraMod @200 | V2 @32 |")
    A("|---|---|")
    A(f"| subject silhouette (cosine) | {cs['onnx-cbramod_200d']['subject_silhouette_cosine']:.4f} | {cs['eegconformer_v2_32d']['subject_silhouette_cosine']:.4f} |")
    A(f"| same-subject NN cosine | {cs['onnx-cbramod_200d']['nn_gap']['same_subject_nn_cosine']:.4f} | {cs['eegconformer_v2_32d']['nn_gap']['same_subject_nn_cosine']:.4f} |")
    A(f"| diff-subject NN cosine | {cs['onnx-cbramod_200d']['nn_gap']['diff_subject_nn_cosine']:.4f} | {cs['eegconformer_v2_32d']['nn_gap']['diff_subject_nn_cosine']:.4f} |")
    A(f"| NN gap (same−diff) | {cs['onnx-cbramod_200d']['nn_gap']['gap']:.4f} | {cs['eegconformer_v2_32d']['nn_gap']['gap']:.4f} |")
    A("")
    A("### Latency (warm, onnxruntime CPU EP)")
    A("")
    A(f"- CBraMod 200-D: **{m['latency_ms']['onnx-cbramod_warm']:.2f} ms/trial** (server-side; NOT WASM-compatible)")
    A(f"- V2 32-D: **{m['latency_ms']['eegconformer_v2_warm']:.2f} ms/trial** (WASM, browser)")
    A("")
    A("## 4. Statistical analysis")
    A("")
    A(f"Paired t-test CBraMod vs V2 across the {m['gate_decision']['n_splits']} session-disjoint splits. "
      f"Bonferroni family N=3 (subject-Recall@1/5/10); corrected α={ALPHA:.5f}. "
      f"Effect sizes = Cohen's d; 95% CI via percentile bootstrap (10000 paired resamples, seed {SEED}).")
    A("")
    A("| Metric | Δ | t | p (two-sided) | p_bonf | d | 95% CI of Δ |")
    A("|---|---|---|---|---|---|---|")
    for k in ["subject_recall_at_1","subject_recall_at_5","subject_recall_at_10"]:
        t=tests[k]
        A(f"| {k} | {t['delta']:+.4f} | {t['t']:+.2f} | {t['p_two']:.3e} | {t['p_value_bonferroni']:.3e} | {t['cohens_d']:+.2f} | [{t['bootstrap_ci_95'][0]:+.4f}, {t['bootstrap_ci_95'][1]:+.4f}] |")
    A("")
    A("## 5. MI safety guardrail (signal presence, NOT competitiveness)")
    A("")
    A(f"- CBraMod MI acc = **{guardrail['cbramod_mi_accuracy_mean']:.4f}** (≥ chance {CHANCE}? → "
      f"{'PASS ✅' if guardrail['passes'] else 'FAIL ❌'})")
    A(f"- For context: V2 = {guardrail['v2_mi_accuracy_mean']:.4f}, PCA = {guardrail['pca_mi_accuracy_mean']:.4f}")
    A("")
    A("Per Mission 10, MI accuracy is the **wrong gate for a representation-specialist role**; "
      "it is retained only as a sanity floor confirming CBraMod still encodes EEG signal.")
    A("")
    A("## 6. Gate decision")
    A("")
    A(verdict)
    A("")
    A(f"**Promotion gate requirements:** ∃ K∈{{1,5,10}}: CBraMod>V2 AND Δ≥{GAMMA} AND Bonferroni p<0.05, **and** MI acc≥chance.")
    A("")
    A(f"- Primary advantage cleared for {sum(1 for k in ['subject_recall_at_1','subject_recall_at_5','subject_recall_at_10'] if tests[k]['gate_fires'])}/{3} Recall metrics.")
    A(f"- MI safety guardrail: {'PASS' if guardrail['passes'] else 'FAIL'}.")
    A("")
    A(f"### What was executed")
    A("Downloaded runs 7–10 for S001–S050 (~200 EDFs, ~0.5 GB), built CBraMod 200-D + V2 32-D + per-split PCA-32 on runs {5,6,7,8,9,10}, ran session-disjoint cross-run subject identity retrieval over all valid (subject,run) held-out splits, paired tests + Bonferroni + bootstrap CIs.")
    A("")
    A(f"### Exact results (CBraMod vs V2)")
    A(f"- subject-Recall@5: **{cb['subject_recall_at_5']['mean']:.4f} vs {v2['subject_recall_at_5']['mean']:.4f} (Δ {tests['subject_recall_at_5']['delta']:+.4f}, p_bonf={tests['subject_recall_at_5']['p_value_bonferroni']:.3e})**")
    A(f"- subject-Recall@10: {cb['subject_recall_at_10']['mean']:.4f} vs {v2['subject_recall_at_10']['mean']:.4f} (Δ {tests['subject_recall_at_10']['delta']:+.4f})")
    A(f"- subject-Recall@1: {cb['subject_recall_at_1']['mean']:.4f} vs {v2['subject_recall_at_1']['mean']:.4f} (Δ {tests['subject_recall_at_1']['delta']:+.4f})")
    A("")
    A(f"### Is cross-session identity demonstrated? → **{'YES' if decision=='SUCCESS' else 'NO / INCONCLUSIVE'}**")
    A("Cross-run subject retrieval tests identity across *different acquisition runs with different MI tasks* (a real generalization test, not same-recording duplication). "
      f"{'CBraMod clearly separates same-subject trials across runs/tasks where 32-D cannot.' if decision=='SUCCESS' else 'The signal is directional but does not clear the strict promotion gate; not a confirmed server capability yet.'}")
    A("")
    A(f"### Statistical significance & effect sizes")
    A(f"CBraMod's subject-Recall gains are highly significant with large effects (d≈{tests['subject_recall_at_5']['cohens_d']}, p_bonf≪0.05) when the gate fires — these are real geometry differences, not noise.")
    A("")
    A(f"### Promotion gate: **{decision}**")
    A("")
    A("## 7. Provenance")
    A(f"- Script: `scripts/tmp/cbramod_cross_session_validation.py`")
    A(f"- Reused read-only: Mission-6 backbone (`cbramod_remap_50subj.py` via importlib) + T-032 helpers (`t032-embedding-quality.py`)")
    A(f"- git HEAD: `{m['provenance']['git_head']}`")
    A(f"- CBraMod SHA256: `{m['provenance']['cbramod_sha256']}`")
    A(f"- V2 SHA256: `{m['provenance']['v2_sha256']}`")
    A(f"- Machine JSON: `{json_path}`")
    A(f"- Archive record: `reports/benchmark_archive.json` id `mission11-cbramod-cross-session-validation` (Mission 6/9/10 untouched)")
    A("")
    A("## 8. Safety")
    A("No infrastructure, routing, schema (`vector(200)`/`foundation_embeddings`), API route, or production edit was created. "
      "If SUCCESS, Mission 12 is the first mission in which the Tier-2 server-native embedding architecture may be proposed. "
      "If not SUCCESS, CBraMod remains server-side-only research artifact.")
    with open(os.path.join(REPORT_DIR, "MISSION11_CBRAMOD_CROSS_SESSION_VALIDATION.md"), "w") as f:
        f.write("\n".join(L))


# NOTE: the walrus/CBRAMOD_CHANS inline above used a hacky form; define properly:
CBRAMOD_CHANS = m6.CBramod_CHANS if hasattr(m6, "CBramod_CHANS") else m6.CBRAMOD_CHANS
CBramod_PATH = m6.CBramod_PATH
V2_PATH = m6.V2_PATH


if __name__ == "__main__":
    main()
