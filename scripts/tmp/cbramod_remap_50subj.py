#!/usr/bin/env python3
"""
Mission 6 (Next Model): CBraMod 19->22 channel remap study + 50-subject LOSO.

Decide whether CBraMod earns a SERVER-SIDE specialist role by comparing it fairly,
on the SAME 50 PhysioNet subjects/splits, against EEGConformer v2 (GA prod default,
22-ch) and the PCA bandpower baseline (22-ch), under the locked T-032 50-subject LOSO
protocol, with Bonferroni-corrected statistics.

Constraints (must NOT violate): V2 production path/rollout/.env untouched; no
retraining of any model; EEGPT/LaBraM/FEMBA/PCA untouched; CBraMod NOT deployed
(wasmCompatible:false). CBraMod artifact read-only (SHA c128ccfd...); V2 read-only (18644de1...).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import importlib.util
import urllib.request
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── Reuse the locked T-032 helpers (import, do NOT copy) ───────────────────────
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("t032", "scripts/t032-embedding-quality.py")
t032 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t032)

CBRAMOD_CHANS = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
                 "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
                 "FZ", "CZ", "PZ"]

CBramod_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"
CBramod_PATH = "public/models/cbramod-encoder.onnx"
V2_PATH = "public/models/eegconformer_finetuned.onnx"
DATA_DIR = os.environ.get("EEGMMIDB_DIR", os.path.join("/tmp", "eegmmidb"))
REPORT_DIR = "reports"
CACHE_PATH = os.path.join(REPORT_DIR, ".cbramod_50subj_cache.npz")
PHYSIONET_BASE = "https://physionet.org/files/eegmmidb/1.0.0"
SUBJECTS = list(range(1, 51))
RUNS = [5, 6]
N_COMPARISONS = 3


def log(msg):
    print(msg, flush=True)


# ── Data: download PhysioNet EEGMMIDB S001-S050 runs 5,6 (idempotent) ────────────

def _dl_one(subj, run):
    scode = f"S{subj:03d}"
    fname = f"{scode}R{run:02d}.edf"
    url = f"{PHYSIONET_BASE}/{scode}/{fname}"
    dst = os.path.join(DATA_DIR, scode, fname)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        return (scode, run, "cached")
    try:
        urllib.request.urlretrieve(url, dst)
        return (scode, run, "ok")
    except Exception as e:
        return (scode, run, f"fail:{e}")


def download_dataset():
    print(f"Downloading PhysioNet EEGMMIDB S001-S050 runs {RUNS} -> {DATA_DIR}", flush=True)
    total = len(SUBJECTS) * len(RUNS)
    present = sum(1 for s in SUBJECTS for r in RUNS
                  if os.path.exists(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf"))
                  and os.path.getsize(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")) > 1024)
    print(f"  already present: {present}/{total}", flush=True)
    if present == total:
        print("  dataset fully cached, skipping download.", flush=True)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_dl_one, s, r): (s, r) for s in SUBJECTS for r in RUNS}
        ok = 0
        for f in concurrent.futures.as_completed(futs):
            scode, run, status = f.result()
            if status == "ok":
                ok += 1
                print(f"    downloaded {scode}R{run:02d}", flush=True)
            elif status.startswith("fail"):
                print(f"    FAIL {scode}R{run:02d}: {status}", file=sys.stderr, flush=True)
    print(f"  downloaded {ok} new files; total now present: {present + ok}/{total}", flush=True)
    missing = [f"S{s:03d}R{r:02d}" for s in SUBJECTS for r in RUNS
               if not (os.path.exists(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf"))
                       and os.path.getsize(os.path.join(DATA_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")) > 1024)]
    if missing:
        raise RuntimeError(f"Missing EDF files after download: {missing[:5]}... ({len(missing)} total)")


# ── Preprocessing (per T-032 pipeline; channel selection only differs) ─────────

def select_channels(trial_data, source_ch_names, chan_list):
    src = {t032.normalize_ch_name(c): i for i, c in enumerate(source_ch_names)}
    selected = []
    for tc in chan_list:
        if tc in src:
            selected.append(trial_data[src[tc]].copy())
        else:
            selected.append(np.zeros(trial_data.shape[1], dtype=np.float32))
    return np.array(selected)


def _finalize_window(n, window_samples=t032.WINDOW_SAMPLES):
    if n.shape[1] >= window_samples:
        s = (n.shape[1] - window_samples) // 2
        n = n[:, s:s + window_samples]
    else:
        n = np.pad(n, ((0, 0), (0, window_samples - n.shape[1])), mode="constant")
    return n.astype(np.float32)


def preprocess_window(trial_data, source_ch_names, chan_list):
    sel = select_channels(trial_data, source_ch_names, chan_list)
    res = t032.resample_160_to_250(sel, 160.0)
    f = t032.bandpass_filter(res, t032.SAMPLE_RATE, t032.BANDPASS[0], t032.BANDPASS[1])
    n = t032.zscore_normalize(f.copy())
    return _finalize_window(n)


def preprocess_for_cbramod(trial_data, source_ch_names):
    return preprocess_window(trial_data, source_ch_names, CBRAMOD_CHANS)


# ── Embeddings ─────────────────────────────────────────────────────────────────

def cbramod_embeddings(session, windows_19):
    inp = session.get_inputs()[0]
    emb_out = next((o for o in session.get_outputs() if o.name == "embedding"),
                   session.get_outputs()[0])
    embs = []
    for w in windows_19:
        r = session.run([emb_out.name], {inp.name: w[np.newaxis, :, :].astype(np.float32)})[0]
        pooled = r.mean(axis=(1, 2)).flatten().astype(np.float32)  # [1,19,5,200]->[200] mean-tokens
        embs.append(pooled)
    embs = np.array(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs / norms, emb_out.name


def v2_embeddings(session, windows_22):
    inp = session.get_inputs()[0]
    emb_out = next((o for o in session.get_outputs() if o.name == "embedding"),
                   session.get_outputs()[0])
    embs = []
    for w in windows_22:
        r = session.run([emb_out.name], {inp.name: w[np.newaxis, :, :].astype(np.float32)})[0]
        embs.append(r.flatten().astype(np.float32))  # [32]
    embs = np.array(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs / norms, emb_out.name


def time_latency(session, windows, output_name, n_warm=3, n_meas=150):
    inp = session.get_inputs()[0]
    for _ in range(n_warm):
        session.run([output_name], {inp.name: windows[0][np.newaxis].astype(np.float32)})
    idxs = np.random.default_rng(42).integers(0, len(windows), size=min(n_meas, len(windows)))
    t0 = time.perf_counter()
    for i in idxs:
        session.run([output_name], {inp.name: windows[i][np.newaxis].astype(np.float32)})
    return float((time.perf_counter() - t0) / len(idxs) * 1000.0)


# ── Clean LOSO (50 folds; passes np arrays -- T-032 metric fns require arrays) ───

def loso_evaluate(emb, labels, subject_ids):
    """LOSO nearest-centroid (cosine) + Recall@K (train-only pool, no self-retrieval).

    emb: [N,D] np.ndarray already L2-normalised. labels/subject_ids: np/list."""
    subj_arr = np.asarray(subject_ids).tolist()
    lab_arr = np.asarray(labels).tolist()
    unique = sorted(set(int(s) for s in subj_arr))
    accs, f1s, r1, r5, r10 = [], [], [], [], []
    for subj in unique:
        test_mask = np.array(subj_arr) == subj
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        tr_emb = np.asarray(emb[train_mask])
        te_emb = np.asarray(emb[test_mask])
        tr_lab = [lab_arr[i] for i in range(len(lab_arr)) if train_mask[i]]
        te_lab = [lab_arr[i] for i in range(len(lab_arr)) if test_mask[i]]
        nc = t032.nearest_centroid_accuracy(tr_emb, tr_lab, te_emb, te_lab)   # np arrays
        rk = t032.recall_at_k(tr_emb, tr_lab, te_emb, te_lab)                # np arrays
        accs.append(nc["accuracy"]); f1s.append(nc["macro_f1"])
        r1.append(rk["recall_at_1"]); r5.append(rk["recall_at_5"]); r10.append(rk["recall_at_10"])
    return {"accuracy": accs, "macro_f1": f1s, "recall_at_1": r1, "recall_at_5": r5,
            "recall_at_10": r10,
            "per_fold_accuracy": [float(a) for a in accs], "per_fold_r1": [float(a) for a in r1],
            "n_folds": len(accs)}


def loso_pca_eval(bandpower, labels, subject_ids):
    """Per-fold train-only PCA(32) -> L2 -> nearest-centroid + Recall@K."""
    subj_arr = np.asarray(subject_ids).tolist()
    lab_arr = np.asarray(labels).tolist()
    unique = sorted(set(int(s) for s in subj_arr))
    from sklearn.decomposition import PCA
    accs, f1s, r1, r5, r10 = [], [], [], [], []
    for subj in unique:
        test_mask = np.array(subj_arr) == subj
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        pca = PCA(n_components=32, random_state=42)
        tr_feat = bandpower[train_mask]; te_feat = bandpower[test_mask]
        tr_emb = pca.fit_transform(tr_feat)        # fit TRAIN ONLY (no leakage)
        te_emb = pca.transform(te_feat)
        tr_emb = tr_emb / (np.linalg.norm(tr_emb, axis=1, keepdims=True) + 1e-9)
        te_emb = te_emb / (np.linalg.norm(te_emb, axis=1, keepdims=True) + 1e-9)
        tr_lab = [lab_arr[i] for i in range(len(lab_arr)) if train_mask[i]]
        te_lab = [lab_arr[i] for i in range(len(lab_arr)) if test_mask[i]]
        nc = t032.nearest_centroid_accuracy(tr_emb, tr_lab, te_emb, te_lab)
        rk = t032.recall_at_k(tr_emb, tr_lab, te_emb, te_lab)
        accs.append(nc["accuracy"]); f1s.append(nc["macro_f1"])
        r1.append(rk["recall_at_1"]); r5.append(rk["recall_at_5"]); r10.append(rk["recall_at_10"])
    return {"accuracy": accs, "macro_f1": f1s, "recall_at_1": r1, "recall_at_5": r5,
            "recall_at_10": r10,
            "per_fold_accuracy": [float(a) for a in accs], "per_fold_r1": [float(a) for a in r1],
            "n_folds": len(accs)}


def aggregate(per_fold, label):
    accs = per_fold["accuracy"]
    m, se, lo, hi = t032.mean_ci(accs)
    m_r1, _, r1_lo, r1_hi = t032.mean_ci(per_fold["recall_at_1"])
    m_r5, _, r5_lo, r5_hi = t032.mean_ci(per_fold["recall_at_5"])
    m_r10, _, r10_lo, r10_hi = t032.mean_ci(per_fold["recall_at_10"])
    return {"model": label, "loso": {
        "n_folds": per_fold["n_folds"],
        "mean_accuracy": float(m), "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "stderr_accuracy": float(se), "ci95_accuracy": [float(lo), float(hi)],
        "mean_macro_f1": float(np.mean(per_fold["macro_f1"])),
        "recall_at_1": {"mean": float(m_r1), "ci95": [float(r1_lo), float(r1_hi)]},
        "recall_at_5": {"mean": float(m_r5), "ci95": [float(r5_lo), float(r5_hi)]},
        "recall_at_10": {"mean": float(m_r10), "ci95": [float(r10_lo), float(r10_hi)]},
        "per_fold_accuracy": per_fold["per_fold_accuracy"],
    }}


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


def try_load_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        d = np.load(CACHE_PATH, allow_pickle=True)
        return {
            "windows_22": d["windows_22"], "windows_19": d["windows_19"],
            "bandpower": d["bandpower"], "labels": d["labels"],
            "subj_ids": d["subj_ids"], "cbramod_emb": d["cbramod_emb"],
            "v2_emb": d["v2_emb"], "cbramod_lat": float(d["cbramod_lat"]),
            "v2_lat": float(d["v2_lat"]),
        }
    except Exception as e:
        print(f"  cache load failed ({e}); recomputing.", flush=True)
        return None


def save_cache(windows_22, windows_19, bandpower, labels, subj_ids,
               cbramod_emb, v2_emb, cbramod_lat, v2_lat):
    np.savez_compressed(CACHE_PATH,
                        windows_22=windows_22, windows_19=windows_19,
                        bandpower=bandpower, labels=labels, subj_ids=subj_ids,
                        cbramod_emb=cbramod_emb, v2_emb=v2_emb,
                        cbramod_lat=cbramod_lat, v2_lat=v2_lat)
    print(f"  cache saved -> {CACHE_PATH}", flush=True)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print("=" * 72, flush=True)
    print("Mission 6: CBraMod 19->22 remap study + 50-subject LOSO", flush=True)
    print("=" * 72, flush=True)
    print(f"Start: {ts_start}", flush=True)

    assert sha256(CBramod_PATH) == CBramod_SHA, f"CBraMod SHA mismatch: {sha256(CBramod_PATH)}"
    assert sha256(V2_PATH) == V2_SHA, f"V2 SHA mismatch: {sha256(V2_PATH)}"
    print(f"Artifact provenance: CBraMod SHA {CBramod_SHA[:12]}... OK; V2 SHA {V2_SHA[:12]}... OK", flush=True)

    download_dataset()
    cache = try_load_cache()
    if cache is not None:
        print("Resuming from cache: ", flush=True)
        windows_22 = cache["windows_22"]; windows_19 = cache["windows_19"]
        bandpower = cache["bandpower"]; labels = cache["labels"]
        subj_ids = cache["subj_ids"]; cbramod_emb = cache["cbramod_emb"]
        v2_emb = cache["v2_emb"]; cbramod_lat = cache["cbramod_lat"]; v2_lat = cache["v2_lat"]
        print(f"  cached trials: {len(windows_22)}", flush=True)
    else:
        print("Loading subjects (T-032 loader)...", flush=True)
        subjects_data = t032.load_physionet_subjects(SUBJECTS, runs=RUNS)
        if not subjects_data:
            raise RuntimeError("No subjects loaded.")
        print(f"Loaded {len(subjects_data)} subjects.", flush=True)

        print("Preprocessing (22-ch for V2/PCA, 19-ch for CBraMod, identical pipeline)...", flush=True)
        windows_22, windows_19, labels, subj_ids = [], [], [], []
        for subj_id in sorted(subjects_data.keys()):
            sd = subjects_data[subj_id]
            for i, trial in enumerate(sd["trials"]):
                w22 = t032.preprocess_for_eegconformer(trial, sd["ch_names"])  # [22,1000]
                w19 = preprocess_for_cbramod(trial, sd["ch_names"])              # [19,1000]
                windows_22.append(w22); windows_19.append(w19)
                labels.append(sd["labels"][i]); subj_ids.append(subj_id)
            print(f"  {subj_id}: {len(sd['trials'])} trials", flush=True)
        windows_22 = np.array(windows_22); windows_19 = np.array(windows_19)
        labels = np.array(labels); subj_ids = np.array(subj_ids)
        print(f"  total trials: {len(windows_22)} | label dist: {np.bincount(labels, minlength=4).tolist()}", flush=True)

        print("Computing bandpower features (110 = 5 bands x 22 ch) for PCA...", flush=True)
        bandpower = np.array([t032.bandpower_features(w) for w in windows_22])

        print("CBraMod ONNX inference (19-ch -> 200-D mean-tokens)...", flush=True)
        cbramod_sess = ort.InferenceSession(CBramod_PATH, providers=["CPUExecutionProvider"])
        cbramod_emb, cbo = cbramod_embeddings(cbramod_sess, windows_19)
        cbramod_lat = time_latency(cbramod_sess, windows_19, cbo)
        print(f"  CBraMod: {cbramod_emb.shape}, latency={cbramod_lat:.2f} ms/trial", flush=True)

        print("EEGConformer v2 ONNX inference (22-ch -> 32-D)...", flush=True)
        v2_sess = ort.InferenceSession(V2_PATH, providers=["CPUExecutionProvider"])
        v2_emb, v2o = v2_embeddings(v2_sess, windows_22)
        v2_lat = time_latency(v2_sess, windows_22, v2o)
        print(f"  V2: {v2_emb.shape}, latency={v2_lat:.2f} ms/trial", flush=True)

        save_cache(windows_22, windows_19, bandpower, labels, subj_ids,
                   cbramod_emb, v2_emb, cbramod_lat, v2_lat)

    # 5. LOSO evaluation (clean 50-fold; train-only PCA fit per fold)
    print("LOSO evaluation (50 folds)...", flush=True)
    pf_cbramod = loso_evaluate(cbramod_emb, labels, subj_ids)
    pf_v2 = loso_evaluate(v2_emb, labels, subj_ids)
    pf_pca = loso_pca_eval(bandpower, labels, subj_ids)
    res_cbramod = aggregate(pf_cbramod, "onnx-cbramod")
    res_v2 = aggregate(pf_v2, "braindecode-eegconformer-prod-v2")
    res_pca = aggregate(pf_pca, "pca-bandpower")
    res_cbramod["loso"]["n_trials"] = res_v2["loso"]["n_trials"] = res_pca["loso"]["n_trials"] = len(labels)
    print(f"  CBraMod  acc={res_cbramod['loso']['mean_accuracy']:.4f} "
          f"R@1={res_cbramod['loso']['recall_at_1']['mean']:.4f} "
          f"R@10={res_cbramod['loso']['recall_at_10']['mean']:.4f}", flush=True)
    print(f"  V2       acc={res_v2['loso']['mean_accuracy']:.4f} "
          f"R@1={res_v2['loso']['recall_at_1']['mean']:.4f} "
          f"R@10={res_v2['loso']['recall_at_10']['mean']:.4f}", flush=True)
    print(f"  PCA      acc={res_pca['loso']['mean_accuracy']:.4f} "
          f"R@1={res_pca['loso']['recall_at_1']['mean']:.4f} "
          f"R@10={res_pca['loso']['recall_at_10']['mean']:.4f}", flush=True)

    # 6. Class separability (Fisher) on full-dataset embeddings
    print("Class separability (Fisher, full dataset)...", flush=True)
    cs_cbramod = t032.class_separability(cbramod_emb, labels.tolist())
    cs_v2 = t032.class_separability(v2_emb, labels.tolist())
    from sklearn.decomposition import PCA
    pca_full = PCA(n_components=32, random_state=42)
    pca_emb_full = pca_full.fit_transform(bandpower)
    pca_emb_full = pca_emb_full / (np.linalg.norm(pca_emb_full, axis=1, keepdims=True) + 1e-9)
    cs_pca = t032.class_separability(pca_emb_full, labels.tolist())

    # 7. Statistical comparison (Bonferroni-corrected)
    print("Statistical comparison (paired t-test + Bonferroni)...", flush=True)
    pairs = [("onnx-cbramod", "braindecode-eegconformer-prod-v2"),
             ("onnx-cbramod", "pca-bandpower"),
             ("braindecode-eegconformer-prod-v2", "pca-bandpower")]
    pf = {"onnx-cbramod": pf_cbramod, "braindecode-eegconformer-prod-v2": pf_v2, "pca-bandpower": pf_pca}
    comps = []
    for a, b in pairs:
        ka = min(len(pf[a]["per_fold_accuracy"]), len(pf[b]["per_fold_accuracy"]))
        aa = np.array(pf[a]["per_fold_accuracy"][:ka]); bb = np.array(pf[b]["per_fold_accuracy"][:ka])
        t, p, d = t032.paired_t_test(aa.tolist(), bb.tolist())
        bonf_p = min(p * N_COMPARISONS, 1.0)
        ra = np.array(pf[a]["per_fold_r1"][:ka]); rb = np.array(pf[b]["per_fold_r1"][:ka])
        _, p_r1, d_r1 = t032.paired_t_test(ra.tolist(), rb.tolist())
        bonf_p_r1 = min(p_r1 * N_COMPARISONS, 1.0)
        comps.append({
            "model_a": a, "model_b": b,
            "metric": "loso_accuracy",
            "mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
            "delta_a_minus_b": float(aa.mean() - bb.mean()),
            "t_statistic": t, "p_value": p, "p_value_bonferroni": bonf_p,
            "significant_bonferroni": bool(bonf_p < 0.05),
            "cohens_d": d,
            "recall_at_1_delta": float(ra.mean() - rb.mean()),
            "recall_at_1_p_bonferroni": bonf_p_r1,
            "n_folds": int(ka),
        })
    for c in comps:
        sig = "SIG" if c["significant_bonferroni"] else "ns"
        print(f"  {c['model_a']} vs {c['model_b']}: Δ={c['delta_a_minus_b']:+.4f} "
              f"p={c['p_value']:.3e} (Bonf p={c['p_value_bonferroni']:.3e}) {sig} d={c['cohens_d']:+.3f}", flush=True)

    # 8. Decision (per MODEL_STRATEGY evidence gate)
    cb = res_cbramod["loso"]["mean_accuracy"]; v2 = res_v2["loso"]["mean_accuracy"]; pc = res_pca["loso"]["mean_accuracy"]
    cb_v2 = next(c for c in comps if c["model_a"] == "onnx-cbramod" and c["model_b"] == "braindecode-eegconformer-prod-v2")
    cb_pca = next(c for c in comps if c["model_a"] == "onnx-cbramod" and c["model_b"] == "pca-bandpower")
    earns = (cb >= pc and cb >= v2 and cb_v2["significant_bonferroni"] and cb_pca["significant_bonferroni"])
    decision = ("EARN server-side specialist role" if earns
                else "DO NOT promote/route CBraMod (negative result)")

    # PCA latency estimate (cheap: fit on a representative train batch, 150 iters).
    t0 = time.perf_counter()
    for _ in range(150):
        PCA(n_components=32, random_state=42).fit_transform(bandpower[:250])
    pca_lat = (time.perf_counter() - t0) / 150 * 1000

    # 9. Assemble + write results
    ts_end = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = {
        "experiment_id": "MISSION6-CBRAMOD-REMAP-50SUBJ",
        "experiment_name": "CBraMod 19->22 channel remap study + 50-subject LOSO validation",
        "timestamp_start": ts_start, "timestamp_end": ts_end,
        "mission": "Mission 6 (Next Model Mission)",
        "author": "zcode-agent",
        "git_head": git_head(),
        "description": "CBraMod 19->22 remap study: CBraMod (native 19-ch, 200-D mean-tokens) vs EEGConformer v2 (22-ch, 32-D) vs PCA bandpower (32-D), on the locked T-032 50-subject LOSO protocol, Bonferroni-corrected.",
        "remap_design": {
            "option": "A: native-montage from a shared raw source",
            "assumption": "CBraMod and V2 are FIXED-shape ONNX (19 and 22 inputs); retraining forbidden. Each model runs its NATIVE montage from the SAME 64-channel PhysioNet raw trial. CBraMod selects its 19 native channels (all present in PhysioNet 64-ch); V2 selects the 22-channel prod subset; PCA uses 110 bandpower features (5x22). The 19<->22 channel-space gap (10 shared / 7 CBraMod-only / 12 prod-only) is documented, NOT interpolated/zero-filled.",
            "shared_channels": 10,
            "cbramod_only_channels": ["O1","O2","F7","F8","FZ","CZ","PZ"],
            "prod_only_channels": ["F5","F6","F1","F2","FC5","FC6","FC3","FC4","C5","C6","P5","P6"],
            "zero_filled_channels": [],
            "interpolated_channels": [],
            "fairness_note": "Each model uses its native channel count; metrics are cosine-based (dimension-agnostic). CBraMod=200-D (mean-tokens), V2/PCA=32-D.",
        },
        "data": {"dataset": "PhysioNet EEGMMIDB 1.0.0", "subjects": "S001-S050 (50)", "runs": RUNS,
                 "classes": t032.CLASS_NAMES, "n_classes": t032.N_CLASSES, "chance_level": t032.CHANCE_LEVEL,
                 "data_dir": DATA_DIR, "n_trials": int(len(labels))},
        "preprocessing": {"channels_cbramod": CBRAMOD_CHANS, "channels_v2_prod": t032.EEGCONFORMER_CHANS,
                          "sample_rate_hz": t032.SAMPLE_RATE, "window_samples": t032.WINDOW_SAMPLES,
                          "bandpass_hz": list(t032.BANDPASS), "normalization": "z-score per channel",
                          "resampling": "160->250 Hz (linear interpolation)",
                          "pca_features": "5 bands x 22 channels = 110", "pca_components": 32,
                          "pca_fit": "train-only, per LOSO fold (no leakage)",
                          "protocol": "LOSO 50 folds, nearest-centroid (cosine), Recall@K train-only pool, no self-retrieval"},
        "artifacts": {
            "cbramod": {"path": CBramod_PATH, "sha256": CBramod_SHA,
                        "input": "eeg[1,19,1000]", "output": "embedding[1,19,5,200]->200 (mean-tokens)",
                        "wasm_compatible": False, "dims": 200},
            "eegconformer_v2": {"path": V2_PATH, "sha256": V2_SHA,
                                "input": "input[1,22,1000]", "output": "embedding[1,32]",
                                "wasm_compatible": True, "dims": 32, "note": "production GA default; read-only"},
            "pca": {"description": "sklearn PCA(32) on 110 bandpower features, random_state=42, train-only per fold", "dims": 32},
        },
        "results": {"onnx-cbramod": res_cbramod, "braindecode-eegconformer-prod-v2": res_v2, "pca-bandpower": res_pca},
        "class_separability": {"onnx-cbramod": cs_cbramod, "braindecode-eegconformer-prod-v2": cs_v2, "pca-bandpower": cs_pca},
        "latency_ms": {"onnx-cbramod_warm": cbramod_lat, "eegconformer_v2_warm": v2_lat,
                       "pca_bandpower_estimate": pca_lat,
                       "engine": "onnxruntime CPU EP (server-side; CBraMod NOT WASM-compatible)"},
        "statistical_comparisons": comps,
        "bonferroni": {"n_comparisons": N_COMPARISONS, "corrected_alpha": 0.05 / N_COMPARISONS},
        "decision": {"cbramod_accuracy": float(cb), "v2_accuracy": float(v2), "pca_accuracy": float(pc),
                     "evidence_gate": "CBraMod acc >= PCA AND >= V2, Bonferroni p<0.05 on 50-subject LOSO",
                     "cbramod_vs_v2_significant": bool(cb_v2["significant_bonferroni"]),
                     "cbramod_vs_pca_significant": bool(cb_pca["significant_bonferroni"]),
                     "result": decision},
        "constraints": {"v2_production": "UNCHANGED (path/rollout/.env=ga, artifact read-only)",
                        "retrain": "NONE (CBraMod, V2, and all other models untouched)",
                        "deploy": "CBraMod NOT deployed (wasmCompatible:false); server-side specialist role gated on this study"},
    }
    out_path = os.path.join(REPORT_DIR, "cbramod_remap_50subj_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written: {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY (50-subj LOSO, Bonferroni-corrected, paired t-test)", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Model':<24}{'Acc':>9}{'R@1':>9}{'R@10':>9}{'Fisher':>9}{'lat(ms)':>9}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'CBraMod@19(200D)':<24}{cb:>9.4f}{res_cbramod['loso']['recall_at_1']['mean']:>9.4f}{res_cbramod['loso']['recall_at_10']['mean']:>9.4f}{cs_cbramod['fisher_score']:>9.4f}{cbramod_lat:>9.2f}", flush=True)
    print(f"{'EEGConformer-v2@22':<24}{v2:>9.4f}{res_v2['loso']['recall_at_1']['mean']:>9.4f}{res_v2['loso']['recall_at_10']['mean']:>9.4f}{cs_v2['fisher_score']:>9.4f}{v2_lat:>9.2f}", flush=True)
    print(f"{'PCA@22':<24}{pc:>9.4f}{res_pca['loso']['recall_at_1']['mean']:>9.4f}{res_pca['loso']['recall_at_10']['mean']:>9.4f}{cs_pca['fisher_score']:>9.4f}{'(fast)':>9}", flush=True)
    print("-" * 72, flush=True)
    print(f"CBraMod vs V2   : Δ={cb_v2['delta_a_minus_b']:+.4f}  p={cb_v2['p_value']:.3e} (Bonf p={cb_v2['p_value_bonferroni']:.3e}) {'SIG' if cb_v2['significant_bonferroni'] else 'ns'} d={cb_v2['cohens_d']:+.3f}", flush=True)
    print(f"CBraMod vs PCA  : Δ={cb_pca['delta_a_minus_b']:+.4f}  p={cb_pca['p_value']:.3e} (Bonf p={cb_pca['p_value_bonferroni']:.3e}) {'SIG' if cb_pca['significant_bonferroni'] else 'ns'} d={cb_pca['cohens_d']:+.3f}", flush=True)
    print(f"DECISION: {decision}", flush=True)
    print(f"Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
