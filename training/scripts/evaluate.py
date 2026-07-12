"""T-010 — EEGConformer empirical validation on BCI-IV-2a holdout.

Reports:
  - intra/inter-class cosine similarity (mean ± std)
  - separation margin (inter - intra, higher is better)
  - recall@10 vs PCA baseline
  - embedding norms and feature variance

Mirrors the runtime benchmark in src/lib/ai/benchmark/index.ts so that
offline numbers are comparable to in-browser numbers. The output JSON is
also consumed as a CI fixture (see training/scripts/validate.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import Paths, load_config


def _capture_embeddings(model, X, device):
    import torch
    feats = {}
    hook = model.fc.register_forward_hook(lambda _m, _i, o: feats.__setitem__("e", o.detach()))
    try:
        with torch.no_grad():
            model(X.to(device))
    finally:
        hook.remove()
    return feats["e"].cpu().numpy()


def _recall_at_k(emb: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    hits = (y[topk] == y[:, None]).any(axis=1)
    return float(hits.mean())


def _cosine_matrix(emb: np.ndarray) -> np.ndarray:
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return n @ n.T


def _intra_inter_class_cosine(emb: np.ndarray, y: np.ndarray) -> dict:
    """Intra-class (same-label) and inter-class (different-label) cosine stats."""
    sim = _cosine_matrix(emb)
    np.fill_diagonal(sim, np.nan)  # exclude self-pairs

    classes = np.unique(y)
    intra_sims: list[float] = []
    inter_sims: list[float] = []

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            s = sim[i, j]
            if np.isnan(s):
                continue
            if y[i] == y[j]:
                intra_sims.append(float(s))
            else:
                inter_sims.append(float(s))

    intra = np.array(intra_sims) if intra_sims else np.array([0.0])
    inter = np.array(inter_sims) if inter_sims else np.array([0.0])
    return {
        "intra_mean": float(intra.mean()),
        "intra_std": float(intra.std()),
        "inter_mean": float(inter.mean()),
        "inter_std": float(inter.std()),
        "separation_margin": float(intra.mean() - inter.mean()),
        "n_intra_pairs": int(len(intra_sims)),
        "n_inter_pairs": int(len(inter_sims)),
    }


def _pca_baseline(emb: np.ndarray, y: np.ndarray, k: int = 10) -> dict:
    """PCA baseline: reduce embeddings to the same dim via PCA, then recall@k.

    This is a sanity check — if EEGConformer embeddings don't beat PCA on
    recall@10, the learned representation isn't adding value.
    """
    # Center and compute covariance.
    centered = emb - emb.mean(axis=0, keepdims=True)
    # SVD-based PCA (handles d > n gracefully).
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    target_dim = min(emb.shape[1], 32)
    reduced = centered @ Vt[:target_dim].T
    recall = _recall_at_k(reduced, y, k)
    return {
        "pca_dim": int(target_dim),
        "recall_at_k": recall,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)

    import torch
    from braindecode.models import EEGConformer

    data = np.load(paths.processed / "holdout.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"])
    y = data["y"]
    sig = cfg["signal"]
    m = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGConformer(
        n_outputs=m["n_classes"],
        n_chans=sig["channels"],
        n_times=sig["window_samples"],
        final_fc_length=m["final_fc_length"],
    ).to(device)
    model.load_state_dict(
        torch.load(paths.artefacts / "eegconformer.pt", map_location=device)
    )
    model.eval()

    emb = _capture_embeddings(model, X, device)
    assert emb.shape[1] == cfg["model"]["embedding_dim"], (
        f"embedding_dim mismatch: got {emb.shape[1]} expected {cfg['model']['embedding_dim']}"
    )

    # --- T-010 metrics ---
    cosine_stats = _intra_inter_class_cosine(emb, y)
    recall = _recall_at_k(emb, y, args.k)
    pca_baseline = _pca_baseline(emb, y, args.k)
    norms = np.linalg.norm(emb, axis=1)

    report = {
        "task": "eegconformer_empirical_validation",
        "dataset": cfg["name"],
        "n": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "embedding_dim": int(emb.shape[1]),
        "recall_at_k": {str(args.k): recall},
        "cosine_analysis": cosine_stats,
        "pca_baseline": pca_baseline,
        "beats_pca": recall > pca_baseline["recall_at_k"],
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "feature_variance_mean": float(emb.var(axis=0).mean()),
    }
    out = paths.artefacts / "evaluation_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"[evaluate] → {out}")


if __name__ == "__main__":
    main()
