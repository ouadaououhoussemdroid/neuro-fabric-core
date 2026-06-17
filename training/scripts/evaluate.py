"""Embedding-quality evaluation: cosine recall@k + embedding stats.

Mirrors the runtime benchmark in src/lib/ai/benchmark/index.ts so that
offline numbers are comparable to in-browser numbers.
"""
from __future__ import annotations

import argparse
import json

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
    # cosine similarity matrix
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = n @ n.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    hits = (y[topk] == y[:, None]).any(axis=1)
    return float(hits.mean())


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
    X = torch.from_numpy(data["X"]); y = data["y"]
    sig = cfg["signal"]; m = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGConformer(
        n_outputs=m["n_classes"],
        n_chans=sig["channels"],
        n_times=sig["window_samples"],
        final_fc_length=m["final_fc_length"],
    ).to(device)
    model.load_state_dict(torch.load(paths.artefacts / "eegconformer.pt", map_location=device))
    model.eval()

    emb = _capture_embeddings(model, X, device)
    assert emb.shape[1] == cfg["model"]["embedding_dim"], (
        f"embedding_dim mismatch: got {emb.shape[1]} expected {cfg['model']['embedding_dim']}"
    )
    norms = np.linalg.norm(emb, axis=1)
    report = {
        "n": int(len(y)),
        "embedding_dim": int(emb.shape[1]),
        "recall_at_k": {str(args.k): _recall_at_k(emb, y, args.k)},
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