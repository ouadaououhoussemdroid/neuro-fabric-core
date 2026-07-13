"""T-018 — Saliency over EEGConformer (Captum).

Computes per-window integrated-gradients attribution maps for the
EEGConformer embedding head, caches them as .npz keyed by the artefact
hash (from T-009's manifest), and emits a JSON sidecar consumable by
the TS-side `/embeddings` route for topomap overlays.

Usage:
    python scripts/compute_saliency.py \
        --checkpoint training/artefacts/eegconformer-bciiv2a-v1/eegconformer.pt \
        --onnx public/models/eegconformer.onnx \
        --out training/artefacts/eegconformer-bciiv2a-v1/saliency.npz

Dependencies (already in training/requirements.txt):
    torch, braindecode, numpy
    Additional: captum (pip install captum)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Captum saliency for EEGConformer")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to eegconformer.pt")
    ap.add_argument("--onnx", type=Path, required=True, help="Path to eegconformer.onnx (for hash)")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz path")
    ap.add_argument("--channels", type=int, default=22)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=5, help="Number of random windows to attribute")
    ap.add_argument("--ig-steps", type=int, default=64, help="Integrated gradients steps")
    return ap.parse_args()


def _artefact_hash(path: Path) -> str:
    """SHA-256 of the ONNX artefact (matches T-009 manifest)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = _parse_args()

    import torch
    from braindecode.models import EEGConformer
    from captum.attr import IntegratedGradients

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGConformer(
        n_outputs=4,
        n_chans=args.channels,
        n_times=args.samples,
        final_fc_length="auto",
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # We attribute w.r.t. the embedding output (model.fc output), not the
    # logits, so the saliency map reflects what the similarity search sees.
    def forward_embedding(x):
        x = torch.unsqueeze(x, dim=1)
        x = model.patch_embedding(x)
        feature = model.transformer(x)
        embedding = model.fc(feature)
        return embedding

    ig = IntegratedGradients(forward_embedding)

    # Generate deterministic random windows.
    rng = np.random.default_rng(42)
    attributions = []
    for i in range(args.n_samples):
        window = torch.from_numpy(
            rng.standard_normal((1, args.channels, args.samples)).astype(np.float32)
        ).to(device)

        # Target: attribute towards the sum of all embedding dimensions
        # (a global "what matters for this embedding" map).
        target = torch.zeros(1, 32, device=device)
        attr, _ = ig.attribute(window, target=target, return_convergence_delta=True, n_steps=args.ig_steps)
        # attr shape: [1, channels, samples] — absolute value for saliency.
        attr_map = attr.abs().squeeze(0).cpu().numpy()  # [C, T]
        attributions.append(attr_map)

    attributions = np.array(attributions, dtype=np.float32)  # [n_samples, C, T]
    mean_attr = attributions.mean(axis=0)  # [C, T]

    artefact_hash = _artefact_hash(args.onnx)

    # Save the attribution maps.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        attributions=attributions,
        mean_attribution=mean_attr,
        artefact_hash=artefact_hash,
        n_samples=args.n_samples,
        channels=args.channels,
        samples=args.samples,
    )

    # Emit a JSON sidecar for the TS route.
    sidecar = {
        "artefact_hash": artefact_hash,
        "saliency_path": str(args.out.relative_to(REPO_ROOT)),
        "n_samples": args.n_samples,
        "channels": args.channels,
        "samples": args.samples,
        # Per-channel mean saliency (for topomap overlay).
        "channel_saliency": mean_attr.mean(axis=1).tolist(),
    }
    sidecar_path = args.out.with_suffix(".json")
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[saliency] artefact_hash={artefact_hash[:16]}…")
    print(f"[saliency] wrote {args.out} ({attributions.shape})")
    print(f"[saliency] sidecar → {sidecar_path}")


if __name__ == "__main__":
    main()
