"""Export Braindecode models to ONNX for Neuro-Fabric.

T-015: extended to support EEGNetv4, ShallowFBCSPNet, and Deep4Net alongside
EEGConformer via the --architecture flag.

Reference artifact-preparation workflow for the selected production EEG
foundation model. Produces an ONNX file consumable by
`registerBraindecodeEEGConformer({ artifact: { kind: 'url', url } })`.

Usage:
    # EEGConformer (default)
    python scripts/export_braindecode_eegconformer.py \
        --architecture EEGConformer \
        --checkpoint path/to/eegconformer.pt \
        --out public/models/eegconformer.onnx

    # EEGNetv4
    python scripts/export_braindecode_eegconformer.py \
        --architecture EEGNetv4 \
        --out public/models/eegnetv4.onnx

    # ShallowFBCSPNet
    python scripts/export_braindecode_eegconformer.py \
        --architecture ShallowFBCSPNet \
        --out public/models/shallowfbcspnet.onnx

    # Deep4Net
    python scripts/export_braindecode_eegconformer.py \
        --architecture Deep4Net \
        --out public/models/deep4net.onnx

Production contract (see docs/audits/2026-06-17_braindecode-model-selection.md):
    - Channels:        22
    - Sample rate:     250 Hz
    - Window samples:  1000  (4 s, EEGConformer)
    - Embedding dim:   32    (attention-pooled features)
    - Opset:           17
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from braindecode.models import EEGConformer


class EEGConformerExportWrapper(nn.Module):
    """Expose ('embedding', 'logits') as named ONNX outputs.

    EEGConformer's forward returns logits only; Neuro-Fabric's similarity
    search needs the 32-dim feature vector produced by the `fc`
    (`_FullyConnected`) module just before the classification head.

    We replicate EEGConformer.forward inline (mirroring braindecode's source:
    unsqueeze → patch_embedding → transformer → fc → final_layer) and return
    *both* tensors directly. Returning them as part of the function's tuple —
    rather than smuggling the embedding out via a forward-hook side-effect —
    is the only shape that survives `torch.onnx.export` with
    `do_constant_folding=True`. The previous hook-based wrapper produced an
    ONNX graph where the `embedding` output was wired to a folded/aliased
    intermediate, yielding PyTorch↔ORT cosine ≈ 0.30 instead of >0.999.
    """

    def __init__(self, model: EEGConformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Mirror braindecode.models.eegconformer.EEGConformer.forward so the
        # tracer sees a single, side-effect-free path to each output.
        x = torch.unsqueeze(x, dim=1)
        x = self.model.patch_embedding(x)
        feature = self.model.transformer(x)
        embedding = self.model.fc(feature)
        logits = self.model.final_layer(embedding)
        return embedding, logits


def build_model(n_channels: int, n_times: int, n_classes: int, architecture: str = "EEGConformer"):
    """Build a Braindecode model by architecture name."""
    from braindecode.models import EEGConformer, EEGNetv4, ShallowFBCSPNet, Deep4Net

    arch = architecture.lower()
    if arch == "eegconformer":
        return EEGConformer(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            final_fc_length="auto",
        )
    elif arch == "eegnetv4":
        return EEGNetv4(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            final_fc_length="auto",
        )
    elif arch == "shallowfbcspnet":
        return ShallowFBCSPNet(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            n_filters_time=40,
            filter_time_length=25,
            final_fc_length="auto",
        )
    elif arch == "deep4net":
        return Deep4Net(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            n_filters_time=25,
            filter_time_length=10,
            final_fc_length="auto",
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", type=str, default="EEGConformer",
                    choices=["EEGConformer", "EEGNetv4", "ShallowFBCSPNet", "Deep4Net"],
                    help="Braindecode architecture to export (T-015 zoo registration).")
    ap.add_argument("--checkpoint", type=Path, required=False,
                    help="Path to a trained .pt state_dict. If omitted, exports a randomly-initialised model (for smoke testing only).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", type=int, default=22)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model(args.channels, args.samples, args.classes, args.architecture)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.eval()

    wrapper = EEGConformerExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, args.channels, args.samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out.as_posix(),
        input_names=["input"],
        output_names=["embedding", "logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # T-023: simplify + shape-infer the exported graph for smaller artefacts
    # and faster cold starts. Falls back to the raw export if the tools are
    # not installed (onnx-simplifier and onnxoptimizer are optional deps).
    _simplify_and_optimize(args.out)

    # Validate the exported graph.
    import onnx
    onnx.checker.check_model(onnx.load(args.out.as_posix()))

    # Smoke-test parity with onnxruntime.
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out.as_posix(), providers=["CPUExecutionProvider"])
    ort_emb, ort_logits = sess.run(None, {"input": dummy.numpy()})
    with torch.no_grad():
        pt_emb, pt_logits = wrapper(dummy)
    cos = torch.nn.functional.cosine_similarity(
        pt_emb.flatten().unsqueeze(0),
        torch.from_numpy(ort_emb).flatten().unsqueeze(0),
    ).item()
    print(f"ONNX export OK: {args.out}  (PyTorch↔ORT cosine = {cos:.6f})")
    assert cos > 0.999, "PyTorch / ONNX parity check failed"


def _simplify_and_optimize(path: Path) -> None:
    """Apply onnx-simplifier + onnxoptimizer to reduce graph size.

    Idempotent: if neither package is installed, the raw export is kept.
    """
    import onnx

    original_size = path.stat().st_size

    # Step 1: onnx-simplifier (constant folding + shape inference).
    try:
        from onnxsim import simplify
        model = onnx.load(path.as_posix())
        simplified, check = simplify(model)
        if check:
            onnx.save(simplified, path.as_posix())
            print(f"[export] onnx-simplifier: {original_size} → {path.stat().st_size} bytes")
    except ImportError:
        print("[export] onnx-simplifier not installed, skipping simplification")
    except Exception as e:
        print(f"[export] onnx-simplifier failed: {e}")

    # Step 2: onnxoptimizer (graph-level optimizations).
    try:
        import onnxoptimizer
        model = onnx.load(path.as_posix())
        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, path.as_posix())
        print(f"[export] onnxoptimizer: applied {len(passes)} passes → {path.stat().st_size} bytes")
    except ImportError:
        print("[export] onnxoptimizer not installed, skipping optimization")
    except Exception as e:
        print(f"[export] onnxoptimizer failed: {e}")


if __name__ == "__main__":
    main()