"""Export Braindecode EEGConformer to ONNX for Neuro-Fabric.

Reference artifact-preparation workflow for the selected production EEG
foundation model. Produces an ONNX file consumable by
`registerBraindecodeEEGConformer({ artifact: { kind: 'url', url } })`.

Usage:
    python scripts/export_braindecode_eegconformer.py \
        --checkpoint path/to/eegconformer.pt \
        --out public/models/eegconformer.onnx

Production contract (see docs/audits/2026-06-17_braindecode-model-selection.md):
    - Channels:        22
    - Sample rate:     250 Hz
    - Window samples:  1000  (4 s)
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


def build_model(n_channels: int, n_times: int, n_classes: int) -> EEGConformer:
    return EEGConformer(
        n_outputs=n_classes,
        n_chans=n_channels,
        n_times=n_times,
        final_fc_length="auto",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=False,
                    help="Path to a trained .pt state_dict. If omitted, exports a randomly-initialised model (for smoke testing only).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", type=int, default=22)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model(args.channels, args.samples, args.classes)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.eval()

    wrapper = EEGConformerExportWrapper(model)
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


if __name__ == "__main__":
    main()