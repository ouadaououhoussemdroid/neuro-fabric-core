#!/usr/bin/env python3
"""T-035: WASM-compatible ONNX re-export of the v2 EEGConformer checkpoint.

Does NOT retrain. Does NOT modify production. Preserves exact v2 checkpoint
weights. Replaces braindecode's MultiHeadAttention torch.einsum calls with
equivalent torch.matmul + transpose operations so the ONNX graph contains no
Einsum ops and is genuinely compatible with onnxruntime-web WASM.

Pipeline:
  1. Load v2 checkpoint into EEGConformer.
  2. Replace all MultiHeadAttention instances with WASM-compatible variants.
  3. Verify PyTorch forward parity (original vs modified attention).
  4. Export to ONNX via EEGConformerWithEmbedding wrapper (same pattern as
     scripts/export_braindecode_eegconformer.py).
  5. Verify PyTorch → ONNX numerical parity (>0.999 cosine).
  6. Verify ONNX graph against the project's WASM compatibility blocklist.
  7. Verify weight preservation: compare new ONNX weights to the existing
     production v2 ONNX (cosine similarity ≥ 0.9999 for all weight tensors).
  8. Run the corrected T-034 50-subject LOSO evaluation.
  9. Compare against the existing v2 result (Acc 0.3250, Fisher 0.0072).

Usage:
    python scripts/t035-reexport-v2-wasm.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnx.numpy_helper as nh
import onnxruntime as ort
from onnx import helper

# ─── Fix braindecode/moabb import ─────────────────────────────────────────────────

import moabb.datasets as mds
if not hasattr(mds, "BNCI2014001"):
    mds.BNCI2014001 = mds.BNCI2014_001
if not hasattr(mds, "HGD"):
    mds.HGD = mds.PhysionetMI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braindecode.models import EEGConformer
from braindecode.modules import MultiHeadAttention

# Import the T-033/T-034 evaluation functions for LOSO
T033_SCRIPT = REPO_ROOT / "scripts" / "t033-embedding-dimension-ablation.py"
import importlib.util

def _load_t033():
    spec = importlib.util.spec_from_file_location("t033_eval", str(T033_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ─── WASM compatibility constants ───────────────────────────────────────────────

# Project's WASM blocker set (from forensic investigation and registry.ts):
# {DFT, ReduceL2, FFT, Complex, GlobalAveragePool, Flatten, Einsum}
WASM_BLOCKERS = {"DFT", "ReduceL2", "FFT", "Complex", "GlobalAveragePool", "Flatten", "Einsum"}

# Also check the full WASM op support from onnxruntime-web docs
# (these ops are NOT supported in WASM execution provider):
WASM_OP_BLOCKLIST = WASM_BLOCKERS | {
    "Asin", "Acosh", "Acos", "Atanh", "Atan", "Cosh", "Sinh", "Tan",
    "ScatterElements", "ScatterND", "GatherND", "Softplus", "Log", "Exp",
    "Sqrt", "Pow", "Sub", "Div", "Reciprocal",  # some of these might be OK actually
}
# Use the conservative, project-aligned blocklist:
WASM_OP_BLOCKLIST = WASM_BLOCKERS

# ─── WASM-compatible MultiHeadAttention ─────────────────────────────────────────


class WASMMultiHeadAttention(nn.Module):
    """MultiHeadAttention that replaces torch.einsum with torch.matmul + transpose.

    braindecode's MultiHeadAttention uses torch.einsum for attention score
    computation and context aggregation, producing ONNX Einsum ops that are
    unsupported by onnxruntime-web WASM. This subclass replaces those with
    explicitly equivalent MatMul + Transpose operations.

    The parameter weights are identical (same Linear layers); only the forward
    computation differs in its op decomposition.
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        from einops.layers.torch import Rearrange
        self.rearrange_stack = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.rearrange_unstack = Rearrange("b h n d -> b n (h d)")

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        queries = self.rearrange_stack(self.queries(x))   # [B, h, q, d]
        keys = self.rearrange_stack(self.keys(x))          # [B, h, k, d]
        values = self.rearrange_stack(self.values(x))      # [B, h, v, d]

        # Replace: torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        # With: matmul(queries, keys.transpose(-2, -1))
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # [B, h, q, k]

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # Replace: torch.einsum("bhal, bhlv -> bhav", att, values)
        # With: matmul(att, values)
        out = torch.matmul(att, values)  # [B, h, q, v]

        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out


def build_wasm_compatible_model():
    """Build an EEGConformer with all MultiHeadAttention modules replaced."""
    from braindecode.models import EEGConformer

    model = EEGConformer(
        n_outputs=4,
        n_chans=22,
        n_times=1000,
        final_fc_length="auto",
        drop_prob=0.5,
        att_drop_prob=0.5,
    )
    # Replace all MultiHeadAttention instances with WASM-compatible versions
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            # Create replacement with same config
            emb_size = module.emb_size
            num_heads = module.num_heads
            dropout = module.att_drop.p
            replacement = WASMMultiHeadAttention(emb_size, num_heads, dropout)
            # Copy all parameters
            replacement.load_state_dict(module.state_dict())
            # Replace in parent
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)

    return model


# ─── EEGConformerWithEmbedding wrapper (same as T-034 training script) ──────────

class EEGConformerWithEmbedding(nn.Module):
    """Expose ('embedding', 'logits') as named ONNX outputs.

    Forward path: unsqueeze → patch_embedding → transformer → fc → final_layer.
    Mirrors braindecode's EEGConformer.forward inline so the tracer sees a
    single, side-effect-free path to each output.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, dim=1)      # [B, 1, 22, 1000]
        x = self.model.patch_embedding(x)  # [B, 61, 40]
        feature = self.model.transformer(x)  # [B, 61, 40]
        embedding = self.model.fc(feature)   # [B, 32]
        logits = self.model.final_layer(embedding)  # [B, 4]
        return embedding, logits


# ─── ONNX helpers ────────────────────────────────────────────────────────────────


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_onnx_ops(model_path):
    """Return set of op types used in the ONNX model."""
    m = onnx.load(model_path)
    return {n.op_type for n in m.graph.node}


# ─── Main re-export pipeline ──────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("T-035: WASM-compatible ONNX re-export of v2 checkpoint")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Config
    ckpt_path = REPO_ROOT / "training" / "artefacts" / "eegconformer-physionet-v2" / "eegconformer.pt"
    existing_v2_onnx = REPO_ROOT / "public" / "models" / "eegconformer_finetuned.onnx"
    output_dir = REPO_ROOT / "training" / "artefacts" / "eegconformer-physionet-v2-wasm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx = output_dir / "eegconformer_finetuned.onnx"

    logs = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(ckpt_path),
        "existing_v2_onnx": str(existing_v2_onnx),
        "output_onnx": str(output_onnx),
        "steps": {},
    }

    # ── Step 1: Load checkpoint ─────────────────────────────────────────────
    print("\n[1] Loading v2 checkpoint...")
    print(f"    checkpoint: {ckpt_path}")
    print(f"    checkpoint SHA-256: {sha256_file(ckpt_path)}")

    torch.manual_seed(42)
    model = build_wasm_compatible_model()
    state = torch.load(ckpt_path, map_location="cpu")
    result = model.load_state_dict(state, strict=True)
    print(f"    Loaded {len(result)} keys — strict=True OK")
    model.eval()

    # ── Step 2: Verify PyTorch parity (original attention vs wasm attention) ─
    print("\n[2] Verifying PyTorch parity (original einsum vs wasm matmul attention)...")

    # Build a reference model with the original braindecode MultiHeadAttention
    ref_model = EEGConformer(
        n_outputs=4,
        n_chans=22,
        n_times=1000,
        final_fc_length="auto",
        drop_prob=0.5,
        att_drop_prob=0.5,
    ).eval()
    ref_model.load_state_dict(state, strict=True)

    wrapper_wasm = EEGConformerWithEmbedding(model)
    wrapper_ref = EEGConformerWithEmbedding(ref_model)
    wrapper_wasm.eval()
    wrapper_ref.eval()

    torch.manual_seed(42)
    test_input = torch.randn(8, 22, 1000)
    with torch.no_grad():
        pt_ref_emb, pt_ref_logits = wrapper_ref(test_input)
        pt_wasm_emb, pt_wasm_logits = wrapper_wasm(test_input)

    cos_emb = F.cosine_similarity(
        pt_ref_emb.flatten().unsqueeze(0),
        pt_wasm_emb.flatten().unsqueeze(0),
    ).item()
    cos_logits = F.cosine_similarity(
        pt_ref_logits.flatten().unsqueeze(0),
        pt_wasm_logits.flatten().unsqueeze(0),
    ).item()
    print(f"    PyTorch parity: embedding cosine={cos_emb:.8f}, logits cosine={cos_logits:.8f}")
    assert cos_emb > 0.9999, f"PyTorch parity failed: {cos_emb}"
    assert cos_logits > 0.9999, f"PyTorch parity failed: {cos_logits}"
    logs["steps"]["pytorch_parity"] = {
        "embedding_cosine": cos_emb,
        "logits_cosine": cos_logits,
        "passed": cos_emb > 0.9999 and cos_logits > 0.9999,
    }
    print(f"    ✓ PyTorch parity: PASS (both > 0.9999)")

    # ── Step 3: Export to ONNX ──────────────────────────────────────────────
    print("\n[3] Exporting to ONNX...")
    dummy = torch.randn(1, 22, 1000)
    torch.onnx.export(
        wrapper_wasm,
        dummy,
        str(output_onnx),
        input_names=["input"],
        output_names=["embedding", "logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy exporter to embed weights in single file (WASM-compatible)
    )
    print(f"    Exported: {output_onnx} ({output_onnx.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"    ONNX SHA-256: {sha256_file(output_onnx)}")

    # ── Step 3b: Ensure all weights are embedded (no external data) ──────────
    # onnxruntime-web WASM supports external data, but the production model format
    # uses a single embedded file. We enforce the same format here.
    m_check = onnx.load(str(output_onnx))
    has_external = any(init.data_location == 1 for init in m_check.graph.initializer)
    if has_external:
        print("    Model has external data references — embedding weights...")
        onnx.save_model(m_check, str(output_onnx), save_as_external_data=False)
        print(f"    Re-saved with embedded weights: {output_onnx.stat().st_size / 1024 / 1024:.2f} MB")

    # Also save with external data for comparison
    onnx.checker.check_model(onnx.load(str(output_onnx)))
    print(f"    ONNX graph validation: OK")

    # ── Step 4: PyTorch → ONNX numerical parity ─────────────────────────────
    print("\n[4] Verifying PyTorch → ONNX numerical parity...")
    sess = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])

    parities = []
    for i in range(10):
        torch.manual_seed(200 + i)
        x = torch.randn(4, 22, 1000)
        with torch.no_grad():
            pt_emb, pt_logits = wrapper_wasm(x)
        ort_emb, ort_logits = sess.run(None, {"input": x.numpy()})

        cos_emb = F.cosine_similarity(
            pt_emb.flatten().unsqueeze(0),
            torch.from_numpy(ort_emb).flatten().unsqueeze(0),
        ).item()
        cos_logits = F.cosine_similarity(
            pt_logits.flatten().unsqueeze(0),
            torch.from_numpy(ort_logits).flatten().unsqueeze(0),
        ).item()
        parities.append((cos_emb, cos_logits))

    mean_emb_cos = sum(p[0] for p in parities) / len(parities)
    min_emb_cos = min(p[0] for p in parities)
    mean_logits_cos = sum(p[1] for p in parities) / len(parities)
    min_logits_cos = min(p[1] for p in parities)

    print(f"    Embedding cosine: mean={mean_emb_cos:.8f}, min={min_emb_cos:.8f}")
    print(f"    Logits cosine:    mean={mean_logits_cos:.8f}, min={min_logits_cos:.8f}")

    parity_ok = min_emb_cos > 0.999
    logs["steps"]["onnx_parity"] = {
        "embedding_cosine_mean": mean_emb_cos,
        "embedding_cosine_min": min_emb_cos,
        "logits_cosine_mean": mean_logits_cos,
        "logits_cosine_min": min_logits_cos,
        "passed": parity_ok,
        "threshold": 0.999,
        "n_tests": len(parities),
    }
    if parity_ok:
        print(f"    ✓ ONNX parity: PASS (min embedding cosine = {min_emb_cos:.8f} > 0.999)")
    else:
        print(f"    ✗ ONNX parity: FAIL (min embedding cosine = {min_emb_cos:.8f})")
        sys.exit(1)

    # ── Step 5: WASM compatibility check ───────────────────────────────────
    print("\n[5] WASM compatibility check...")
    ops_used = get_onnx_ops(str(output_onnx))
    print(f"    ONNX ops used ({len(ops_used)}): {sorted(ops_used)}")

    wasm_blockers_found = WASM_BLOCKERS & ops_used
    if wasm_blockers_found:
        print(f"    ✗ WASM BLOCKERS found: {wasm_blockers_found}")
        logs["steps"]["wasm_check"] = {
            "wasm_compatible": False,
            "ops_used": sorted(ops_used),
            "blockers_found": sorted(wasm_blockers_found),
            "blocker_set": sorted(WASM_BLOCKERS),
        }
        sys.exit(1)
    else:
        print(f"    ✓ No WASM blockers — model is WASM-compatible")
        logs["steps"]["wasm_check"] = {
            "wasm_compatible": True,
            "ops_used": sorted(ops_used),
            "blockers_found": [],
            "blocker_set": sorted(WASM_BLOCKERS),
        }

    # ── Step 6: Cross-check against onnxruntime-web WASM provider ────────────
    print("\n[6] Cross-checking: ONNX Runtime WASM provider availability...")
    try:
        # Check if ORT WASM is available in this Python environment
        # (onnxruntime WASM is a browser build; in Python we check if the op
        # types are in the supported set for ORT minimal/WASM runtime)
        available_providers = ort.get_available_providers()
        print(f"    Available ORT providers: {available_providers}")
        # The CPU provider supports all ops; WASM provider would be listed if available
        # In Python we can't run the actual WASM provider, but we verify ops
        wasm_ops_supported = ops_used.issubset({
            "Add", "AveragePool", "Concat", "Conv", "Div", "Elu", "Erf",
            "Gemm", "LayerNormalization", "MatMul", "Mul", "Reshape",
            "Shape", "Slice", "Softmax", "Transpose", "Unsqueeze",
            "Sub", "Sqrt", "Reciprocal", "ReduceMean", "ReduceSum",
            "Cast", "Constant", "Gather", "Unsqueeze", "Pad",
        })
        if wasm_ops_supported:
            print(f"    ✓ All ops are in the ORT-Web WASM supported set")
        else:
            unsupported = ops_used - {
                "Add", "AveragePool", "Concat", "Conv", "Div", "Elu", "Erf",
                "Gemm", "LayerNormalization", "MatMul", "Mul", "Reshape",
                "Shape", "Slice", "Softmax", "Transpose", "Unsqueeze",
                "Sub", "Sqrt", "Reciprocal", "ReduceMean", "ReduceSum",
                "Cast", "Constant", "Gather", "Unsqueeze", "Pad",
            }
            print(f"    ! Possibly unsupported WASM ops: {unsupported}")
        logs["steps"]["wasm_crosscheck"] = {
            "providers": available_providers,
            "all_ops_in_wasm_supported_set": wasm_ops_supported,
        }
    except Exception as e:
        print(f"    Warning: could not cross-check ORT providers: {e}")

    # ── Step 7: Weight preservation verification ─────────────────────────────
    # Compare new ONNX initializers against the PyTorch checkpoint as ground truth.
    # The legacy ONNX exporter names Conv weights as 'onnx::Conv_*' and Linear
    # weight matrices as 'onnx::MatMul_*', while bias terms retain their PyTorch
    # path. We use bipartite matching: match by name first, then greedily match
    # remaining unmatched items by shape + cosine similarity.
    #
    # IMPORTANT: ONNX MatMul stores Linear weights transposed relative to PyTorch.
    # PyTorch nn.Linear(in, out).weight has shape (out, in), but ONNX exporter
    # writes the initializer as (in, out) so that Y = X @ W works (no extra
    # Transpose node). Conv2d weights are NOT transposed.
    print("\n[7] Verifying weight preservation vs PyTorch checkpoint...")
    pt_state = state  # from Step 1
    onnx_model = onnx.load(str(output_onnx))

    # Build list of ONNX initializers
    onnx_inits = []
    for init in onnx_model.graph.initializer:
        arr = nh.to_array(init)
        onnx_inits.append((init.name, arr))

    # Separate into named (bias/weights that retain PyTorch names) and opaque
    named_onnx = [(n, a) for n, a in onnx_inits if not n.startswith("onnx::")]
    opaque_onnx = [(n, a) for n, a in onnx_inits if n.startswith("onnx::")]

    matched = 0
    mismatched = []

    def cosine_similarity(a, b):
        a_flat = a.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)
        return float(np.dot(a_flat, b_flat) /
                     (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))

    # Phase 1: Match PyTorch params to named ONNX initializers by name
    used_named = set()
    named_lookup = {n: a for n, a in named_onnx}
    for pt_name, pt_tensor in pt_state.items():
        onnx_key = f"model.{pt_name}"
        if onnx_key in named_lookup:
            cos = cosine_similarity(pt_tensor.numpy(), named_lookup[onnx_key])
            if cos > 0.9999:
                matched += 1
                used_named.add(onnx_key)
            else:
                mismatched.append({"name": pt_name, "onnx_name": onnx_key, "cosine": cos})

    # Phase 2: Match remaining PyTorch weight matrices to opaque ONNX inits
    # by greedy best-shape+cosine matching.
    # Exclude BN/buffer params (folded into Conv by ONNX constant folding).
    # BatchNorm2d params and adjacent Conv params are folded into Conv by ONNX
    # constant folding. The BN module is named with "shallow.net.2" in the state
    # dict (where the module name is "shallow" + "net" concatenated without a
    # dot — braindecode's ShallowConv2d internals). Its gamma/beta and running
    # stats don't appear as separate ONNX initializers, and the preceding Conv2d's
    # weights are transformed by the BN fold. We skip all of these.
    #
    # Extract the exact BN module prefix from the checkpoint to avoid spelling issues.
    bn_prefix = None
    for k in pt_state:
        if ".running_mean" in k and "shallow" in k and "net" in k and ".2." in k:
            bn_prefix = k[: k.index(".running_mean")]
            break
    if bn_prefix is None:
        # Fallback: find any running_mean under shallow.net.2
        for k in pt_state:
            if "running_mean" in k:
                bn_prefix = k.replace("running_mean", "").replace("running_mean", "")
                # Extract the module path
                parts = k.replace(".running_mean", "").replace(".running_mean", "")
                # Just try: take everything before a known pattern
                if "bn" in k.lower() or "shallow" in k:
                    bn_prefix = k.replace(".running_mean", "").replace("running_mean", "").rstrip(".")

    foldable_keys = set()
    if bn_prefix:
        foldable_keys.update([
            bn_prefix + ".weight",
            bn_prefix + ".bias",
            bn_prefix + ".running_mean",
            bn_prefix + ".running_var",
            bn_prefix + ".num_batches_tracked",
        ])
        # Also add the preceding Conv2d params (folded with BN)
        # BN module is ...net.2, Conv is ...net.1
        if bn_prefix.endswith(".2"):
            conv_prefix = bn_prefix[:-1] + "1"  # change ".2" to ".1"
            foldable_keys.update([
                conv_prefix + ".weight",
                conv_prefix + ".bias",
            ])

    remaining_pt = []
    skipped_folded = []
    for n, t in pt_state.items():
        if f"model.{n}" not in used_named:
            if n in foldable_keys:
                skipped_folded.append(n)
            else:
                remaining_pt.append((n, t))
    remaining_onnx = [(n, a) for n, a in opaque_onnx]
    print(f"    Skipped foldable BN/buffer params: {len(skipped_folded)}")
    print(f"    Remaining PT weight matrices to match: {len(remaining_pt)}")

    # Greedy matching: for each PT weight, find the best ONNX init match.
    # Key insight: ONNX MatMul inits store Linear weights TRANSPOSED relative
    # to PyTorch. PT nn.Linear(in,out).weight has shape (out, in), but ONNX
    # writes the initializer as (in, out) so Y = X @ W works without a
    # Transpose node. Conv weights are NOT transposed.
    #
    # For square MatMul weights (e.g. 40×40 attention), both PT and ONNX have
    # the same shape, so we must try BOTH orientations and keep whichever gives
    # higher cosine. For non-square MatMul, the shape tells us which orientation
    # to use.
    for pt_name, pt_tensor in remaining_pt:
        pt_arr = pt_tensor.numpy().astype(np.float32)
        best_cos = -1.0
        best_onnx = None
        best_idx = -1
        best_transposed = False

        for idx, (onnx_name, onnx_arr) in enumerate(remaining_onnx):
            onnx_arr_f32 = onnx_arr.astype(np.float32)
            is_matmul = onnx_name.startswith("onnx::MatMul")

            # Always try direct comparison (works for Conv, and for MatMul if
            # shapes happen to be the same — though for MatMul they may be
            # transposed, so we also try the transpose below)
            if onnx_arr_f32.shape == pt_arr.shape:
                cos = cosine_similarity(pt_arr, onnx_arr_f32)
                if cos > best_cos:
                    best_cos = cos
                    best_onnx = onnx_name
                    best_idx = idx
                    best_transposed = False
                # For MatMul, also try transposed (shapes match but values
                # might match better when transposed)
                if is_matmul and pt_arr.ndim == 2:
                    cos_t = cosine_similarity(pt_arr, onnx_arr_f32.T)
                    if cos_t > best_cos:
                        best_cos = cos_t
                        best_onnx = onnx_name
                        best_idx = idx
                        best_transposed = True
            # Try transposed match (for MatMul: PT (out,in) vs ONNX (in,out))
            elif is_matmul and pt_arr.ndim == 2 and onnx_arr_f32.shape == pt_arr.shape[::-1]:
                cos = cosine_similarity(pt_arr, onnx_arr_f32.T)
                if cos > best_cos:
                    best_cos = cos
                    best_onnx = onnx_name
                    best_idx = idx
                    best_transposed = True

        if best_onnx is not None and best_cos > 0.9999:
            matched += 1
            tag = " (transposed)" if best_transposed else ""
            remaining_onnx.pop(best_idx)  # Remove matched item
        elif best_onnx is not None:
            mismatched.append({"name": pt_name, "onnx_name": best_onnx, "cosine": best_cos, "transposed": best_transposed})
        else:
            mismatched.append({"name": pt_name, "reason": "no shape match in ONNX (even transposed)"})

    total_pt_params = len(pt_state)
    total_onnx_inits = len(onnx_inits)
    unmatched_onnx_names = [n for n, _ in remaining_onnx]

    print(f"    PyTorch checkpoint params: {total_pt_params}")
    print(f"    ONNX initializers: {total_onnx_inits}")
    print(f"    Matched (name or shape+cosine > 0.9999): {matched}/{total_pt_params}")
    print(f"    Unmatched ONNX initializers: {len(unmatched_onnx_names)}")

    if mismatched:
        for m in mismatched[:5]:
            print(f"    MISMATCH: {m}")
    for n in unmatched_onnx_names[:5]:
        print(f"    UNUSED ONNX init: {n}")

    # Embedding parity is the ultimate proof of weight preservation (computed
    # in Step 8 from actual inference). Weights are preserved iff:
    #   (1) all name-matched params have cosine > 0.9999
    #   (2) all non-folded PT params found a match by shape+cosine
    #   (3) any unused ONNX inits are only Conv inits (folded with BN)
    #   (4) embedding parity (new vs existing ONNX) > 0.9999
    # emb_cos is computed in Step 8; will be stored in weight_preservation log
    # right after it's computed (before JSON save).

    # Unused ONNX inits are expected for Conv weights/biases that were folded
    # with adjacent BatchNorm (they don't match the original PT Conv params).
    unmatched_conv = [n for n in unmatched_onnx_names if n.startswith("onnx::Conv")]
    unmatched_matmul = [n for n in unmatched_onnx_names if n.startswith("onnx::MatMul")]
    print(f"    Unused Conv inits (folded with BN): {len(unmatched_conv)}")
    print(f"    Unused MatMul inits: {len(unmatched_matmul)}")

    all_matched = (
        len(mismatched) == 0
        and len(unmatched_matmul) == 0  # all MatMul inits must be consumed
    )

    # ── BN folding reconstruction ──────────────────────────────────────────
    # For Conv+BN pairs that were folded, verify by reconstructing the folded
    # weights from PT Conv + BN params and comparing to the ONNX Conv inits.
    print(f"    Verifying folded Conv weights (BN reconstruction)...")
    bn_reconstruction_checks = []
    folded_conv_onnx = [n for n in unmatched_onnx_names if n.startswith("onnx::Conv")]
    bn_eps = 1e-5

    # Reconstruct folded spatial Conv (BN conv + BN)
    # Use the dynamically extracted bn_prefix and conv_prefix
    conv_w_name = conv_prefix + ".weight"
    conv_b_name = conv_prefix + ".bias"
    bn_w_name = bn_prefix + ".weight"  # gamma
    bn_b_name = bn_prefix + ".bias"    # beta
    bn_rm_name = bn_prefix + ".running_mean"
    bn_rv_name = bn_prefix + ".running_var"

    if all(n in pt_state for n in [conv_w_name, conv_b_name, bn_w_name, bn_b_name, bn_rm_name, bn_rv_name]):
        pt_conv_w = pt_state[conv_w_name].numpy().astype(np.float32)
        pt_conv_b = pt_state[conv_b_name].numpy().astype(np.float32)
        pt_bn_gamma = pt_state[bn_w_name].numpy().astype(np.float32)
        pt_bn_beta = pt_state[bn_b_name].numpy().astype(np.float32)
        pt_bn_rm = pt_state[bn_rm_name].numpy().astype(np.float32)
        pt_bn_rv = pt_state[bn_rv_name].numpy().astype(np.float32)

        # BN folding: folded_weight = weight * (gamma / sqrt(var + eps))
        #              folded_bias = (bias - mean) * (gamma / sqrt(var + eps)) + beta
        std = np.sqrt(pt_bn_rv + bn_eps)
        scale = pt_bn_gamma / std  # shape (40,)

        folded_weight = pt_conv_w * scale.reshape(40, 1, 1, 1)
        folded_bias = (pt_conv_b - pt_bn_rm) * scale + pt_bn_beta

        # Compare with ONNX Conv inits
        onnx_lookup = {n: a for n, a in onnx_inits}
        for onnx_name in folded_conv_onnx:
            onnx_arr = onnx_lookup[onnx_name]
            if onnx_arr.shape == folded_weight.shape:
                cos = cosine_similarity(folded_weight, onnx_arr)
                bn_reconstruction_checks.append({
                    "onnx_name": onnx_name,
                    "pt_weight_name": conv_w_name,
                    "type": "conv_weight",
                    "cosine": cos,
                    "passed": cos > 0.9999,
                })
            elif onnx_arr.shape == folded_bias.shape:
                cos = cosine_similarity(folded_bias, onnx_arr)
                bn_reconstruction_checks.append({
                    "onnx_name": onnx_name,
                    "pt_weight_name": conv_b_name,
                    "type": "conv_bias",
                    "cosine": cos,
                    "passed": cos > 0.9999,
                })

    bn_reconstruction_passed = all(c["passed"] for c in bn_reconstruction_checks) if bn_reconstruction_checks else True
    if bn_reconstruction_checks:
        for c in bn_reconstruction_checks:
            status = "✓" if c["passed"] else "✗"
            print(f"    {status} Folded Conv {c['type']} ({c['onnx_name']}): cosine={c['cosine']:.8f}")

    all_matched = all_matched and bn_reconstruction_passed
    logs["steps"]["weight_preservation"] = {
        "checkpoint_path": str(ckpt_path),
        "new_onnx": str(output_onnx),
        "total_pt_params": total_pt_params,
        "total_onnx_inits": total_onnx_inits,
        "matched": matched,
        "mismatched": mismatched,
        "unmatched_onnx": unmatched_onnx_names,
        "skipped_folded_params": len(skipped_folded),
        "folded_param_names": skipped_folded,
        "passed": all_matched,
    }
    if all_matched:
        print(f"    ✓ All {matched} trainable checkpoint params matched in ONNX weights")
        print(f"    ✓ {len(skipped_folded)} BN/buffer params folded into Conv (expected)")
        print(f"    ✓ Embedding parity (see Step 8 for actual value)")
    else:
        print(f"    ✗ Weight mismatch: {len(mismatched)} mismatches, {len(unmatched_onnx_names)} unused ONNX inits")

    # ── Step 8: Run corrected T-034 50-subject LOSO evaluation ───────────────
    print("\n[8] Running T-034 50-subject LOSO evaluation...")
    print("    Loading data and extracting embeddings...")

    t = _load_t033()

    subjects_data = t.load_physionet_subjects(list(range(1, 51)), runs=[5, 6])
    print(f"    Loaded {len(subjects_data)} subjects")

    all_windows = []
    all_labels = []
    all_subject_ids = []
    for subj_id in sorted(subjects_data.keys()):
        sd = subjects_data[subj_id]
        for i, trial in enumerate(sd["trials"]):
            win = t.preprocess_for_eegconformer(trial, sd["ch_names"])
            all_windows.append(win)
            all_labels.append(sd["labels"][i])
            all_subject_ids.append(subj_id)

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    print(f"    Total trials: {len(all_windows)}")

    # Extract embeddings from the new WASM-compatible ONNX
    new_embs, _ = t.batched_onnx_inference(str(output_onnx), all_windows, intermediate_tensor=None)
    print(f"    New WASM model 32-D embeddings: {new_embs.shape}")

    # Also extract embeddings from the existing v2 ONNX for comparison
    existing_embs, _ = t.batched_onnx_inference(str(existing_v2_onnx), all_windows, intermediate_tensor=None)
    print(f"    Existing v2 32-D embeddings: {existing_embs.shape}")

    # Verify embedding parity between new and existing ONNX (same weights → same output)
    emb_cos = float(np.dot(new_embs.flatten(), existing_embs.flatten()) /
                    (np.linalg.norm(new_embs.flatten()) * np.linalg.norm(existing_embs.flatten()) + 1e-12))
    print(f"    Embedding cosine (new vs existing ONNX): {emb_cos:.8f}")

    # Store embedding parity in weight_preservation log (ultimate proof of
    # weight preservation — same weights produce same embeddings)
    logs["steps"]["weight_preservation"]["embedding_parity_cosine"] = emb_cos

    # Run LOSO on the new model
    loso_new = t.run_loso(new_embs, all_labels, all_subject_ids, needs_pca=False)
    r_new = loso_new["loso"]
    cs_new = t.class_separability(new_embs, all_labels.tolist())

    # Run LOSO on the existing v2 model for direct comparison
    loso_existing = t.run_loso(existing_embs, all_labels, all_subject_ids, needs_pca=False)
    r_existing = loso_existing["loso"]
    cs_existing = t.class_separability(existing_embs, all_labels.tolist())

    print("\n    LOSO Results (50-subject):")
    print(f"    {'Model':<24} {'Acc':>10} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'Fisher':>10} {'Intra':>8} {'Inter':>8}")
    print(f"    {'':24} {'':>10} {'':>8} {'':>8} {'':>8} {'':>10} {'':>8} {'':>8}")
    print(f"    {'Existing v2 (Einsum)':<24} {r_existing['mean_accuracy']:>10.4f} {r_existing['recall_at_1']['mean']:>8.4f} "
          f"{r_existing['recall_at_5']['mean']:>8.4f} {r_existing['recall_at_10']['mean']:>8.4f} {cs_existing['fisher_score']:>10.4f} "
          f"{cs_existing['intra_class_cosine_mean']:>8.4f} {cs_existing['inter_class_cosine_mean']:>8.4f}")
    print(f"    {'New WASM (MatMul)':<24} {r_new['mean_accuracy']:>10.4f} {r_new['recall_at_1']['mean']:>8.4f} "
          f"{r_new['recall_at_5']['mean']:>8.4f} {r_new['recall_at_10']['mean']:>8.4f} {cs_new['fisher_score']:>10.4f} "
          f"{cs_new['intra_class_cosine_mean']:>8.4f} {cs_new['inter_class_cosine_mean']:>8.4f}")

    # Statistical comparison
    t_stat, p_val, cohen_d = t.paired_t_test(
        r_existing["per_fold_accuracy"],
        r_new["per_fold_accuracy"],
    )

    logs["steps"]["evaluation"] = {
        "n_subjects": 50,
        "n_trials": len(all_labels),
        "label_distribution": np.bincount(all_labels, minlength=4).tolist(),
        "existing_v2": {
            "accuracy": r_existing["mean_accuracy"],
            "accuracy_ci95": r_existing["ci95_accuracy"],
            "r1": r_existing["recall_at_1"]["mean"],
            "r1_ci95": r_existing["recall_at_1"]["ci95"],
            "r5": r_existing["recall_at_5"]["mean"],
            "r10": r_existing["recall_at_10"]["mean"],
            "fisher": cs_existing["fisher_score"],
            "intra_cosine": cs_existing["intra_class_cosine_mean"],
            "inter_cosine": cs_existing["inter_class_cosine_mean"],
        },
        "new_wasm": {
            "accuracy": r_new["mean_accuracy"],
            "accuracy_ci95": r_new["ci95_accuracy"],
            "r1": r_new["recall_at_1"]["mean"],
            "r1_ci95": r_new["recall_at_1"]["ci95"],
            "r5": r_new["recall_at_5"]["mean"],
            "r10": r_new["recall_at_10"]["mean"],
            "fisher": cs_new["fisher_score"],
            "intra_cosine": cs_new["intra_class_cosine_mean"],
            "inter_cosine": cs_new["inter_class_cosine_mean"],
        },
        "paired_t_test": {
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohen_d,
            "delta_accuracy": r_new["mean_accuracy"] - r_existing["mean_accuracy"],
            "delta_fisher": cs_new["fisher_score"] - cs_existing["fisher_score"],
        },
        "embedding_parity_cosine": emb_cos,
    }

    # ── Step 9: Stability check on new model ─────────────────────────────────
    print("\n[9] Running embedding stability check on new WASM model...")
    sess_wasm = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    inp = sess_wasm.get_inputs()[0]
    session_info = {"session": sess_wasm, "input_name": inp.name, "output_name": "embedding"}
    np.random.seed(42)
    test_indices = np.random.choice(len(all_windows), 15, replace=False)
    stability_results = []
    for idx in test_indices:
        w = all_windows[idx]
        s = t.embedding_stability_onnx(session_info, w)
        stability_results.append(s)

    avg_det = np.mean([s["determinism"]["max_pairwise_cosine"] for s in stability_results])
    avg_scale = np.mean([s["amplitude_scaling"]["mean_cosine"] for s in stability_results])
    avg_noise = np.mean([s["noise_robustness"]["cosine_similarity"] for s in stability_results])
    avg_shift = np.nanmean([s["window_boundary_shift"]["mean_cosine"] for s in stability_results])

    print(f"    Determinism (max pairwise cos): {avg_det:.8f}")
    print(f"    Amplitude ±10%:                 cosine={avg_scale:.4f}")
    print(f"    Noise (SNR=20dB):               cosine={avg_noise:.4f}")
    print(f"    Window shift ±40ms:            cosine={avg_shift:.4f}")

    logs["steps"]["stability"] = {
        "determinism_max_cosine": float(avg_det),
        "deterministic": bool(avg_det > 0.9999),
        "amplitude_scaling_mean_cosine": float(avg_scale),
        "noise_robustness_cosine": float(avg_noise),
        "window_boundary_shift_mean_cosine": float(avg_shift),
    }

    # ── Step 10: Richness check ──────────────────────────────────────────────
    print("\n[10] Checking embedding richness...")
    er_new = t.embedding_richness(new_embs)
    er_existing = t.embedding_richness(existing_embs)
    print(f"    New WASM:    effective_rank={er_new['effective_rank_participation_ratio']:.2f}, "
          f"90%var@D={er_new['cumulative_variance_90_at_dim']}, "
          f"dead_dims={er_new['n_dead_dimensions']}")
    print(f"    Existing v2: effective_rank={er_existing['effective_rank_participation_ratio']:.2f}, "
          f"90%var@D={er_existing['cumulative_variance_90_at_dim']}, "
          f"dead_dims={er_existing['n_dead_dimensions']}")

    logs["steps"]["richness"] = {
        "new_wasm": {k: v for k, v in er_new.items() if k not in ("explained_variance_ratio",)},
        "existing_v2": {k: v for k, v in er_existing.items() if k not in ("explained_variance_ratio",)},
    }

    # ── Save logs and report ─────────────────────────────────────────────────
    logs["output_onnx_sha256"] = sha256_file(str(output_onnx))
    logs["output_onnx_size"] = output_onnx.stat().st_size

    # Summary
    print("\n" + "=" * 70)
    print("T-035 SUMMARY")
    print("=" * 70)
    wp = logs["steps"]["weight_preservation"]
    print(f"  Weights preserved: {wp['matched']}/{wp['total_pt_params']} checkpoint params matched in ONNX")
    print(f"  ONNX SHA-256: {logs['output_onnx_sha256']}")
    print(f"  ONNX size: {logs['output_onnx_size'] / 1024 / 1024:.2f} MB")
    print(f"  ONNX ops: {sorted(ops_used)}")
    print(f"  Einsum present: {'Einsum' in ops_used}")
    print(f"  WASM compatible: {len(wasm_blockers_found) == 0}")
    print(f"  ONNX parity (min cosine): {min_emb_cos:.8f}")
    print(f"  PyTorch parity (emb cosine): {cos_emb:.8f}")
    print(f"  Embedding parity (new vs existing): {emb_cos:.8f}")
    print(f"  Existing v2:    Acc={r_existing['mean_accuracy']:.4f}, Fisher={cs_existing['fisher_score']:.4f}")
    print(f"  New WASM model:   Acc={r_new['mean_accuracy']:.4f}, Fisher={cs_new['fisher_score']:.4f}")
    print(f"  Delta:            ΔAcc={r_new['mean_accuracy'] - r_existing['mean_accuracy']:+.6f}, "
          f"ΔFisher={cs_new['fisher_score'] - cs_existing['fisher_score']:+.6f}")
    print(f"  Stability: deterministic={avg_det > 0.9999}")

    wp = logs["steps"]["weight_preservation"]
    all_passed = (
        parity_ok
        and len(wasm_blockers_found) == 0
        and wp["passed"]
        and emb_cos > 0.9999
    )

    # Update weight preservation log with embedding parity as ultimate proof
    wp["embedding_parity_cosine"] = emb_cos
    print(f"\n  VERDICT: {'✅ ALL CHECKS PASSED' if all_passed else '❌ CHECKS FAILED'}")
    print("=" * 70)

    # ── Save logs (after all steps complete) ───────────────────────────────
    log_path = output_dir / "t035_export_log.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2, default=str)
    print(f"[log] Export log saved to: {log_path}")

    return logs


if __name__ == "__main__":
    main()
