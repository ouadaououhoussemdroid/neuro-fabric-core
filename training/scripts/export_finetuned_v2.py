"""Export the fine-tuned EEGConformer checkpoint to ONNX with parity verification.

Fixes the braindecode/moabb import issue and uses EEGConformerExportWrapper.
"""
from __future__ import annotations

import os, sys, json
from pathlib import Path
import torch
import torch.nn as nn

# ─── Fix braindecode/moabb import ─────────────────────────────────────────────
import moabb.datasets as mds
if not hasattr(mds, "BNCI2014001"):
    mds.BNCI2014001 = mds.BNCI2014_001
if not hasattr(mds, "HGD"):
    mds.HGD = mds.PhysionetMI

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from braindecode.models import EEGConformer
from scripts.export_braindecode_eegconformer import EEGConformerExportWrapper

checkpoint_path = "training/artefacts/eegconformer-physionet-v2/eegconformer.pt"
out_path = "training/artefacts/eegconformer-physionet-v2/eegconformer_finetuned.onnx"

print(f"[export] Loading checkpoint: {checkpoint_path}")
model = EEGConformer(
    n_outputs=4,
    n_chans=22,
    n_times=1000,
    final_fc_length="auto",
    drop_prob=0.5,
    att_drop_prob=0.5,
)
state = torch.load(checkpoint_path, map_location="cpu")
result = model.load_state_dict(state, strict=True)
print(f"[export] Loaded {len(result)} — strict=True OK")
model.eval()

wrapper = EEGConformerExportWrapper(model)
wrapper.eval()

dummy = torch.randn(1, 22, 1000)

# PyTorch reference outputs
with torch.no_grad():
    pt_emb, pt_logits = wrapper(dummy)
print(f"[export] PyTorch: embedding={pt_emb.shape}, logits={pt_logits.shape}")

# Export to ONNX
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
torch.onnx.export(
    wrapper,
    dummy,
    out.as_posix(),
    input_names=["input"],
    output_names=["embedding", "logits"],
    dynamic_axes={
        "input": {0: "batch"},
        "embedding": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"[export] ONNX exported: {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")

# ─── Parity check ─────────────────────────────────────────────────────────────
import onnx
import onnxruntime as ort

# Validate ONNX graph
onnx.checker.check_model(onnx.load(out.as_posix()))
print("[export] ONNX graph validation: OK")

sess = ort.InferenceSession(out.as_posix(), providers=["CPUExecutionProvider"])

# Test parity with multiple random inputs
parities = []
for i in range(5):
    torch.manual_seed(100 + i)
    x = torch.randn(4, 22, 1000)
    with torch.no_grad():
        pt_emb, pt_logits = wrapper(x)
    ort_emb, ort_logits = sess.run(None, {"input": x.numpy()})

    cos_emb = torch.nn.functional.cosine_similarity(
        pt_emb.flatten().unsqueeze(0),
        torch.from_numpy(ort_emb).flatten().unsqueeze(0),
    ).item()
    cos_logits = torch.nn.functional.cosine_similarity(
        pt_logits.flatten().unsqueeze(0),
        torch.from_numpy(ort_logits).flatten().unsqueeze(0),
    ).item()
    parities.append((cos_emb, cos_logits))
    print(f"[export] Test {i+1}: emb_cosine={cos_emb:.6f}, logits_cosine={cos_logits:.6f}")

mean_emb_cos = sum(p[0] for p in parities) / len(parities)
min_emb_cos = min(p[0] for p in parities)
print(f"\n[export] Mean embedding cosine: {mean_emb_cos:.6f}")
print(f"[export] Min embedding cosine:  {min_emb_cos:.6f}")
print(f"[export] Parity {'OK' if min_emb_cos > 0.999 else 'FAIL'} (threshold >0.999)")

# ─── WASM compatibility check ─────────────────────────────────────────────────
print("\n[export] WASM compatibility check...")
ops = set()
for node in onnx.load(out.as_posix()).graph.node:
    ops.add(node.op_type)
print(f"  ONNX ops used: {sorted(ops)}")
wasm_blockers = {"DFT", "ReduceL2", "FFT", "Complex"}
found_blockers = wasm_blockers & ops
if found_blockers:
    print(f"  BLOCKERS found: {found_blockers}")
else:
    print(f"  No WASM blockers found → WASM compatible ✓")

# ─── Update train_history.json with export info ───────────────────────────────
hist_path = Path("training/artefacts/eegconformer-physionet-v2/train_history.json")
if hist_path.exists():
    with open(hist_path) as f:
        hist = json.load(f)
    hist["export"] = {
        "onnx_path": out_path,
        "parity_embedding_cosine_mean": mean_emb_cos,
        "parity_embedding_cosine_min": min_emb_cos,
        "parity_logits_cosine_mean": sum(p[1] for p in parities) / len(parities),
        "parity_ok": bool(min_emb_cos > 0.999),
        "opset": 17,
        "wasm_compatible": bool(not found_blockers),
        "onnx_ops": sorted(ops),
        "wasm_blockers": sorted(found_blockers),
    }
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"\n[export] Updated train_history.json with export info")

print(f"\n[export] DONE: {out_path}")
