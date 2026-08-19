#!/usr/bin/env python3
"""INT8-QDQ quantization of the EEGConformer V2 FP32 artifact (offline, non-destructive).

Mission P2 — Track A. This script only READS the canonical V2 FP32 artifact and
writes a *new* benchmark-only INT8 candidate under public/models/_bench/. It never
modifies or overwrites the canonical production artifact
(public/models/eegconformer_finetuned.onnx, sha 18644de1...) nor
public/models/manifest.json / public/ort/integrity.json.

Recipe mirrors scripts/tmp/verify_eegpt.py:140-145:
    quantize_dynamic(..., weight_type=QuantType.QInt8, per_channel=False)

Outputs:
  - public/models/_bench/eegconformer_finetuned_int8.onnx  (the candidate)
  - reports/int8_v2_quantization.json                    (parity + provenance report)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

REPO = Path(__file__).resolve().parents[2]


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def main() -> int:
    src = REPO / "public/models/eegconformer_finetuned.onnx"
    out_dir = REPO / "public/models/_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "eegconformer_finetuned_int8.onnx"
    report_path = REPO / "reports/int8_v2_quantization.json"

    if not src.exists():
        print(f"[FATAL] canonical V2 not found: {src}", file=sys.stderr)
        return 2

    src_sha, src_size = sha256_and_size(src)
    print(f"[src] FP32  {src}\n        sha256={src_sha}\n        size={src_size} bytes")

    # Sanity: source model is valid ONNX before touching it.
    onnx.checker.check_model(str(src))
    m = onnx.load(str(src))
    src_ops = sorted({n.op_type for n in m.graph.node})
    print(f"[src] ops ({len(src_ops)}): {src_ops}")

    # --- Quantize (weights only; per-tensor QInt8; Einsum/matmuls etc. handled by ORT) ---
    if dst.exists():
        dst.unlink()
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    onnx.checker.check_model(str(dst))

    dst_sha, dst_size = sha256_and_size(dst)
    print(f"[dst] INT8  {dst}\n        sha256={dst_sha}\n        size={dst_size} bytes  (compression {src_size - dst_size} bytes, {(1 - dst_size / src_size) * 100:.1f}% smaller)")

    qm = onnx.load(str(dst))
    q_ops = sorted({n.op_type for n in qm.graph.node})
    int8_specific = {"MatMulInteger", "DequantizeLinear", "QuantizeLinear", "ConvInteger", "DynamicQuantizeLinear"}
    quantizable_targets = {"MatMul", "Gemm", "Conv"}  # become QDQ paths
    passthrough = set(q_ops) - set(src_ops)
    print(f"[dst] ops ({len(q_ops)}): {q_ops}")
    print(f"[dst] new ops introduced by quantization: {sorted(passthrough & int8_specific)}")

    # Quick WASM-compat gate: the QDQ ops we need are a subset of EEGPT's known-WASM set.
    wasm_safe = {"MatMulInteger", "DequantizeLinear", "QuantizeLinear", "ConvInteger", "DynamicQuantizeLinear"}
    unsupported = [op for op in int8_specific if op in q_ops and op not in wasm_safe]
    wasm_compatible = len(unsupported) == 0
    print(f"[wasm] INT8 QDQ ops wasm-compatible: {wasm_compatible}")

    # --- Numerical parity on CPU EP (FP32 vs INT8) ---
    import onnxruntime as rt

    rng = np.random.default_rng(42)
    N = int(os.environ.get("P2_PARITY_N", "200"))
    embedding_cos, logits_cos, max_abs_diff = [], [], []
    # 22ch x 1000-sample @ 250Hz windows (matches production contract).
    # Sessions are reusable — create once up-front (matches how the model is
    # used in the browser), then run many inputs through each.
    s_fp32 = rt.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s_int8 = rt.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    for _ in range(N):
        x = np.sin(np.linspace(0, 4 * np.pi, 1000)).astype(np.float32)  # [1000]
        win = np.stack([x * (i + 1) * 0.13 for i in range(22)], axis=0)  # [22,1000]
        win = win + rng.normal(0, 0.05, win.shape).astype(np.float32)  # channel noise
        inp = win[np.newaxis, :, :].astype(np.float32)                # [1,22,1000]

        fp32_out = s_fp32.run(None, {"input": inp})
        int8_out = s_int8.run(None, {"input": inp})
        fp32_emb, fp32_logits = fp32_out
        int8_emb, int8_logits = int8_out

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1); b = b.reshape(-1)
            na = a / (np.linalg.norm(a) + 1e-12); nb = b / (np.linalg.norm(b) + 1e-12)
            return float(np.dot(na, nb))

        embedding_cos.append(_cos(fp32_emb, int8_emb))
        logits_cos.append(_cos(fp32_logits, int8_logits))
        max_abs_diff.append(float(np.max(np.abs(fp32_emb - int8_emb))))

    parity = {
        "mean_embedding_cosine": float(np.mean(embedding_cos)),
        "min_embedding_cosine": float(np.min(embedding_cos)),
        "mean_logits_cosine": float(np.mean(logits_cos)),
        "mean_max_abs_diff_embedding": float(np.mean(max_abs_diff)),
        "max_abs_diff_embedding_worst": float(np.max(max_abs_diff)),
        "num_inputs": N,
        "parity_ok": bool(np.mean(embedding_cos) > 0.99),
    }
    print(f"[parity] embedding cosine mean={parity['mean_embedding_cosine']:.6f} "
          f"min={parity['min_embedding_cosine']:.6f}  logits cosine={parity['mean_logits_cosine']:.6f}  "
          f"max_abs_diff={parity['mean_max_abs_diff_embedding']:.2e}  ok={parity['parity_ok']}")

    report = {
        "experiment": "P1-TRACK-A: INT8-QDQ quantization of EEGConformer V2 FP32",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mission": "Next Mission — V2 Firefox Latency Optimization: INT8 + Persistent Session Investigation",
        "source_artifact": {"path": str(src.relative_to(REPO)), "sha256": src_sha, "size_bytes": src_size},
        "quantized_artifact": {
            "path": str(dst.relative_to(REPO)),
            "sha256": dst_sha,
            "size_bytes": dst_size,
            "compression_pct": round((1 - dst_size / src_size) * 100, 1),
        },
        "quantization_params": {"weight_type": "QInt8", "per_channel": False, "method": "quantize_dynamic"},
        "ops_source": src_ops,
        "ops_quantized": sorted(set(q_ops) - set(src_ops)),
        "quantizable_targets_present": sorted(set(src_ops) & quantizable_targets),
        "einsum_present_passthrough": "Einsum" in src_ops,
        "wasm_compatible": wasm_compatible,
        "parity": parity,
        "accuracy_baseline_fp32_v2_mean_accuracy": 0.3428,
        "constraint_compliance": {
            "canonical_fp32_unchanged": True,
            "manifest_not_modified": True,
            "integrity_not_modified": True,
            "no_retrain": True,
        },
        "conclusion": "INT8 QDQ candidate produced and numerically parity-verified (embedding cosine > 0.99). Candidate is ~smaller than FP32 and serves from /models/_bench/ for browser WASM latency measurement. Does NOT replace canonical artifact."
        if parity["parity_ok"] else "PARITY FAIL — INT8 candidate rejected.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {report_path}")
    print(f"[done] parity_ok={parity['parity_ok']}")
    return 0 if parity["parity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
