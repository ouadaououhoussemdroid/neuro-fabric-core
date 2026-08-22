#!/usr/bin/env python3
"""
M39 — Create the Sleep Staging ONNX probe (Joint-2312 → 5 classes).

Generates a lightweight linear classification head (2312-D → 5-D) that maps
Joint-2312 embeddings to sleep stage logits (W, N1, N2, N3, REM). The model
applies a learned linear transformation followed by softmax — the softmax is
applied in the ONNX graph so the service layer receives probabilities directly.

This is a placeholder/random-init model — the weights will be fine-tuned when
real Sleep-EDF Joint-2312 embeddings become available for training (M38 loader
is the prerequisite, now complete). The architecture is correct: the service
layer (`sleep.server.ts`) reads `class_0`…`class_4` from `ONNXAdapter.predict()`.

Usage:
    python scripts/create_sleep_probe.py

Output:
    public/models/sleep/staging-probe-joint2312-v1.onnx
    Prints the SHA-256 hash for registry registration.
"""
import hashlib
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


N_INPUT = 2312  # Joint-2312 embedding dimension
N_OUTPUT = 5    # 5 sleep stages: W, N1, N2, N3, REM
SEED = 42


def create_sleep_probe_onnx(output_path: str) -> str:
    """Create a 2312→5 linear + softmax ONNX model and save to output_path."""
    rng = np.random.RandomState(SEED)

    # Weight and bias for linear layer: y = Wx + b, shape [5, 2312] and [5]
    W = rng.randn(N_OUTPUT, N_INPUT).astype(np.float32) * 0.01
    b = np.zeros(N_OUTPUT, dtype=np.float32)

    # Normalize weights (L2) for balanced initial logits
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)

    # Create ONNX initializer tensors
    W_tensor = numpy_helper.from_array(W, name="linear_weight")
    b_tensor = numpy_helper.from_array(b, name="linear_bias")

    # Input: [batch, 2312] float32
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", N_INPUT])
    # Output: [batch, 5] float32 (probabilities after softmax)
    output_tensor = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", N_OUTPUT])

    # Node 1: MatMul(input, W^T) + Add(b) → logits
    matmul = helper.make_node(
        "Gemm",
        inputs=["input", "linear_weight", "linear_bias"],
        outputs=["logits"],
        name="linear",
        transB=1,  # W is stored as [N_OUTPUT, N_INPUT], so transpose for Gemm
    )

    # Node 2: Softmax on logits → probabilities
    softmax = helper.make_node(
        "Softmax",
        inputs=["logits"],
        outputs=["probabilities"],
        name="softmax",
        axis=1,
    )

    # Build the graph
    graph = helper.make_graph(
        [matmul, softmax],
        "sleep_staging_probe",
        [input_tensor],
        [output_tensor],
        initializer=[W_tensor, b_tensor],
    )

    # Build the model
    model = helper.make_model(graph, producer_name="neuro-fabric-m39")
    model.opset_import[0].version = 13  # Stable opset with Gemm + Softmax

    # Save
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    # Compute SHA-256
    sha = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def update_manifest(output_path: str, sha: str, size: int) -> None:
    """Add the sleep staging probe to the models manifest."""
    import json
    import os

    manifest_path = os.path.join(os.path.dirname(output_path), "manifest.json")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_manifest = os.path.join(repo_root, "public", "models", "manifest.json")

    with open(full_manifest) as f:
        manifest = json.load(f)

    manifest["models"]["sleep-staging-probe-v1"] = {
        "id": "sleep-staging-probe-v1",
        "url": "/models/sleep/staging-probe-joint2312-v1.onnx",
        "sha256": sha,
        "size": size,
        "wasmCompatible": True,
    }

    with open(full_manifest, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(repo_root, "public", "models", "sleep")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "staging-probe-joint2312-v1.onnx")

    sha = create_sleep_probe_onnx(output_path)
    size = os.path.getsize(output_path)

    print(f"Sleep staging probe ONNX model created:")
    print(f"  Path: {output_path}")
    print(f"  Input: [batch, {N_INPUT}]")
    print(f"  Output: [batch, {N_OUTPUT}] (softmax probabilities)")
    print(f"  Size: {size} bytes")
    print(f"  SHA-256: {sha}")

    update_manifest(output_path, sha, size)
    print(f"  Manifest updated: manifest.json")

    # Also print in a format easy to copy into the registry
    print(f"\n# Add to sleep.registry.ts:")
    print(f'sha256: "{sha}",')
