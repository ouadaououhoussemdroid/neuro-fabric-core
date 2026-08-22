#!/usr/bin/env python3
"""
M40 — Create the Sleep Quality ONNX probe (Joint-2312 → 1-D regression).

Generates a lightweight linear regression head (2312-D → 1-D) that maps
Joint-2312 embeddings to a normalized sleep quality score in [0, 1].
The output is a single scalar in [0, 1] (representing sleep quality,
where 1 = excellent, 0 = poor).

This is a placeholder/random-init model — weights will be fine-tuned when
real Sleep-EDF Joint-2312 embeddings become available for training. The
architecture is correct: the service layer reads `class_0` from
`ONNXAdapter.predict()` and clamps to [0, 1].

Usage:
    python scripts/create_sleep_quality_probe.py

Output:
    public/models/sleep/quality-probe-joint2312-v1.onnx
    Prints the SHA-256 hash for registry registration.
"""
import hashlib
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


N_INPUT = 2312  # Joint-2312 embedding dimension
N_OUTPUT = 1     # Single regression output (normalized sleep quality [0, 1])
SEED = 42


def create_sleep_quality_probe_onnx(output_path: str) -> str:
    """Create a 2312→1 linear regression ONNX model and save to output_path."""
    rng = np.random.RandomState(SEED)

    # Weight and bias for linear layer: y = Wx + b, shape [1, 2312] and [1]
    W = rng.randn(N_OUTPUT, N_INPUT).astype(np.float32) * 0.01
    b = np.array([0.5], dtype=np.float32)  # Default midpoint (0.5 = moderate quality)

    # L2-normalize the weight row for balanced initial predictions
    W = W / (np.linalg.norm(W) + 1e-8)
    # Scale so initial predictions land near 0.5
    W = W * 0.01

    # Create ONNX initializer tensors
    W_tensor = numpy_helper.from_array(W, name="linear_weight")
    b_tensor = numpy_helper.from_array(b, name="linear_bias")

    # Input: [batch, 2312] float32
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", N_INPUT])
    # Output: [batch, 1] float32 (regression value, clamped to [0, 1] in service)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", N_OUTPUT])

    # Node 1: Gemm(input, W^T) + b → logits (transB=1 for [1, 2312]-shaped W)
    matmul = helper.make_node(
        "Gemm",
        inputs=["input", "linear_weight", "linear_bias"],
        outputs=["output"],
        name="linear",
        transB=1,
    )

    # Build the graph (no softmax — this is a regression head)
    graph = helper.make_graph(
        [matmul],
        "sleep_quality_probe",
        [input_tensor],
        [output_tensor],
        initializer=[W_tensor, b_tensor],
    )

    # Build the model
    model = helper.make_model(graph, producer_name="neuro-fabric-m40")
    model.opset_import[0].version = 13  # Stable opset with Gemm

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
    """Add the sleep quality probe to the models manifest."""
    import json

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_manifest = os.path.join(repo_root, "public", "models", "manifest.json")

    with open(full_manifest) as f:
        manifest = json.load(f)

    manifest["models"]["sleep-quality-probe-v1"] = {
        "id": "sleep-quality-probe-v1",
        "url": "/models/sleep/quality-probe-joint2312-v1.onnx",
        "sha256": sha,
        "size": size,
        "wasmCompatible": True,
    }

    with open(full_manifest, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(repo_root, "public", "models", "sleep")
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, "quality-probe-joint2312-v1.onnx")

    sha = create_sleep_quality_probe_onnx(output_path)
    size = os.path.getsize(output_path)

    print(f"Sleep quality probe ONNX model created:")
    print(f"  Path: {output_path}")
    print(f"  Input: [batch, {N_INPUT}]")
    print(f"  Output: [batch, {N_OUTPUT}] (regression, clamped to [0,1] in service)")
    print(f"  Size: {size} bytes")
    print(f"  SHA-256: {sha}")

    update_manifest(output_path, sha, size)
    print(f"  Manifest updated: manifest.json")

    # Also print in a format easy to copy into the registry
    print(f"\n# Add to sleep.registry.ts:")
    print(f'sha256: "{sha}",')
