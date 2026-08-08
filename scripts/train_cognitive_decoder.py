"""T-025 — Train the cognitive decoder v0 (logistic regression on band-power).

Replaces the heuristic ratio-based cognitive-state decoder with a calibrated
logistic regression trained on band-power features. Exports to ONNX with a
single tensor output (`probabilities`, shape [None, 3]) so it runs cleanly
through the existing onnxruntime-web path without non-tensor SEQUENCE outputs
that onnxruntime-web cannot materialise.

Usage:
    python scripts/train_cognitive_decoder.py --out public/models/cognitive-decoder-v0.onnx

The model takes 5 band-power features (δ, θ, α, β, γ) as input and
outputs 3 calibrated probabilities: attention, workload, arousal.

Training data: since no public attention/workload dataset is bundled,
the script generates a synthetic calibration set derived from known
band-power→cognitive-state relationships (documented in the EEG
literature). Replace with real data when available (T-019 manifest).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic band-power features with known cognitive labels.

    The labels follow documented EEG relationships:
      - attention ~ β/(α+θ), high when beta power is high
      - workload ~ θ/α, high when theta is high relative to alpha
      - arousal ~ β+γ, high when beta+gamma power is high
    """
    rng = np.random.default_rng(seed)
    # Base band powers (δ, θ, α, β, γ) — normalized to unit sum.
    raw = rng.dirichlet([1, 1, 1, 1, 1], size=n) * 100  # scale to percentage
    X = raw.astype(np.float32)

    # Generate labels from the known relationships + noise.
    attention = 1 / (1 + np.exp(-(X[:, 3] / (X[:, 2] + X[:, 1] + 0.1) - 0.5) * 5))
    workload = 1 / (1 + np.exp(-(X[:, 1] / (X[:, 2] + 0.1) - 0.5) * 5))
    arousal = np.clip((X[:, 3] + X[:, 4]) / 100, 0, 1)

    # Add noise and binarize for classification (threshold at 0.5).
    y = np.column_stack([
        (attention + rng.normal(0, 0.05, n) > 0.5).astype(int),
        (workload + rng.normal(0, 0.05, n) > 0.5).astype(int),
        (arousal + rng.normal(0, 0.05, n) > 0.5).astype(int),
    ])
    return X, y


def _train_three_models(X: np.ndarray, y: np.ndarray) -> list:
    """Train three independent LogisticRegression pipelines (scaler + LR).

    Using separate models instead of MultiOutputClassifier lets us build a
    clean ONNX graph with a single tensor output, avoiding the non-tensor
    SEQUENCE output that MultiOutputClassifier + skl2onnx produces and that
    onnxruntime-web cannot materialise.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    models = []
    for i, name in enumerate(["attention", "workload", "arousal"]):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        clf.fit(X, y[:, i])
        models.append(clf)
    return models


def _export_onnx(models, args) -> None:
    """Build a clean ONNX model from the trained pipeline parameters.

    The graph implements: for each of the 3 outputs,
      scaled = (input - mean) / scale
      logit  = scaled @ coef^T + intercept   (Gemm with transB=True)
      prob   = sigmoid(logit)
    A final Concat (axis=1) produces the [None, 3] `probabilities` tensor.

    This avoids skl2onnx's MultiOutputClassifier conversion, which wraps
    probabilities in a non-tensor SEQUENCE that onnxruntime-web cannot read.
    """
    import onnx
    import onnx.helper as helper
    import onnx.numpy_helper as numpy_helper

    scalers = [m.named_steps["scaler"] for m in models]
    lrs = [m.named_steps["clf"] for m in models]

    input_info = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, 5])
    output_info = helper.make_tensor_value_info("probabilities", onnx.TensorProto.FLOAT, [None, 3])

    nodes = []
    prob_names = []
    for i in range(3):
        mean = scalers[i].mean_.astype(np.float32)
        scale = scalers[i].scale_.astype(np.float32)
        coef = lrs[i].coef_[0].astype(np.float32)            # shape [5]
        intercept = np.array([lrs[i].intercept_[0]], dtype=np.float32)  # shape [1]

        nodes.append(helper.make_node("Constant", [], [f"mean_{i}"], value=numpy_helper.from_array(mean)))
        nodes.append(helper.make_node("Constant", [], [f"scale_{i}"], value=numpy_helper.from_array(scale)))
        nodes.append(helper.make_node("Constant", [], [f"coef_{i}"], value=numpy_helper.from_array(coef.reshape(1, 5))))
        nodes.append(helper.make_node("Constant", [], [f"intercept_{i}"], value=numpy_helper.from_array(intercept)))

        # (X - mean) / scale
        nodes.append(helper.make_node("Sub", ["input", f"mean_{i}"], [f"sub_{i}"]))
        nodes.append(helper.make_node("Div", [f"sub_{i}", f"scale_{i}"], [f"scaled_{i}"]))
        # logits = scaled @ coef^T + intercept  (Gemm with transB=True)
        nodes.append(helper.make_node("Gemm", [f"scaled_{i}", f"coef_{i}", f"intercept_{i}"], [f"logit_{i}"], transB=True))
        nodes.append(helper.make_node("Sigmoid", [f"logit_{i}"], [f"prob_{i}"]))
        prob_names.append(f"prob_{i}")

    # Concat [batch,1] × 3 → [batch, 3] along axis 1
    nodes.append(helper.make_node("Concat", prob_names, ["probabilities"], axis=1))

    graph = helper.make_graph(nodes, "cognitive_decoder", [input_info], [output_info])
    opset = [helper.make_opsetid("", args.opset)]
    model = helper.make_model(graph, opset_imports=opset, producer_name="neuro-fabric-cognitive-decoder")
    model.ir_version = 8
    onnx.checker.check_model(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(model.SerializeToString())
    print(f"[decoder] exported → {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cognitive decoder v0")
    ap.add_argument("--out", type=Path, required=True, help="Output .onnx path")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    X, y = _generate_synthetic_data(args.n_samples, args.seed)
    print(f"[decoder] synthetic data: X={X.shape} y={y.shape}")

    models = _train_three_models(X, y)
    print("[decoder] training complete")

    # Cross-validation accuracy per output.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    for i, name in enumerate(["attention", "workload", "arousal"]):
        scores = cross_val_score(LogisticRegression(max_iter=1000), X, y[:, i], cv=5, scoring="accuracy")
        print(f"[decoder] {name}: cv_accuracy={scores.mean():.4f} ± {scores.std():.4f}")

    # Export to ONNX with a clean single-tensor output.
    try:
        _export_onnx(models, args)
    except ImportError:
        print("[decoder] onnx package not installed — skipping ONNX export")
        print("[decoder] Install with: pip install onnx onnxruntime")
        return

    # Validate with onnxruntime (Python) and verify parity with sklearn.
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out.as_posix(), providers=["CPUExecutionProvider"])
    print(f"[decoder] ONNX inputs: {sess.get_inputs()}")
    print(f"[decoder] ONNX outputs: {sess.get_outputs()}")

    test_input = X[:5].astype(np.float32)
    outputs = sess.run(None, {"input": test_input})
    print(f"[decoder] ONNX output shapes: {[np.asarray(o).shape for o in outputs]}")

    # Verify parity with sklearn predict_proba.
    max_diff = 0.0
    for i in range(5):
        for j in range(3):
            sklearn_prob = models[j].predict_proba(test_input)[i, 1]
            onnx_prob = outputs[0][i, j]
            max_diff = max(max_diff, abs(float(sklearn_prob) - float(onnx_prob)))
    print(f"[decoder] max |sklearn - onnx| parity diff: {max_diff:.8f}")
    assert max_diff < 1e-5, f"Parity check failed: max diff {max_diff}"
    print("[decoder] ONNX smoke test + parity validation passed")


if __name__ == "__main__":
    main()
