"""T-025 — Train the cognitive decoder v0 (logistic regression on band-power).

Replaces the heuristic ratio-based cognitive-state decoder with a calibrated
logistic regression trained on band-power features. Exports to ONNX via
skl2onnx so it runs through the existing onnxruntime-web path.

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cognitive decoder v0")
    ap.add_argument("--out", type=Path, required=True, help="Output .onnx path")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.multi_output import MultiOutputClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = _generate_synthetic_data(args.n_samples, args.seed)
    print(f"[decoder] synthetic data: X={X.shape} y={y.shape}")

    # Pipeline: standardize → logistic regression (one per output).
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000, C=1.0))),
    ])

    pipeline.fit(X, y)
    print("[decoder] training complete")

    # Cross-validation accuracy per output.
    for i, name in enumerate(["attention", "workload", "arousal"]):
        scores = cross_val_score(
            LogisticRegression(max_iter=1000), X, y[:, i], cv=5, scoring="accuracy"
        )
        print(f"[decoder] {name}: cv_accuracy={scores.mean():.4f} ± {scores.std():.4f}")

    # Export to ONNX via skl2onnx.
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, 5]))]
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=args.opset,
            options={id(pipeline): {"zipmap": False}},
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[decoder] exported → {args.out}")

        # Validate with onnxruntime.
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out.as_posix(), providers=["CPUExecutionProvider"])
        test_input = X[:5].astype(np.float32)
        outputs = sess.run(None, {"input": test_input})
        print(f"[decoder] ONNX output shapes: {[o.shape for o in outputs]}")
        print(f"[decoder] ONNX smoke test passed")

    except ImportError:
        print("[decoder] skl2onnx not installed — skipping ONNX export")
        print("[decoder] Install with: pip install skl2onnx onnxmltools")


if __name__ == "__main__":
    main()
