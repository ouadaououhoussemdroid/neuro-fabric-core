#!/usr/bin/env python3
"""
T-036 / Mission 13 — on-demand real-EDF serving cross-check bridge documenter.

Companion to the committed TS serving test (`foundation-serving-m13.test.ts`),
which downloads a real PhysioNet EEGMMIDB EDF (S001R05), runs the FULL Tier-2
serving pipeline (parseEDF -> selectCbraModChannels -> resampleSignal(250) ->
preprocess -> embedFoundationWindows over the real 22 MB cbramod-encoder.onnx),
and asserts the resulting 200-D vector lands in the cached CBraMod manifold.

This script does NOT retrain or re-embed: it reuses the Mission-11/13 cached real
embeddings (`.cbramod_cross_session_cache.npz`) and the dumped subset
(`m13_embedding_subset.json`) plus the already-validated retrieval results
(`m13_retrieval_results.json`) to document the subset-level bridge statistics
and reference the TS real-EDF forward numbers. Run on demand:

    python scripts/tmp/m13_edf_serving_xcheck.py

Output: reports/m13_edf_serving_xcheck.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NPZ = ROOT / "reports" / ".cbramod_cross_session_cache.npz"
SUBSET = REPORTS / "m13_embedding_subset.json"
RETRIEVAL = REPORTS / "m13_retrieval_results.json"
OUT = REPORTS / "m13_edf_serving_xcheck.json"


def _import_numpy():
    try:
        import numpy as np  # noqa: F401
        return np
    except Exception:
        return None


def main() -> None:
    np = _import_numpy()
    report: dict = {
        "script": "scripts/tmp/m13_edf_serving_xcheck.py",
        "purpose":
            "Document the subset-level bridge (cached real embeddings) + reference "
            "the committed TS real-EDF serving test (S001R05 -> 22MB ONNX forward -> "
            "200-D -> manifold landing). Does NOT retrain or re-embed.",
        "ts_real_edf_serving_test":
            "src/lib/ai/inference/__tests__/foundation-serving-m13.test.ts",
        "ts_real_edf_result": {
            "edf": "S001R05.edf (PhysioNet EEGMMIDB, ~2.6MB, HTTP 200)",
            "pipeline":
                "parseEDF -> selectCbraModChannels(19) -> resampleSignal(250) -> "
                "preprocess(bandpass[4,38],segment{4,0.5}) -> embedFoundationWindows "
                "(onnxruntime-node CPU EP, cbramod-encoder.onnx)",
            "sha256_verified": "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
            "artifact_bytes": 22018587,
            "query_vector_dim": 200,
            "query_L2_norm": 1.0,
            "max_sim_to_subset": 0.9922281170875898,
            "mean_cosine_to_subject_1_vectors": 0.9867780556646127,
            "interpretation":
                "The real-EDF 200-D query lands in the learned CBraMod manifold "
                "(maxSim 0.9922, consistent with the ~0.99 subset cluster band).",
        },
    }

    # Subset-level bridge stats (recompute from the dumped subset, no model work).
    if SUBSET.exists():
        sub = json.loads(SUBSET.read_text())
        report["subset"] = {
            "n_vectors": len(sub["vectors"]),
            "dim": sub["dim"],
            "note": "Real CBraMod 200-D vectors dumped from .cbramod_cross_session_cache.npz",
        }
        if np is not None:
            vecs = np.array([v["vector"] for v in sub["vectors"]], dtype=np.float64)
            subj = np.array([v["meta"]["subject"] for v in sub["vectors"]], dtype=int)
            # L2-normalise (idempotent) then LOO Recall@K + NN gap.
            n0 = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / n0
            r1 = r5 = r10 = 0
            same_s = diff_s = 0.0
            ns = nd = 0
            for q in range(len(vecs)):
                sims = vecs @ vecs[q]
                sims[q] = -np.inf
                order = np.argsort(-sims)
                if subj[order[0]] == subj[q]:
                    r1 += 1
                if (subj[order[:5]] == subj[q]).any():
                    r5 += 1
                if (subj[order[:10]] == subj[q]).any():
                    r10 += 1
                if subj[order[0]] == subj[q]:
                    same_s += sims[order[0]]; ns += 1
                else:
                    diff_s += sims[order[0]]; nd += 1
            report["subset"]["leave_one_out_recall"] = {
                "recall_at_1": r1 / len(vecs),
                "recall_at_5": r5 / len(vecs),
                "recall_at_10": r10 / len(vecs),
            }
            report["subset"]["nn_gap"] = {
                "same_subject_nn_cosine": same_s / ns,
                "diff_subject_nn_cosine": diff_s / nd,
                "gap": (same_s / ns) - (diff_s / nd),
                "note": "Positive gap => same-subject NN is more similar (CBraMod separates subjects).",
            }

    # Reference the already-validated Python retrieval results.
    if RETRIEVAL.exists():
        rr = json.loads(RETRIEVAL.read_text())
        report["validated_retrieval_results"] = {
            "recall_at_5": {
                "cbramod_200": rr["recall_at_k"]["cbramod_200"]["recall_at_5"],
                "v2_32": rr["recall_at_k"]["v2_32"]["recall_at_5"],
                "pca_32": rr["recall_at_k"]["pca_32"]["recall_at_5"],
            },
            "nn_gap": rr.get("cosine_descriptors"),
            "statistical_comparisons": rr.get("statistical_comparisons"),
            "retrieval_quality_gate": rr.get("retrieval_quality_gate"),
            "source": "scripts/tmp/m13_tier2_retrieval_benchmark.py (reran Mission-11 harness on cached embeddings)",
        }

    # Note the pgvector RPC leg explicitly as INCONCLUSIVE (no DB in this env).
    report["pgvector_rpc_leg"] = {
        "exercised": False,
        "verdict": "INCONCLUSIVE",
        "blocker": "Docker daemon unavailable (npipe:////./pipe/dockerDesktopLinuxEngine) -> `npx supabase start` cannot run a real pgvector; no psql/postgres binary. The match_foundation_embeddings RPC could NOT be executed against a live database.",
        "compensating_control":
            "The identical cosine metric is validated via the NeuralVectorIndex in-memory "
            "fallback (this script + foundation-retrieval.m13.test.ts), and the RPC name "
            "is asserted wired to the foundation namespace in the route test.",
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
