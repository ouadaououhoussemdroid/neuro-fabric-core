# ADR 0001 — EEG Foundation-Model Execution Strategy

- Status: **Accepted** (2026-06-17)
- Owners: Neuro-Fabric AI layer
- Supersedes: placeholder Braindecode bridge from `docs/braindecode-integration.md`

## Context

Phases 1–4 of the AI Foundation Layer left the platform with a real ONNX
runtime, a registry, validation, fallback chain, and a stub Braindecode
adapter that throws unless an external bridge is injected. We now need real
neural embeddings while preserving every existing pipeline (PCA, ONNX, vector
search, dashboards).

Four execution strategies were considered.

## Options

| #   | Strategy                                         | Where it runs | Real model? | Browser cost                                   | SSR safe?                      | Maintenance                                                |
| --- | ------------------------------------------------ | ------------- | ----------- | ---------------------------------------------- | ------------------------------ | ---------------------------------------------------------- |
| A   | Braindecode via Pyodide + PyTorch wheel          | Browser       | Yes         | ~80 MB cold start, no WebGPU, single-threaded  | No (window/document at import) | High — PyTorch on Pyodide is unofficial, frequent breakage |
| B   | Braindecode exported to ONNX → ONNX Runtime Web  | Browser       | Yes         | 5–30 MB per model, WASM SIMD + optional WebGPU | Yes (lazy import)              | Low — same artefact format as every other ONNX model       |
| C   | Hybrid: browser ONNX + optional server inference | Both          | Yes         | Same as B in browser; server path is opt-in    | Yes                            | Medium — requires a server endpoint contract               |
| D   | Hosted-only inference (REST)                     | Server        | Yes         | 0                                              | Yes                            | Highest infra cost; no offline path                        |

### Decision Matrix (1 = poor, 5 = excellent)

| Criterion                 | A: Pyodide | B: ONNX | C: Hybrid | D: Server-only |
| ------------------------- | :--------: | :-----: | :-------: | :------------: |
| Technical feasibility     |     2      |    5    |     4     |       5        |
| Browser compatibility     |     2      |    5    |     5     |       5        |
| Performance (P50 latency) |     2      |    4    |     4     |       3        |
| Memory consumption        |     1      |    4    |     4     |       5        |
| Maintainability           |     2      |    5    |     3     |       4        |
| Scalability               |     3      |    4    |     5     |       4        |
| Production readiness      |     2      |    5    |     4     |       5        |
| Fits current architecture |     2      |    5    |     4     |       3        |
| **Total / 40**            |   **16**   | **37**  |  **33**   |     **34**     |

## Decision

**Option B — Braindecode exported to ONNX, executed through the existing
`ONNXAdapter`.**

The Braindecode JS adapter is preserved but now satisfied by a real
`BraindecodeBridge` backed by `onnxruntime-web` (see
`src/lib/ai/adapters/braindecode-onnx-bridge.ts`). Operators export a
trained Braindecode model to ONNX once (offline, with `torch.onnx.export`),
host the file, and call `registerBraindecodeONNX({ artifact, ... })`.

Option C remains available: the bridge accepts any `artifact` source
(URL, ArrayBuffer, bytes), so a future server-side fetch returning ONNX
bytes plugs in without changing the adapter.

## Consequences

- No new runtime dependency: reuses already-installed `onnxruntime-web`.
- Same fallback chain applies automatically: Braindecode-ONNX → ONNX → PCA.
- PCA, generic ONNX, vector search, dashboards untouched.
- Pyodide path is not removed — `setBraindecodeBridge()` still accepts a
  custom bridge for research builds.

## References

- `src/lib/ai/adapters/braindecode-onnx-bridge.ts`
- `src/lib/ai/inference/embed-eeg.ts`
- `docs/ai-layer-architecture.md`
- `docs/braindecode-integration.md`
