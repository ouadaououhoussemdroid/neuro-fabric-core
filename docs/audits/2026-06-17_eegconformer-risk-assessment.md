# EEGConformer — Risk Assessment

- Date: 2026-06-17
- Scope: risks introduced by obtaining and deploying the first real
  EEGConformer artefact (Track A from the acquisition report).
- Out of scope: EEGPT, EEG2IMG, architectural redesign.

## Risk matrix

| #   | Risk                                                               | Category    |          Likelihood           | Impact |   Severity   | Mitigation                                                                                                                                 | Owner      |
| --- | ------------------------------------------------------------------ | ----------- | :---------------------------: | :----: | :----------: | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| R1  | No public pretrained checkpoint exists; in-house training required | Acquisition |             High              | Medium |   **Med**    | Train on BCI-IV-2a (Braindecode tutorial), 1–2 d effort                                                                                    | ML         |
| R2  | Dataset licence prevents weight redistribution                     | Legal       | Low (BCI-IV-2a is permissive) |  High  |     Med      | Restrict v1 training to BCI-IV-2a + PhysioNet MI; document in MODEL_CARD                                                                   | ML + Legal |
| R3  | ONNX export drift — attention op incompatibility                   | Technical   |              Low              | Medium |     Low      | Pin `torch==2.3`, `opset=17`; `onnx.checker` + parity smoke test in `scripts/export_braindecode_eegconformer.py`                           | ML         |
| R4  | Browser ORT version mismatch (opset 17 attention)                  | Technical   |              Low              | Medium |     Low      | Pin `onnxruntime-web>=1.18`; CI smoke test on Chromium + Safari                                                                            | FE         |
| R5  | First-call latency (~500 ms cold load) noticed by users            | UX          |             High              |  Low   |     Low      | Warm session on app boot when flag enabled                                                                                                 | FE         |
| R6  | Heap pressure on low-memory devices                                | UX          |            Medium             | Medium |     Med      | Capability probe + automatic PCA fallback (already wired)                                                                                  | FE         |
| R7  | Overfitting to BCI-IV-2a degrades cross-paradigm recall            | Quality     |             High              | Medium |     Med      | Plan Track-B (PhysioNet MI → BCI-IV-2a fine-tune); gate GA on cross-subject holdout                                                        | ML         |
| R8  | Mixed-model vectors contaminate similarity results                 | Data        |            Medium             |  High  | **Med-High** | `NeuralVectorIndex` tags entries with `modelId`; queries filter by id (already implemented)                                                | BE         |
| R9  | Artefact host outage or CORS misconfiguration                      | Ops         |              Low              |  High  |     Med      | Host on Lovable Cloud Storage with immutable hashed path; CORS allow-list to prod + preview origins; PCA fallback covers any fetch failure | Ops        |
| R10 | Wrong electrode montage upstream                                   | Data        |            Medium             |  High  | **Med-High** | Adapter validates `[22, 1000]` shape and throws precise error; upstream tutorial documents BCI-IV-2a montage                               | BE         |
| R11 | Silent regression vs PCA on existing benchmarks                    | Quality     |              Low              | Medium |     Low      | Always benchmark both adapters in CI; fail build if EEGConformer recall@10 < PCA recall@10                                                 | ML         |
| R12 | Rollback leaves orphaned vectors                                   | Data        |              Low              |  Low   |     Low      | Vectors are model-tagged; `unregisterModel()` is a soft toggle; no migration required                                                      | BE         |
| R13 | Artefact tampering / supply-chain                                  | Security    |              Low              |  High  |     Med      | Pin sha256 in artefact registry; verify after fetch; serve over HTTPS only                                                                 | Sec        |
| R14 | PII leakage through embeddings                                     | Privacy     |              Low              |  High  |     Med      | 32-D attention-pooled features are not invertible to raw EEG; no PII in logs; document in security review                                  | Sec        |
| R15 | Licence misclassification on publish                               | Legal       |              Low              |  High  |     Med      | Pre-publish MODEL_CARD checklist (dataset, code, weights, intended use)                                                                    | ML + Legal |

## Top three risks to actively manage

1. **R8 / R10 — Data integrity around model-tagged vectors and montage.**
   Already mitigated in code; the action is to keep these invariants
   **tested** in CI (existing suite covers them — do not regress).
2. **R7 — Cross-paradigm quality.** Bound the v1 claim to BCI-IV-2a-like
   inputs in the MODEL_CARD; schedule Track-B before broadening claims.
3. **R2 — Licence posture.** Easy to get right on day one, expensive to
   undo after publishing weights. Lock to BCI-IV-2a / PhysioNet MI for
   v1 and review before any TUH / SEED inclusion.

## Acceptance gates before GA

- ≥ 1 000 staging inferences with < 0.5 % fallback rate.
- P95 latency < 600 ms on target devices.
- Cosine recall@10 ≥ PCA + 15 pp on BCI-IV-2a holdout.
- MODEL_CARD reviewed and signed off.
- Rollback drill (`unregisterModel`) executed in staging in < 5 min.

## Residual risk after mitigations

**Low-to-Medium.** No risk in this register is a blocker; all
mitigations rely on infrastructure that already exists in the
repository. The deployment is a one-artefact + one-flag change with PCA
as a permanent safety net.
