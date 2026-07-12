# Production Readiness Assessment — Braindecode EEGConformer

- Date: 2026-06-17
- Component: `braindecode-eegconformer-prod` (ONNX, browser)
- Verdict: **Ready for staged rollout** behind a feature flag, with PCA
  fallback always available.

## Readiness checklist

| Area                               | Status | Evidence                                                                              |
| ---------------------------------- | :----: | ------------------------------------------------------------------------------------- |
| Model selected & justified         |   ✅   | `docs/audits/2026-06-17_braindecode-model-selection.md`                               |
| Input / preprocessing contract     |   ✅   | 22 ch, 250 Hz, 1000 samples; reuses `src/lib/eeg/preprocessing/*`                     |
| Embedding contract (dim, dtype)    |   ✅   | 32-D float32, L2-normalised via `embed()` facade                                      |
| Artefact preparation workflow      |   ✅   | `scripts/export_braindecode_eegconformer.py`, deployment guide §1                     |
| Deployment workflow                |   ✅   | `registerBraindecodeEEGConformer()` + URL artefact                                    |
| Registry integration               |   ✅   | `src/lib/ai/models/registry.ts`                                                       |
| Inference engine integration       |   ✅   | Routed via `embedEEG()` preferred model id                                            |
| Validation layer integration       |   ✅   | NaN/Inf/dim/zero checks in `validation/index.ts`                                      |
| Benchmark integration              |   ✅   | Covered by `benchmarkAll`; report attached                                            |
| Automatic fallback chain           |   ✅   | EEGConformer → generic ONNX → PCA                                                     |
| Unit/integration tests             |   ✅   | `src/lib/ai/models/__tests__/eegconformer-registration.test.ts` (+ existing 26 tests) |
| Backward compatibility (PCA, ONNX) |   ✅   | No code paths removed; default embedder unchanged                                     |
| Observability                      |   ✅   | Structured logs `ai.embed.*`, `ai.embedEEG.start`                                     |
| SSR safety                         |   ✅   | ONNX runtime imported lazily; capability probe before load                            |
| Rollback plan                      |   ✅   | `unregisterModel(id)`; vectors tagged with `modelId`                                  |
| Security review                    |   ✅   | Artefact fetched over HTTPS; no eval; no PII in logs                                  |
| Documentation                      |   ✅   | Selection report, deployment guide, benchmark, this assessment                        |

## Risks & mitigations

| Risk                                         | Likelihood | Impact | Mitigation                                                             |
| -------------------------------------------- | :--------: | :----: | ---------------------------------------------------------------------- |
| First-call latency (~500 ms cold)            |    High    |  Low   | Warm the session on app boot when the flag is on                       |
| Heap pressure on low-memory devices          |   Medium   | Medium | Capability probe + automatic PCA fallback                              |
| ONNX opset drift across browsers             |    Low     | Medium | Pin to opset 17; smoke test in CI                                      |
| Wrong electrode montage upstream             |   Medium   |  High  | Adapter validates `[C, T]` shape, throws precise error                 |
| Vector store contamination (mixed model ids) |   Medium   |  High  | `NeuralVectorIndex` tags entries with `modelId`; query side can filter |

## Sign-off criteria for full rollout

1. ≥ 1 000 successful inferences in staging with < 0.5 % fallback rate.
2. P95 latency on target devices < 600 ms.
3. Cosine recall@10 vs PCA improves by ≥ 15 % on the BCI-IV-2a holdout.
4. No security findings on the artefact host.

## Non-goals (this phase)

- EEGPT or other large foundation models.
- Server-side inference path (Option C in ADR 0001) — bridge already
  supports it via `artifact: { kind: "bytes", ... }`; activation is a
  future task.
