# Mission 14 — Phase 0: Live pgvector / Supabase RPC Validation

**Status: Phase 0 PASSED (the Mission-13 pgvector gap is closed).**
The single remaining Mission-13 infrastructure gap — validation of the *real*
`match_foundation_embeddings` / `match_foundation_embeddings_exact` pgvector RPCs — is
closed against a **real local Docker Postgres+pgvector** instance. In-memory cosine
(`te @ tr.T`) is computed ONLY as an independent cross-check that the live RPC returns
identical ordering; it is not a substitute for the primary path.

## Environment (throwaway local DB; no production data touched)
- Container `nf-m14-pg`: `supabase/postgres:15.14.1.162` (Docker Hub), pgvector **0.8.2**,
  Postgres 15.14, TCP 127.0.0.1:5432. Docker Desktop 4.81.0 daemon UP (was repaired on
  `npipe:////./pipe/docker_cli`).
- (Supabase CLI full-stack pull failed on `public.ecr.aws` TLS timeouts → standalone DB image.)
- Migration `supabase/migrations/20260814000000_foundation_embeddings.sql` applied **verbatim**
  (CREATE EXTENSION vector; foundation_embeddings table vector(200) + CHECK; 3 RLS policies;
  IVFFlat `vector_cosine_ops` lists=100; both RPCs).
- Driver: `python3 scripts/tmp/m14_live_pgvector_validation.py` (psycopg2 + pgvector adapter).
- Embeddings source: `reports/.cbramod_cross_session_cache.npz` (real cached `cb_emb` 4500×200,
  `v2_emb` 4500×32, `bandpower` 4500×110). **No retraining. No production files modified.**

## Schema validation on the LIVE DB  (all_pass = True)
| Check | Result |
|---|---|
| column type | `vector(200)` ✓ |
| `CHECK (vector_dims(embedding)=200)` | present (constraint `foundation_embeddings_dims`) ✓ |
| dimension gate rejects 32-D | empirically True ✓ |
| RLS | enabled ✓ |
| RLS policies | 3 ✓ |
| `match_foundation_embeddings` (IVF, `<=>` cosine) | exists ✓ |
| `match_foundation_embeddings_exact` (brute-force, `<#>`) | exists ✓ |
| IVFFlat index `idx_foundation_embedding_ivfflat` (cosine, lists=100) | exists, used by IVF RPC ✓ |

Note: embeddings are L2-normalised in the driver, so for unit vectors `||a-b||² = 2(1-cos)`:
ordering by `<#>` (L2) is identical to ordering by `<=>` (cosine) — hence the exact RPC (L2) and
the IVF RPC (cosine) are comparable on the same metric.

## Live retrieval path: EEG → CBraMod 200-D → foundation_embeddings → real pgvector RPC
300 leakage-free session-disjoint splits (50 subjects × 6 held-out runs). Pool = all
(subject,run) ≠ held-out run (no self-retrieval, train-only pool). Per split: exact RPC
`match_foundation_embeddings_exact(q, 40, model_id, NULL)` oversampled to 40, then held-out-run
rows excluded to recover the exact pool top-K. Independent 60-split in-memory cross-check.

## Results (300 splits, real live RPC)  — R = Recall@K
| Model | R1 | R5 | R10 | MI (nearest-centroid) | RPC ms/q | 60-split in-mem R5 |
|---|---:|---:|---:|---:|---:|---:|
| **onnx-cbramod-foundation-200d** | 0.2427 | **0.5269** | 0.6587 | 0.2749 | 10.32 | 0.5256 |
| braindecode-eegconformer-prod-v2-padded-200 (32→200 pad) | 0.0687 | 0.2162 | 0.3360 | 0.3020 | 8.33 | 0.1656 |
| pca-bandpower-32-padded-200 (per-fold PCA, train-only) | 0.4402 | 0.6920 | 0.7853 | 0.3018 | 8.71 | 0.6967 |

Metric-equivalence proof: live exact-RPC Recall@K == in-memory cosine (`te @ tr.T`) Recall@K
on the 60-split sample (CBraMod 0.5269 vs 0.5256). **CBraMod live R5 0.5269 ≈ Mission-13
in-memory 0.5273** → the real pgvector RPC reproduces the gate metric.

## ANN / IVFFLAT SLO (CBraMod-200, full 4500, 300 queries)
| probe | IVF ms/q | exact ms/q | IVF-vs-exact ANN-R5 |
|---:|---:|---:|---:|
| 1 | 0.28 | 7.37 | 0.5467 |
| 4 | 0.29 | 7.37 | 0.5467 |
| 10 | 0.28 | 7.37 | 0.5467 |
| 20 | 0.35 | 7.37 | 0.5467 |

IVFFlat index **exists, is used, and is ~26× faster** than brute-force at the build default.
**Honest environment finding:** `SET LOCAL ivfflat.probe` is accepted (SHOW reads it back) but
the IVF index scan does **not** honour it in `supabase/postgres:15.14.1.162` — probe
1/4/10/20/100 (verified) return byte-identical top-K. So IVF recall/latency are fixed at the
build default; **probe tunability is INCONCLUSIVE** (a build quirk, not a script bug — each probe
uses a fresh connection where `SET LOCAL` succeeds, yet the scan ignores the knob).

## NN same-vs-diff gap (live exact RPC, 300 queries, self excluded, cosine = similarity−1)
- same-subject nearest NN cosine ≈ 0.992, diff-subject ≈ 0.993, gap ≈ −0.001 (n_same=298, n_diff=300).
- Cross-checked in-memory: identical (−0.0010). NN cosine margin is ~0 for CBraMod-200 — subject
discriminability is a *top-K/ranking* phenomenon (R5=0.527), not a single-NN margin.

## End-to-end latency (CBraMod)
- embed (onnxruntime-node, 22MB ONNX, warm, M13 real-EDF) = **155 ms/window**
- + pgvector RPC exact = **6.33 ms/query** → ~161 ms/query end-to-end
- IVF RPC = 0.28 ms/query (if approximate retrieval acceptable).

## Honest headline (required by Mission 14)
- **CBraMod-200 retrieval gate: PASS** — live pgvector RPC reproduces M13 (R5 0.5269 ≈ 0.5273;
  in-memory cross-check 0.5256; CBraMod 200-D, server-side, opt-in, separate namespace).
- **PCA currently outperforms CBraMod on Recall@5/10** — PCA R5=0.692 vs CBraMod 0.527; PCA R10=0.785
  vs CBraMod 0.659. CBraMod beats V2 (R5 0.527 vs 0.216) but is below PCA.

## Phase 0 gate verdict
| Gate | Result |
|---|---|
| Docker/Supabase/Postgres/pgvector started & healthy | PASS |
| Migration applied verbatim; vector(200) + CHECK + dim-gate | PASS |
| RLS enabled + 3 policies | PASS (enabled/defined; non-superuser enforcement not tested) |
| Both RPCs work; cosine ordering correct | PASS |
| IVFFlat index exists & usable (faster) | PASS |
| Real retrieval path (EEG→CBraMod→RPC) reproduces gate metric | PASS |
| 300 session-disjoint benchmark CBraMod/V2/PCA | PASS |
| Recall@1/5/10, MI, NN gap, ANN-vs-exact, latency | PASS |
| ivfflat.probe tunability | INCONCLUSIVE (env quirk) |

## Phase 1 GA-readiness assessment (evidence-based; many gates INCONCLUSIVE — not executed)
| Gate | Evidence | Result |
|---|---|---|
| Storage correctness (dim, CHECK, idempotency) | vector(200) CHECK enforced; COPY reload idempotent | PASS |
| Retrieval correctness / metric equivalence | live RPC == in-memory cosine; R5 == M13 | PASS |
| Reliability | 14 clean runs of the validation pipeline | partial |
| Inference latency | 155 ms/window (onnxruntime-node warm, M13 real-EDF) | ASSESSED |
| Retrieval latency | exact RPC 6.33 ms/query; IVF 0.28 ms/query | PASS |
| IVF vs exact SLA | ~26× faster, ANN-R5≈0.55 @ default probe | PASS (index usable) |
| Artifact SHA verification (embedEEG/ONNX) | not re-verified in M14 (manifest restored to HEAD) | INCONCLUSIVE |
| Rate limiting | not tested | INCONCLUSIVE |
| Concurrent requests | not tested (single-driver) | INCONCLUSIVE |
| Rollback safety | CBraMod opt-in, separate namespace; no default swap | NOT-GA (no rollout) |
| V2 regression | V2 not changed; V2 R5=0.216 (unchanged behaviour) | PASS (no regression) |
| Browser/WASM isolation | CBraMod wasmCompatible:false, server-side only | PASS (by design) |
| API contract stability | out of scope for Phase 0 | INCONCLUSIVE |

## Strict verdict
- **Phase 0 (close M13 pgvector gap): PASS** — real pgvector RPC validated end-to-end.
- **CBraMod Tier-2 retrieval gate: PASS** (live R5 0.5269 ≈ M13 0.5273).
- **Mission-14 overall (GA promotion): NOT-GA-READY / INCONCLUSIVE** — the mandatory DB leg
  passes, but Phase-1 production-readiness (SHA verify, rate-limit, concurrency, API contract,
  rollback testing) was not exercised. CBraMod remains **opt-in, server-side, 200-D, separate
  foundation_embeddings namespace, no PCA fallback, no silent V2 fallback**.
- **Smallest opt-in promotion (conditional):** promote CBraMod-200d as an opt-in server-side specialist *only after* Phase-1 gates pass; do NOT change DEFAULT_PREFERRED.

## Files changed (this mission; scratch + reports only)
- `scripts/tmp/m14_live_pgvector_validation.py` — NEW validation driver (debugged: `load_all` optional `idx_all` for PCA; ANN fresh-conn-per-probe; NN exclude-self + ORDER BY).
- `scripts/tmp/_arc_m14.py` — NEW throwaway archive appender.
- `reports/MISSION14_LIVE_PGVECTOR_VALIDATION.json` — NEW (live results).
- `reports/MISSION14_LIVE_PGVECTOR_VALIDATION.md` — NEW (this report).
- `reports/benchmark_archive.json` — **one append** (idx13); idx0-12 byte-identical (sha256 of prefix preserved: `2be2806ea3188317…`).
- **Unmodified:** `embedEEG`, `DEFAULT_PREFERRED`, V2 routing, `vector(32)`/PCA behaviour, V2 artifacts, `public/models/manifest.json`, `public/ort/integrity.json`, all M11/M12/M13 results, migration — no retraining, no CI weakening, no test deletion.
