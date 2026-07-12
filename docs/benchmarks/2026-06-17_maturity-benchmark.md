# NeuroWeave Maturity Benchmark

- **Date:** 2026-06-17
- **Purpose:** Permanent, time-series record of platform maturity. New rows
  should be appended (never overwritten) at each future audit so the trend
  is preserved.
- **Companion audit:** `docs/audits/2026-06-17_strategic-progress-audit.md`

## 1. Maturity Time Series (0–10 unless noted)

| Snapshot | Date               | Architecture |  AI | EEG | Embeddings | Reconstruction | Cognitive | Security | Overall (/100) |
| -------- | ------------------ | -----------: | --: | --: | ---------: | -------------: | --------: | -------: | -------------: |
| Audit #1 | 2026-06-06         |          3.0 | 0.0 | 7.0 |        2.0 |            0.0 |       3.0 |      1.0 |             22 |
| Audit #2 | 2026-06-17 (early) |          5.5 | 3.0 | 7.5 |        3.0 |            1.0 |       3.0 |      5.0 |             41 |
| Current  | 2026-06-17 (late)  |          7.5 | 6.5 | 7.5 |        6.0 |            1.0 |       3.0 |      5.0 |         **58** |
| T-028    | 2026-07-12         |          8.5 | 8.0 | 8.5 |        8.0 |            1.0 |       4.0 |      7.0 |         **72** |

## 2. Readiness Time Series (%)

| Snapshot | EEG Platform | AI Infra | Foundation Model |    API |   SaaS |
| -------- | -----------: | -------: | ---------------: | -----: | -----: |
| Audit #1 |           35 |        0 |                0 |     20 |     10 |
| Audit #2 |           60 |       20 |                5 |     40 |     30 |
| Current  |       **70** |   **85** |           **75** | **45** | **40** |
| T-028    |       **80** |   **90** |           **85** | **65** | **50** |

## 3. Blueprint Completion Time Series (%)

| Snapshot |  Score |
| -------- | -----: |
| Audit #1 |     14 |
| Audit #2 |     28 |
| Current  | **42** |
| T-028    | **65** |

## 4. Evolution Charts (ASCII)

```text
Overall Maturity         AI Infra (0–10)          Foundation Model %
A1 ██████        22      A1                0.0    A1                   0
A2 ███████████   41      A2 █████          3.0    A2 █                 5
Now ████████████████ 58  Now ████████████  6.5    Now ████████████████ 75
T28 ██████████████████████ 72 T28 ████████████████ 8.0  T28 █████████████████ 85
```

## 5. Methodology

- Scores are repository-evidence driven (file presence, test pass count,
  module surface area, doc coverage).
- Weighting for the Blueprint score is fixed (see audit §9) so trends are
  comparable across snapshots.
- New snapshots **append** to each table; never rewrite history.

## 6. Append-Only Protocol

At the next audit:

1. Add a new row to §1, §2, §3 with the snapshot date.
2. Extend the ASCII chart in §4 with a new line.
3. Do not edit prior rows even if scoring criteria evolve — record the change
   in a footnote instead.
