# T-035 — Beta Deployment Report (Mission 5)

**Date:** 2026-08-12T22:36:38Z  
**Status:** ✅ Beta promotion complete, monitoring active, all gates passing  
**Rollout Stage:** Beta (50% cohort routing)

---

## 1. What Was Changed

### Server Configuration
- **Env file:** `.env.staging.beta` — `AI_EEGCONFORMER_ENABLED=beta` (50% rollout)
- **Server:** Vite dev server (`--mode staging.beta`) on port 5173
- **Env var injection:** `AI_EEGCONFORMER_ENABLED=beta` set via shell environment (Nitro/TanStack Start loads `.env` files independently of Vite mode, so explicit shell env is required for `process.env` access in server handlers)

### Code Changes (Staging Infrastructure Only)
1. **`src/lib/metrics/index.ts`** — Added finer-grained histogram buckets between 400ms–1000ms (25ms granularity at 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950ms) to enable accurate P95 interpolation near the 600ms gate. Old buckets had a 500ms gap between 500ms and 1000ms buckets, causing P95 interpolation to overestimate latencies by up to 200ms.

2. **`src/routes/api/public/staging/metrics.ts`** — Removed redundant structured-metrics processing from the POST handler. The handler was previously processing both the Prometheus text (via `mergeBrowserMetrics`) and the structured metrics object, causing counter values to double-count. Now only the Prometheus text is processed (it contains all metrics: counters, gauges, histograms).

3. **`scripts/staging_traffic_generator.py`** — Increased benchmark iterations from 20 to 100 per cycle for statistically stable P95 estimates (20 samples were sensitive to outliers; 100 samples provide a representative percentile). Also changed `STAGING_URL` default to port 5173 (matching the beta server).

### No Changes to Production
- **Production `.env`** remains `AI_EEGCONFORMER_ENABLED=off` — unchanged
- **No model retraining** — T-035 artifact used as-is
- **No GA promotion** — beta is 50%, not 100%
- **No unrelated models modified** (EEGPT, CBraMod, etc.)

### Artifact Provenance
| Field | Value |
|-------|-------|
| Model ID | `eegconformer_finetuned` |
| Registry ID | `braindecode-eegconformer-prod-v2` |
| SHA-256 | `18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931` |
| WASM Compatible | ✅ true |
| Manifest Match | ✅ Verified |

---

## 2. Beta Rollout Percentage

- **50%** (beta stage)
- **Cohort routing:** djb2 hash of `userId` → 0–99, threshold = 50 (`hashUserId(userId) < 50`)
- **Rollout stages:** off (0%) → canary (5%) → beta (50%) → ga (100%)
- **Current:** `beta` = 50% of authenticated users receive EEGConformer; remaining 50% fall back to PCA
- **Fallback chain:** EEGConformer → ONNX → PCA (`DEFAULT_EMBEDDER_ID = "pca-legacy-v1"`)
- **Rollback:** `fallbackToPCA: true` by default in `embed()` — if EEGConformer fails, system automatically falls back to PCA

---

## 3. Current Metrics (Latest Snapshot)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| P50 latency | 267.65 ms | < 400 ms | ✅ PASS |
| P95 latency | 400.00 ms | < 600 ms | ✅ PASS |
| Mean latency | 272.53 ms | — | — |
| Fallback rate | 0% | < 0.5% | ✅ PASS |
| Artifact verification failures | 0 | 0 | ✅ PASS |
| Total embeds (cumulative) | 1,266 | — | — |
| Model selection fallbacks | 0 | — | ✅ PASS |
| Artifact verifications | 1,266 pass, 0 fail | — | ✅ PASS |
| All gates pass | **true** | — | ✅ |

### Browser-side benchmark (100-sample cycles, 3 warmup excluded):

| Cycle | P50 | P95 | P99 | Min | Max | Mean | Fallbacks |
|-------|-----|-----|-----|-----|-----|------|-----------|
| 1 | 219.9ms | 369.6ms | 581.6ms | 188.5ms | 722.5ms | 249.7ms | 0 |
| 2 | 220.7ms | 414.9ms | 523.8ms | 186ms | 744ms | 246.6ms | 0 |
| 3 | 240.4ms | 397.1ms | 562.8ms | 186.9ms | 610.4ms | 268.8ms | 0 |
| 4 | 254.5ms | 540.7ms | — | — | — | 303.4ms | 0 |

All cycles show P95 < 600ms and P50 < 400ms. No fallbacks, no verification failures.

---

## 4. Beta Monitoring Status

| Component | Status | Details |
|-----------|--------|---------|
| Observation script | ✅ Running | PID 3164, polling `GET /api/public/staging/metrics` every 60s for 24h |
| Traffic generator | ✅ Running | PID 3165, launching headless Chromium every 10 min (600s) for 24h |
| Staging server | ✅ Running | Port 5173, `--mode staging.beta`, `AI_EEGCONFORMER_ENABLED=beta` |
| Metrics endpoint | ✅ Active | `GET /api/public/staging/metrics` (CRON_SECRET auth) |
| Metrics POST | ✅ Active | POST merges browser WASM metrics into server registry |

**Observation files:**
- `reports/t035_staging_observation.jsonl` — 60-second polling snapshots
- `reports/t035_traffic_gen_observation.jsonl` — per-cycle benchmark results

---

## 5. What Remains Before GA

1. **Complete 24-hour observation period** — Continue collecting metrics with sustained traffic to verify stability under sustained load (currently ~6 cycles / ~10 min in)

2. **Run `promote_ga.sh` dry-run** — Verify GA promotion readiness with artifact provenance check and gate validation:
   ```bash
   ./scripts/promote_beta.sh --dry-run --env-file=.env.staging.ga
   ```

3. **Cohort routing validation with real user IDs** — The traffic generator calls `embedEEG()` directly with explicit model IDs, bypassing the djb2 hash cohort check. Beta routing (50% threshold) will be validated when real users access the app through the API route handler, which sets `setRolloutStage` per-request from the env var.

4. **Firefox compatibility** — WASM SIMD threading is not fully optimized in Firefox's WebAssembly runtime (~4.6x slower than Chrome's V8/TurboFan). All benchmarks are calibrated on Chrome 124. Firefox is tested in CI but does not meet the same latency gate. Production recommendation: Chrome/Chromium-only for EEGConformer WASM inference.

5. **Error rate tracking** — Currently tracking fallback rate (0%) as the proxy for inference errors. A dedicated `inference_error_rate` counter should be added to distinguish WASM runtime errors from model fallbacks.

---

## Rollback Procedure

If beta metrics degrade:
```bash
# 1. Set env to canary (5%) or off (0%)
# 2. Or kill the beta server — fallback chain EEGConformer → ONNX → PCA activates automatically
# 3. The promote_beta.sh script provides --dry-run verification before any env change
```

Rollback is fully available via the `fallbackToPCA` default in `embed()` and the `AI_EEGCONFORMER_ENABLED` env var override.
