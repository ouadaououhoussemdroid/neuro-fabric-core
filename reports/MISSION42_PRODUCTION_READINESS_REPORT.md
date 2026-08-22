# M42 — Production Readiness & Browser WASM Deployment (Tier-4)

## Mission Summary

**M42 extends M15's operational validation gates (A1–A4) to Tier-2/Tier-3 sleep task routes** and adds browser WASM smoke tests for sleep task heads.

After M41's multi-task fusion completed Tier-3, the user approved **Option A** — Production Readiness & Browser WASM Deployment. M42 closes the 4 operational gates that M15 Phase 1–Phase 5 validated for the CBraMod foundation path, but which were still **INCONCLUSIVE** for the sleep staging, sleep quality, and fusion routes:

| Gate | Description | M15 Status (foundation) | M42 Action |
|------|-------------|------------------------|------------|
| A1 | Browser WASM runtime | PASS (Tier-2 path) | Add sleep task head browser WASM smoke tests |
| A2 | Signed artifact SHA under load | PASS (Tier-2 path) | Extend to sleep probes + fusion (SHA already verified) |
| A3 | Rate-limit / concurrency | PASS (Tier-2 path) | Add live rate-limit tests for sleep + fusion routes |
| A4 | Real JWT auth | PASS (Tier-2 path) | Add live JWT auth tests for sleep + fusion routes |

## Validation Gates Extended

### Gate A4 — Real JWT Authentication (live)

Three new live test files extend M15 Phase 1's real JWT/auth validation pattern to the Tier-2/Tier-3 routes:

| Route | Test File | Tests | Mocked | Real |
|-------|-----------|-------|--------|------|
| POST /api/joint2312/sleep/decode | `-decode-jwt-auth-live.test.ts` | 8 | `decodeSleepState` | `authenticateRequest`, `checkRateLimit`, PG/RSL on `joint_embeddings_2312` |
| POST /api/joint2312/sleep/quality | `-quality-jwt-auth-live.test.ts` | 8 | `decodeSleepQuality` | `authenticateRequest`, `checkRateLimit`, PG/RSL on `joint_embeddings_2312` |
| POST /api/joint2312/fusion | `-fusion-jwt-auth-live.test.ts` | 9 | `decodeJoint2312` | `authenticateRequest`, `checkRateLimit`, PG/RSL on `joint_embeddings_2312` |

Each test file verifies:
1. **Valid JWT → 200**: Real GoTrue JWT validation + decode service invoked with authenticated `userId`
2. **No Bearer token → 401**: Auth gate enforces before any inference
3. **Invalid JWT (tampered signature) → 401**: GoTrue rejects tampered tokens
4. **Expired JWT → 401**: GoTrue enforces expiry (`exp` claim)
5. **RLS isolation**: User A cannot read User B's embeddings via `embedding_id` (PG RLS `auth.uid() = user_id`)
6. **User ID override rejected**: Client-supplied `userId` field in JSON body ignored — only JWT identity used

### Gate A3 — Real Rate-Limit Testing (live)

Three new live test files extend M15 Phase 2's rate-limit pattern:

| Route | Test File | Tests | RPC |
|-------|-----------|-------|-----|
| POST /api/joint2312/sleep/decode | `-decode-rate-limit-live.test.ts` | 5 | `check_rate_limit` (real PostgREST) |
| POST /api/joint2312/sleep/quality | `-quality-rate-limit-live.test.ts` | 5 | `check_rate_limit` (real PostgREST) |
| POST /api/joint2312/fusion | `-fusion-rate-limit-live.test.ts` | 5 | `check_rate_limit` (real PostgREST) |

Each test file verifies:
1. **20 within budget → 200**: All 20 requests succeed
2. **21st → 429 + `retry_after_ms`**: Atomic UPSERT race-free boundary
3. **Per-user isolation**: User B has independent 20 req/min budget
4. **Concurrent bypass blocked**: 21 simultaneous → exactly 20 succeed, 1 gets 429
5. **Invalid JWT → 401, not 429**: Auth fails before rate-limit check; budget not consumed

### Gate A1 — Browser WASM Smoke Tests

**New file**: `tests/browser/sleep-task-heads-wasm-smoke-firefox.test.ts`

Tests the real browser-compatible sleep task head decoders inside actual Chromium and Firefox browsers.

#### Group 1: Sleep Staging (`detectSleepFromV2Embedding`)
- Valid 5-class softmax prediction from V2-32 embedding
- Stage ID ↔ label mapping (W=0, N1=1, N2=2, N3=3, REM=4)
- Probabilities sum to 1.0 (softmax contract)
- Confidence = max probability in [0, 1]
- Dimension validation (throws on non-32-D)

#### Group 2: Sleep Quality (`browserSleepQuality`)
- Score clamped to [0, 1] from V2-32 embedding
- Quality band derivation (poor < 0.4, fair < 0.6, good < 0.8, excellent ≥ 0.8)
- Confidence from distance to nearest band boundary
- Dimension validation (throws on wrong dim)

#### Group 3: End-to-End (`embedEEG` → V2-32 → sleep decoders)
- Real EEGConformer V2 WASM inference → 32-D embedding
- Feed real embedding to `detectSleepFromV2Embedding` → valid staging
- Feed real embedding to `browserSleepQuality` → valid quality
- SHA-256 verification pass asserted (artifact integrity)
- WASM binary loaded from `/ort/` (200 response + Performance API)
- Canary metrics: `modelSelectedTotal` + `artifactVerificationTotal{pass}` incremented

#### Group 4: Trained Probe Weight Injection
- `setBrowserSleepWeights`: injects 5×32 linear probe weights
- `setBrowserSleepQualityWeights`: injects 32-D quality weights
- Wrong-dimension weights rejected (warning + no-op)

### Harness Update

**Modified**: `src/testing/harness.ts`

Added exports for the browser-compatible sleep decoders:
- `detectSleepFromV2Embedding` (V2-32 → 5-class sleep staging)
- `browserSleepQuality` (V2-32 → sleep quality score [0, 1])
- `setBrowserSleepWeights` (load trained sleep staging probe weights)
- `setBrowserSleepQualityWeights` (load trained sleep quality probe weights)
- `BROWSER_SLEEP_INPUT_DIM`, `BROWSER_SLEEP_OUTPUT_DIM`

These are the real production implementations from `src/lib/ai/decoders/sleep.browser.ts` — no stubs or duplicates.

## Architecture

```
                    POST /api/joint2312/sleep/decode          (Tier-2)
                    POST /api/joint2312/sleep/quality        (Tier-2)
                    POST /api/joint2312/fusion               (Tier-3)
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  CORS → Auth(Bearer JWT) → Rate-Limit    │
        │  (20 req/min) → Security Headers → 10s    │
        └──────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  decodeSleepState / decodeSleepQuality   │
        │  / decodeJoint2312                       │
        │  ─ Resolves embedding (embed-once-      │
        │    reuse-many on joint_embeddings_2312)  │
        │  ─ Runs ONNX probe(s) via onnxruntime    │
        │    (mocked in tests only)                │
        │  ─ Builds shared provenance via          │
        │    buildServiceProvenance()              │
        └──────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────┐
        │  Real local Supabase stack               │
        │  (GoTrue + PostgREST + Postgres +        │
        │   pgvector + RLS)                        │
        │  joint_embeddings_2312: vector(2312)     │
        │  check_rate_limit RPC (real)             │
        └──────────────────────────────────────────┘
```

## Security Pattern

All three routes follow the same M15 security pattern:

```
Auth (Bearer JWT) → rate-limit (20 req/min) → CORS → security headers → timeout (10s)
```

- **JWT**: Real GoTrue `/auth/v1/user` validation (signature + expiry)
- **Rate limit**: Real `check_rate_limit` RPC via PostgREST (atomic UPSERT, per-user isolation)
- **RLS**: `joint_embeddings_2312` table has `auth.uid() = user_id` row-level security
- **No silent fallback**: 401 on auth failure; 429 on rate limit; 400 on decode errors

## Constraints Honored

| Constraint | Status |
|-----------|--------|
| No model weights modified | ✅ |
| No ONNX artifacts modified | ✅ |
| No default preferred changed | ✅ |
| No PCA behavior modified | ✅ |
| Browser WASM bundle unchanged | ✅ (only test harness extended) |
| Live tests require local Supabase stack | ✅ (real GoTrue + PostgREST + Postgres) |

## Test Inventory

### Live JWT Auth Tests (24 tests)
- `src/routes/api/joint2312/sleep/__tests__/-decode-jwt-auth-live.test.ts` (8 tests)
- `src/routes/api/joint2312/sleep/__tests__/-quality-jwt-auth-live.test.ts` (8 tests)
- `src/routes/api/joint2312/__tests__/-fusion-jwt-auth-live.test.ts` (8 tests + 1 partial heads test)

### Live Rate-Limit Tests (15 tests)
- `src/routes/api/joint2312/sleep/__tests__/-decode-rate-limit-live.test.ts` (5 tests)
- `src/routes/api/joint2312/sleep/__tests__/-quality-rate-limit-live.test.ts` (5 tests)
- `src/routes/api/joint2312/__tests__/-fusion-rate-limit-live.test.ts` (5 tests)

### Browser WASM Smoke Tests (11 tests, browser-driven)
- `tests/browser/sleep-task-heads-wasm-smoke-firefox.test.ts` (4 groups, 11 tests)

### Validation Script
- `scripts/tmp/m42_production_readiness_validation.py`

## File Summary

### Created
| File | Purpose |
|------|---------|
| `src/routes/api/joint2312/sleep/__tests__/-decode-jwt-auth-live.test.ts` | Live JWT/RSL tests for sleep staging route |
| `src/routes/api/joint2312/sleep/__tests__/-quality-jwt-auth-live.test.ts` | Live JWT/RSL tests for sleep quality route |
| `src/routes/api/joint2312/__tests__/-fusion-jwt-auth-live.test.ts` | Live JWT/RSL tests for fusion route |
| `src/routes/api/joint2312/sleep/__tests__/-decode-rate-limit-live.test.ts` | Live rate-limit tests for sleep staging route |
| `src/routes/api/joint2312/sleep/__tests__/-quality-rate-limit-live.test.ts` | Live rate-limit tests for sleep quality route |
| `src/routes/api/joint2312/__tests__/-fusion-rate-limit-live.test.ts` | Live rate-limit tests for fusion route |
| `tests/browser/sleep-task-heads-wasm-smoke-firefox.test.ts` | Browser WASM smoke tests for sleep task heads |
| `scripts/tmp/m42_production_readiness_validation.py` | M42 validation script (code checks + archive checks + test runner) |
| `reports/MISSION42_PRODUCTION_READINESS_REPORT.md` | This report |

### Modified
| File | Change |
|------|--------|
| `src/testing/harness.ts` | Added sleep browser decoder exports (`detectSleepFromV2Embedding`, `browserSleepQuality`, `setBrowserSleepWeights`, `setBrowserSleepQualityWeights`) |
| `reports/benchmark_archive.json` | Appended `m42-production-readiness` record |

## Regressions

All existing tests (M39: 38, M40: 32, M41: 35 = 105 pre-existing unit + route tests) remain compatible — M42 adds new live and browser test files without modifying existing test logic.
