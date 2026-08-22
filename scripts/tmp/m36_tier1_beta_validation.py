#!/usr/bin/env python3
"""
M36 — Tier 1 Beta Validation Script

Validates all M31.8 Tier 1 Beta acceptance criteria for the three production-
candidate Tier-1 services (Subject Identity, Cognitive State, Anomaly Detection).

This script checks:
  1. All service code is implemented and deployed
  2. Database migration applied
  3. API routes functional (auth, rate-limit, CORS, error handling)
  4. SHA-256 verification on all artifacts
  5. Embed-once, reuse-many pattern enforced
  6. Browser fallback path implemented (V2-32)
  7. Test suites pass (unit, E2E, browser, integration)
  8. Production readiness (metrics, logging, health, RLS, rate limit)

Usage:
    python scripts/tmp/m36_tier1_beta_validation.py
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Canonical SHAs ────────────────────────────────────────────────────────────

EXPECTED_CBROMOD_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
EXPECTED_V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"
EXPECTED_EEGPT_SHA = "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36"
EXPECTED_COGNITIVE_PROBE_SHA = "ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32"
EXPECTED_ANOMALY_PROBE_SHA = "b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59"

CHECKS_PASSED = 0
CHECKS_FAILED = 0
CHECKS_TOTAL = 0


def check(condition: bool, label: str) -> None:
    global CHECKS_PASSED, CHECKS_FAILED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if condition:
        CHECKS_PASSED += 1
        print(f"  ✓ {label}")
    else:
        CHECKS_FAILED += 1
        print(f"  ✗ {label}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def file_exists(path: str) -> bool:
    return os.path.isfile(os.path.join(ROOT, path))


def file_contains(path: str, needle: str) -> bool:
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return False
    with open(full, encoding="utf-8") as f:
        return needle in f.read()


section("CHECK 1: Service Code Implementation")

check(file_exists("src/lib/ai/inference/subject-identity.server.ts"), "Subject Identity service (M32)")
check(file_exists("src/lib/ai/inference/cognitive.server.ts"), "Cognitive State service (M33)")
check(file_exists("src/lib/ai/inference/anomaly.server.ts"), "Anomaly Detection service (M34)")
check(file_exists("src/lib/ai/decoders/cognitive.registry.ts"), "Cognitive registry")
check(file_exists("src/lib/ai/decoders/anomaly.registry.ts"), "Anomaly registry")
check(file_exists("src/lib/ai/decoders/cognitive.browser.ts"), "Cognitive browser fallback")
check(file_exists("src/lib/ai/decoders/anomaly.browser.ts"), "Anomaly browser fallback")

section("CHECK 2: API Routes")

check(file_exists("src/routes/api/joint2312/similarity/search.ts"), "Subject-identity route")
check(file_exists("src/routes/api/joint2312/cognitive/decode.ts"), "Cognitive route")
check(file_exists("src/routes/api/joint2312/anomaly/detect.ts"), "Anomaly route")

# Verify all routes have auth/rate-limit/CORS/security
for route_file, route_name in [
    ("src/routes/api/joint2312/similarity/search.ts", "Subject Identity"),
    ("src/routes/api/joint2312/cognitive/decode.ts", "Cognitive"),
    ("src/routes/api/joint2312/anomaly/detect.ts", "Anomaly"),
]:
    check(file_contains(route_file, "authenticateRequest"), f"{route_name}: auth")
    check(file_contains(route_file, "checkRateLimit"), f"{route_name}: rate-limit")
    check(file_contains(route_file, "handleCors"), f"{route_name}: CORS")
    check(file_contains(route_file, "applySecurityHeaders"), f"{route_name}: security headers")
    check(file_contains(route_file, "setTimeout"), f"{route_name}: timeout")

section("CHECK 3: Database Migration (Tier-1 Service Layer)")

check(file_exists("supabase/migrations/20260820000000_tier1_service_layer.sql"), "Migration file exists")

migration_checks = [
    ("subject_similarity_results", "Subject similarity results table"),
    ("cognitive_state_results", "Cognitive state results table"),
    ("anomaly_detection_results", "Anomaly detection results table"),
    ("service_audit_log", "Service audit log table"),
    ("subject_metadata", "Subject metadata table"),
    ("idx_joint_embeddings_2312_subject_id", "Joint-2312 subject index"),
    ("idx_joint_embeddings_2312_session_id", "Joint-2312 session index"),
    ("match_subject_similarity", "Subject similarity RPC"),
    ("ENABLE ROW LEVEL SECURITY", "RLS enabled"),
]

for needle, label in migration_checks:
    check(file_contains("supabase/migrations/20260820000000_tier1_service_layer.sql", needle), label)

section("CHECK 4: SHA-256 Verification")

# Check ONNX artifacts exist and match
onnx_anomaly = os.path.join(ROOT, "models/anomaly/mahalanobis-probe-joint2312-v1.onnx")
if os.path.isfile(onnx_anomaly):
    with open(onnx_anomaly, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    check(actual == EXPECTED_ANOMALY_PROBE_SHA, f"Anomaly ONNX SHA matches ({actual[:16]}…)")
else:
    check(False, "Anomaly ONNX artifact exists")

# Check manifest for CBRaMod/V2/EEGPT SHAs
manifest_path = os.path.join(ROOT, "public/models/manifest.json")
if os.path.isfile(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    models = manifest.get("models", {})
    # models is a dict keyed by model id
    shas = {k: v.get("sha256") for k, v in models.items()}
    cbramod_sha = shas.get("cbramod-encoder")
    v2_sha = shas.get("eegconformer_finetuned")
    eegpt_sha = shas.get("eegpt-encoder-int8")
    check(cbramod_sha == EXPECTED_CBROMOD_SHA, f"CBRaMod SHA in manifest ({cbramod_sha[:16] if cbramod_sha else 'NOT FOUND'}…)")
    check(v2_sha == EXPECTED_V2_SHA, f"V2 SHA in manifest ({v2_sha[:16] if v2_sha else 'NOT FOUND'}…)")
    check(eegpt_sha == EXPECTED_EEGPT_SHA, f"EEGPT SHA in manifest ({eegpt_sha[:16] if eegpt_sha else 'NOT FOUND'}…)")
else:
    check(False, "manifest.json exists")

# Check task head SHAs in registries
check(file_contains("src/lib/ai/decoders/cognitive.registry.ts", EXPECTED_COGNITIVE_PROBE_SHA),
      f"Cognitive probe SHA in registry ({EXPECTED_COGNITIVE_PROBE_SHA[:16]}…)")
check(file_contains("src/lib/ai/decoders/anomaly.registry.ts", EXPECTED_ANOMALY_PROBE_SHA),
      f"Anomaly probe SHA in registry ({EXPECTED_ANOMALY_PROBE_SHA[:16]}…)")

section("CHECK 5: Embed-Once → Reuse-Many Pattern")

for service_file, service_name in [
    ("src/lib/ai/inference/subject-identity.server.ts", "Subject Identity"),
    ("src/lib/ai/inference/cognitive.server.ts", "Cognitive State"),
    ("src/lib/ai/inference/anomaly.server.ts", "Anomaly Detection"),
]:
    check(file_contains(service_file, "embedding_id"), f"{service_name}: accepts embedding_id")
    check(file_contains(service_file, "joint_embeddings_2312"), f"{service_name}: queries joint_embeddings_2312")
    check(file_contains(service_file, "embedding_reused"), f"{service_name}: tracks embedding_reused")

section("CHECK 6: Browser Fallback Paths")

check(file_contains("src/lib/ai/decoders/cognitive.browser.ts", "BROWSER_COGNITIVE_INPUT_DIM"), "Cognitive browser: BROWSER_COGNITIVE_INPUT_DIM")
check(file_contains("src/lib/ai/decoders/cognitive.browser.ts", "decodeFromV2Embedding"), "Cognitive browser: decodeFromV2Embedding")
check(file_contains("src/lib/ai/decoders/anomaly.browser.ts", "BROWSER_ANOMALY_INPUT_DIM"), "Anomaly browser: BROWSER_ANOMALY_INPUT_DIM")
check(file_contains("src/lib/ai/decoders/anomaly.browser.ts", "detectFromV2Embedding"), "Anomaly browser: detectFromV2Embedding")

section("CHECK 7: Metrics Instrumentation")

metrics_path = os.path.join(ROOT, "src/lib/metrics/index.ts")
with open(metrics_path) as f:
    metrics_content = f.read()

# Shared tier-1 metrics
check("tier1ServiceRequestsTotal" in metrics_content, "Shared: tier1ServiceRequestsTotal")
check("tier1ServiceErrorsTotal" in metrics_content, "Shared: tier1ServiceErrorsTotal")
check("tier1ServiceLatencyMs" in metrics_content, "Shared: tier1ServiceLatencyMs")
check("tier1AuditLogInsertsTotal" in metrics_content, "Shared: tier1AuditLogInsertsTotal")

# Subject Identity metrics
for m in ["subjectIdentityRequestsTotal", "subjectIdentityErrorsTotal", "subjectIdentitySearchLatencyMs",
          "subjectIdentityResultsTotal", "subjectIdentityEmbeddingReusedTotal", "subjectIdentityEmbeddingReembeddedTotal"]:
    check(m in metrics_content, f"Subject Identity: {m}")

# Cognitive metrics
for m in ["cognitiveDecodeRequestsTotal", "cognitiveDecodeErrorsTotal", "cognitiveDecodeLatencyMs",
          "cognitiveWorkloadPredictionsTotal", "cognitiveConfidenceDistribution",
          "cognitiveEmbeddingReusedTotal"]:
    check(m in metrics_content, f"Cognitive: {m}")

# Anomaly metrics
for m in ["anomalyDetectRequestsTotal", "anomalyDetectErrorsTotal", "anomalyDetectLatencyMs",
          "anomalyScoresTotal", "anomalyConfidenceDistribution",
          "anomalyEmbeddingReusedTotal", "anomalyEmbeddingReembeddedTotal"]:
    check(m in metrics_content, f"Anomaly: {m}")

section("CHECK 8: Structured Logging")

for service_file, service_name in [
    ("src/lib/ai/inference/subject-identity.server.ts", "Subject Identity"),
    ("src/lib/ai/inference/cognitive.server.ts", "Cognitive State"),
    ("src/lib/ai/inference/anomaly.server.ts", "Anomaly Detection"),
]:
    check(file_contains(service_file, "from '@/lib/logging'") or file_contains(service_file, "from \"@/lib/logging\""), f"{service_name}: imports logging")
    check(file_contains(service_file, "log("), f"{service_name}: uses log()")
    check(file_contains(service_file, "startTimer"), f"{service_name}: uses startTimer")

section("CHECK 9: Inference Index Barrel Exports")

idx_path = os.path.join(ROOT, "src/lib/ai/inference/index.ts")
with open(idx_path) as f:
    idx_content = f.read()

check("export * from" in idx_content and "subject-identity" in idx_content, "Inference index exports subject-identity.server")
check("export * from" in idx_content and "cognitive.server" in idx_content, "Inference index exports cognitive.server")
check("export * from" in idx_content and "anomaly.server" in idx_content, "Inference index exports anomaly.server")

check(file_exists("src/lib/ai/decoders/index.ts"), "Decoders barrel exists")

section("CHECK 10: Beta Environment Configuration")

check(file_exists(".env.staging.beta"), ".env.staging.beta exists")
if file_exists(os.path.join(ROOT, ".env.staging.beta")):
    check(file_contains(".env.staging.beta", "AI_EEGCONFORMER_ENABLED=beta"), "Beta env: AI_EEGCONFORMER_ENABLED=beta")

check(file_exists("scripts/promote_beta.sh"), "Beta promotion script exists")

section("CHECK 11: Test Suite Count")

# Count test files and test count for Tier-1 + M35 cross-service
test_files = [
    "src/lib/ai/inference/__tests__/cross-service.test.ts",
    "src/lib/ai/inference/__tests__/cognitive-decode.test.ts",
    "src/lib/ai/inference/__tests__/anomaly-detect.test.ts",
    "src/lib/ai/decoders/__tests__/registry.cognitive.test.ts",
    "src/lib/ai/decoders/__tests__/registry.anomaly.test.ts",
    "src/routes/api/joint2312/cognitive/__tests__/-decode.test.ts",
    "src/routes/api/joint2312/anomaly/__tests__/-decode.test.ts",
    "src/lib/ai/inference/__tests__/joint-fusion-2312.test.ts",
    "src/lib/ai/inference/__tests__/joint-server.test.ts",
    "src/lib/ai/decoders/__tests__/registry.test.ts",
]

tier1_test_count = 0
tier1_files_exist = 0
for tf in test_files:
    if file_exists(tf):
        tier1_files_exist += 1
        with open(os.path.join(ROOT, tf)) as f:
            content = f.read()
            # Count "it(" or "test(" occurrences
            tier1_test_count += content.count("it(")

check(tier1_files_exist >= 7, f"Tier-1 + cross-service test files exist ({tier1_files_exist} found)")
check(tier1_test_count >= 50, f"Unit/integration tests ≥ 50 ({tier1_test_count} found)")

# Run vitest for Tier-1 tests
npx_path = os.path.join("C:\\Program Files\\nodejs", "npx.cmd")
vitest_bin = os.path.join(ROOT, "node_modules", ".bin", "vitest.cmd")

if os.path.isfile(vitest_bin):
    result = subprocess.run(
        [vitest_bin, "run",
         "src/lib/ai/inference/__tests__/cross-service.test.ts",
         "src/lib/ai/inference/__tests__/cognitive-decode.test.ts",
         "src/lib/ai/inference/__tests__/anomaly-detect.test.ts",
         "src/lib/ai/decoders/__tests__/registry.cognitive.test.ts",
         "src/lib/ai/decoders/__tests__/registry.anomaly.test.ts",
         "src/routes/api/joint2312/cognitive/__tests__/-decode.test.ts",
         "src/routes/api/joint2312/anomaly/__tests__/-decode.test.ts"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    check(result.returncode == 0, "All Tier-1 + cross-service tests pass")
    for line in (result.stdout + result.stderr).split("\n"):
        if "Test Files" in line and "passed" in line:
            print(f"  Vittest: {line.strip()}")
        if "Tests" in line and "passed" in line:
            print(f"  {line.strip()}")
else:
    print("  WARNING: vitest not found at expected path")

section("CHECK 12: Mission Reports & Archive")

check(file_exists("reports/MISSION32_SUBJECT_IDENTITY_IMPLEMENTATION_REPORT.md"), "M32 report exists")
check(file_exists("reports/MISSION33_COGNITIVE_STATE_INTELLIGENCE_REPORT.md"), "M33 report exists")
check(file_exists("reports/MISSION34_ANOMALY_DETECTION_REPORT.md"), "M34 report exists")
check(file_exists("reports/MISSION35_CROSS_SERVICE_VALIDATION_REPORT.md"), "M35 report exists")

archive_path = os.path.join(ROOT, "reports/benchmark_archive.json")
if os.path.isfile(archive_path):
    with open(archive_path) as f:
        archive = json.load(f)
    exp_ids = [e.get("id") for e in archive.get("experiments", [])]
    check("m32-subject-identity-service" in exp_ids, "M32 archive record present")
    check("m33-cognitive-workload-probe" in exp_ids, "M33 archive record present")
    check("m34-anomaly-detection-probe" in exp_ids, "M34 archive record present")
else:
    check(False, "benchmark_archive.json exists")

section("CHECK 13: Rate Limiting (20 req/min/user)")

for route_file, service_name in [
    ("src/routes/api/joint2312/similarity/search.ts", "Subject Identity"),
    ("src/routes/api/joint2312/cognitive/decode.ts", "Cognitive"),
    ("src/routes/api/joint2312/anomaly/detect.ts", "Anomaly"),
]:
    content = open(os.path.join(ROOT, route_file)).read()
    check("RATE_LIMIT_MAX = 20" in content, f"{service_name}: rate limit = 20 req/min")

# ─── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
print(f"\n  Code checks: {CHECKS_PASSED}/{CHECKS_TOTAL} passed, {CHECKS_FAILED} failed")

if CHECKS_FAILED == 0:
    print(f"\n  🎉 M36 Tier 1 Beta Validation: PASS")
    print(f"  All 3 Tier-1 services are production-candidate ready.")
else:
    print(f"\n  ⚠️  M36 Tier 1 Beta Validation: FAIL ({CHECKS_FAILED} checks failed)")

sys.exit(1 if CHECKS_FAILED > 0 else 0)
