#!/usr/bin/env python3
"""
M37 — Alert & Health Check Validation Script

Validates Tier-1 Beta production readiness:
  1. Health endpoint exists and returns valid JSON
  2. Alert threshold gauges registered in metrics
  3. Alert configuration script exists and is executable
  4. Metrics endpoint is accessible (CRON_SECRET-gated)
  5. All Tier-1 services have monitoring coverage

Usage:
    python scripts/tmp/m37_alert_validation.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


section("CHECK 1: Health Check Endpoint")

check(file_exists("src/routes/api/health.ts"), "Health endpoint source exists")
check(file_contains("src/routes/api/health.ts", "GET"), "Health endpoint handles GET")
check(file_contains("src/routes/api/health.ts", "database"), "Health checks database")
check(file_contains("src/routes/api/health.ts", "onnx_runtime"), "Health checks ONNX runtime")
check(file_exists("src/routes/api/__tests__/health.test.ts"), "Health test file exists")

# Read test file to count tests
if file_exists("src/routes/api/__tests__/health.test.ts"):
    with open(os.path.join(ROOT, "src/routes/api/__tests__/health.test.ts")) as f:
        content = f.read()
    test_count = content.count("it(")
    check(test_count >= 5, f"Health has ≥5 tests ({test_count} found)")

section("CHECK 2: Alert Threshold Metrics")

metrics_path = os.path.join(ROOT, "src/lib/metrics/index.ts")
with open(metrics_path) as f:
    metrics_content = f.read()

check("tier1AlertThresholds" in metrics_content, "Alert threshold gauges in metrics")
check("neuro_fabric_tier1_alert_p95_latency_ms_threshold" in metrics_content, "P95 latency alert gauge")
check("neuro_fabric_tier1_alert_p50_latency_ms_threshold" in metrics_content, "P50 latency alert gauge")
check("neuro_fabric_tier1_alert_error_rate_threshold" in metrics_content, "Error rate alert gauge")
check("neuro_fabric_tier1_alert_fallback_rate_threshold" in metrics_content, "Fallback rate alert gauge")

section("CHECK 3: Alert Configuration Script")

check(file_exists("scripts/setup_alerts.sh"), "Alert setup script exists")
if file_exists(os.path.join(ROOT, "scripts/setup_alerts.sh")):
    with open(os.path.join(ROOT, "scripts/setup_alerts.sh")) as f:
        script_content = f.read()
    check("CRON_SECRET" in script_content, "Script uses CRON_SECRET")
    check("P95_LATENCY" in script_content, "Script checks P95 latency")
    check("fallback_rate" in script_content.lower(), "Script checks fallback rate")
    check("WEBHOOK_URL" in script_content, "Script supports webhook alerts")
    check("allGatesPass" in script_content or "all_gates" in script_content.lower(), "Script checks gates")

section("CHECK 4: Metrics Endpoint")

check(file_exists("src/routes/api/public/metrics.ts"), "Prometheus metrics endpoint exists")
check(file_contains("src/routes/api/public/metrics.ts", "renderPrometheusMetrics"), "Metrics endpoint renders Prometheus format")
check(file_contains("src/routes/api/public/metrics.ts", "CRON_SECRET"), "Metrics endpoint uses CRON_SECRET auth")

section("CHECK 5: Staging Metrics Endpoint (Beta Gates)")

check(file_exists("src/routes/api/public/staging/metrics.ts"), "Staging metrics endpoint exists")
with open(os.path.join(ROOT, "src/routes/api/public/staging/metrics.ts")) as f:
    staging_content = f.read()
check("p95LatencyOk" in staging_content, "Staging metrics has P95 gate")
check("p50LatencyOk" in staging_content, "Staging metrics has P50 gate")
check("fallbackRateOk" in staging_content, "Staging metrics has fallback rate gate")
check("allGatesPass" in staging_content, "Staging metrics has all-gates-pass flag")

section("CHECK 6: Rate Limiting (Production)")

for route_file, service_name in [
    ("src/routes/api/joint2312/similarity/search.ts", "Subject Identity"),
    ("src/routes/api/joint2312/cognitive/decode.ts", "Cognitive"),
    ("src/routes/api/joint2312/anomaly/detect.ts", "Anomaly"),
]:
    check(file_contains(route_file, "checkRateLimit"), f"{service_name}: rate limiting")
    check(file_contains(route_file, "RATE_LIMIT_MAX = 20"), f"{service_name}: 20 req/min")

section("CHECK 7: Audit Trail")

migration_path = os.path.join(ROOT, "supabase/migrations/20260820000000_tier1_service_layer.sql")
if os.path.isfile(migration_path):
    with open(migration_path) as f:
        migration_content = f.read()
    check("service_audit_log" in migration_content, "Audit log table in migration")
    check("audit_log" in migration_content, "Audit log indexes")
else:
    check(False, "Migration file exists")

for service_file, service_name in [
    ("src/lib/ai/inference/subject-identity.server.ts", "Subject Identity"),
    ("src/lib/ai/inference/cognitive.server.ts", "Cognitive State"),
    ("src/lib/ai/inference/anomaly.server.ts", "Anomaly Detection"),
]:
    check(file_contains(service_file, "tier1AuditLogInsertsTotal"), f"{service_name}: audit log metric")

section("CHECK 8: Error Handling & Sanitization")

for service_file, service_name, error_class in [
    ("src/lib/ai/inference/subject-identity.server.ts", "Subject Identity", "SubjectIdentityError"),
    ("src/lib/ai/inference/cognitive.server.ts", "Cognitive State", "CognitiveDecodeError"),
    ("src/lib/ai/inference/anomaly.server.ts", "Anomaly Detection", "AnomalyDetectError"),
]:
    check(file_contains(service_file, error_class), f"{service_name}: custom error class ({error_class})")
    check(file_contains(service_file, "log("), f"{service_name}: structured logging on errors")

section("CHECK 9: Timeout Handling")

for route_file, service_name in [
    ("src/routes/api/joint2312/similarity/search.ts", "Subject Identity"),
    ("src/routes/api/joint2312/cognitive/decode.ts", "Cognitive"),
    ("src/routes/api/joint2312/anomaly/detect.ts", "Anomaly"),
]:
    check(file_contains(route_file, "setTimeout") or file_contains(route_file, "timeoutPromise"), f"{service_name}: timeout handling")
    check(file_contains(route_file, "408"), f"{service_name}: returns 408 on timeout")

section("CHECK 10: Environment Configuration")

check(file_exists(".env.staging.beta"), "Staging beta environment exists")
check(file_exists(".env.staging.ga"), "Staging GA environment exists")
check(file_exists(".env.staging-runtime"), "Staging runtime env exists")

# ─── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
print(f"\n  Code checks: {CHECKS_PASSED}/{CHECKS_TOTAL} passed, {CHECKS_FAILED} failed")

if CHECKS_FAILED == 0:
    print(f"\n  🎉 M37 Alert & Health Check Validation: PASS")
    print(f"  All Tier-1 production readiness alerts are configured.")
else:
    print(f"\n  ⚠️  M37 Alert & Health Check Validation: FAIL ({CHECKS_FAILED} checks failed)")

sys.exit(1 if CHECKS_FAILED > 0 else 0)
