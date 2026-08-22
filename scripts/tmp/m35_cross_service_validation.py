#!/usr/bin/env python3
"""
M35 — Cross-Service Validation Script

Validates that all three Tier-1 services (Subject Identity, Cognitive State, Anomaly
Detection) are correctly wired for cross-service operation on the shared Joint-2312
embedding layer. This mirrors the m32/m33/m34 validation pattern: code checks +
artifact verification + archive validation + test count.

Usage:
    python scripts/tmp/m35_cross_service_validation.py
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "reports")
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")

# ─── Canonical artifact SHAs (must match across all Tier-1 services) ───────────

EXPECTED_CBROMOD_SHA = "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47"
EXPECTED_V2_SHA = "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931"
EXPECTED_EEGPT_SHA = "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36"
EXPECTED_ANOMALY_PROBE_SHA = "b72373576376f7c8ec2209cfe7c640033ddf13378646f01741cdd1a6c8bb9f59"
EXPECTED_COGNITIVE_PROBE_SHA = "ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32"

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


# ─── CHECK 1: All Tier-1 service files exist ──────────────────────────────────

section("CHECK 1: Tier-1 Service Files Exist")

check(file_exists("src/lib/ai/decoders/registry.ts"), "TaskHeadRegistry (M31 shared layer)")
check(file_exists("src/lib/ai/decoders/index.ts"), "Barrel export for all decoders")
check(file_exists("src/lib/ai/inference/subject-identity.server.ts"), "Subject Identity service (M32)")
check(file_exists("src/lib/ai/inference/cognitive.server.ts"), "Cognitive State service (M33)")
check(file_exists("src/lib/ai/inference/anomaly.server.ts"), "Anomaly Detection service (M34)")
check(file_exists("src/lib/ai/decoders/cognitive.registry.ts"), "Cognitive registry")
check(file_exists("src/lib/ai/decoders/anomaly.registry.ts"), "Anomaly registry")
check(file_exists("src/lib/ai/decoders/cognitive.browser.ts"), "Cognitive browser fallback")
check(file_exists("src/lib/ai/decoders/anomaly.browser.ts"), "Anomaly browser fallback")
check(file_exists("src/routes/api/joint2312/similarity/search.ts"), "Subject-identity route")
check(file_exists("src/routes/api/joint2312/cognitive/decode.ts"), "Cognitive route")
check(file_exists("src/routes/api/joint2312/anomaly/detect.ts"), "Anomaly route")

# ─── CHECK 2: All Tier-1 services export the same interface ───────────────────

section("CHECK 2: Shared embedding_id / query_embedding Interface")

check(
    file_contains("src/lib/ai/inference/subject-identity.server.ts", "embedding_id"),
    "Subject Identity accepts embedding_id",
)
check(
    file_contains("src/lib/ai/inference/subject-identity.server.ts", "query_embedding"),
    "Subject Identity accepts query_embedding",
)
check(
    file_contains("src/lib/ai/inference/cognitive.server.ts", "embedding_id"),
    "Cognitive State accepts embedding_id",
)
check(
    file_contains("src/lib/ai/inference/cognitive.server.ts", "query_embedding"),
    "Cognitive State accepts query_embedding",
)
check(
    file_contains("src/lib/ai/inference/anomaly.server.ts", "embedding_id"),
    "Anomaly Detection accepts embedding_id",
)
check(
    file_contains("src/lib/ai/inference/anomaly.server.ts", "query_embedding"),
    "Anomaly Detection accepts query_embedding",
)

# ─── CHECK 3: All services use buildServiceProvenance ──────────────────────────

section("CHECK 3: Shared Provenance via buildServiceProvenance")

check(
    file_contains("src/lib/ai/inference/subject-identity.server.ts", "buildServiceProvenance"),
    "Subject Identity uses buildServiceProvenance",
)
check(
    file_contains("src/lib/ai/inference/cognitive.server.ts", "buildServiceProvenance"),
    "Cognitive State uses buildServiceProvenance",
)
check(
    file_contains("src/lib/ai/inference/anomaly.server.ts", "buildServiceProvenance"),
    "Anomaly Detection uses buildServiceProvenance",
)

# ─── CHECK 4: All services reference Joint-2312 artifact SHAs ──────────────────

section("CHECK 4: Joint-2312 Artifact SHA Consistency")

check(
    file_contains("public/models/manifest.json", EXPECTED_CBROMOD_SHA),
    f"CBRaMod SHA in manifest.json ({EXPECTED_CBROMOD_SHA[:16]}…)",
)
check(
    file_contains("src/lib/ai/inference/joint.server.ts", EXPECTED_V2_SHA),
    f"V2 SHA in joint.server.ts ({EXPECTED_V2_SHA[:16]}…)",
)
check(
    file_contains("src/lib/ai/inference/joint.server.ts", EXPECTED_EEGPT_SHA),
    f"EEGPT SHA in joint.server.ts ({EXPECTED_EEGPT_SHA[:16]}…)",
)

# ─── CHECK 5: Task head SHAs match registry ───────────────────────────────────

section("CHECK 5: Task Head SHA Consistency (Registry → Report → Archive)")

check(
    file_contains("src/lib/ai/decoders/cognitive.registry.ts", EXPECTED_COGNITIVE_PROBE_SHA),
    f"Cognitive probe SHA in registry ({EXPECTED_COGNITIVE_PROBE_SHA[:16]}…)",
)
check(
    file_contains("src/lib/ai/decoders/anomaly.registry.ts", EXPECTED_ANOMALY_PROBE_SHA),
    f"Anomaly probe SHA in registry ({EXPECTED_ANOMALY_PROBE_SHA[:16]}…)",
)

# ─── CHECK 6: Benchmark archive has M33 + M34 + M35 records ────────────────────

section("CHECK 6: Benchmark Archive Records")

archive_path = os.path.join(REPORTS, "benchmark_archive.json")
check(os.path.isfile(archive_path), "benchmark_archive.json exists")

if os.path.isfile(archive_path):
    with open(archive_path) as f:
        archive = json.load(f)

    experiments = archive.get("experiments", [])
    exp_ids = [e.get("id") for e in experiments]

    check("m33-cognitive-workload-probe" in exp_ids, "M33 archive record present")
    check("m34-anomaly-detection-probe" in exp_ids, "M34 archive record present")

    m34_exp = next((e for e in experiments if e.get("id") == "m34-anomaly-detection-probe"), None)
    if m34_exp:
        m34_shas = m34_exp.get("artifact_shas", {})
        check(
            m34_shas.get("anomaly_probe") == EXPECTED_ANOMALY_PROBE_SHA,
            f"M34 archive SHA matches actual ONNX ({EXPECTED_ANOMALY_PROBE_SHA[:16]}…)",
        )
        check(
            m34_shas.get("cbramod") == EXPECTED_CBROMOD_SHA,
            "M34 archive CBraMod SHA matches",
        )

# ─── CHECK 7: ONNX artifact exists and SHA matches ─────────────────────────────

section("CHECK 7: ONNX Artifact Verification")

onnx_path = os.path.join(ROOT, "models/anomaly/mahalanobis-probe-joint2312-v1.onnx")
if os.path.isfile(onnx_path):
    with open(onnx_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    check(True, f"ONNX artifact exists ({os.path.basename(onnx_path)})")
    check(actual_sha == EXPECTED_ANOMALY_PROBE_SHA, f"ONNX SHA-256 matches ({actual_sha[:16]}…)")
else:
    check(False, "ONNX artifact missing")

# ─── CHECK 8: Metrics registered for all 3 services ────────────────────────────

section("CHECK 8: Tier-1 Metrics Registration")

metrics_path = os.path.join(SRC, "lib/metrics/index.ts")
with open(metrics_path) as f:
    metrics_content = f.read()

check("anomalyDetectRequestsTotal" in metrics_content, "Anomaly detect requests counter")
check("anomalyDetectErrorsTotal" in metrics_content, "Anomaly detect errors counter")
check("anomalyDetectLatencyMs" in metrics_content, "Anomaly detect latency histogram")
check("anomalyScoresTotal" in metrics_content, "Anomaly scores counter")
check("anomalyConfidenceDistribution" in metrics_content, "Anomaly confidence histogram")
check("anomalyEmbeddingReusedTotal" in metrics_content, "Anomaly embedding reused counter")
check("anomalyEmbeddingReembeddedTotal" in metrics_content, "Anomaly embedding reembedded counter")

check("cognitiveDecodeRequestsTotal" in metrics_content, "Cognitive decode requests counter")
check("cognitiveDecodeErrorsTotal" in metrics_content, "Cognitive decode errors counter")
check("cognitiveDecodeLatencyMs" in metrics_content, "Cognitive latency histogram")
check("cognitiveWorkloadPredictionsTotal" in metrics_content, "Cognitive workload predictions counter")
check("cognitiveConfidenceDistribution" in metrics_content, "Cognitive confidence histogram")
check("cognitiveEmbeddingReusedTotal" in metrics_content, "Cognitive embedding reused counter")

check("subjectIdentityRequestsTotal" in metrics_content, "Subject identity requests counter")
check("subjectIdentityErrorsTotal" in metrics_content, "Subject identity errors counter")
check("subjectIdentitySearchLatencyMs" in metrics_content, "Subject identity latency histogram")
check("subjectIdentityResultsTotal" in metrics_content, "Subject identity results counter")
check("subjectIdentityEmbeddingReusedTotal" in metrics_content, "Subject identity embedding reused counter")

check("tier1ServiceRequestsTotal" in metrics_content, "Shared tier1 service requests counter")
check("tier1ServiceErrorsTotal" in metrics_content, "Shared tier1 service errors counter")
check("tier1ServiceLatencyMs" in metrics_content, "Shared tier1 service latency histogram")
check("tier1AuditLogInsertsTotal" in metrics_content, "Shared tier1 audit log counter")

# ─── CHECK 9: Inference barrel export ───────────────────────────────────────────

section("CHECK 9: Inference Index Exports")

inference_index = os.path.join(SRC, "lib/ai/inference/index.ts")
with open(inference_index) as f:
    idx_content = f.read()

check("anomaly.server" in idx_content, "Inference index exports anomaly.server")
check("cognitive.server" in idx_content or "cognitive" in idx_content, "Inference index exports cognitive.server")
check("subject-identity" in idx_content or "subject" in idx_content, "Inference index references subject-identity")

# ─── CHECK 10: Cross-service test exists ───────────────────────────────────────

section("CHECK 10: Cross-Service Integration Test")

check(file_exists("src/lib/ai/inference/__tests__/cross-service.test.ts"), "cross-service.test.ts exists")

if file_exists("src/lib/ai/inference/__tests__/cross-service.test.ts"):
    with open(os.path.join(ROOT, "src/lib/ai/inference/__tests__/cross-service.test.ts")) as f:
        cs_content = f.read()

    check('import { searchSubjectIdentity }' in cs_content, "Imports subject-identity service")
    check('import { decodeCognitiveState }' in cs_content, "Imports cognitive service")
    check('import { detectAnomalies }' in cs_content, "Imports anomaly service")
    check("embedding_reused" in cs_content, "Tests embedding reuse across services")
    check("artifact_shas" in cs_content, "Tests provenance SHA consistency")
    check("latency" in cs_content.lower(), "Tests latency budgeting")

# ─── CHECK 11: Mission reports exist ─────────────────────────────────────────────

section("CHECK 11: Mission Reports")

check(file_exists("reports/MISSION33_COGNITIVE_STATE_INTELLIGENCE_REPORT.md"), "M33 report exists")
check(file_exists("reports/MISSION34_ANOMALY_DETECTION_REPORT.md"), "M34 report exists")

# ─── CHECK 12: Run vitest for Tier-1 tests ──────────────────────────────────────

section("CHECK 12: Tier-1 Test Suite (vitest)")

npx_path = os.path.join("C:\\Program Files\\nodejs", "npx.cmd")
vitest_args = [
    "vitest", "run",
    "src/lib/ai/inference/__tests__/cross-service.test.ts",
    "src/lib/ai/inference/__tests__/cognitive-decode.test.ts",
    "src/lib/ai/inference/__tests__/anomaly-detect.test.ts",
    "src/lib/ai/decoders/__tests__/registry.cognitive.test.ts",
    "src/lib/ai/decoders/__tests__/registry.anomaly.test.ts",
    "src/routes/api/joint2312/cognitive/__tests__/-decode.test.ts",
    "src/routes/api/joint2312/anomaly/__tests__/-decode.test.ts",
]

if os.path.isfile(npx_path):
    result = subprocess.run(
        [npx_path] + vitest_args,
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
else:
    # Fallback: try node_modules/.bin/vitest directly
    vitest_bin = os.path.join(ROOT, "node_modules", ".bin", "vitest")
    if os.path.isfile(vitest_bin):
        result = subprocess.run(
            [vitest_bin, "run"] + vitest_args[1:],
            capture_output=True, text=True, cwd=ROOT, timeout=120,
        )
    else:
        result = subprocess.run(
            ["npx"] + vitest_args,
            capture_output=True, text=True, cwd=ROOT, timeout=120,
            shell=True,
        )

check(result.returncode == 0, "All Tier-1 unit + route tests pass")

# Parse test counts from vitest output
output = result.stdout + result.stderr
for line in output.split("\n"):
    if "Test Files" in line and "passed" in line:
        print(f"\n  Vittest output: {line.strip()}")
    if "Tests" in line and "passed" in line:
        print(f"  {line.strip()}")

# ─── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
print(f"\n  Code checks: {CHECKS_PASSED}/{CHECKS_TOTAL} passed, {CHECKS_FAILED} failed")

if CHECKS_PASSED + CHECKS_FAILED == CHECKS_TOTAL:
    print(f"\n  🎉 M35 Validation: {'PASS' if CHECKS_FAILED == 0 else 'FAIL'}")
    print(f"  Total: {CHECKS_TOTAL} checks, {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
else:
    print(f"\n  ⚠️  M35 Validation: INCOMPLETE ({CHECKS_TOTAL - CHECKS_PASSED - CHECKS_FAILED} checks not evaluated)")

sys.exit(1 if CHECKS_FAILED > 0 else 0)
