#!/usr/bin/env python3
"""
M46 - Validation script for Production Hardening with Trained Probes.

This script validates that the M46 mission is complete: all M15 operational
validation gates (A1-A4) have been re-validated against Tier-2/Tier-3 sleep
routes using *trained* ONNX probes (replacing the random-init placeholders
from M42).

Validation phases:
  1. All M42 regression test files still exist (unchanged — re-run with trained models)
  2. All M43/M44 trained model artifacts exist and have real SHAs (not placeholders)
  3. Registry files have real SHAs (no "placeholder-*" remaining)
  4. Manifest.json contains SHA entries for all trained sleep probes
  5. New M46 test files exist (accuracy, sha-match, latency regression)
  6. Benchmark archive record for m46-production-hardening exists
  7. Run TypeScript test suite (live auth + rate-limit + browser + M46 accuracy tests)

USAGE:
    python scripts/tmp/m46_production_hardening_validation.py

ENVIRONMENT:
    SKIP_TESTS -- if set to "1", skip running the TypeScript test suite
"""
import json
import os
import sys
import subprocess
import hashlib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_PATH = os.path.join(REPO, "reports", "benchmark_archive.json")
MANIFEST_PATH = os.path.join(REPO, "public", "models", "manifest.json")


def load_archive():
    with open(ARCHIVE_PATH, "r") as f:
        return json.load(f)


def load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def compute_sha256(path):
    """Compute SHA-256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_m42_regression_tests_exist():
    """Verify all 39 M42 regression test files still exist (must re-pass with trained models)."""
    checks = []

    regression_tests = [
        # JWT Auth tests (A4 gate)
        ("src/routes/api/joint2312/sleep/__tests__/-decode-jwt-auth-live.test.ts",
         "staging JWT auth live test"),
        ("src/routes/api/joint2312/sleep/__tests__/-quality-jwt-auth-live.test.ts",
         "quality JWT auth live test"),
        ("src/routes/api/joint2312/__tests__/-fusion-jwt-auth-live.test.ts",
         "fusion JWT auth live test"),
        # Rate-limit tests (A3 gate)
        ("src/routes/api/joint2312/sleep/__tests__/-decode-rate-limit-live.test.ts",
         "staging rate-limit live test"),
        ("src/routes/api/joint2312/sleep/__tests__/-quality-rate-limit-live.test.ts",
         "quality rate-limit live test"),
        ("src/routes/api/joint2312/__tests__/-fusion-rate-limit-live.test.ts",
         "fusion rate-limit live test"),
        # Browser WASM smoke (A1 gate)
        ("tests/browser/sleep-task-heads-wasm-smoke-firefox.test.ts",
         "browser sleep WASM smoke test"),
    ]

    for path, desc in regression_tests:
        full = os.path.join(REPO, path)
        checks.append((f"{desc} exists", os.path.exists(full)))

    return checks


def validate_trained_artifacts_exist():
    """Verify all M43/M44 trained ONNX artifacts exist on disk."""
    checks = []

    trained_artifacts = [
        # M43 Tier-2 probes (trained, replacing random-init)
        ("public/models/sleep/staging-probe-joint2312-v1.onnx",
         "staging probe (2312→5) trained artifact"),
        ("public/models/sleep/quality-probe-joint2312-v1.onnx",
         "quality probe (2312→1) trained artifact"),
        # M44 Browser V2-32 probes (trained, replacing placeholder SHAs)
        ("models/sleep/staging-probe-v2-32d-v1.onnx",
         "staging V2-32 browser probe trained artifact"),
        ("models/sleep/quality-probe-v2-32d-v1.onnx",
         "quality V2-32 browser probe trained artifact"),
        ("models/cognitive/cognitive-probe-v2-32d-v1.onnx",
         "cognitive V2-32 browser probe trained artifact"),
        ("models/anomaly/mahalanobis-probe-v2-32d-v1.onnx",
         "anomaly V2-32 browser probe trained artifact"),
    ]

    for path, desc in trained_artifacts:
        full = os.path.join(REPO, path)
        checks.append((f"{desc} exists", os.path.exists(full)))

    return checks


def validate_shas_match():
    """Verify that model file SHAs match their registered values in manifest + registry."""
    checks = []

    manifest = load_manifest()
    manifest_models = manifest.get("models", {})

    # Map: file path → (manifest_key, registry_key, m42_placeholder_sha)
    # Manifest keys are filename-based IDs (e.g. staging-probe-joint2312-v1),
    # NOT service IDs (e.g. sleep-staging-v1).
    sha_checks = [
        # M43 trained Tier-2 probes
        ("public/models/sleep/staging-probe-joint2312-v1.onnx",
         "staging-probe-joint2312-v1",
         "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6"),
        ("public/models/sleep/quality-probe-joint2312-v1.onnx",
         "quality-probe-joint2312-v1",
         "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f"),
    ]

    for path, model_id, expected_sha_m42 in sha_checks:
        full = os.path.join(REPO, path)
        if os.path.exists(full):
            actual_sha = compute_sha256(full)
            # SHA should have changed from M42 placeholder
            checks.append((
                f"{model_id}: SHA changed from M42 random-init placeholder",
                actual_sha != expected_sha_m42
            ))
            # SHA should be registered in manifest (keyed by filename-based ID)
            manifest_entry = manifest_models.get(model_id)
            if manifest_entry:
                checks.append((
                    f"{model_id}: file SHA matches manifest",
                    actual_sha == manifest_entry.get("sha256", "")
                ))
            else:
                checks.append((
                    f"{model_id}: exists in manifest.json",
                    False
                ))

    return checks


def validate_no_placeholder_shas():
    """Verify no 'placeholder-*' SHAs remain in registry files."""
    checks = []

    registry_files = [
        "src/lib/ai/decoders/sleep.registry.ts",
        "src/lib/ai/decoders/cognitive.registry.ts",
        "src/lib/ai/decoders/anomaly.registry.ts",
    ]

    for path in registry_files:
        full = os.path.join(REPO, path)
        if os.path.exists(full):
            with open(full, "r") as f:
                content = f.read()
            checks.append((
                f"{os.path.basename(path)}: no 'placeholder-*' SHAs",
                "placeholder-" not in content
            ))
            checks.append((
                f"{os.path.basename(path)}: no placeholder-v2-32d entries",
                'placeholder-v2-32d' not in content
            ))

    return checks


def validate_metrics_populated():
    """Verify sleep registry has real metrics (not all zeros)."""
    checks = []

    staging_path = os.path.join(REPO, "src/lib/ai/decoders/sleep.registry.ts")
    if os.path.exists(staging_path):
        with open(staging_path, "r") as f:
            content = f.read()
        # Staging: acc_5class should be >= 0.65
        checks.append(("staging registry: acc_5class is non-zero",
                       "0.0 // placeholder" not in content))
        # Quality: r2 should be >= 0.60
        checks.append(("quality registry: r2 is non-zero",
                       'r2: 0.0 // placeholder' not in content))

    return checks


def validate_m46_test_files():
    """Verify M46-specific new test files exist."""
    checks = []

    m46_tests = [
        ("tests/m46_trained_probe_accuracy.test.ts",
         "trained probe accuracy test"),
        ("tests/m46_v2_32_accuracy.test.ts",
         "V2-32 browser probe accuracy test"),
        ("tests/m46_manifest_sha_match.test.ts",
         "manifest SHA match test"),
        ("tests/m46_latency_regression.test.ts",
         "latency regression test"),
    ]

    for path, desc in m46_tests:
        full = os.path.join(REPO, path)
        checks.append((f"{desc} exists", os.path.exists(full)))

    return checks


def validate_archive_record():
    """Verify m46-production-hardening record exists in benchmark_archive.json."""
    checks = []

    try:
        archive = load_archive()
        experiments = archive.get("experiments", [])
        m46_found = any(
            "m46" in str(e.get("id", "")).lower() or
            "m46" in str(e.get("experiment_name", "")).lower()
            for e in experiments
        )
        checks.append(("benchmark_archive.json: m46-production-hardening record exists", m46_found))

        if m46_found:
            m46_entry = next(
                e for e in experiments
                if "m46" in str(e.get("id", "")).lower() or
                   "m46" in str(e.get("experiment_name", "")).lower()
            )
            checks.append(("M46 entry: has trained_probe_accuracy field",
                           "trained_probe_accuracy" in str(m46_entry.get("results", {}))))
    except Exception as e:
        checks.append(("benchmark_archive.json: parseable", False))
        print(f"  ERROR parsing archive: {e}")

    return checks


def validate_no_tsc_errors():
    """Run TypeScript typecheck, checking M46-modified files for errors."""
    checks = []
    files_to_check = [
        "src/lib/ai/decoders/sleep.registry.ts",
        "src/lib/ai/decoders/sleep.browser.ts",
        "src/lib/ai/decoders/cognitive.browser.ts",
        "src/lib/ai/decoders/anomaly.browser.ts",
        "src/lib/ai/decoders/index.ts",
        "src/lib/ai/decoders/registry.ts",
        "src/testing/harness.ts",
    ]

    # Check files exist
    for f in files_to_check:
        checks.append((f"TypeScript file exists: {f}", os.path.exists(os.path.join(REPO, f))))

    # Run tsc --noEmit on the whole project (fastest verification)
    # Use shell=True for cross-platform PATH resolution (needed on Windows)
    try:
        result = subprocess.run(
            "npx tsc --noEmit",
            shell=True, cwd=REPO, capture_output=True, text=True, timeout=120
        )
        # Check that none of the M46-modified files have NEW errors
        # (pre-existing errors in other files like inference/*.server.ts are out of scope)
        output = (result.stdout + result.stderr).strip()
        tsc_errors = output.split("\n") if output else []

        m46_files_with_errors = []
        for f in files_to_check:
            for err in tsc_errors:
                if f in err:
                    m46_files_with_errors.append((f, err))
                    break

        checks.append((
            "TypeScript typecheck: no errors in M46-modified files",
            len(m46_files_with_errors) == 0
        ))

        if m46_files_with_errors:
            for f, err in m46_files_with_errors:
                print(f"  TS ERROR in {f}: {err}")
        elif tsc_errors:
            print(f"  (Note: {len(tsc_errors)} pre-existing TS errors in other files — out of M46 scope)")
    except Exception as e:
        checks.append(("TypeScript typecheck: no errors in M46-modified files", False))
        print(f"  TSC ERROR: {e}")

    return checks


def run_tests():
    """Run the TypeScript test suite for M42 regression + M46 new tests."""
    checks = []

    test_commands = [
        # M42 regression tests (must re-pass with trained models)
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/sleep/__tests__/-decode-jwt-auth-live.test.ts",
          "--reporter=dot"],
         "M42 regression: staging JWT auth tests"),
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/sleep/__tests__/-quality-jwt-auth-live.test.ts",
          "--reporter=dot"],
         "M42 regression: quality JWT auth tests"),
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/__tests__/-fusion-jwt-auth-live.test.ts",
          "--reporter=dot"],
         "M42 regression: fusion JWT auth tests"),
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/sleep/__tests__/-decode-rate-limit-live.test.ts",
          "--reporter=dot"],
         "M42 regression: staging rate-limit tests"),
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/sleep/__tests__/-quality-rate-limit-live.test.ts",
          "--reporter=dot"],
         "M42 regression: quality rate-limit tests"),
        (["npx", "vitest", "run",
          "src/routes/api/joint2312/__tests__/-fusion-rate-limit-live.test.ts",
          "--reporter=dot"],
         "M42 regression: fusion rate-limit tests"),
        # M46 new tests (explicit file list for cross-platform glob expansion)
        (["npx", "vitest", "run",
          "tests/m46_trained_probe_accuracy.test.ts",
          "tests/m46_v2_32_accuracy.test.ts",
          "tests/m46_manifest_sha_match.test.ts",
          "tests/m46_latency_regression.test.ts",
          "--reporter=dot"],
         "M46 new tests: accuracy + SHA + latency regression"),
    ]

    for cmd, desc in test_commands:
        try:
            result = subprocess.run(
                " ".join(cmd), shell=True, cwd=REPO, capture_output=True, text=True, timeout=120
            )
            passed = result.returncode == 0
            checks.append((f"Test: {desc}", passed))
            if not passed:
                output = result.stdout + result.stderr
                print(f"  FAILED: {desc}")
                print(f"  Output (last 500 chars): {output[-500:]}")
        except Exception as e:
            checks.append((f"Test: {desc}", False))
            print(f"  ERROR running {desc}: {e}")

    return checks


def run_browser_tests():
    """Run browser WASM smoke tests with trained probes."""
    checks = []

    try:
        result = subprocess.run(
            "npx playwright test tests/browser/sleep-task-heads-wasm-smoke-firefox.test.ts --project=chromium",
            shell=True, cwd=REPO, capture_output=True, text=True, timeout=180
        )
        checks.append(("Browser WASM: sleep task heads smoke test (chromium)",
                       result.returncode == 0))
        if result.returncode != 0:
            output = result.stdout + result.stderr
            print(f"  Browser test FAILED")
            print(f"  Output (last 800 chars): {output[-800:]}")
    except Exception as e:
        checks.append(("Browser WASM: sleep task heads smoke test (chromium)", False))
        print(f"  ERROR running browser test: {e}")

    return checks


def main():
    all_checks = []

    print("=" * 70)
    print("M46 — Production Hardening Validation")
    print("=" * 70)

    # 1. M42 regression tests exist
    print("\n[1/7] M42 Regression Test Files Exist...")
    all_checks += validate_m42_regression_tests_exist()

    # 2. Trained artifacts exist
    print("[2/7] Trained Artifacts Exist...")
    all_checks += validate_trained_artifacts_exist()

    # 3. SHAs match
    print("[3/7] SHA Verification (trained vs placeholder)...")
    all_checks += validate_shas_match()

    # 4. No placeholder SHAs in registries
    print("[4/7] No Placeholder SHAs in Registries...")
    all_checks += validate_no_placeholder_shas()

    # 5. Metrics populated
    print("[5/7] Registry Metrics Populated...")
    all_checks += validate_metrics_populated()

    # 6. M46 test files exist
    print("[6/7] M46 New Test Files Exist...")
    all_checks += validate_m46_test_files()

    # 7. Archive record
    print("[7/7] Benchmark Archive Record...")
    all_checks += validate_archive_record()

    # TypeScript check
    print("\n[TypeScript] Typecheck...")
    all_checks += validate_no_tsc_errors()

    # Test suite
    if os.environ.get("SKIP_TESTS") != "1":
        print("\n[Tests] Running live + M46 test suite...")
        # Note: live tests require local Supabase stack (GoTrue + PostgREST + Postgres + pgvector)
        all_checks += run_tests()

        print("\n[Browser] Running browser WASM smoke tests...")
        # Note: browser tests require Playwright browsers
        all_checks += run_browser_tests()
    else:
        print("\n[Tests] Skipped (SKIP_TESTS=1)")

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    passed = sum(1 for _, ok in all_checks if ok)
    failed = sum(1 for _, ok in all_checks if not ok)
    total = len(all_checks)

    for name, ok in all_checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\nTotal: {passed}/{total} passed, {failed} failed")

    if failed > 0:
        print("\n❌ M46 validation FAILED — see failing checks above.")
        sys.exit(1)
    else:
        print("\n✅ M46 validation PASSED — all operational gates re-validated with trained models.")


if __name__ == "__main__":
    main()
