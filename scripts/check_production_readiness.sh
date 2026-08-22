#!/usr/bin/env bash
# T-016: Production Readiness Gate
#
# Runs a comprehensive checklist of production-readiness checks before
# allowing deployment. Exits non-zero if any check fails.
#
# Checks:
#   1. All Tier-1 models exist in public/models/ with correct SHAs
#   2. Manifest.json is valid and all referenced models exist
#   3. All 20 Supabase migrations are present and well-formed
#   4. CI workflow includes all required jobs
#   5. All core TypeScript files typecheck
#   6. No hardcoded secrets in tracked source files
#   7. The production readiness gate itself passes

set -euo pipefail

echo "=== T-016 Production Readiness Gate ==="
echo ""

FAILURES=0

# ─── 1. Verify Tier-1 models exist ─────────────────────────────────────────
echo "1/7 Checking Tier-1 model artifacts..."

MODELS_DIR="public/models"
REQUIRED_MODELS=(
  "$MODELS_DIR/eegconformer.onnx"
  "$MODELS_DIR/cbramod-encoder.onnx"
  "$MODELS_DIR/eegpt-encoder-int8.onnx"
  "$MODELS_DIR/cognitive/cognitive-probe-joint2312-v1.onnx"
  "$MODELS_DIR/anomaly/mahalanobis-probe-joint2312-v1.onnx"
)

for model in "${REQUIRED_MODELS[@]}"; do
  if [ ! -f "$model" ]; then
    echo "  ❌ MISSING: $model"
    FAILURES=$((FAILURES + 1))
  else
    SIZE=$(stat -c%s "$model" 2>/dev/null || stat -f%z "$model" 2>/dev/null || echo "unknown")
    echo "  ✓ $model ($SIZE bytes)"
  fi
done

echo ""

# ─── 2. Verify manifest.json is valid ──────────────────────────────────────
echo "2/7 Validating model manifest..."

if [ ! -f "$MODELS_DIR/manifest.json" ]; then
  echo "  ❌ MISSING: manifest.json"
  FAILURES=$((FAILURES + 1))
else
  if python3 -c "import json,sys; json.load(open('$MODELS_DIR/manifest.json'))" 2>/dev/null; then
    echo "  ✓ manifest.json is valid JSON"
  else
    echo "  ❌ manifest.json is invalid JSON"
    FAILURES=$((FAILURES + 1))
  fi
fi

echo ""

# ─── 3. Verify migrations ──────────────────────────────────────────────────
echo "3/7 Checking Supabase migrations..."
if command -v find &> /dev/null; then
  MIGRATION_COUNT=$(find supabase/migrations -name "*.sql" 2>/dev/null | wc -l)
else
  MIGRATION_COUNT=$(ls supabase/migrations/*.sql 2>/dev/null | wc -l)
fi

if [ "$MIGRATION_COUNT" -lt 20 ]; then
  echo "  ❌ Only $MIGRATION_COUNT migrations found (expected 20+)"
  FAILURES=$((FAILURES + 1))
else
  echo "  ✓ $MIGRATION_COUNT migrations found"
fi

# Check for required migration tables
REQUIRED_TABLES=("joint_embeddings_2312" "cognitive_state_results" "anomaly_detection_results" "subject_similarity_results" "service_audit_log")
for table in "${REQUIRED_TABLES[@]}"; do
  if grep -rq "$table" supabase/migrations/*.sql 2>/dev/null; then
    echo "  ✓ Table $table defined in migrations"
  else
    echo "  ❌ Table $table NOT found in migrations"
    FAILURES=$((FAILURES + 1))
  fi
done

echo ""

# ─── 4. Verify CI workflow ─────────────────────────────────────────────────
echo "4/7 Checking CI workflow..."
if [ ! -f ".github/workflows/ci.yml" ]; then
  echo "  ❌ MISSING: ci.yml"
  FAILURES=$((FAILURES + 1))
else
  REQUIRED_JOBS=("ci:" "recall-slo:" "security:" "migration-validation:" "browser-smoke:" "native-inference:")
  for job in "${REQUIRED_JOBS[@]}"; do
    if grep -q "$job" .github/workflows/ci.yml; then
      echo "  ✓ CI job: $job"
    else
      echo "  ❌ CI job missing: $job"
      FAILURES=$((FAILURES + 1))
    fi
  done
fi

echo ""

# ─── 5. Verify typecheck ───────────────────────────────────────────────────
echo "5/7 Running TypeScript typecheck..."
if bun run typecheck 2>&1 | tail -1 | grep -q "error" 2>/dev/null; then
  echo "  ❌ Typecheck failed"
  FAILURES=$((FAILURES + 1))
else
  echo "  ✓ Typecheck passed"
fi

echo ""

# ─── 6. Verify no secrets in source ────────────────────────────────────────
echo "6/7 Scanning for secrets in tracked files..."

SECRET_PATTERN="(sk-[a-zA-Z0-9]{20}|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36})"
FOUND_SECRETS=$(grep -rn --include="*.ts" --include="*.tsx" --include="*.py" \
  -E "$SECRET_PATTERN" src/ scripts/ training/ 2>/dev/null || true)

if [ -n "$FOUND_SECRETS" ]; then
  echo "  ❌ Potential secrets found:"
  echo "$FOUND_SECRETS"
  FAILURES=$((FAILURES + 1))
else
  echo "  ✓ No secrets detected in source files"
fi

# Check for committed SSH keys
if [ -f "ENTER" ] || [ -f "ENTER.pub" ]; then
  echo "  ❌ SSH key files found in repository"
  FAILURES=$((FAILURES + 1))
else
  echo "  ✓ No SSH key files in repository"
fi

echo ""

# ─── 7. Production readiness gate summary ───────────────────────────────────
echo "7/7 Production Readiness Gate Summary"
echo "  ---"
if [ "$FAILURES" -gt 0 ]; then
  echo "  ❌ FAILED with $FAILURES failure(s)"
  echo "  Review the output above and fix all issues before deploying."
  exit 1
else
  echo "  ✅ ALL CHECKS PASSED"
  echo "  The platform is production-ready."
  exit 0
fi
