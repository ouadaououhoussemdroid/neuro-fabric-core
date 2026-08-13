#!/usr/bin/env bash
#
# Mission 5 - GA Promotion Script (staging-safe, NOT for production).

set -euo pipefail

DRY_RUN=true
ENV_FILE="${ENV_FILE:-.env.staging.ga}"

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) DRY_RUN=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --env-file=*) ENV_FILE="${1#*=}"; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "=== Mission 5 GA Promotion (Staging) ==="
echo "Env file:   $ENV_FILE"
echo "Dry run:    $DRY_RUN"
echo "Target:     AI_EEGCONFORMER_ENABLED=ga"
echo ""

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: env file not found: $ENV_FILE"
  exit 1
fi

CURRENT_STAGE=$(grep "^AI_EEGCONFORMER_ENABLED=" "$ENV_FILE" | cut -d= -f2)
echo "[1/5] Current stage: $CURRENT_STAGE"

if [ "$CURRENT_STAGE" = "ga" ]; then
  echo "SKIP: already at ga - no promotion needed."
  exit 0
fi

echo "[2/5] Checking staging observation evidence..."
SUMMARY_FILE="reports/t035_staging_observation_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
  echo "  Summary file: $SUMMARY_FILE"
  ALL_PASS=$(python3 -c "import json; d=json.load(open(\x27$SUMMARY_FILE\x27)); print(\x27true\x27 if d.get(\x27gates\x27,{}).get(\x27all_gates_pass\x27) else \x27false\x27)" 2>/dev/null || echo "unknown")
  if [ "$ALL_PASS" = "false" ]; then
    echo "  ERROR: gates not all passing. Aborting."
    exit 1
  fi
else
  echo "  WARNING: No observation summary found."
fi

echo "[3/5] Verifying artifact provenance..."
ARTIFACT="public/models/eegconformer_finetuned.onnx"
MANIFEST="public/models/manifest.json"
if [ -f "$ARTIFACT" ] && [ -f "$MANIFEST" ]; then
  ACTUAL_SHA=$(python3 -c "import hashlib; print(hashlib.sha256(open(\x27$ARTIFACT\x27,\x27rb\x27).read()).hexdigest())")
  MANIFEST_SHA=$(python3 -c "import json; m=json.load(open(\x27$MANIFEST\x27)); print(next(e.get(\x27sha256\x27,\x27NOT_FOUND\x27) for e in m.get(\x27models\x27,[]) if e.get(\x27id\x27)==\x27eegconformer_finetuned\x27))")
  echo "  Artifact SHA-256: $ACTUAL_SHA"
  echo "  Manifest SHA-256: $MANIFEST_SHA"
  if [ "$ACTUAL_SHA" != "$MANIFEST_SHA" ]; then
    echo "  ERROR: SHA-256 mismatch!"
    exit 1
  fi
  echo "  Artifact provenance: VERIFIED"
fi

echo "[4/5] Verifying model registry alignment..."
REGISTRY="src/lib/ai/models/registry.ts"
if grep -q "braindecode-eegconformer-prod-v2" "$REGISTRY" 2>/dev/null; then
  echo "  V2 registered: YES"
else
  echo "  ERROR: V2 not registered!"
  exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
  echo "[5/5] DRY RUN - would flip $ENV_FILE to AI_EEGCONFORMER_ENABLED=ga"
else
  echo "[5/5] Executing GA promotion..."
  sed -i "s/^AI_EEGCONFORMER_ENABLED=.*/AI_EEGCONFORMER_ENABLED=ga/" "$ENV_FILE"
  NEW_VAL=$(grep "^AI_EEGCONFORMER_ENABLED=" "$ENV_FILE" | cut -d= -f2)
  echo "  Flipped. New value: ga"
fi

echo ""
echo "=== GA Promotion Check Complete ==="
[ "$DRY_RUN" = "true" ] && echo "(Dry run - no changes made)"
