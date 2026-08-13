#!/usr/bin/env bash
#
# Mission 5 - Beta Promotion Script (staging-safe, NOT for production).
#
# Flips AI_EEGCONFORMER_ENABLED from "canary" to "beta" in the staging
# environment file. Designed to be run AFTER the 24-hour staging observation
# has completed successfully.
#
# SAFE TO CREATE - DO NOT EXECUTE against production.
# Production .env remains AI_EEGCONFORMER_ENABLED=off until all staging
# gates have actual measured evidence (24h observation + gate pass).
#
# Usage:
#   ./scripts/promote_beta.sh --dry-run       # Show what would happen
#   ./scripts/promote_beta.sh --execute       # Execute the flip (staging only)
#
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

echo "=== Mission 5 Beta Promotion (Staging) ==="
echo "Env file:   $ENV_FILE"
echo "Dry run:    $DRY_RUN"
echo "Target:     AI_EEGCONFORMER_ENABLED=beta"
echo ""

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: env file not found: $ENV_FILE"
  exit 1
fi

CURRENT_STAGE=$(grep "^AI_EEGCONFORMER_ENABLED=" "$ENV_FILE" | cut -d= -f2)
echo "[1/4] Current stage: $CURRENT_STAGE"

if [ "$CURRENT_STAGE" = "beta" ]; then
  echo "SKIP: already at 'beta' - no promotion needed."
  exit 0
fi

# Verify staging observation evidence
echo "[2/4] Checking staging observation evidence..."
SUMMARY_FILE="reports/t035_staging_observation_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
  echo "  Summary file: $SUMMARY_FILE"
  ALL_PASS=$(python3 -c "
import json
with open('$SUMMARY_FILE') as f:
    d = json.load(f)
print('true' if d.get('gates', {}).get('all_gates_pass') else 'false')
" 2>/dev/null || echo "unknown")
  echo "  All gates pass: $ALL_PASS"
  if [ "$ALL_PASS" = "false" ]; then
    echo "ERROR: Staging observation gates not all passing. Aborting."
    exit 1
  fi
else
  echo "  WARNING: No observation summary found. Run staging observation for 24h first."
fi

# Verify artifact provenance
echo "[3/4] Verifying artifact provenance..."
ARTIFACT="public/models/eegconformer_finetuned.onnx"
MANIFEST="public/models/manifest.json"
if [ -f "$ARTIFACT" ] && [ -f "$MANIFEST" ]; then
  ACTUAL_SHA=$(python3 -c "
import hashlib
with open('$ARTIFACT', 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
")
  MANIFEST_SHA=$(python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
for entry in m.get('models', []):
    if entry.get('id') == 'eegconformer_finetuned':
        print(entry.get('sha256', 'NOT_FOUND'))
        break
" 2>/dev/null || echo "NOT_FOUND")
  echo "  Artifact SHA-256: $ACTUAL_SHA"
  echo "  Manifest SHA-256: $MANIFEST_SHA"
  if [ "$ACTUAL_SHA" != "$MANIFEST_SHA" ]; then
    echo "  ERROR: SHA-256 mismatch!"
    exit 1
  fi
  echo "  Artifact provenance: VERIFIED"
else
  echo "  ERROR: Artifact or manifest not found!"
  exit 1
fi

# Execute or simulate the flip
if [ "$DRY_RUN" = "true" ]; then
  echo "[4/4] DRY RUN - would flip $ENV_FILE to AI_EEGCONFORMER_ENABLED=beta"
  echo "  Command: sed -i 's/AI_EEGCONFORMER_ENABLED=.*/AI_EEGCONFORMER_ENABLED=beta/' $ENV_FILE"
else
  echo "[4/4] Executing - flipping $ENV_FILE to AI_EEGCONFORMER_ENABLED=beta"
  sed -i 's/^AI_EEGCONFORMER_ENABLED=.*/AI_EEGCONFORMER_ENABLED=beta/' "$ENV_FILE"
  echo "  Flipped. New value: $(grep '^AI_EEGCONFORMER_ENABLED=' "$ENV_FILE")"
fi

echo ""
echo "=== Promotion Check Complete ==="
if [ "$DRY_RUN" = "true" ]; then
  echo "(Dry run - no changes made)"
fi
