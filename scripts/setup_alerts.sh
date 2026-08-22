#!/usr/bin/env bash
#
# M37 — Tier 1 Beta Alert Configuration Script
#
# Configures production readiness alerts for the three Tier-1 services:
#   - Subject Identity (M32)
#   - Cognitive State (M33)
#   - Anomaly Detection (M34)
#
# Reads metrics from the Prometheus endpoint and checks alert thresholds.
# Designed to run as a cron job every 5 minutes in staging/beta environments.
#
# Exit codes:
#   0 = all checks pass
#   1 = one or more alert thresholds breached
#   2 = configuration error (CRON_SECRET missing, endpoint unreachable, etc.)
#
set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────────────────

METRICS_URL="${METRICS_URL:-http://localhost:3000/api/public/staging/metrics}"
CRON_SECRET="${CRON_SECRET:-}"
WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

# Alert thresholds (per M31 §26.5 + T-035 staging gates)
P95_LATENCY_MS_THRESHOLD="${P95_LATENCY_MS_THRESHOLD:-2000}"    # Server P95
P50_LATENCY_MS_THRESHOLD="${P50_LATENCY_MS_THRESHOLD:-400}"     # Browser P50
ERROR_RATE_THRESHOLD="${ERROR_RATE_THRESHOLD:-0.05}"             # 5% error rate
RATE_LIMIT_BREACH_THRESHOLD="${RATE_LIMIT_BREACH_THRESHOLD:-10}" # 10+ rate-limited per cycle
FALLBACK_RATE_THRESHOLD="${FALLBACK_RATE_THRESHOLD:-0.005}"     # 0.5% fallback rate

# ─── Functions ─────────────────────────────────────────────────────────────────

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

fail() {
    log "ERROR: $*"
    exit 2
}

check_dependency() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fail "Required dependency '$1' not found in PATH"
    fi
}

# ─── Pre-flight Checks ─────────────────────────────────────────────────────────

check_dependency "curl"
check_dependency "python3"

if [ -z "$CRON_SECRET" ]; then
    # Try to read from .env file if not set
    if [ -f ".env.staging.beta" ]; then
        CRON_SECRET=$(grep "^CRON_SECRET=" .env.staging.beta | cut -d= -f2)
    fi
fi

if [ -z "$CRON_SECRET" ]; then
    fail "CRON_SECRET is not configured. Set it in environment or .env.staging.beta"
fi

# ─── Fetch Metrics ─────────────────────────────────────────────────────────────

log "Fetching staging metrics from $METRICS_URL..."

response=$(curl -sf \
    -H "Authorization: Bearer ${CRON_SECRET}" \
    -H "Accept: application/json" \
    "$METRICS_URL" 2>/dev/null) || fail "Failed to fetch metrics from $METRICS_URL"

# Parse JSON with python3 (jq may not be available)
parse_metric() {
    local key="$1"
    echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
keys = '${key}'.split('.')
val = data
for k in keys:
    val = val.get(k, {}) if isinstance(val, dict) else None
    if val is None:
        print('null', end='')
        sys.exit(0)
print(str(val) if not isinstance(val, (dict, list)) else json.dumps(val), end='')
" 2>/dev/null
}

# ─── Alert Checks ──────────────────────────────────────────────────────────────

ALERTS_TRIGGERED=0

log ""
log "=== Tier-1 Alert Threshold Checks ==="
log ""

# Check 1: P95 latency
p95_latency=$(parse_metric "gates.p95LatencyMs")
check_name="Server P95 latency"
if [ "$p95_latency" = "null" ] || [ -z "$p95_latency" ]; then
    log "  ⚠️  $check_name: no data (skipping)"
elif python3 -c "exit(0 if float('$p95_latency') < $P95_LATENCY_MS_THRESHOLD else 1)" 2>/dev/null; then
    log "  ✓ $check_name: ${p95_latency}ms < ${P95_LATENCY_MS_THRESHOLD}ms"
else
    log "  ✗ $check_name: ${p95_latency}ms ≥ ${P95_LATENCY_MS_THRESHOLD}ms — ALERT"
    ALERTS_TRIGGERED=$((ALERTS_TRIGGERED + 1))
fi

# Check 2: P50 latency (browser)
p50_latency=$(parse_metric "gates.p50LatencyMs")
check_name="Browser P50 latency"
if [ "$p50_latency" = "null" ] || [ -z "$p50_latency" ]; then
    log "  ⚠️  $check_name: no data (skipping)"
elif python3 -c "exit(0 if float('$p50_latency') < $P50_LATENCY_MS_THRESHOLD else 1)" 2>/dev/null; then
    log "  ✓ $check_name: ${p50_latency}ms < ${P50_LATENCY_MS_THRESHOLD}ms"
else
    log "  ✗ $check_name: ${p50_latency}ms ≥ ${P50_LATENCY_MS_THRESHOLD}ms — ALERT"
    ALERTS_TRIGGERED=$((ALERTS_TRIGGERED + 1))
fi

# Check 3: Fallback rate
fallback_rate=$(parse_metric "fallbackRate")
check_name="Fallback rate"
if [ "$fallback_rate" = "null" ] || [ -z "$fallback_rate" ]; then
    log "  ⚠️  $check_name: no data (skipping)"
elif python3 -c "exit(0 if float('$fallback_rate') < $FALLBACK_RATE_THRESHOLD else 1)" 2>/dev/null; then
    log "  ✓ $check_name: ${fallback_rate} < ${FALLBACK_RATE_THRESHOLD} (${FALLBACK_RATE_THRESHOLD} = 0.5%)"
else
    log "  ✗ $check_name: ${fallback_rate} ≥ ${FALLBACK_RATE_THRESHOLD} — ALERT"
    ALERTS_TRIGGERED=$((ALERTS_TRIGGERED + 1))
fi

# Check 4: Artifact verification failures
has_failures=$(parse_metric "gates.hasVerificationFailures")
check_name="Artifact verification failures"
if [ "$has_failures" = "True" ] || [ "$has_failures" = "true" ]; then
    log "  ✗ $check_name: failures detected — ALERT"
    ALERTS_TRIGGERED=$((ALERTS_TRIGGERED + 1))
else
    log "  ✓ $check_name: no failures"
fi

# Check 5: All gates pass
all_pass=$(parse_metric "gates.allGatesPass")
check_name="All staging gates pass"
if [ "$all_pass" = "True" ] || [ "$all_pass" = "true" ]; then
    log "  ✓ $check_name: all gates PASS"
else
    log "  ✗ $check_name: some gates FAIL — ALERT"
    ALERTS_TRIGGERED=$((ALERTS_TRIGGERED + 1))
fi

# ─── Alert Summary ─────────────────────────────────────────────────────────────

log ""
log "=== Alert Summary ==="
log "  Alerts triggered: $ALERTS_TRIGGERED"

if [ "$ALERTS_TRIGGERED" -gt 0 ]; then
    log "  ⚠️  $ALERTS_TRIGGERED alert(s) triggered — investigate immediately."

    # Optionally POST to webhook
    if [ -n "$WEBHOOK_URL" ]; then
        curl -sf -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "$(python3 -c "
import json, sys
response = json.loads('''$response''')
print(json.dumps({
    'alert': 'tier1_beta_threshold_breach',
    'severity': 'warning',
    'triggers': $ALERTS_TRIGGERED,
    'metrics': {
        'p95_latency_ms': ''' + str(p95_latency) + ''' if ''' + str(p95_latency) + ''' != 'null' else None,
        'fallback_rate': ''' + str(fallback_rate) + ''' if ''' + str(fallback_rate) + ''' != 'null' else None,
        'verification_failures': ''' + str(has_failures) + ''',
        'all_gates_pass': ''' + str(all_pass) + ''',
    },
    'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
}))
")" 2>/dev/null || log "  WARNING: Failed to POST alert to webhook $WEBHOOK_URL"
    fi

    exit 1
fi

log "  ✅ All alert checks passed."
exit 0
