#!/usr/bin/env python3
"""
Mission 5 — 24-hour staging observation script.

Polls GET /api/public/staging/metrics every POLL_INTERVAL seconds for
OBSERVATION_DURATION seconds, recording latency percentiles, fallback rate,
and artifact verification counts.

This runs against the staging server (http://localhost:5173/api/public/staging/metrics).
The CRON_SECRET must be set in the environment for authentication.

Usage:
  python scripts/staging_observation.py
  python scripts/staging_observation.py --duration 86400 --interval 300

Requires: requests (pip install requests)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

DEFAULT_DURATION = 86400  # 24 hours
DEFAULT_INTERVAL = 300     # 5 minutes
STAGING_URL = os.environ.get("STAGING_URL", "http://localhost:5173/api/public/staging/metrics")
CRON_SECRET = os.environ.get("CRON_SECRET", "NeuroFabric_2026_8vK92LmPq7Rs4XtWc91AaZ5HdQeM")
OUTPUT_FILE = "reports/t035_staging_observation.jsonl"


def poll_metrics(url: str, cron_secret: str) -> dict | None:
    """Fetch one snapshot from the staging metrics endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {cron_secret}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 5 24h staging observation")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Observation duration in seconds (default: 86400 = 24h)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Polling interval in seconds (default: 300 = 5min)")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="Output JSONL file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Poll once and exit (no waiting)")
    args = parser.parse_args()

    print(f"=== Mission 5 Staging Observation ===")
    print(f"URL:       {STAGING_URL}")
    print(f"Duration:  {args.duration}s ({args.duration/3600:.1f}h)")
    print(f"Interval:  {args.interval}s")
    print(f"Output:    {args.output}")
    print(f"Dry-run:   {args.dry_run}")
    print(f"Started:   {datetime.now(timezone.utc).isoformat()}")
    print()

    snapshots = []
    start = time.time()

    while True:
        snapshot = poll_metrics(STAGING_URL, CRON_SECRET)
        if snapshot:
            snapshots.append(snapshot)
            if "error" not in snapshot:
                # Log key metrics
                print(f"[{snapshot['timestamp']}] "
                      f"stage={snapshot.get('rolloutStage', '?')} "
                      f"fallbackRate={snapshot.get('fallbackRate', '?')} "
                      f"p95={snapshot.get('latency', {}).get('p95', '?')} "
                      f"p50={snapshot.get('latency', {}).get('p50', '?')} "
                      f"verify_pass={snapshot.get('artifactVerification', {}).get('pass', '?')} "
                      f"verify_fail={snapshot.get('artifactVerification', {}).get('fail', '?')}")
            else:
                print(f"[{snapshot['timestamp']}] ERROR: {snapshot['error']}")

        # Save after each poll (incremental persistence)
        with open(args.output, "w") as f:
            for s in snapshots:
                f.write(json.dumps(s) + "\n")

        if args.dry_run:
            print("\nDry run complete — one snapshot taken.")
            break

        elapsed = time.time() - start
        if elapsed >= args.duration:
            print(f"\nObservation complete ({elapsed:.0f}s elapsed).")
            break

        remaining = args.duration - elapsed
        sleep_time = min(args.interval, remaining)
        print(f"Sleeping {sleep_time}s... ({remaining:.0f}s remaining)")
        time.sleep(sleep_time)

    # Write final summary
    summary_file = args.output.replace(".jsonl", "_summary.json")
    if snapshots:
        valid = [s for s in snapshots if "error" not in s]
        if valid:
            summary = {
                "experiment": "T-035 Staging Observation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_polls": len(snapshots),
                "valid_polls": len(valid),
                "rollout_stage": valid[-1].get("rolloutStage", "unknown"),
                "latency_p95_mean": sum(float(s["latency"]["p95"]) for s in valid if s.get("latency", {}).get("p95", "0") != "0") / max(1, sum(1 for s in valid if s.get("latency", {}).get("p95", "0") != "0")),
                "latency_p50_mean": sum(float(s["latency"]["p50"]) for s in valid if s.get("latency", {}).get("p50", "0") != "0") / max(1, sum(1 for s in valid if s.get("latency", {}).get("p50", "0") != "0")),
                "fallback_rate_max": max(float(s["fallbackRate"]) for s in valid),
                "verification_failures": sum(s["artifactVerification"].get("fail", 0) for s in valid),
                "verification_passes": sum(s["artifactVerification"].get("pass", 0) for s in valid),
                "gates": {
                    "p95_latency_gt_600ms": any(float(s["latency"]["p95"]) >= 600 for s in valid if s.get("latency", {}).get("p95", "0") != "0"),
                    "p50_latency_gt_400ms": any(float(s["latency"]["p50"]) >= 400 for s in valid if s.get("latency", {}).get("p50", "0") != "0"),
                    "fallback_rate_exceeds_0_5_percent": max(float(s["fallbackRate"]) for s in valid) >= 0.005,
                    "has_verification_failures": any(s["artifactVerification"].get("fail", 0) > 0 for s in valid),
                },
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary written to: {summary_file}")
            print(json.dumps(summary, indent=2))

    sys.exit(0 if not snapshots else 0)


if __name__ == "__main__":
    main()
