#!/usr/bin/env python3
"""
Mission 5 - Automated Staging Observation Completion Checker.

Reads reports/t035_staging_observation.jsonl and reports whether:
  - 24 hours have elapsed since the first snapshot
  - All staging gates have passed (allGatesPass == true for all valid polls)
  - No fallback rate exceedances
  - No verification failures

This script ONLY READS the observation output - it does NOT modify
the running observation script, the staging server, or any production
configuration.

Usage:
  python scripts/check_staging_observation.py
  python scripts/check_staging_observation.py --input reports/t035_staging_observation.jsonl

Exit codes:
  0 = all gates pass AND 24h elapsed
  1 = observation incomplete (duration not yet reached)
  2 = gates failing
  3 = file not found or parse error
"""
import argparse
import json
import sys
from datetime import datetime, timezone

DEFAULT_INPUT = "reports/t035_staging_observation.jsonl"
OBSERVATION_DURATION_SECONDS = 86400  # 24 hours


def parse_timestamp(ts: str) -> datetime:
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def check_observation(input_file: str) -> dict:
    with open(input_file) as f:
        snapshots = [json.loads(line) for line in f if line.strip()]

    if not snapshots:
        return {"status": "waiting", "error": "no snapshots", "snapshots_total": 0}

    first = parse_timestamp(snapshots[0]["timestamp"])
    last = parse_timestamp(snapshots[-1]["timestamp"])
    duration = (last - first).total_seconds()

    valid = [s for s in snapshots if "error" not in s]
    invalid = [s for s in snapshots if "error" in s]

    all_gates_pass = True
    gates_failing = set()
    max_fallback_rate = 0.0
    verification_failures = 0
    verification_passes = 0

    for s in valid:
        gates = s.get("gates", {})
        for gate_name, gate_pass in gates.items():
            if isinstance(gate_pass, bool) and not gate_pass:
                all_gates_pass = False
                gates_failing.add(gate_name)

        fr = float(s.get("fallbackRate", 0))
        if fr > max_fallback_rate:
            max_fallback_rate = fr

        av = s.get("artifactVerification", {})
        verification_failures += int(av.get("fail", 0))
        verification_passes += int(av.get("pass", 0))

    duration_complete = duration >= OBSERVATION_DURATION_SECONDS
    gates_ok = all_gates_pass and verification_failures == 0

    if duration_complete and gates_ok:
        status = "complete_pass"
    elif duration_complete and not gates_ok:
        status = "complete_fail"
    else:
        status = "waiting"

    return {
        "status": status,
        "duration_seconds": duration,
        "duration_remaining_seconds": max(0, OBSERVATION_DURATION_SECONDS - duration),
        "snapshots_total": len(snapshots),
        "snapshots_valid": len(valid),
        "snapshots_with_errors": len(invalid),
        "all_gates_pass": all_gates_pass,
        "gates_failing": list(gates_failing),
        "max_fallback_rate": max_fallback_rate,
        "verification_failures": verification_failures,
        "verification_passes": verification_passes,
        "first_snapshot": snapshots[0]["timestamp"],
        "last_snapshot": snapshots[-1]["timestamp"],
        "rollout_stage": valid[-1].get("rolloutStage", "unknown") if valid else "unknown",
        "checks": {
            "duration_complete_24h": duration_complete,
            "all_gates_pass": all_gates_pass,
            "no_verification_failures": verification_failures == 0,
            "fallback_rate_under_05_percent": max_fallback_rate < 0.005,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mission 5 staging observation checker")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Path to JSONL observation file")
    args = parser.parse_args()

    result = check_observation(args.input)

    print("=== Mission 5 Staging Observation Check ===")
    print(f"Input file: {args.input}")
    print(f"Status: {result.get('status', 'unknown')}")

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(3)

    print(f"Duration: {result['duration_seconds']:.0f}s "
          f"({result['duration_seconds']/3600:.1f}h / 24.0h required)")
    print(f"Remaining: {result['duration_remaining_seconds']:.0f}s "
          f"({result['duration_remaining_seconds']/3600:.1f}h)")
    print(f"Snapshots: {result['snapshots_total']} total, "
          f"{result['snapshots_valid']} valid")
    print(f"Rollout stage: {result['rollout_stage']}")
    print(f"All gates pass: {result['all_gates_pass']}")
    print(f"Max fallback rate: {result['max_fallback_rate']:.4f}")
    print(f"Verification failures: {result['verification_failures']}")
    print(f"First snapshot: {result['first_snapshot']}")
    print(f"Last snapshot: {result['last_snapshot']}")

    for check_name, check_pass in result["checks"].items():
        print(f"  {check_name}: {'PASS' if check_pass else 'FAIL'}")

    if result["status"] == "complete_pass":
        print("\nSTAGING OBSERVATION COMPLETE - all gates passed!")
        sys.exit(0)
    elif result["status"] == "complete_fail":
        print("\nSTAGING OBSERVATION COMPLETE - gates failing!")
        sys.exit(2)
    else:
        print(f"\nObservation still running - "
              f"{result['duration_remaining_seconds']/3600:.1f}h remaining.")
        sys.exit(1)


if __name__ == "__main__":
    main()
