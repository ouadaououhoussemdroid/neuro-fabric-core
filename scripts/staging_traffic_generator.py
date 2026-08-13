#!/c/Users/pc/AppData/Local/Programs/Python/Python313/python.exe -B
"""
Mission 5 - Staging Traffic Generator.
Generates REAL browser WASM inference traffic against the staging server.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

STAGING_URL = os.environ.get("STAGING_URL", "http://localhost:5177")
HARNESS_PATH = "/staging-harness.html"
METRICS_ENDPOINT = "/api/public/staging/metrics"
CRON_SECRET = os.environ.get("CRON_SECRET", "NeuroFabric_2026_8vK92LmPq7Rs4XtWc91AaZ5HdQeM")
MODEL_ID = "braindecode-eegconformer-prod-v2"
ITERATIONS = 100

JS_BENCHMARK_CODE = r"""
async () => {
    const CRON_SECRET = "__CRON_SECRET__";
    const input = window.__stagingTest.makeSyntheticInput(22, 1000, 250);
    window.__stagingTest.resetMetrics();
    window.__stagingTest.__resetManifestCache();
    window.__stagingTest.setRolloutStage("beta");
    const samples = await window.__stagingTest.runLatencyBenchmark(
        input,
        { preferredModelId: "braindecode-eegconformer-prod-v2", normalize: false },
        __ITERATIONS__
    );
    const stats = window.__stagingTest.latencyPercentiles(samples);
    const snapshot = window.__stagingTest.collectMetricsSnapshot();
    await window.__stagingTest.reportMetricsToStagingServer(snapshot, CRON_SECRET);
    for (let i = 0; i < 5; i++) {
        await window.__stagingTest.embedEEG(input, {
            preferredModelId: "braindecode-eegconformer-prod-v2",
            normalize: false
        });
    }
    const finalSnapshot = window.__stagingTest.collectMetricsSnapshot();
    await window.__stagingTest.reportMetricsToStagingServer(finalSnapshot, CRON_SECRET);
    return {
        stats,
        samples: samples.length,
        fallbackCount: samples.filter(s => s.fellBack).length,
        fallbackRate: stats.fallbackRate,
        p95: stats.p95,
        p50: stats.p50,
        mean: stats.mean,
        min: stats.min,
        max: stats.max
    };
}
""".replace("__CRON_SECRET__", CRON_SECRET).replace("__ITERATIONS__", str(ITERATIONS))


async def run_single_poll():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()
        harness_url = STAGING_URL + HARNESS_PATH
        await page.goto(harness_url, wait_until="networkidle", timeout=60000)
        await page.wait_for_function("window.__stagingTest !== undefined", timeout=30000)
        result = await page.evaluate(JS_BENCHMARK_CODE)
        req = urllib.request.Request(STAGING_URL + METRICS_ENDPOINT)
        req.add_header("Authorization", "Bearer " + CRON_SECRET)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                server_metrics = json.loads(resp.read().decode())
        except Exception as e:
            server_metrics = {"error": str(e)}
        await browser.close()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "browser": "chromium",
            "model": MODEL_ID,
            "benchmark": result,
            "server_metrics": server_metrics,
        }


def check_staging_server():
    try:
        req = urllib.request.Request(STAGING_URL + METRICS_ENDPOINT)
        req.add_header("Authorization", "Bearer " + CRON_SECRET)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return True, data.get("rolloutStage", "unknown")
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Mission 5 staging traffic generator")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--duration", type=int, default=86400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== Mission 5 Staging Traffic Generator ===")
    print("Server:   " + STAGING_URL)
    print("Interval: {}s".format(args.interval))
    print("Duration: {}s ({:.1f}h)".format(args.duration, args.duration / 3600))
    print("Dry run:  {}".format(args.dry_run))
    print()

    reachable, info = check_staging_server()
    if not reachable:
        print("ERROR: Staging server not reachable at " + STAGING_URL)
        print("  " + str(info))
        sys.exit(1)
    print("Staging server: reachable (stage={})".format(info))
    print()

    start = time.time()
    cycles = 0
    obs_file = "reports/t035_traffic_gen_observation.jsonl"

    while True:
        try:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print("[{}] Running browser WASM benchmark cycle ({})...".format(ts, cycles + 1))
            result = asyncio.run(run_single_poll())
            cycles += 1
            b = result["benchmark"]
            print("  Benchmark: {} samples, P95={:.1f}ms, P50={:.1f}ms, fallbacks={}, mean={:.1f}ms".format(
                b["samples"], b["p95"], b["p50"], b["fallbackCount"], b["mean"]))
            sm = result["server_metrics"]
            embeds = sum(v.get("total", 0) for v in sm.get("modelSelected", {}).values())
            print("  Server: stage={}, embeds={}, fallback={}, allGates={}".format(
                sm.get("rolloutStage", "?"), embeds, sm.get("fallbackRate", "?"),
                sm.get("gates", {}).get("allGatesPass", "?")))

            combined = {
                "timestamp": result["timestamp"],
                "rolloutStage": sm.get("rolloutStage", "ga"),
                "cohortChecks": sm.get("cohortChecks", {"hit": 0, "miss": 0}),
                "modelSelected": sm.get("modelSelected", {}),
                "fallbackRate": sm.get("fallbackRate", 0),
                "artifactVerification": sm.get("artifactVerification", {"pass": 0, "fail": 0, "attempt": 0}),
                "latency": sm.get("latency", {"p50": "0", "p95": "0", "mean": "0"}),
                "browser_benchmark": b,
                "gates": sm.get("gates", {}),
            }
            with open(obs_file, "a") as f:
                f.write(json.dumps(combined) + "\n")
        except Exception as e:
            print("  ERROR: " + str(e))
            with open(obs_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                }) + "\n")

        if args.dry_run:
            print("\nDry run complete - one cycle executed.")
            break

        elapsed = time.time() - start
        if elapsed >= args.duration:
            print("\nTraffic generation complete ({:.0f}s, {} cycles).".format(elapsed, cycles))
            break

        remaining = args.duration - elapsed
        sleep_time = min(args.interval, remaining)
        print("  Sleeping {}s... ({:.0f}s remaining, {} cycles)".format(sleep_time, remaining, cycles))
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
