/**
 * Mission 5 — Staging validation metrics endpoint.
 *
 * Route: GET /api/public/staging/metrics
 *
 * Returns aggregated staging validation metrics (latency percentiles,
 * fallback rate, cohort distribution, artifact verification counts) in JSON.
 * Used by the staging dashboard to gate the beta→GA promotion.
 *
 * Protected by CRON_SECRET (same auth as /api/public/metrics).
 */
import { createFileRoute } from "@tanstack/react-router";
import { renderPrometheusMetrics, metrics } from "@/lib/metrics";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";

interface StagingMetricsResponse {
  timestamp: string;
  rolloutStage: string;
  cohortChecks: { hit: number; miss: number };
  modelSelected: Record<string, { total: number; fellBack: number }>;
  fallbackRate: number;
  artifactVerification: { pass: number; fail: number; attempt: number };
  latency: { p50: string; p95: string; mean: string };
  gates: {
    p95LatencyMs: number;
    p50LatencyMs: number;
    fallbackRate: number;
    hasVerificationFailures: boolean;
    cohortHitRate: number;
    p95LatencyOk: boolean;
    p50LatencyOk: boolean;
    fallbackRateOk: boolean;
    allGatesPass: boolean;
  };
}

/** Extract latency percentiles from the Prometheus histogram buckets. */
function extractLatencyPercentiles(prometheus: string): { p50: string; p95: string; mean: string } {
  const lines = prometheus.split("\n");
  const buckets: { le: number; count: number }[] = [];
  let sum = 0;
  let count = 0;

  for (const line of lines) {
    const histMatch = line.match(/^neuro_fabric_upload_embed_ms_bucket\{le="([^"]+)"\s*\} (\d+)$/);
    if (histMatch) {
      buckets.push({ le: parseFloat(histMatch[1]), count: parseInt(histMatch[2]) });
    }
    const sumMatch = line.match(/^neuro_fabric_upload_embed_ms_sum\s+(\d+(?:\.\d+)?)$/);
    if (sumMatch) sum = parseFloat(sumMatch[1]);
    const countMatch = line.match(/^neuro_fabric_upload_embed_ms_count\s+(\d+)$/);
    if (countMatch) count = parseInt(countMatch[1]);
  }

  buckets.sort((a, b) => a.le - b.le);
  const getPercentile = (p: number): string => {
    if (buckets.length === 0 || count === 0) return "0";
    const target = (p / 100) * count;
    for (let i = 0; i < buckets.length; i++) {
      if (buckets[i].count >= target) {
        if (i === 0) return buckets[0].le.toFixed(2);
        // Linear interpolation between previous and current bucket for
        // more accurate percentile estimates on coarse bucket boundaries.
        const prev = buckets[i - 1];
        const curr = buckets[i];
        if (curr.count === prev.count) return prev.le.toFixed(2);
        const ratio = (target - prev.count) / (curr.count - prev.count);
        return (prev.le + ratio * (curr.le - prev.le)).toFixed(2);
      }
    }
    return buckets[buckets.length - 1]?.le.toFixed(2) ?? "0";
  };

  return {
    p50: getPercentile(50),
    p95: getPercentile(95),
    mean: count > 0 ? (sum / count).toFixed(2) : "0",
  };
}

/**
 * Merge browser-side Prometheus metrics into the server-side registry.
 * Accepts the JSON snapshot from collectMetricsSnapshot() in the staging harness.
 */
function mergeBrowserMetrics(prometheusText: string): void {
  const lines = prometheusText.split("\n");

  // Collect histogram bucket data for batch application after the loop.
  // Prometheus formats histograms as: name_bucket{le="500"} count  (cumulative)
  // followed by name_sum value and name_count value.
  const histBuckets = new Map<string, { labels: string; bucketCounts: Map<number, number> }>();

  for (const line of lines) {
    if (line.startsWith("#") || !line.trim()) continue;

    // Counter: neuro_fabric_model_selected_total{model="x",fell_back="true"} 5
    // Counter: neuro_fabric_artifact_verification_total{result="pass"} 3
    // Counter: neuro_fabric_eegconformer_cohort_checks_total{result="hit"} 10
    // Counter: neuro_fabric_embed_fallback_total{model="x"} 1
    const counterMatch = line.match(/^(\w+)\{([^}]*)\}\s+(\d+(?:\.\d+)?)$/);
    if (counterMatch) {
      const [, name, labelStr, val] = counterMatch;
      const labels: Record<string, string> = {};
      const labelPairs = labelStr.split(",");
      for (const pair of labelPairs) {
        const [k, v] = pair.trim().split("=");
        if (k && v) labels[k.trim()] = v.trim().replace(/^"(.*)"$/, "$1");
      }
      const value = parseInt(val);
      const counterObj = (metrics as any)[
        Object.keys(metrics).find((k) => (metrics as any)[k]?.name === name) || ""
      ];
      if (counterObj && typeof counterObj.inc === "function") {
        counterObj.inc(labels, value);
      }
    }

    // Histogram bucket: neuro_fabric_upload_embed_ms_bucket{le="500"} 5
    const histMatch = line.match(/^(\w+)_bucket\{le="([^"]+)"(?:\s*,([^}]*))?\}\s+(\d+)$/);
    if (histMatch) {
      const [, baseName, leStr, extraLabels, countStr] = histMatch;
      const le = parseFloat(leStr);
      const bucketCount = parseInt(countStr);
      const labels: Record<string, string> = {};
      if (extraLabels) {
        for (const pair of extraLabels.split(",")) {
          const [k, v] = pair.trim().split("=");
          if (k && v) labels[k.trim()] = v.trim().replace(/^"(.*)"$/, "$1");
        }
      }
      const labelKey = formatLabelStr(labels);
      const mapKey = baseName + labelKey;
      if (!histBuckets.has(mapKey)) {
        histBuckets.set(mapKey, { labels: labelKey, bucketCounts: new Map() });
      }
      histBuckets.get(mapKey)!.bucketCounts.set(le, bucketCount);
    }
  }

  // Apply collected histogram data
  for (const [mapKey, data] of histBuckets) {
    const baseNameMatch = mapKey.match(/^(\w+)_/);
    const baseName = baseNameMatch ? mapKey.substring(0, mapKey.length - data.labels.length) : mapKey;
    // Extract base histogram name by removing the label suffix
    const histObj = (metrics as any)[
      Object.keys(metrics).find((k) => (metrics as any)[k]?.name === baseName) || ""
    ];
    if (histObj && typeof histObj.setAggregated === "function") {
      // Find sum and count lines in the prometheus text
      const labelPattern = data.labels.replace(/[\{}]/g, "");
      const sumLine = lines.find(
        (l) => l.startsWith(`${baseName}_sum${data.labels}`) || l.startsWith(`${baseName}_sum`),
      );
      const countLine = lines.find((l) => l.match(new RegExp(`^${baseName}_count${data.labels ? "\\" + data.labels : "(?:\\{[^}]*\\}\\s*)?"} \\d+$`)));
      const sumVal = sumLine ? parseFloat(sumLine.split(/\s+/).pop()!) : 0;
      const countVal = countLine ? parseInt(countLine.split(/\s+/).pop()!, 10) : 0;
      histObj.setAggregated({}, { sum: sumVal, count: countVal, bucketCounts: data.bucketCounts });
    }
  }
}

function formatLabelStr(labels: Record<string, string>): string {
  const entries = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return "";
  return "{" + entries.map(([k, v]) => `${k}="${v}"`).join(",") + "}";
}

/** Extract a single counter value from Prometheus text format. */
function extractCounter(prometheus: string, name: string, labels: Record<string, string> = {}): number {
  const labelStr = Object.entries(labels)
    .map(([k, v]) => `${k}="${v}"`)
    .join(",");
  const labelPattern = labelStr
    ? `\{${labelStr.replace(/,/g, ", ")}\}\\s*`
    : "(?:\{[^}]*\}\s*)?";
  const regex = new RegExp("^" + name + labelPattern + "(\\d+)$", "m");
  const match = prometheus.match(regex);
  return match ? parseInt(match[1], 10) : 0;
}

/** Extract model selection counts grouped by model id and fell_back label. */
function extractModelSelected(prometheus: string): Record<string, { total: number; fellBack: number }> {
  const result: Record<string, { total: number; fellBack: number }> = {};
  const lineRegex = /^neuro_fabric_model_selected_total\{([^}]*)\}\s+(\d+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = lineRegex.exec(prometheus)) !== null) {
    const labels = m[1];
    const value = parseInt(m[2], 10);
    const modelMatch = labels.match(/model="([^"]*)"/);
    const fellBackMatch = labels.match(/fell_back="([^"]*)"/);
    if (modelMatch) {
      const model = modelMatch[1];
      if (!result[model]) result[model] = { total: 0, fellBack: 0 };
      result[model].total += value;
      if (fellBackMatch && fellBackMatch[1] === "true") {
        result[model].fellBack += value;
      }
    }
  }
  return result;
}

export const Route = createFileRoute("/api/public/staging/metrics")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        const cronSecret = process.env.CRON_SECRET;
        if (!cronSecret) {
          return new Response(JSON.stringify({ error: "CRON_SECRET not configured" }), {
            status: 500,
            headers: { "content-type": "application/json" },
          });
        }

        const auth = request.headers.get("authorization") ?? "";
        if (auth !== `Bearer ${cronSecret}`) {
          return new Response(JSON.stringify({ error: "Unauthorized" }), {
            status: 401,
            headers: { "content-type": "application/json" },
          });
        }

        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const origin = request.headers.get("origin");
        const corsHeaders = getCorsHeadersForResponse(origin);

        const stage = (process.env.AI_EEGCONFORMER_ENABLED ?? "off") as "off" | "canary" | "beta" | "ga";
        const prometheus = renderPrometheusMetrics();

        const cohortHit = extractCounter(prometheus, "neuro_fabric_eegconformer_cohort_checks_total", { result: "hit" });
        const cohortMiss = extractCounter(prometheus, "neuro_fabric_eegconformer_cohort_checks_total", { result: "miss" });
        const modelSelected = extractModelSelected(prometheus);

        const totalEmbeds = Object.values(modelSelected).reduce((s, v) => s + v.total, 0);
        const totalFallbacks = Object.values(modelSelected).reduce((s, v) => s + v.fellBack, 0);
        const fallbackRate = totalEmbeds > 0 ? totalFallbacks / totalEmbeds : 0;

        const artifactPass = extractCounter(prometheus, "neuro_fabric_artifact_verification_total", { result: "pass" });
        const artifactFail = extractCounter(prometheus, "neuro_fabric_artifact_verification_total", { result: "fail" });
        const artifactAttempt = extractCounter(prometheus, "neuro_fabric_artifact_verification_total", { result: "attempt" });
        const latency = extractLatencyPercentiles(prometheus);

        const cohortTotal = cohortHit + cohortMiss;
        const cohortHitRate = cohortTotal > 0 ? cohortHit / cohortTotal : 0;

        const p95Ms = parseFloat(latency.p95);
        const p50Ms = parseFloat(latency.p50);

        // GA exit criteria gates
        const gates = {
          p95LatencyMs: p95Ms,
          p50LatencyMs: p50Ms,
          fallbackRate,
          hasVerificationFailures: artifactFail > 0,
          cohortHitRate,
          p95LatencyOk: p95Ms > 0 ? p95Ms < 600 : true,
          p50LatencyOk: p50Ms > 0 ? p50Ms < 400 : true,
          fallbackRateOk: totalEmbeds > 0 ? fallbackRate < 0.005 : true,
          allGatesPass:
            (p95Ms === 0 || p95Ms < 600) &&
            (p50Ms === 0 || p50Ms < 400) &&
            (totalEmbeds === 0 || fallbackRate < 0.005) &&
            artifactFail === 0,
        };

        const response: StagingMetricsResponse = {
          timestamp: new Date().toISOString(),
          rolloutStage: stage,
          cohortChecks: { hit: cohortHit, miss: cohortMiss },
          modelSelected,
          fallbackRate,
          artifactVerification: { pass: artifactPass, fail: artifactFail, attempt: artifactAttempt },
          latency,
          gates,
        };

        return applySecurityHeaders(
          new Response(JSON.stringify(response, null, 2), {
            status: 200,
            headers: {
              "content-type": "application/json",
              "cache-control": "no-cache, no-store, must-revalidate",
              ...corsHeaders,
            },
          }),
        );
      },
      POST: async ({ request }) => {
        // Accept browser-side metrics snapshots from the staging harness.
        // This bridges browser WASM inference metrics (collected in-browser via
        // collectMetricsSnapshot()) into the server-side Prometheus registry,
        // making them visible to GET /api/public/staging/metrics and the
        // 24-hour observation script.
        const cronSecret = process.env.CRON_SECRET;
        if (!cronSecret) {
          return new Response(JSON.stringify({ error: "CRON_SECRET not configured" }), {
            status: 500,
            headers: { "content-type": "application/json" },
          });
        }

        const auth = request.headers.get("authorization") ?? "";
        if (auth !== `Bearer ${cronSecret}`) {
          return new Response(JSON.stringify({ error: "Unauthorized" }), {
            status: 401,
            headers: { "content-type": "application/json" },
          });
        }

        let body: unknown;
        try {
          body = await request.json();
        } catch {
          return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
            status: 400,
            headers: { "content-type": "application/json" },
          });
        }

        const snapshot = body as { prometheus?: string; metrics?: Record<string, any[]> };
        if (!snapshot.prometheus && !snapshot.metrics) {
          return new Response(JSON.stringify({ error: "Missing prometheus or metrics field" }), {
            status: 400,
            headers: { "content-type": "application/json" },
          });
        }

        // Merge browser-side Prometheus text into server-side registry.
        // The Prometheus text from renderPrometheusMetrics() contains ALL
        // counters, gauges, and histograms, so we don't also process the
        // structured metrics object — doing so would double-count counters.
        if (snapshot.prometheus) {
          mergeBrowserMetrics(snapshot.prometheus);
        }

        return new Response(JSON.stringify({ status: "merged", timestamp: new Date().toISOString() }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      },
    },
  },
});
