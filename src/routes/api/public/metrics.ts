/**
 * T-008 — Prometheus-format metrics endpoint.
 *
 * Route: /api/public/metrics
 *
 * Returns all in-process metrics (counters, gauges, histograms) in Prometheus
 * text exposition format. Protected by CRON_SECRET to prevent information
 * leakage to unauthenticated callers. In production, a sidecar or metrics
 * scraper (Prometheus, Grafana Agent) polls this endpoint; per-isolate state
 * is aggregated by the scraper.
 *
 * Note: on Cloudflare Workers with multiple isolates, each isolate reports its
 * own counters. The Prometheus `sum()` aggregation operator should be used
 * when querying (e.g., `sum(rate(neuro_fabric_upload_requests_total[5m]))`).
 */
import { createFileRoute } from "@tanstack/react-router";
import { renderPrometheusMetrics } from "@/lib/metrics";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";

export const Route = createFileRoute("/api/public/metrics")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        // Fail closed: if CRON_SECRET is not configured, block all access
        // rather than allowing unauthenticated metrics exposure.
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

        const body = renderPrometheusMetrics();
        return applySecurityHeaders(
          new Response(body, {
            status: 200,
            headers: {
              "content-type": "text/plain; version=0.0.4; charset=utf-8",
              "cache-control": "no-cache, no-store, must-revalidate",
              ...corsHeaders,
            },
          }),
        );
      },
    },
  },
});
