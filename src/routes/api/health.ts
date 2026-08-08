/**
 * Production health check endpoint.
 *
 * Route: GET /api/health
 *
 * Lightweight liveness/readiness probe. No authentication required.
 * Returns 200 when all critical dependencies are available, 503 otherwise.
 */
import { createFileRoute } from "@tanstack/react-router";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { isONNXRuntimeAvailable, __resetONNXCapabilityProbe } from "@/lib/ai/adapters/onnx-adapter";
import { log } from "@/lib/logging";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";

const VERSION = "1.0.0";

interface HealthCheck {
  name: string;
  status: "ok" | "degraded" | "down";
}

interface HealthResponse {
  status: "ok" | "degraded" | "down";
  timestamp: string;
  version: string;
  checks: HealthCheck[];
}

export const Route = createFileRoute("/api/health")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        // CORS pre-flight / origin check.
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const checks: HealthCheck[] = [];

        // Application availability — always ok if we got this far.
        checks.push({ name: "application", status: "ok" });

        // Supabase connectivity.
        const dbCheck = await checkDatabase();
        checks.push(dbCheck);

        // ONNX runtime availability.
        const onnxCheck = await checkONNXRuntime();
        checks.push(onnxCheck);

        const anyDown = checks.some((c) => c.status === "down");

        // HTTP status: 200 when healthy or degraded (system still serves
        // requests via fallbacks), 503 only when a dependency is fully down.
        const httpStatus = anyDown ? 503 : 200;

        const response: HealthResponse = {
          status: anyDown ? "down" : checks.every((c) => c.status === "ok") ? "ok" : "degraded",
          timestamp: new Date().toISOString(),
          version: VERSION,
          checks,
        };

        const corsHeaders = getCorsHeadersForResponse(request.headers.get("origin"));
        return applySecurityHeaders(
          new Response(JSON.stringify(response, null, 2), {
            status: httpStatus,
            headers: {
              "content-type": "application/json",
              ...corsHeaders,
            },
          }),
        );
      },
    },
  },
});

async function checkDatabase(): Promise<HealthCheck> {
  try {
    if (!supabaseAdmin) {
      return { name: "database", status: "down" };
    }
    // Cheap connectivity probe: read-only RPC that returns 1.
    // Cast to bypass generated types that don't know about health_check yet.
    const { data, error } = await (
      supabaseAdmin as unknown as {
        rpc: (
          fn: string,
          args?: Record<string, unknown>,
        ) => Promise<{ data: unknown; error: unknown }>;
      }
    ).rpc("health_check");
    if (
      error ||
      !data ||
      !Array.isArray(data) ||
      data.length === 0 ||
      (data[0] as { ok?: boolean })?.ok !== true
    ) {
      log("warn", "health.db_error", {
        error: (error as { message?: string } | null)?.message ?? "no data or unhealthy",
      });
      return { name: "database", status: "down" };
    }
    return { name: "database", status: "ok" };
  } catch (e) {
    log("error", "health.db_exception", { error: (e as Error).message });
    return { name: "database", status: "down" };
  }
}

async function checkONNXRuntime(): Promise<HealthCheck> {
  try {
    // Reset the cached probe so each health check reflects current state.
    __resetONNXCapabilityProbe();
    const available = await isONNXRuntimeAvailable();
    if (available) {
      return { name: "onnx_runtime", status: "ok" };
    }
    // ONNX not available is a degraded state, not a hard failure —
    // the platform degrades to PCA/heuristic fallbacks.
    return { name: "onnx_runtime", status: "degraded" };
  } catch (e) {
    log("error", "health.onnx_exception", { error: (e as Error).message });
    return { name: "onnx_runtime", status: "degraded" };
  }
}

function json(body: HealthResponse, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });
}
