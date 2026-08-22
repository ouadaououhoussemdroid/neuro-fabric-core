/**
 * M41 — Multi-Task Fusion on Joint-2312.
 *
 * Route: POST /api/joint2312/fusion
 *
 * Runs all 4 Tier-1+Tier-2 task probes (cognitive, anomaly, sleep-staging,
 * sleep-quality) in parallel on a single Joint-2312 embedding. Embed-once-
 * reuse-many at the batch level: one embedding lookup → 4 ONNX probes →
 * unified response with shared provenance.
 *
 * Subject Identity (M32) is excluded (requires vector RPC, not single-vector
 * inference). Call /api/joint2312/search/subject-identity separately.
 */
import { createFileRoute } from "@tanstack/react-router";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import {
  decodeJoint2312,
  type JointFusionRequest,
  type JointFusionResponse,
  JOINT_FUSION_TIMEOUT_MS,
} from "@/lib/ai/inference/joint-fusion.server";
import { metrics } from "@/lib/metrics";

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;
const PROCESSING_TIMEOUT_MS = JOINT_FUSION_TIMEOUT_MS;

const res = (origin: string | null, body: unknown, init: ResponseInit = {}) =>
  applySecurityHeaders(
    new Response(JSON.stringify(body), {
      ...init,
      headers: {
        "content-type": "application/json",
        ...getCorsHeadersForResponse(origin),
        ...init.headers,
      },
    }),
  );

export const Route = createFileRoute("/api/joint2312/fusion")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const overall = startTimer("joint.decode.total");
        void overall;
        metrics.tier1ServiceRequestsTotal.inc({ service: "joint-fusion" });

        let timeoutId: ReturnType<typeof setTimeout>;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(new Error(`Joint fusion timeout (${PROCESSING_TIMEOUT_MS}ms)`));
          }, PROCESSING_TIMEOUT_MS);
        });

        try {
          return await Promise.race([processDecode(), timeoutPromise]);
        } catch (err) {
          const msg = (err as Error).message;
          if (err instanceof AuthError) {
            return res(requestOrigin, { error: err.message }, { status: err.status });
          }
          if (msg.includes("timeout")) {
            log("error", "joint.fusion.timeout_exceeded", { timeoutMs: PROCESSING_TIMEOUT_MS });
            return res(requestOrigin, { error: "Processing timeout. Please try a smaller file." }, { status: 408 });
          }
          log("error", "joint.fusion.failed", { error: msg });
          metrics.sleepDecodeErrorsTotal.inc({ error: "fusion_failed" });
          metrics.tier1ServiceErrorsTotal.inc({ service: "joint-fusion" });
          return res(requestOrigin, { error: "An error occurred during joint fusion decode." }, { status: 500 });
        } finally {
          clearTimeout(timeoutId!);
        }

        async function processDecode(): Promise<Response> {
          // Authentication
          let userId: string;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let supabase: any;
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            if (authErr instanceof AuthError) {
              return res(requestOrigin, { error: authErr.message }, { status: authErr.status });
            }
            log("error", "joint.fusion.auth_unexpected", {
              error: (authErr as Error).message,
            });
            return res(requestOrigin, { error: "Authentication failed." }, { status: 401 });
          }

          // Rate limiting
          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            log("error", "joint.fusion.rate_limit_check_failed", {
              error: (rlErr as Error).message,
              userId,
            });
            return res(requestOrigin, { error: "Rate limit service unavailable." }, { status: 503 });
          }
          if (!rl.allowed) {
            return res(
              requestOrigin,
              {
                error: "Rate limit exceeded. Try again shortly.",
                retry_after_ms: rl.retryAfterMs,
              },
              { status: 429 },
            );
          }

          // Parse request body
          const ct = request.headers.get("content-type") ?? "";
          if (!ct.includes("application/json")) {
            return res(requestOrigin, { error: "expected application/json body" }, { status: 400 });
          }

          let body: unknown;
          try {
            body = await request.json();
          } catch {
            return res(requestOrigin, { error: "Invalid JSON body" }, { status: 400 });
          }

          const req = body as JointFusionRequest;

          // Validate that either embedding_id or query_embedding is provided
          if (!req.embedding_id && !req.query_embedding) {
            return res(
              requestOrigin,
              { error: "Either embedding_id or query_embedding must be provided" },
              { status: 400 },
            );
          }

          // Validate query_embedding dimension if provided
          if (req.query_embedding && req.query_embedding.length !== 2312) {
            return res(
              requestOrigin,
              {
                error: `query_embedding must be 2312-dimensional, got ${req.query_embedding.length}`,
              },
              { status: 400 },
            );
          }

          // Validate optional heads parameter
          const validHeads = ["cognitive", "anomaly", "sleep-staging", "sleep-quality"];
          if (req.heads) {
            for (const h of req.heads) {
              if (!validHeads.includes(h)) {
                return res(
                  requestOrigin,
                  { error: `Invalid head "${h}". Must be one of: ${validHeads.join(", ")}` },
                  { status: 400 },
                );
              }
            }
          }

          // Run the fusion decode
          const tDecode = startTimer("joint.decode.inference");
          let response: JointFusionResponse;
          try {
            response = await decodeJoint2312(req, supabase, userId);
          } catch (decodeErr) {
            const err = decodeErr as Error & { code?: string };
            log("warn", "joint.fusion.decode_error", {
              code: err.code,
              error: err.message,
              userId,
            });
            return res(
              requestOrigin,
              { error: err.message, code: err.code ?? "DECODE_ERROR" },
              { status: 400 },
            );
          }
          tDecode.end({
            heads_run: response.metadata.heads_run.length,
          });

          return res(requestOrigin, response, { status: 200 });
        }
      },
    },
  },
});
