/**
 * M39 — Sleep Staging decode endpoint.
 *
 * Route: POST /api/joint2312/sleep/decode
 *
 * Decodes 5-class sleep staging (W, N1, N2, N3, REM) from a Joint-2312
 * embedding using a linear probe (2312→5).
 *
 * **Embed Once → Reuse Many:** if the caller provides `embedding_id`, the
 * existing Joint-2312 embedding stored in `joint_embeddings_2312` is reused —
 * no recomputation of Joint-2312. If `query_embedding` is provided, it is
 * used directly. The upstream `/api/eeg/embed/foundation?model=joint-2312`
 * route is responsible for computing and storing embeddings.
 *
 * Security: same auth/rate-limit/CORS pattern as `/api/eeg/embed/foundation`.
 * All results are RLS-scoped to the authenticated user.
 */
import { createFileRoute } from "@tanstack/react-router";
import { log, startTimer } from "@/lib/logging";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import {
  decodeSleepState,
  type SleepDecodeRequest,
  type SleepDecodeResponse,
  SleepDecodeError,
  SLEEP_TIMEOUT_MS,
} from "@/lib/ai/inference/sleep.server";
import { metrics } from "@/lib/metrics";

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;
const PROCESSING_TIMEOUT_MS = SLEEP_TIMEOUT_MS;

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

export const Route = createFileRoute("/api/joint2312/sleep/decode")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const overall = startTimer("sleep.decode.total");
        void overall;
        metrics.tier1ServiceRequestsTotal.inc({ service: "sleep-staging" });

        let timeoutId: ReturnType<typeof setTimeout>;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(new Error(`Sleep decode timeout (${PROCESSING_TIMEOUT_MS}ms)`));
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
            log("error", "sleep.decode.timeout_exceeded", { timeoutMs: PROCESSING_TIMEOUT_MS });
            return res(requestOrigin, { error: "Processing timeout. Please try a smaller file." }, { status: 408 });
          }
          log("error", "sleep.decode.failed", { error: msg });
          metrics.sleepDecodeErrorsTotal.inc({ error: "decode_failed" });
          metrics.tier1ServiceErrorsTotal.inc({ service: "sleep-staging" });
          return res(requestOrigin, { error: "An error occurred during sleep staging." }, { status: 500 });
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
            log("error", "sleep.decode.auth_unexpected", {
              error: (authErr as Error).message,
            });
            return res(requestOrigin, { error: "Authentication failed." }, { status: 401 });
          }

          // Rate limiting
          let rl: { allowed: boolean; retryAfterMs: number };
          try {
            rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          } catch (rlErr) {
            log("error", "sleep.decode.rate_limit_check_failed", {
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

          const req = body as SleepDecodeRequest;

          // Validate query_type
          const validQueryTypes = ["sleep-stages"];
          if (!req.query_type || !validQueryTypes.includes(req.query_type)) {
            return res(
              requestOrigin,
              { error: `query_type must be "sleep-stages" (or omit for default)` },
              { status: 400 },
            );
          }

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

          // Run the decode
          const tDecode = startTimer("sleep.decode.inference");
          let response: SleepDecodeResponse;
          try {
            response = await decodeSleepState(req, supabase, userId);
          } catch (decodeErr) {
            if (decodeErr instanceof SleepDecodeError) {
              log("warn", "sleep.decode.decode_error", {
                code: decodeErr.code,
                error: decodeErr.message,
                userId,
              });
              return res(
                requestOrigin,
                { error: decodeErr.message, code: decodeErr.code },
                { status: 400 },
              );
            }
            throw decodeErr;
          }
          tDecode.end({
            query_type: req.query_type,
            results: response.results.length,
          });

          return res(requestOrigin, response, { status: 200 });
        }
      },
    },
  },
});
