/**
 * T-028 — Cross-Subject Validation (LOSO) API endpoint.
 *
 * Route: POST /api/evaluate/cross-subject
 *
 * Runs leave-one-subject-out cross-validation over a set of embedding samples
 * and returns per-subject metrics + aggregate statistics.
 *
 * Request body (JSON):
 *   {
 *     "samples": [
 *       { "subjectId": "sub-001", "embedding": [0.1, 0.2, ...], "label": 0 },
 *       ...
 *     ],
 *     "options": { "k": 1, "confidence": 0.95 }
 *   }
 *
 * Response:
 *   {
 *     "nFolds": 5, "nSubjects": 5, "nSamples": 100, "nClasses": 4,
 *     "chanceAccuracy": 0.25,
 *     "perSubject": { "sub-001": { "accuracy": 0.8, ... }, ... },
 *     "aggregate": {
 *       "meanAccuracy": 0.72, "stdAccuracy": 0.08,
 *       "accuracyCI": { "lower": 0.64, "upper": 0.80, "margin": 0.08 },
 *       "tTest": { "t": 3.2, "pValue": 0.018, "significant": true },
 *       "effectSize": { "d": 2.1, "interpretation": "large" }
 *     },
 *     "pcaBaseline": { "meanAccuracy": 0.25, ... }
 *   }
 *
 * Authentication: Supabase JWT required (same pattern as /api/eeg/upload).
 * Rate limit: 5 requests / 60 s per user.
 */
import { createFileRoute } from "@tanstack/react-router";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { log } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { startTimer } from "@/lib/logging";
import { evaluateLOSO } from "@/lib/evaluation";
import type { LOFOSSample, LOSOOptions } from "@/lib/evaluation";

const RATE_LIMIT_MAX = 5;
const RATE_LIMIT_WINDOW_SECONDS = 60;

function json(body: unknown, status = 200, origin: string | null = null): Response {
  const corsHeaders = getCorsHeadersForResponse(origin);
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: {
        "content-type": "application/json",
        ...corsHeaders,
      },
    }),
  );
}

function makeJson(origin: string | null) {
  return (body: unknown, status = 200): Response => json(body, status, origin);
}

interface CrossSubjectRequest {
  samples: LOFOSSample[];
  options?: LOSOOptions;
}

export const Route = createFileRoute("/api/evaluate/cross-subject")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        // CORS pre-flight / origin check.
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

        const timer = startTimer("evaluate.cross_subject.total");
        void timer;

        try {
          // Authentication.
          let userId: string;
          let supabase: Awaited<ReturnType<typeof authenticateRequest>>["supabase"];
          try {
            const auth = await authenticateRequest(request);
            userId = auth.userId;
            supabase = auth.supabase;
          } catch (authErr) {
            if (authErr instanceof AuthError) {
              return res({ error: authErr.message }, authErr.status);
            }
            log("error", "evaluate.cross_subject.auth_error", {
              error: (authErr as Error).message,
            });
            return res({ error: "Authentication failed." }, 401);
          }

          // Rate limiting.
          try {
            const rl = await checkRateLimit(
              supabase,
              userId,
              RATE_LIMIT_MAX,
              RATE_LIMIT_WINDOW_SECONDS,
            );
            if (!rl.allowed) {
              return res({ error: "Rate limit exceeded. Try again later." }, 429);
            }
          } catch (rlErr) {
            log("warn", "evaluate.cross_subject.rate_limit_error", {
              error: (rlErr as Error).message,
            });
          }

          // Parse request body.
          let body: CrossSubjectRequest;
          try {
            body = (await request.json()) as CrossSubjectRequest;
          } catch {
            return res({ error: "Invalid JSON body." }, 400);
          }

          // Validate input.
          if (!Array.isArray(body.samples) || body.samples.length === 0) {
            return res({ error: "Field 'samples' must be a non-empty array." }, 400);
          }
          if (body.samples.length < 2) {
            return res({ error: "At least 2 samples are required for cross-validation." }, 400);
          }

          // Validate each sample.
          for (let i = 0; i < body.samples.length; i++) {
            const s = body.samples[i];
            if (!s.subjectId || typeof s.subjectId !== "string") {
              return res({ error: `Sample ${i}: missing or invalid 'subjectId'.` }, 400);
            }
            if (!Array.isArray(s.embedding) || s.embedding.length === 0) {
              return res({ error: `Sample ${i}: missing or empty 'embedding'.` }, 400);
            }
            if (typeof s.label !== "number" || !Number.isInteger(s.label)) {
              return res({ error: `Sample ${i}: 'label' must be an integer.` }, 400);
            }
          }

          // Ensure all embeddings have the same dimension.
          const dim = body.samples[0].embedding.length;
          for (const s of body.samples) {
            if (s.embedding.length !== dim) {
              return res({ error: `All embeddings must have dimension ${dim}.` }, 400);
            }
          }

          // Run evaluation.
          const result = evaluateLOSO(body.samples, body.options);

          log("info", "evaluate.cross_subject.completed", {
            userId,
            nFolds: result.nFolds,
            meanAccuracy: result.aggregate.meanAccuracy,
            meanRecallAtK: result.aggregate.meanRecallAtK,
          });

          if (metrics.evaluationRequestsTotal) metrics.evaluationRequestsTotal.inc();

          return res(result, 200);
        } catch (err) {
          log("error", "evaluate.cross_subject.failed", {
            error: (err as Error).message,
          });
          if (metrics.evaluationErrorsTotal) metrics.evaluationErrorsTotal.inc();
          return res({ error: "An error occurred during evaluation." }, 500);
        }
      },
    },
  },
});
