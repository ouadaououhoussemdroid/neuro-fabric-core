/**
 * T-028 — Ground Truth Annotation API endpoints.
 *
 * Routes:
 *   POST /api/annotations          — submit one or more ground-truth labels
 *   GET  /api/annotations          — list annotations (filter by analysisId or subjectId)
 *
 * Annotations link to `eeg_analyses.id` via `analysisId` and to a subject
 * via `subjectId`.  They are validated against the `eeg_analyses` table
 * to ensure referential integrity.
 */
import { createFileRoute } from "@tanstack/react-router";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { log } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { startTimer } from "@/lib/logging";
import { summarizeAnnotations } from "@/lib/evaluation";
import type { GroundTruthLabel } from "@/lib/evaluation";

const RATE_LIMIT_MAX = 20;
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

interface AnnotationSubmitBody {
  annotations: Omit<GroundTruthLabel, "id" | "createdAt">[];
}

interface AnnotationListQuery {
  analysisId?: string;
  subjectId?: string;
}

export const Route = createFileRoute("/api/annotations/")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);
        const timer = startTimer("annotations.submit.total");
        void timer;

        if (metrics.evaluationRequestsTotal) metrics.evaluationRequestsTotal.inc();

        try {
          // Auth
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
            log("error", "annotations.auth_error", {
              error: (authErr as Error).message,
            });
            return res({ error: "Authentication failed." }, 401);
          }

          // Rate limit
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
            log("warn", "annotations.rate_limit_error", {
              error: (rlErr as Error).message,
            });
          }

          // Parse body
          let body: AnnotationSubmitBody;
          try {
            body = (await request.json()) as AnnotationSubmitBody;
          } catch {
            return res({ error: "Invalid JSON body." }, 400);
          }

          if (!Array.isArray(body.annotations) || body.annotations.length === 0) {
            return res({ error: "Field 'annotations' must be a non-empty array." }, 400);
          }

          // Validate and insert each annotation.
          const inserted: GroundTruthLabel[] = [];
          for (const ann of body.annotations) {
            // Verify the analysis exists and belongs to the user.
            const { data: analysis, error: lookupErr } = await supabase
              .from("eeg_analyses")
              .select("id")
              .eq("id", ann.analysisId)
              .eq("user_id", userId)
              .single();

            if (lookupErr || !analysis) {
              return res(
                { error: `Analysis '${ann.analysisId}' not found or not accessible.` },
                404,
              );
            }

            // Validate fields.
            if (!ann.subjectId || typeof ann.subjectId !== "string") {
              return res({ error: "Each annotation must have a 'subjectId'." }, 400);
            }
            if (typeof ann.value !== "number") {
              return res({ error: "Each annotation must have a numeric 'value'." }, 400);
            }
            if (!ann.annotatorUserId) {
              return res({ error: "Each annotation must have an 'annotatorUserId'." }, 400);
            }
            if (ann.confidence !== undefined && (ann.confidence < 0 || ann.confidence > 1)) {
              return res({ error: "'confidence' must be between 0 and 1." }, 400);
            }

            const { data: insertedRow, error: insertErr } = await supabase
              .from("ground_truth_labels")
              .insert({
                analysis_id: ann.analysisId,
                subject_id: ann.subjectId,
                type: ann.type,
                value: ann.value,
                start_sample: ann.startSample,
                end_sample: ann.endSample,
                annotator_user_id: ann.annotatorUserId,
                confidence: ann.confidence ?? 1,
              })
              .select()
              .single();

            if (insertErr) {
              log("error", "annotations.insert_failed", {
                error: insertErr.message,
                analysisId: ann.analysisId,
              });
              return res({ error: "Failed to save annotation." }, 500);
            }

            inserted.push({
              id: insertedRow.id,
              analysisId: insertedRow.analysis_id,
              subjectId: insertedRow.subject_id,
              type: insertedRow.type,
              value: insertedRow.value,
              startSample: insertedRow.start_sample ?? undefined,
              endSample: insertedRow.end_sample ?? undefined,
              annotatorUserId: insertedRow.annotator_user_id,
              confidence: insertedRow.confidence,
              createdAt: insertedRow.created_at,
            });
          }

          log("info", "annotations.submitted", {
            userId,
            nAnnotations: inserted.length,
          });

          return res({ annotations: inserted, summaries: summarizeAnnotations(inserted) }, 200);
        } catch (err) {
          log("error", "annotations.submit_failed", {
            error: (err as Error).message,
          });
          if (metrics.evaluationErrorsTotal) metrics.evaluationErrorsTotal.inc();
          return res({ error: "An error occurred while submitting annotations." }, 500);
        }
      },

      GET: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);
        void requestOrigin;

        try {
          // Auth
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
            return res({ error: "Authentication failed." }, 401);
          }

          // Parse query parameters.
          const url = new URL(request.url);
          const analysisId = url.searchParams.get("analysisId");
          const subjectId = url.searchParams.get("subjectId");

          let query = supabase
            .from("ground_truth_labels")
            .select("*")
            .eq("annotator_user_id", userId);

          if (analysisId) query = query.eq("analysis_id", analysisId);
          if (subjectId) query = query.eq("subject_id", subjectId);

          const { data, error: fetchErr } = await query;

          if (fetchErr) {
            log("error", "annotations.fetch_failed", {
              error: fetchErr.message,
            });
            return res({ error: "Failed to fetch annotations." }, 500);
          }

          const labels: GroundTruthLabel[] = (data ?? []).map((row) => ({
            id: row.id,
            analysisId: row.analysis_id,
            subjectId: row.subject_id,
            type: row.type,
            value: row.value,
            startSample: row.start_sample ?? undefined,
            endSample: row.end_sample ?? undefined,
            annotatorUserId: row.annotator_user_id,
            confidence: row.confidence,
            createdAt: row.created_at,
          }));

          return res(
            {
              annotations: labels,
              summaries: summarizeAnnotations(labels),
            },
            200,
          );
        } catch (err) {
          log("error", "annotations.fetch_failed", {
            error: (err as Error).message,
          });
          return res({ error: "An error occurred while fetching annotations." }, 500);
        }
      },
    },
  },
});
