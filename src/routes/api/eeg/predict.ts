/**
 * M48 — Predictive Neural Coding API endpoint.
 *
 * Route: POST /api/eeg/predict
 *
 * Runs the predictive coding engine on a Joint-2312 embedding or raw EEG signal:
 *   1. If embedding_id provided → fetch stored 2312-D embedding from DB
 *   2. If raw EEG provided → embed via embedEEG() → Joint-2312
 *   3. Run LSTM autoregressive prediction (or CPU AR fallback)
 *   4. Compute prediction error (surprise) + band-limited surprise scores
 *   5. Flag anomalies (surprise > kσ above baseline)
 *
 * Uses the "Embed Once → Reuse Many" principle: if the caller provides
 * an embedding_id, the existing Joint-2312 embedding is reused without
 * recomputation.
 */
import { createFileRoute } from "@tanstack/react-router";
import { randomUUID } from "node:crypto";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";
import {
  predictSignal,
  PREDICTIVE_CODING_SERVICE,
  type PredictiveCodingOptions,
  PredictiveCodingError,
} from "@/lib/ai/inference/predictive-coding.server";
import { embedEEG } from "@/lib/ai/inference/embed-eeg";
import { parseEDF, parseCSV } from "@/lib/eeg/parsers";
import { preprocess } from "@/lib/eeg/preprocessing";
import type { EEGSignal } from "@/lib/eeg/types";

function json(body: unknown, status = 200, origin: string | null = null): Response {
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json", ...getCorsHeadersForResponse(origin) },
    }),
  );
}

/** Create a json() closure bound to a request origin for CORS headers. */
function makeJson(origin: string | null) {
  return (body: unknown, status = 200): Response => json(body, status, origin);
}

const RATE_LIMIT_MAX = 30;
const RATE_LIMIT_WINDOW_SECONDS = 60;
const PROCESSING_TIMEOUT_MS = 30_000;
const MAX_FILE_BYTES = 50 * 1024 * 100; // 5 MB for prediction (lighter than full embed)

/** Parse EEG signal from uploaded file or JSON body. */
async function parseSignal(
  request: Request,
): Promise<{ signal: EEGSignal; embeddingId?: string } | { error: string; status: number }> {
  const contentType = request.headers.get("content-type") || "";

  if (contentType.includes("multipart/form-data")) {
    // Parse multipart: file + JSON fields
    const formData = await request.formData();

    const embeddingId = formData.get("embedding_id")?.toString();
    if (embeddingId) {
      // Reuse existing embedding — no file needed
      return { signal: { channels: [], data: [], sampleRate: 250 }, embeddingId };
    }

    const file = formData.get("file") as File | null;
    if (!file) {
      return { error: "No file provided", status: 400 };
    }

    if (file.size > MAX_FILE_BYTES) {
      return { error: "File too large", status: 413 };
    }

      const sampleRate = parseInt(formData.get("sampleRate")?.toString() ?? "250", 10);
      const arrayBuffer = await file.arrayBuffer();

      const ext = file.name.split(".").pop()?.toLowerCase();
      if (ext === "csv") {
        const signal = parseCSV(new TextDecoder().decode(arrayBuffer), sampleRate);
        if (!signal) return { error: "Failed to parse CSV", status: 400 };
        return { signal };
      }
      if (ext === "edf") {
        const signal = parseEDF(arrayBuffer);
        if (!signal) return { error: "Failed to parse EDF", status: 400 };
        return { signal };
      }
      return { error: `Unsupported file type: .${ext}`, status: 400 };
  }

  if (contentType.includes("application/json")) {
    const body = await request.json() as Record<string, unknown>;

    if (body.embedding_id) {
      return { signal: { channels: [], data: [], sampleRate: 250 }, embeddingId: body.embedding_id as string };
    }

    if (body.signal && Array.isArray(body.signal)) {
      const signal: EEGSignal = {
        channels: body.channels as string[] ?? [],
        data: body.signal as number[][],
        sampleRate: body.sampleRate as number ?? 250,
      };
      return { signal };
    }

    return { error: "No signal or embedding_id provided", status: 400 };
  }

  return { error: "Content-Type must be multipart/form-data or application/json", status: 400 };
}

export const Route = createFileRoute("/api/eeg/predict")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

        const overall = startTimer("predict.total");
        void overall;

        // Parse request
        const parsed = await parseSignal(request);
        if ("error" in parsed) {
          return res({ error: parsed.error }, parsed.status);
        }
        const { signal, embeddingId } = parsed;

        if (embeddingId) {
          // TODO: fetch 2312-D embedding from DB and reconstruct EEG signal
          // For now, return an error indicating this path needs the embedding
          return res(
            {
              error:
                "embedding_id lookup not yet implemented. Provide raw EEG via file upload or JSON.",
            },
            501,
          );
        }

        // Authenticate
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

        // Rate limit
        const rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
        if (!rl.allowed) {
          return res(
            { error: "Rate limit exceeded", retryAfterMs: rl.retryAfterMs },
            429,
          );
        }

        // Parse prediction options from form/JSON
        let opts: PredictiveCodingOptions = {};
        try {
          const horizon = parseInt(
            (signal as any)._horizon?.toString() ?? "8",
            10,
          );
          opts = { horizon: isNaN(horizon) ? 8 : horizon };
        } catch {
          /* use defaults */
        }

        // Run timeout-protected prediction
        let timeoutId: ReturnType<typeof setTimeout>;
        const timeoutPromise = new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(new Error(`Predictive coding timeout (${PROCESSING_TIMEOUT_MS}ms)`));
          }, PROCESSING_TIMEOUT_MS);
        });

        try {
          const result = await Promise.race([
            predictSignal(signal, opts),
            timeoutPromise,
          ]);

          void overall;
          metrics.tier1ServiceRequestsTotal.inc({ service: PREDICTIVE_CODING_SERVICE });

          return res({
            service: PREDICTIVE_CODING_SERVICE,
            version: "v0.1.0",
            user_id: userId,
            request_id: randomUUID(),
            results: {
              channels: result.channels,
              overallSurprise: result.overallSurprise,
              isAnomalous: result.isAnomalous,
              anomalyScore: result.anomalyScore,
              forecastHorizon: result.forecastHorizon,
            },
            used_model: result.usedModel,
            provenance: result.provenance,
            timing: {
              total_ms: result.durationMs,
            },
          });
        } catch (err) {
          if (err instanceof PredictiveCodingError) {
            metrics.tier1ServiceErrorsTotal.inc({ service: PREDICTIVE_CODING_SERVICE });
            log("error", "predict.failed", { error: err.message, code: err.code, userId });
            return res({ error: err.message, code: err.code }, 500);
          }
          if ((err as Error).message.includes("timeout")) {
            metrics.tier1ServiceErrorsTotal.inc({ service: PREDICTIVE_CODING_SERVICE });
            log("error", "predict.timeout", { userId, timeoutMs: PROCESSING_TIMEOUT_MS });
            return res({ error: "Processing timeout" }, 408);
          }
          metrics.tier1ServiceErrorsTotal.inc({ service: PREDICTIVE_CODING_SERVICE });
          log("error", "predict.failed_unexpected", {
            error: (err as Error).message,
            userId,
          });
          return res({ error: "An error occurred during prediction." }, 500);
        } finally {
          clearTimeout(timeoutId!);
        }
      },
    },
  },
});
