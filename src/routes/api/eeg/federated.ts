/**
 * M49 — Federated Brain Learning API endpoint.
 *
 * Routes:
 *   POST /api/eeg/federated/round  — aggregate client updates (FedAvg)
 *   GET  /api/eeg/federated/model/:task — distribute global model weights
 *   POST /api/eeg/federated/validate — validate a client update before submission
 *
 * Flow:
 *   1. Client trains local V2-32 probe on its data (no raw EEG leaves the browser)
 *   2. Client computes weight delta (gradient) locally
 *   3. Client POSTs to /validate → server checks dimensions/NaN
 *   4. Client POSTs to /round → server clips L2 norm, applies DP, FedAvg aggregation
 *   5. Server updates global model → client GETs updated weights for next round
 */
import { createFileRoute } from "@tanstack/react-router";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";
import { authenticateRequest, AuthError } from "@/integrations/supabase/request-auth";
import { checkRateLimit } from "@/integrations/supabase/rate-limit";

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
import {
  runFederatedRound,
  getGlobalModelWeights,
  validateClientUpdate,
  FEDERATED_SERVICE,
  type ClientUpdate,
  type FederatedTask,
  FederatedLearningError,
} from "@/lib/ai/inference/federated-learning.server";

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_SECONDS = 60;
const ROUND_TIMEOUT_MS = 60_000;

/** Validate that a task string is a recognized FederatedTask. */
function isValidTask(task: string): task is FederatedTask {
  return ["sleep-staging", "sleep-quality", "cognitive-workload", "anomaly-detection"].includes(task);
}

/** Parse a client update from JSON body. */
function parseClientUpdate(body: Record<string, unknown>): ClientUpdate {
  return {
    clientId: body.client_id as string,
    task: body.task as FederatedTask,
    weightDelta: body.weight_delta as number[][],
    biasDelta: body.bias_delta as number[],
    sampleCount: body.sample_count as number,
    loss: body.loss as number,
    accuracy: body.accuracy as number,
    epochs: body.epochs as number,
  };
}

export const Route = createFileRoute("/api/eeg/federated")({
  server: {
    handlers: {
      POST: async ({ request, params }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

        // Route sub-path: /round, /validate
        const path = new URL(request.url).pathname.split("/").filter(Boolean);
        const subPath = path[path.length - 1];

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

        if (subPath === "validate") {
          // Validate a client update before submission
          const body = (await request.json()) as Record<string, unknown>;
          const update = parseClientUpdate(body);

          const validation = validateClientUpdate(update);
          return res({
            valid: validation.valid,
            reason: validation.reason ?? null,
          });
        }

        if (subPath === "round") {
          // Rate limit
          const rl = await checkRateLimit(supabase, userId, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SECONDS);
          if (!rl.allowed) {
            return res({ error: "Rate limit exceeded", retryAfterMs: rl.retryAfterMs }, 429);
          }

          const body = (await request.json()) as Record<string, unknown>;
          const task = body.task as string;
          const updates = (body.updates as Record<string, unknown>[]) ?? [];

          if (!isValidTask(task)) {
            return res({ error: `Invalid task: ${task}` }, 400);
          }

          const clientUpdates: ClientUpdate[] = updates.map(parseClientUpdate);

          // Timeout-protected round
          let timeoutId: ReturnType<typeof setTimeout>;
          const timeoutPromise = new Promise<never>((_, reject) => {
            timeoutId = setTimeout(() => {
              reject(new Error(`Federated round timeout (${ROUND_TIMEOUT_MS}ms)`));
            }, ROUND_TIMEOUT_MS);
          });

          try {
            const result = await Promise.race([
              runFederatedRound(clientUpdates, task as FederatedTask, body.options ?? {}),
              timeoutPromise,
            ]);

            metrics.tier1ServiceRequestsTotal.inc({ service: FEDERATED_SERVICE });

            return res({
              service: FEDERATED_SERVICE,
              version: "v0.1.0",
              user_id: userId,
              round: result.round,
              task: result.task,
              participant_count: result.participantCount,
              total_samples: result.totalSamples,
              mean_loss: result.meanLoss,
              mean_accuracy: result.meanAccuracy,
              convergence: result.convergence,
              duration_ms: result.durationMs,
              provenance: result.provenance,
            });
          } catch (err) {
            if (err instanceof FederatedLearningError) {
              metrics.tier1ServiceErrorsTotal.inc({ service: FEDERATED_SERVICE });
              log("error", "federated.round_failed", {
                error: err.message,
                code: err.code,
                userId,
              });
              return res({ error: err.message, code: err.code }, 500);
            }
            if ((err as Error).message.includes("timeout")) {
              metrics.tier1ServiceErrorsTotal.inc({ service: FEDERATED_SERVICE });
              return res({ error: "Processing timeout" }, 408);
            }
            metrics.tier1ServiceErrorsTotal.inc({ service: FEDERATED_SERVICE });
            log("error", "federated.round_unexpected_error", {
              error: (err as Error).message,
              userId,
            });
            return res({ error: "An error occurred during federated round." }, 500);
          } finally {
            clearTimeout(timeoutId!);
          }
        }

        return res({ error: "Unknown endpoint. Use /round or /validate." }, 404);
      },

      GET: async ({ request, params }) => {
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const requestOrigin = request.headers.get("origin");
        const res = makeJson(requestOrigin);

        // GET /api/eeg/federated/model/:task → return global weights
        const path = new URL(request.url).pathname.split("/").filter(Boolean);
        // path: ["api", "eeg", "federated", "model", "sleep-staging"]
        if (path[3] === "model" && path[4]) {
          const task = path[4];
          if (!isValidTask(task)) {
            return res({ error: `Invalid task: ${task}` }, 400);
          }

          const { weights, bias, round } = getGlobalModelWeights(task);
          return res({
            task,
            round,
            weights,
            bias,
          });
        }

        // GET /api/eeg/federated → list available tasks
        return res({
          service: FEDERATED_SERVICE,
          version: "v0.1.0",
          tasks: ["sleep-staging", "sleep-quality", "cognitive-workload", "anomaly-detection"],
        });
      },
    },
  },
});
