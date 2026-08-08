/**
 * T-012 — Recall@10 SLO cron route.
 *
 * Route: /api/public/cron/recall
 *
 * Called by pg_cron or an external scheduler (e.g. GitHub Actions nightly).
 * Samples labelled embeddings from the `embeddings` table, computes
 * recall@10 against exact brute-force, and emits an {@link SLOReport}.
 *
 * Authentication: the route expects a CRON_SECRET Bearer token (set via
 * the CRON_SECRET environment variable) to prevent unauthorised access.
 *
 * The SLO harness logic lives in `src/lib/vector-search/recall-slo.ts`
 * and is fully unit-tested. This route wires it to the production
 * database.
 */
import { createFileRoute } from "@tanstack/react-router";
import {
  runRecallSLO,
  formatSLOAlert,
  DEFAULT_SLO_CONFIG,
  type SLOSample,
} from "@/lib/vector-search/recall-slo";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import type { Json } from "@/integrations/supabase/types";
import { log } from "@/lib/logging";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";

/** Maximum number of samples to pull from the embeddings table per run. */
const MAX_SAMPLES = 500;

/** Metadata key that carries the class label (set by the upload/graph pipeline). */
const LABEL_METADATA_KEY = "label";

export const Route = createFileRoute("/api/public/cron/recall")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        // CORS pre-flight / origin check.
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        // Authenticate with CRON_SECRET.
        const cronSecret = process.env.CRON_SECRET;
        if (!cronSecret) {
          return json({ error: "CRON_SECRET not configured" }, 500);
        }

        const auth = request.headers.get("authorization") ?? "";
        if (auth !== `Bearer ${cronSecret}`) {
          return json({ error: "Unauthorized" }, 401);
        }

        const origin = request.headers.get("origin");
        try {
          // Sample labelled embeddings from the database.
          const samples = await sampleEmbeddings();

          if (samples.length === 0) {
            // No labelled embeddings in the DB — return a "no data" report
            // rather than a false SLO failure.
            log("info", "slo.recall.no_data", {
              reason: "No labelled embeddings found in the embeddings table",
            });
            return json(
              {
                timestamp: new Date().toISOString(),
                n: 0,
                passed: false,
                message:
                  "No labelled embeddings found. Upload EEG data with labels " +
                  "to populate the embeddings table, then re-run this SLO check.",
              },
              200,
              origin,
            );
          }

          // Measure real ANN recall via pgvector: for each sample, run the
          // match_embeddings RPC and compare top-k neighbour labels against
          // the ground-truth (same-label) set.  Falls back to 0 when no DB.
          let annRecall = 0;
          if (supabaseAdmin) {
            annRecall = await measureANNGatherRecalls(samples, DEFAULT_SLO_CONFIG.k);
            log("info", "slo.recall.ann_measured", { n: samples.length, recall: annRecall });
          }

          // Build the final SLO report (computes both ANN and brute-force recall).
          const report = runRecallSLO(samples, annRecall, DEFAULT_SLO_CONFIG);

          if (!report.passed) {
            log("warn", "slo.recall.alert", { alert: formatSLOAlert(report) });
          } else {
            log("info", "slo.recall.ok", {
              n: report.n,
              annRecall: report.pgvectorRecall,
              bruteForceRecall: report.bruteForceRecall,
            });
          }

          return json(report, 200, origin);
        } catch (err) {
          log("error", "slo.recall.failed", { error: (err as Error).message });
          return json({ error: "SLO run failed", detail: (err as Error).message }, 500, origin);
        }
      },
    },
  },
});

/**
 * Sample labelled embeddings from the `embeddings` table.
 *
 * Pulls up to MAX_SAMPLES rows that have a `label` in their metadata JSONB.
 * Returns an array of {@link SLOSample} with the embedding vector, label,
 * and model id.
 */
async function sampleEmbeddings(): Promise<SLOSample[]> {
  const { data, error } = await supabaseAdmin
    .from("embeddings")
    .select("id, embedding, model_id, metadata")
    .not("metadata->label", "is", null)
    .limit(MAX_SAMPLES);

  if (error) {
    log("warn", "slo.recall.db_error", { error: error.message });
    return [];
  }

  if (!data || data.length === 0) {
    return [];
  }

  return (data as Array<{ id: string; embedding: number[]; model_id: string; metadata: Json }>).map(
    (row) => ({
      id: row.id,
      embedding: row.embedding,
      label: extractLabel(row.metadata),
      modelId: row.model_id,
    }),
  );
}

/** Extract the numeric label from the metadata JSONB. */
function extractLabel(metadata: Json): number {
  if (typeof metadata !== "object" || metadata === null) return -1;
  const obj = metadata as Record<string, unknown>;
  const label = obj[LABEL_METADATA_KEY];
  if (typeof label === "number") return label;
  if (typeof label === "string") {
    const parsed = parseInt(label, 10);
    if (Number.isFinite(parsed)) return parsed;
  }
  return -1;
}

/**
 * Measure pgvector ANN recall@K against the ground-truth same-label set.
 *
 * For each sample, we run the `match_embeddings` RPC (which uses the ivfflat
 * index) and collect the top-K neighbour IDs. We then compare which of those
 * neighbours share the sample's label — both against a per-sample ground-truth
 * set (the K nearest same-label neighbours in the full sample set, computed
 * with exact cosine in JS) and across all samples aggregate.
 *
 * Returns the aggregate recall@K (fraction of ground-truth same-label
 * neighbours that the ANN top-K retrieved).
 */
async function measureANNGatherRecalls(samples: SLOSample[], k: number): Promise<number> {
  if (samples.length === 0) return 0;

  // Pre-compute the ground-truth: for each sample, the set of the K nearest
  // *same-label* neighbours (by exact cosine distance, excluding self).
  const labels = samples.map((s) => s.label);
  const embeddings = samples.map((s) => s.embedding);
  const kClamped = Math.min(k, samples.length - 1);

  let hits = 0;
  let total = 0;

  for (let i = 0; i < samples.length; i++) {
    const query = samples[i];
    const queryLabel = labels[i];

    // Ground-truth: the k nearest same-label neighbours (excluding self).
    const gtSameLabel: { idx: number; dist: number }[] = [];
    for (let j = 0; j < samples.length; j++) {
      if (i === j || labels[j] !== queryLabel) continue;
      const d = cosineDistance(query.embedding, embeddings[j]);
      gtSameLabel.push({ idx: j, dist: d });
    }
    gtSameLabel.sort((a, b) => a.dist - b.dist);
    const gtTopK = gtSameLabel.slice(0, kClamped).map((g) => g.idx);

    // ANN result: run match_embeddings RPC via Supabase.
    const { data, error } = await supabaseAdmin.rpc("match_embeddings", {
      query_embedding: query.embedding,
      match_count: kClamped,
      filter_model_id: query.modelId,
      filter_user_id: null,
    });

    if (error || !data || !Array.isArray(data)) {
      log("warn", "slo.recall.ann_rpc_error", {
        sampleId: query.id,
        error: (error as Error)?.message ?? "no data",
      });
      continue;
    }

    // The ANN result returns IDs in ranked order. We don't have labels on the
    // RPC response, so we check whether each returned ID corresponds to a
    // same-label sample by cross-referencing against our samples array.
    const returnedIds = new Set<string>();
    for (const row of data as Array<{ id: string }>) {
      returnedIds.add(row.id);
    }

    // Count how many ground-truth same-label neighbours appear in the ANN top-K.
    for (const gtIdx of gtTopK) {
      total++;
      if (returnedIds.has(samples[gtIdx].id)) {
        hits++;
      }
    }
  }

  return total > 0 ? hits / total : 0;
}

/** Simple cosine distance (1 - cosine similarity) for ground-truth computation. */
function cosineDistance(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom === 0) return 1;
  return 1 - dot / denom;
}

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
