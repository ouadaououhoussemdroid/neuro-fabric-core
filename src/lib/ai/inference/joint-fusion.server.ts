/**
 * M41 — Multi-Task Fusion on Joint-2312.
 *
 * Provides a single `decodeJoint2312()` entry point that resolves a Joint-2312
 * embedding once and runs all Tier-1+Tier-2 task heads in parallel on the same
 * embedding vector. This realizes the "Embed Once → Reuse Many" principle at
 * the batch level: one embedding → 4 task probes → unified response with a
 * single shared provenance block.
 *
 * The 4 ONNX probes that operate directly on the 2312-D embedding:
 *   1. Cognitive State (M33):  2312→1 regression (workload)
 *   2. Anomaly Detection (M34): 2312→1 Mahalanobis distance
 *   3. Sleep Staging (M39):    2312→5 classification (softmax)
 *   4. Sleep Quality (M40):    2312→1 regression (clamped [0,1])
 *
 * Subject Identity (M32) is excluded from the fusion batch — it requires an
 * RPC vector search against the full embedding table, not a single-vector
 * ONNX pass. Callers can invoke `searchSubjectIdentity()` separately.
 *
 * Security: same auth/rate-limit/CORS pattern as other Tier-1 routes.
 */
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance, type ServiceProvenance } from "@/lib/ai/services/provenance.server";
import {
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";
import { decodeCognitiveState } from "@/lib/ai/inference/cognitive.server";
import { detectAnomalies } from "@/lib/ai/inference/anomaly.server";
import { decodeSleepState, decodeSleepQuality } from "@/lib/ai/inference/sleep.server";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const JOINT_FUSION_SERVICE = "joint-fusion";
/** Service version. */
export const JOINT_FUSION_VERSION = "v0.1.0";

/** Processing timeout — same as other Tier-1 services. */
export const JOINT_FUSION_TIMEOUT_MS = 10_000;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export interface JointFusionRequest {
  /** Raw 2312-D embedding vector. If omitted, must provide embedding_id. */
  query_embedding?: number[];
  /** Existing Joint-2312 embedding row id to reuse (Embed Once → Reuse Many). */
  embedding_id?: string;
  /** Optional task heads to run. Defaults to all 4. */
  heads?: ("cognitive" | "anomaly" | "sleep-staging" | "sleep-quality")[];
}

export interface JointFusionResponse {
  service: string;
  model: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: {
    cognitive?: ReturnType<Awaited<ReturnType<typeof decodeCognitiveState>>>["results"];
    anomaly?: ReturnType<Awaited<ReturnType<typeof detectAnomalies>>>["results"];
    sleep_staging?: ReturnType<Awaited<ReturnType<typeof decodeSleepState>>>["results"];
    sleep_quality?: ReturnType<Awaited<ReturnType<typeof decodeSleepQuality>>>["results"];
  };
  metadata: {
    embedding_reused: boolean;
    heads_run: string[];
    probes: Array<{ id: string; sha256: string }>;
  };
  timings: {
    embed_ms?: number;
    inference_ms: number;
    total_ms: number;
  };
}

// ─────────────────────────────────────────────────────────────────────
// Core fusion decode
// ─────────────────────────────────────────────────────────────────────

/**
 * Decode all Tier-1+Tier-2 task heads from a single Joint-2312 embedding.
 *
 * **Embed Once → Reuse Many (batch):**
 * - If `embedding_id` is provided, the existing embedding is fetched once
 *   from `joint_embeddings_2312` and reused across all 4 probes.
 * - If `query_embedding` is provided, it is used directly.
 *
 * All 4 ONNX probes run in parallel on `Promise.all()` since they are
 * independent linear/regression ops on the same 2312-D vector.
 *
 * @param opts - Fusion decode parameters.
 * @param supabase - Authenticated Supabase client (user-scoped).
 * @param userId - Authenticated user id (for RLS + audit).
 * @returns The unified multi-task decode response.
 */
export async function decodeJoint2312(
  opts: JointFusionRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<JointFusionResponse> {
  const tStart = startTimer("joint.decode.total");

  metrics.tier1ServiceRequestsTotal.inc({ service: JOINT_FUSION_SERVICE });

  // Determine which heads to run (default: all 4)
  const requestedHeads = opts.heads ?? ["cognitive", "anomaly", "sleep-staging", "sleep-quality"];

  // ─── 1. Resolve the embedding once ──────────────────────────────────
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    const tEmbed = startTimer("joint.decode.reuse_embedding");
    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new Error("EMBEDDING_NOT_FOUND: embedding_id not found or access denied");
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.sleepEmbeddingReusedTotal.inc(); // reuse counter is shared across sleep+quality
  } else if (opts.query_embedding) {
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    metrics.sleepEmbeddingReembeddedTotal.inc();
  } else {
    throw new Error("INVALID_REQUEST: Either query_embedding or embedding_id must be provided");
  }

  // Validate embedding dimension.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new Error(
      `DIMENSION_MISMATCH: Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
    );
  }

  // ─── 2. Run all requested task probes in parallel ───────────────────
  const tInfer = startTimer("joint.decode.inference", { heads: requestedHeads.length });

  const results: JointFusionResponse["results"] = {};
  const probeInfos: Array<{ id: string; sha256: string }> = [];
  const headsRun: string[] = [];

  // Shared embedding vector passed to all probes (embed-once-reuse-many)
  const sharedEmbedding = { query_embedding: queryEmbedding } as const;

  const promises: Promise<void>[] = [];

  if (requestedHeads.includes("cognitive")) {
    promises.push(
      decodeCognitiveState(
        { ...sharedEmbedding, query_type: "workload" } as Parameters<typeof decodeCognitiveState>[0],
        supabase,
        userId,
      )
        .then((res) => {
          results.cognitive = res.results;
          headsRun.push("cognitive");
          probeInfos.push({ id: res.head, sha256: res.metadata.probe_sha256 ?? "" });
        })
        .catch((e) => {
          log("warn", "joint.fusion.cognitive_failed", { error: (e as Error).message, userId });
        }),
    );
  }

  if (requestedHeads.includes("anomaly")) {
    promises.push(
      detectAnomalies(
        { ...sharedEmbedding, query_type: "artifact" } as Parameters<typeof detectAnomalies>[0],
        supabase,
        userId,
      )
        .then((res) => {
          results.anomaly = res.results;
          headsRun.push("anomaly");
          probeInfos.push({ id: res.head, sha256: res.metadata.probe_sha256 ?? "" });
        })
        .catch((e) => {
          log("warn", "joint.fusion.anomaly_failed", { error: (e as Error).message, userId });
        }),
    );
  }

  if (requestedHeads.includes("sleep-staging")) {
    promises.push(
      decodeSleepState(
        { ...sharedEmbedding, query_type: "sleep-stages" } as Parameters<typeof decodeSleepState>[0],
        supabase,
        userId,
      )
        .then((res) => {
          results.sleep_staging = res.results;
          headsRun.push("sleep-staging");
          probeInfos.push({ id: res.head, sha256: res.metadata.probe_sha256 ?? "" });
        })
        .catch((e) => {
          log("warn", "joint.fusion.staging_failed", { error: (e as Error).message, userId });
        }),
    );
  }

  if (requestedHeads.includes("sleep-quality")) {
    promises.push(
      decodeSleepQuality(
        { ...sharedEmbedding, query_type: "sleep-quality" } as Parameters<typeof decodeSleepQuality>[0],
        supabase,
        userId,
      )
        .then((res) => {
          results.sleep_quality = res.results;
          headsRun.push("sleep-quality");
          probeInfos.push({ id: res.head, sha256: res.metadata.probe_sha256 ?? "" });
        })
        .catch((e) => {
          log("warn", "joint.fusion.quality_failed", { error: (e as Error).message, userId });
        }),
    );
  }

  await Promise.all(promises);

  const inferMs = tInfer.end({ heads: headsRun.length });
  metrics.sleepDecodeLatencyMs.observe({ query_type: "joint-fusion" }, inferMs);

  const totalMs = tStart.end({
    heads_run: headsRun.length,
    embedding_reused: embeddingReused,
  });

  // ─── 3. Build unified provenance ──────────────────────────────────────
  const provenance = buildServiceProvenance({
    service: JOINT_FUSION_SERVICE,
    serviceVersion: JOINT_FUSION_VERSION,
    taskHeadId: "joint-fusion-all-v1",
    taskHeadVersion: "0.1.0",
    taskHeadSha256: "multi-probe-fusion",
    taskHeadDataset: "Sleep-EDF + PhysioNet EEGMMIDB (shared Joint-2312)",
    taskHeadMetrics: {
      targets: "[cognitive, anomaly, sleep-staging, sleep-quality]",
      shared_backbone: "joint-2312",
      input_dim: JOINT_2312_EMBEDDING_DIM,
    },
    experimentId: "m41-multi-task-fusion",
  });

  return {
    service: JOINT_FUSION_SERVICE,
    model: JOINT_2312_MODEL_ID,
    head_version: JOINT_FUSION_VERSION,
    ...(embeddingId ? { embedding_id: embeddingId } : {}),
    provenance,
    results,
    metadata: {
      embedding_reused: embeddingReused,
      heads_run: headsRun,
      probes: probeInfos,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      inference_ms: inferMs,
      total_ms: totalMs,
    },
  };
}
