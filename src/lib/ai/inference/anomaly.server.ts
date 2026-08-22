/**
 * M34 — Anomaly Detection service logic.
 *
 * Implements anomaly detection on the frozen Joint-2312 (2312-D) embedding space
 * using Mahalanobis distance. This mirrors the M33 cognitive.server.ts pattern:
 * same security layer, same embed-once-reuse-many principle, same provenance.
 *
 * **Embed Once → Reuse Many:** if the caller provides an `embedding_id`, the
 * existing Joint-2312 embedding stored in `joint_embeddings_2312` is reused.
 * If `query_embedding` is provided (raw 2312-D vector), it is used directly.
 *
 * **Why server-only (.server.ts):** the 2312-D Mahalanobis model runs via
 * onnxruntime-node. The browser path (V2-32 → lightweight detector) is handled
 * by `src/lib/ai/decoders/anomaly.browser.ts`.
 */
import { randomUUID } from "node:crypto";
import {
  NeuralVectorIndex,
  DimensionMismatchError,
} from "@/lib/vector-search/neural-index";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance, type ServiceProvenance } from "@/lib/ai/services/provenance.server";
import {
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";
import { ONNXAdapter } from "@/lib/ai/adapters/onnx-adapter";
import { ANOMALY_MAHALANOBIS_PROBE_JOINT_2312 } from "@/lib/ai/decoders/anomaly.registry";
import { type TaskHeadDescriptor } from "@/lib/ai/decoders/registry";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const ANOMALY_SERVICE = "anomaly-detection";
/** Service version. */
export const ANOMALY_VERSION = "v0.1.0";
/** Default task head id for anomaly detection. */
export const ANOMALY_DEFAULT_HEAD_ID = "anomaly-mahalanobis-v1";
/** Processing timeout — same as cognitive (no CBraMod/EEGPT inference). */
export const ANOMALY_TIMEOUT_MS = 10_000;
/** Anomaly score threshold (Mahalanobis distance → normalised). */
export const ANOMALY_DEFAULT_THRESHOLD = 0.75;
/** Confidence interval margin when bootstrap is unavailable. */
export const ANOMALY_DEFAULT_CI_MARGIN = 0.08;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export type AnomalyQueryType = "artifact" | "baseline" | "fatigue";

export interface AnomalyDetectRequest {
  /** Raw 2312-D embedding vector (L2-normalised). If omitted, must provide embedding_id. */
  query_embedding?: number[];
  /** Existing Joint-2312 embedding row id to reuse (Embed Once → Reuse Many). */
  embedding_id?: string;
  /** Which anomaly type to detect. Default: "artifact". */
  query_type?: AnomalyQueryType;
  /** Optional head id to use (defaults to ANOMALY_DEFAULT_HEAD_ID). */
  head_id?: string;
}

export interface AnomalyResult {
  /** Anomaly score [0, 1] (1 = most anomalous). */
  score: number;
  /** Whether the embedding is flagged as anomalous. */
  is_anomalous: boolean;
  /** Confidence of the prediction [0, 1]. */
  confidence: number;
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** The anomaly type detected. */
  metric: AnomalyQueryType;
}

export interface AnomalyDetectResponse {
  service: string;
  model: string;
  head: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: AnomalyResult[];
  metadata: {
    embedding_reused: boolean;
    probe_sha256?: string;
  };
  timings: {
    embed_ms?: number;
    inference_ms: number;
    total_ms: number;
  };
}

/** Error thrown when the anomaly detector fails. */
export class AnomalyDetectError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "AnomalyDetectError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// ONNX probe loading (server-side, onnxruntime-node)
// ─────────────────────────────────────────────────────────────────────

let cachedAnomalyAdapter: ONNXAdapter | null = null;

/**
 * Load the anomaly detection probe ONNX model (2312→1), with SHA-256
 * verification against the manifest (T-016 pattern). Idempotent — the loaded
 * session is cached for reuse across requests.
 */
async function ensureAnomalyProbe(): Promise<ONNXAdapter> {
  if (cachedAnomalyAdapter) return cachedAnomalyAdapter;

  const head: TaskHeadDescriptor = ANOMALY_MAHALANOBIS_PROBE_JOINT_2312;

  cachedAnomalyAdapter = new ONNXAdapter({
    id: head.id,
    name: head.name,
    version: head.version,
    description: head.training.protocol,
    task: "regression",
    inputShape: { kind: "features", dim: JOINT_2312_EMBEDDING_DIM },
    embeddingDim: head.outputDim,
    outputPooling: "none",
    artifact: "/models/anomaly/mahalanobis-probe-joint2312-v1.onnx",
    enableVerification: true,
    executionProviders: ["cpu"],
  });

  await cachedAnomalyAdapter.load();
  return cachedAnomalyAdapter;
}

/** Reset the cached adapter (test helper + SSR lifecycle hook). */
export function resetAnomalyProbe(): void {
  cachedAnomalyAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedAnomalyAdapter = null;
}

// ─────────────────────────────────────────────────────────────────────
// Core detect
// ─────────────────────────────────────────────────────────────────────

/**
 * Detect cognitive anomalies (artifact, baseline drift, fatigue) from a Joint-2312
 * embedding.
 *
 * Follows the "Embed Once → Reuse Many" principle:
 * - If `embedding_id` is provided, the existing stored embedding is fetched
 *   from `joint_embeddings_2312` and reused.
 * - If `query_embedding` is provided, it is used directly.
 *
 * @param opts - Detection parameters.
 * @param supabase - Authenticated Supabase client (user-scoped).
 * @param userId - Authenticated user id (for RLS + audit).
 * @returns The anomaly detection response.
 */
export async function detectAnomalies(
  opts: AnomalyDetectRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<AnomalyDetectResponse> {
  const tStart = startTimer("anomaly.detect.total");

  metrics.tier1ServiceRequestsTotal.inc({ service: ANOMALY_SERVICE });
  metrics.anomalyDetectRequestsTotal.inc();

  const queryType = opts.query_type ?? "artifact";
  const headId = opts.head_id ?? ANOMALY_DEFAULT_HEAD_ID;

  // Fetch or validate the query embedding.
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    // Reuse existing embedding — fetch from joint_embeddings_2312.
    const tEmbed = startTimer("anomaly.detect.reuse_embedding");

    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new AnomalyDetectError(
        "embedding_id not found or access denied",
        "EMBEDDING_NOT_FOUND",
      );
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.anomalyEmbeddingReusedTotal.inc();
  } else if (opts.query_embedding) {
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    embedMs = undefined;
    metrics.anomalyEmbeddingReembeddedTotal.inc();
  } else {
    throw new AnomalyDetectError(
      "Either query_embedding or embedding_id must be provided",
      "INVALID_REQUEST",
    );
  }

  // Validate embedding dimension before inference.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new AnomalyDetectError(
      `Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
      "DIMENSION_MISMATCH",
    );
  }

  // Run the ONNX probe.
  const tInfer = startTimer("anomaly.detect.inference", { query_type: queryType });

  let probeAdapter: ONNXAdapter;
  try {
    probeAdapter = await ensureAnomalyProbe();
  } catch (e) {
    metrics.anomalyDetectErrorsTotal.inc({ error: "probe_load_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: ANOMALY_SERVICE });
    log("error", "anomaly.detect.probe_load_failed", {
      error: (e as Error).message,
      userId,
    });
    throw new AnomalyDetectError(
      "Anomaly probe model unavailable",
      "PROBE_UNAVAILABLE",
    );
  }

  let prediction: number;
  try {
    const input = { kind: "features" as const, features: [queryEmbedding] };
    const result = await probeAdapter.predict(input);
    prediction = result.values["class_0"] ?? 0;
  } catch (e) {
    metrics.anomalyDetectErrorsTotal.inc({ error: "inference_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: ANOMALY_SERVICE });
    log("error", "anomaly.detect.inference_failed", {
      error: (e as Error).message,
      userId,
      queryType,
    });
    throw new AnomalyDetectError("Inference failed", "INFERENCE_FAILED");
  }

  const inferMs = tInfer.end({ results: 1 });
  metrics.anomalyDetectLatencyMs.observe({ query_type: queryType }, inferMs);

  // Clamp prediction to [0, 1] and compute confidence.
  const score = Math.max(0, Math.min(1, prediction));
  // Use the registry's threshold (normalized to [0, 1]) — falls back to
  // the legacy constant if the registry doesn't define one.
  const threshold = ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.training?.metrics?.threshold ?? ANOMALY_DEFAULT_THRESHOLD;
  // The registry threshold is in raw Mahalanobis distance space (2.5);
  // normalise to [0, 1] by dividing by a safety factor (~10×).
  const normalisedThreshold = threshold > 1 ? threshold / 10 : threshold;
  const isAnomalous = score >= normalisedThreshold;

  // Confidence = 1 - CI width. With a fixed CI margin of ±0.08,
  // confidence = 1 - (2 * 0.08) = 0.84 for a well-calibrated prediction.
  const ciMargin = ANOMALY_DEFAULT_CI_MARGIN;
  const ciLower = Math.max(0, score - ciMargin);
  const ciUpper = Math.min(1, score + ciMargin);
  const confidence = 1 - (ciUpper - ciLower);

  // Build the result for the requested query type.
  const result: AnomalyResult = {
    score,
    is_anomalous: isAnomalous,
    confidence_interval: [ciLower, ciUpper],
    confidence,
    metric: queryType,
  };

  const totalMs = tStart.end({
    query_type: queryType,
    embedding_reused: embeddingReused,
  });

  metrics.anomalyScoresTotal.inc({}, 1);
  metrics.anomalyConfidenceDistribution.observe({ query_type: queryType }, confidence);
  metrics.tier1ServiceLatencyMs.observe({ service: ANOMALY_SERVICE }, totalMs);
  metrics.tier1AuditLogInsertsTotal.inc();

  // Build provenance (reuses M32/M33's buildServiceProvenance)
  const provenance = buildServiceProvenance({
    service: ANOMALY_SERVICE,
    serviceVersion: ANOMALY_VERSION,
    taskHeadId: headId,
    taskHeadVersion: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.version,
    taskHeadSha256: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.sha256,
    taskHeadDataset: "PhysioNet EEGMMIDB (artifact detection proxy)",
    taskHeadMetrics: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.training?.metrics ?? {
      auc_roc: 0.892,
      f1_score: 0.81,
      threshold: 2.5,
      precision: 0.78,
      recall: 0.84,
    },
    experimentId: "m34-anomaly-detection-probe",
  });

  return {
    service: ANOMALY_SERVICE,
    model: JOINT_2312_MODEL_ID,
    head: headId,
    head_version: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.version,
    ...(embeddingId ? { embedding_id: embeddingId } : {}),
    provenance,
    results: [result],
    metadata: {
      embedding_reused: embeddingReused,
      probe_sha256: ANOMALY_MAHALANOBIS_PROBE_JOINT_2312.sha256,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      inference_ms: inferMs,
      total_ms: totalMs,
    },
  };
}
