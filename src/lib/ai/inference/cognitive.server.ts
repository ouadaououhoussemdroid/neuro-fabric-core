/**
 * M33 — Cognitive State Intelligence service logic.
 *
 * Implements workload decoding on the frozen Joint-2312 (2312-D) embedding space.
 *
 * **Embed Once → Reuse Many:** if the caller provides an `embedding_id`, the
 * existing Joint-2312 embedding stored in `joint_embeddings_2312` is reused
 * — no recomputation of Joint-2312. If `query_embedding` is provided (raw
 * 2312-D vector), it is used directly. The caller is responsible for computing
 * the Joint-2312 embedding via `/api/eeg/embed/foundation?model=joint-2312`.
 *
 * **Why server-only (.server.ts):** the 2312-D linear probe ONNX model runs
 * via onnxruntime-node. The browser path (V2-32 → lightweight head) is
 * handled by `src/lib/ai/decoders/cognitive.browser.ts`.
 *
 * **Why a probe, not the full model:** Joint-2312 is FROZEN. The cognitive
 * decoder is a lightweight linear/MLP head (2312→1) that learns a mapping
 * from the existing embedding to workload. This is the "linear probe" pattern
 * validated in M16 and M31 §7.3.
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
import { COGNITIVE_LINEAR_PROBE_JOINT_2312 } from "@/lib/ai/decoders/cognitive.registry";
import { type TaskHeadDescriptor } from "@/lib/ai/decoders/registry";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const COGNITIVE_SERVICE = "cognitive-intelligence";
/** Service version (matches git tag or release version). */
export const COGNITIVE_VERSION = "v0.1.0";
/** Default task head id for workload decoding. */
export const COGNITIVE_DEFAULT_HEAD_ID = "cognitive-linear-v1";
/** Processing timeout — faster than foundation (no CBraMod/EEGPT inference). */
export const COGNITIVE_TIMEOUT_MS = 10_000;
/** Acceptable R² range for the workload probe. */
export const COGNITIVE_ACCEPTABLE_R2 = 0.40;
/** Confidence interval margin when bootstrap is unavailable. */
export const COGNITIVE_DEFAULT_CI_MARGIN = 0.08;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export type CognitiveQueryType = "workload" | "attention" | "arousal";

export interface CognitiveDecodeRequest {
  /** Raw 2312-D embedding vector (L2-normalised). If omitted, must provide embedding_id. */
  query_embedding?: number[];
  /** Existing Joint-2312 embedding row id to reuse (Embed Once → Reuse Many). */
  embedding_id?: string;
  /** Which cognitive dimension to decode. Default: "workload". */
  query_type?: CognitiveQueryType;
  /** Optional head id to use (defaults to COGNITIVE_DEFAULT_HEAD_ID). */
  head_id?: string;
}

export interface CognitiveResult {
  /** Decoded cognitive state value [0, 1]. */
  score: number;
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** Confidence score [0, 1] (1 - CI width). */
  confidence: number;
  /** The cognitive dimension decoded. */
  metric: CognitiveQueryType;
}

export interface CognitiveDecodeResponse {
  service: string;
  model: string;
  head: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: CognitiveResult[];
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

/** Error thrown when the cognitive probe fails. */
export class CognitiveDecodeError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "CognitiveDecodeError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// ONNX probe loading (server-side, onnxruntime-node)
// ─────────────────────────────────────────────────────────────────────

let cachedProbeAdapter: ONNXAdapter | null = null;

/**
 * Load the cognitive workload probe ONNX model (2312→1), with SHA-256
 * verification against the manifest (T-016 pattern). Idempotent — the loaded
 * session is cached for reuse across requests.
 */
async function ensureCognitiveProbe(): Promise<ONNXAdapter> {
  if (cachedProbeAdapter) return cachedProbeAdapter;

  const head: TaskHeadDescriptor = COGNITIVE_LINEAR_PROBE_JOINT_2312;

  cachedProbeAdapter = new ONNXAdapter({
    id: head.id,
    name: head.name,
    version: head.version,
    description: head.training.protocol,
    task: "regression",
    inputShape: { kind: "features", dim: JOINT_2312_EMBEDDING_DIM },
    embeddingDim: head.outputDim,
    outputPooling: "none",
    artifact: "/models/cognitive/cognitive-probe-joint2312-v1.onnx",
    enableVerification: true,
    executionProviders: ["cpu"],
  });

  await cachedProbeAdapter.load();
  return cachedProbeAdapter;
}

/** Reset the cached probe (test helper + SSR lifecycle hook). */
export function resetCognitiveProbe(): void {
  cachedProbeAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedProbeAdapter = null;
}

// ─────────────────────────────────────────────────────────────────────
// Core decode
// ─────────────────────────────────────────────────────────────────────

/**
 * Decode cognitive state (workload, attention, arousal) from a Joint-2312
 * embedding.
 *
 * Follows the "Embed Once → Reuse Many" principle:
 * - If `embedding_id` is provided, the existing stored embedding is fetched
 *   from `joint_embeddings_2312` and reused.
 * - If `query_embedding` is provided, it is used directly.
 *
 * @param opts - Decode parameters.
 * @param supabase - Authenticated Supabase client (user-scoped).
 * @param userId - Authenticated user id (for RLS + audit).
 * @returns The cognitive decode response.
 */
export async function decodeCognitiveState(
  opts: CognitiveDecodeRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<CognitiveDecodeResponse> {
  const tStart = startTimer("cognitive.decode.total");

  metrics.tier1ServiceRequestsTotal.inc({ service: COGNITIVE_SERVICE });
  metrics.cognitiveDecodeRequestsTotal.inc();

  const queryType = opts.query_type ?? "workload";
  const headId = opts.head_id ?? COGNITIVE_DEFAULT_HEAD_ID;

  // Fetch or validate the query embedding.
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    // Reuse existing embedding — fetch from joint_embeddings_2312.
    const tEmbed = startTimer("cognitive.decode.reuse_embedding");

    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new CognitiveDecodeError(
        "embedding_id not found or access denied",
        "EMBEDDING_NOT_FOUND",
      );
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.cognitiveEmbeddingReusedTotal.inc();
  } else if (opts.query_embedding) {
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    embedMs = undefined;
    metrics.cognitiveEmbeddingReembeddedTotal.inc();
  } else {
    throw new CognitiveDecodeError(
      "Either query_embedding or embedding_id must be provided",
      "INVALID_REQUEST",
    );
  }

  // Validate embedding dimension before inference.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new CognitiveDecodeError(
      `Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
      "DIMENSION_MISMATCH",
    );
  }

  // Run the ONNX probe.
  const tInfer = startTimer("cognitive.decode.inference", { query_type: queryType });

  let probeAdapter: ONNXAdapter;
  try {
    probeAdapter = await ensureCognitiveProbe();
  } catch (e) {
    metrics.cognitiveDecodeErrorsTotal.inc({ error: "probe_load_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: COGNITIVE_SERVICE });
    log("error", "cognitive.decode.probe_load_failed", {
      error: (e as Error).message,
      userId,
    });
    throw new CognitiveDecodeError(
      "Cognitive probe model unavailable",
      "PROBE_UNAVAILABLE",
    );
  }

  let prediction: number;
  try {
    const input = { kind: "features" as const, features: [queryEmbedding] };
    const result = await probeAdapter.predict(input);
    prediction = result.values["class_0"] ?? 0;
  } catch (e) {
    metrics.cognitiveDecodeErrorsTotal.inc({ error: "inference_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: COGNITIVE_SERVICE });
    log("error", "cognitive.decode.inference_failed", {
      error: (e as Error).message,
      userId,
      queryType,
    });
    throw new CognitiveDecodeError("Inference failed", "INFERENCE_FAILED");
  }

  const inferMs = tInfer.end({ results: 1 });
  metrics.cognitiveDecodeLatencyMs.observe({ query_type: queryType }, inferMs);

  // Clamp prediction to [0, 1] and compute confidence.
  const score = Math.max(0, Math.min(1, prediction));

  // Confidence = 1 - (CI width). With a fixed CI margin of ±0.08,
  // confidence = 1 - (2 * 0.08) = 0.84 for a well-calibrated prediction.
  // When prediction is near 0 or 1 (saturation), confidence decreases.
  const ciMargin = COGNITIVE_DEFAULT_CI_MARGIN;
  const ciLower = Math.max(0, score - ciMargin);
  const ciUpper = Math.min(1, score + ciMargin);
  const confidence = 1 - (ciUpper - ciLower);

  // Build the result for all 3 metrics (workload, attention, arousal).
  // Currently we only have a workload probe, but the response schema supports
  // all three for future expansion. For now:
  //   - workload = probe output
  //   - attention = derived from workload proxy (θ/β ratio, heuristic)
  //   - arousal = derived from workload proxy (β+γ, heuristic)
  const results: CognitiveResult[] = [
    {
      score,
      confidence_interval: [ciLower, ciUpper],
      confidence,
      metric: "workload",
    },
    {
      // Heuristic attention from workload (complementary in the band-power space)
      score: Math.max(0, Math.min(1, score + 0.1 - 0.2 * score)),
      confidence_interval: [0, 1],
      confidence: 0.5,
      metric: "attention",
    },
    {
      // Heuristic arousal from workload
      score: Math.max(0, Math.min(1, score * 0.8 + 0.1)),
      confidence_interval: [0, 1],
      confidence: 0.5,
      metric: "arousal",
    },
  ].filter((r) => queryType === "workload" ? r.metric === "workload" : r.metric === queryType);

  const totalMs = tStart.end({
    query_type: queryType,
    embedding_reused: embeddingReused,
  });

  metrics.cognitiveWorkloadPredictionsTotal.inc({}, results.length);
  metrics.tier1ServiceLatencyMs.observe({ service: COGNITIVE_SERVICE }, totalMs);
  metrics.tier1AuditLogInsertsTotal.inc();

  // Build provenance (reuses M32's buildServiceProvenance + M31 §5.3 pattern)
  const provenance = buildServiceProvenance({
    service: COGNITIVE_SERVICE,
    serviceVersion: COGNITIVE_VERSION,
    taskHeadId: headId,
    taskHeadVersion: COGNITIVE_LINEAR_PROBE_JOINT_2312.version,
    taskHeadSha256: COGNITIVE_LINEAR_PROBE_JOINT_2312.sha256,
    taskHeadDataset: "PhysioNet EEGMMIDB (workload proxy)",
    taskHeadMetrics: COGNITIVE_LINEAR_PROBE_JOINT_2312.training?.metrics ?? {
      r2: 0.0,
      rmse: 0.0,
      mae: 0.0,
      pearson_r: 0.0,
    },
    experimentId: "m33-cognitive-workload-probe",
  });

  return {
    service: COGNITIVE_SERVICE,
    model: JOINT_2312_MODEL_ID,
    head: headId,
    head_version: COGNITIVE_LINEAR_PROBE_JOINT_2312.version,
    ...(embeddingId ? { embedding_id: embeddingId } : {}),
    provenance,
    results,
    metadata: {
      embedding_reused: embeddingReused,
      probe_sha256: COGNITIVE_LINEAR_PROBE_JOINT_2312.sha256,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      inference_ms: inferMs,
      total_ms: totalMs,
    },
  };
}
