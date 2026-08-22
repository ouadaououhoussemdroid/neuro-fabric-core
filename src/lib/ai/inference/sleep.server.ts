/**
 * M39 — Sleep Staging service logic.
 *
 * Implements 5-class sleep staging (W, N1, N2, N3, REM) on the frozen Joint-2312
 * (2312-D) embedding space. This mirrors the M33 cognitive.server.ts and M34
 * anomaly.server.ts patterns: same security layer, same embed-once-reuse-many
 * principle, same provenance.
 *
 * **Embed Once → Reuse Many:** if the caller provides an `embedding_id`, the
 * existing Joint-2312 embedding stored in `joint_embeddings_2312` is reused.
 * If `query_embedding` is provided (raw 2312-D vector), it is used directly.
 * The caller is responsible for computing the Joint-2312 embedding via
 * `/api/eeg/embed/foundation?model=joint-2312`.
 *
 * **Why server-only (.server.ts):** the 2312-D linear probe ONNX model runs
 * via onnxruntime-node. The browser path (V2-32 → lightweight head) is handled
 * by `src/lib/ai/decoders/sleep.browser.ts`.
 */
import { randomUUID } from "node:crypto";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance, type ServiceProvenance } from "@/lib/ai/services/provenance.server";
import {
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";
import { ONNXAdapter } from "@/lib/ai/adapters/onnx-adapter";
import {
  SLEEP_STAGING_PROBE_JOINT_2312,
  SLEEP_QUALITY_PROBE_JOINT_2312,
} from "@/lib/ai/decoders/sleep.registry";
import { type TaskHeadDescriptor } from "@/lib/ai/decoders/registry";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const SLEEP_SERVICE = "sleep-staging";
/** Service version. */
export const SLEEP_VERSION = "v0.1.0";
/** Default task head id for sleep staging. */
export const SLEEP_DEFAULT_HEAD_ID = "sleep-staging-v1";
/** Default task head id for sleep quality. */
export const SLEEP_QUALITY_DEFAULT_HEAD_ID = "sleep-quality-v1";
/** Processing timeout — same as other Tier-1 services. */
export const SLEEP_TIMEOUT_MS = 10_000;
/** Confidence interval margin when bootstrap is unavailable. */
export const SLEEP_DEFAULT_CI_MARGIN = 0.08;
/** Sleep quality score range (normalized [0, 1]). */
export const SLEEP_QUALITY_MIN = 0;
export const SLEEP_QUALITY_MAX = 1;

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/** Sleep stage labels in order: W(0), N1(1), N2(2), N3(3), REM(4). */
export const SLEEP_STAGES_5 = ["W", "N1", "N2", "N3", "REM"] as const;
export type SleepStageLabel = (typeof SLEEP_STAGES_5)[number];

export type SleepQueryType = "sleep-stages";

export interface SleepDecodeRequest {
  /** Raw 2312-D embedding vector (L2-normalised). If omitted, must provide embedding_id. */
  query_embedding?: number[];
  /** Existing Joint-2312 embedding row id to reuse (Embed Once → Reuse Many). */
  embedding_id?: string;
  /** Which sleep analysis to perform. Default: "sleep-stages". */
  query_type?: SleepQueryType;
  /** Optional head id to use (defaults to SLEEP_DEFAULT_HEAD_ID). */
  head_id?: string;
}

export interface SleepResult {
  /** Predicted sleep stage index (0=W, 1=N1, 2=N2, 3=N3, 4=REM). */
  stage_id: number;
  /** Predicted sleep stage label. */
  stage: SleepStageLabel | "UNKNOWN";
  /** Per-stage softmax probabilities [P(W), P(N1), P(N2), P(N3), P(REM)]. */
  probabilities: [number, number, number, number, number];
  /** Confidence score [0, 1] (max probability). */
  confidence: number;
  /** Confidence interval [lower, upper]. */
  confidence_interval: [number, number];
  /** Metric type decoded. */
  metric: SleepQueryType;
}

export interface SleepDecodeResponse {
  service: string;
  model: string;
  head: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: SleepResult[];
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

// ─────────────────────────────────────────────────────────────────────
// Sleep Quality (M40)
// ─────────────────────────────────────────────────────────────────────

export interface SleepQualityDecodeRequest {
  query_embedding?: number[];
  embedding_id?: string;
  query_type?: "sleep-quality";
  head_id?: string;
}

export interface SleepQualityResult {
  /** Normalized sleep quality score [0, 1]. */
  score: number;
  /** Quality band label for quick interpretation. */
  band: "poor" | "fair" | "good" | "excellent";
  /** Confidence interval [lower, upper] around the score. */
  confidence_interval: [number, number];
  /** Confidence [0, 1] (inverse of prediction uncertainty). */
  confidence: number;
  /** Metric type decoded. */
  metric: "sleep-quality";
}

export interface SleepQualityDecodeResponse {
  service: string;
  model: string;
  head: string;
  head_version: string;
  embedding_id?: string;
  provenance: ServiceProvenance;
  results: SleepQualityResult[];
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

/** Error thrown when the sleep staging probe fails. */
export class SleepDecodeError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "SleepDecodeError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// ONNX probe loading (server-side, onnxruntime-node)
// ─────────────────────────────────────────────────────────────────────

let cachedSleepAdapter: ONNXAdapter | null = null;

/**
 * Load the sleep staging probe ONNX model (2312→5), with SHA-256
 * verification against the manifest (T-016 pattern). Idempotent — the loaded
 * session is cached for reuse across requests.
 */
async function ensureSleepProbe(): Promise<ONNXAdapter> {
  if (cachedSleepAdapter) return cachedSleepAdapter;

  const head: TaskHeadDescriptor = SLEEP_STAGING_PROBE_JOINT_2312;

  cachedSleepAdapter = new ONNXAdapter({
    id: head.id,
    name: head.name,
    version: head.version,
    description: head.training.protocol,
    task: "classification",
    inputShape: { kind: "features", dim: JOINT_2312_EMBEDDING_DIM },
    embeddingDim: head.outputDim,
    outputPooling: "none",
    artifact: "/models/sleep/staging-probe-joint2312-v1.onnx",
    enableVerification: true,
    executionProviders: ["cpu"],
  });

  await cachedSleepAdapter.load();
  return cachedSleepAdapter;
}

/** Reset the cached sleep staging probe (test helper + SSR lifecycle hook). */
export function resetSleepProbe(): void {
  cachedSleepAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedSleepAdapter = null;
}

// ─────────────────────────────────────────────────────────────────────
// Quality probe loading (M40 — regression, 2312→1)
// ─────────────────────────────────────────────────────────────────────

let cachedQualityAdapter: ONNXAdapter | null = null;

/**
 * Load the sleep quality probe ONNX model (2312→1), with SHA-256
 * verification. Idempotent — session cached for reuse.
 */
async function ensureSleepQualityProbe(): Promise<ONNXAdapter> {
  if (cachedQualityAdapter) return cachedQualityAdapter;

  const head = SLEEP_QUALITY_PROBE_JOINT_2312;

  cachedQualityAdapter = new ONNXAdapter({
    id: head.id,
    name: head.name,
    version: head.version,
    description: head.training.protocol,
    task: "regression",
    inputShape: { kind: "features", dim: JOINT_2312_EMBEDDING_DIM },
    embeddingDim: head.outputDim,
    outputPooling: "none",
    artifact: "/models/sleep/quality-probe-joint2312-v1.onnx",
    enableVerification: true,
    executionProviders: ["cpu"],
  });

  await cachedQualityAdapter.load();
  return cachedQualityAdapter;
}

/** Reset the cached quality probe (test helper). */
export function resetSleepQualityProbe(): void {
  cachedQualityAdapter?.unload().catch(() => {
    /* best-effort on teardown */
  });
  cachedQualityAdapter = null;
}

// ─────────────────────────────────────────────────────────────────────
// Softmax helper
// ─────────────────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exp = logits.map((l) => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((e) => e / sum);
}

// ─────────────────────────────────────────────────────────────────────
// Core decode
// ─────────────────────────────────────────────────────────────────────

/**
 * Decode sleep stage (W, N1, N2, N3, REM) from a Joint-2312 embedding.
 *
 * Follows the "Embed Once → Reuse Many" principle:
 * - If `embedding_id` is provided, the existing stored embedding is fetched
 *   from `joint_embeddings_2312` and reused.
 * - If `query_embedding` is provided, it is used directly.
 *
 * @param opts - Decode parameters.
 * @param supabase - Authenticated Supabase client (user-scoped).
 * @param userId - Authenticated user id (for RLS + audit).
 * @returns The sleep staging decode response.
 */
export async function decodeSleepState(
  opts: SleepDecodeRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<SleepDecodeResponse> {
  const tStart = startTimer("sleep.decode.total");

  metrics.tier1ServiceRequestsTotal.inc({ service: SLEEP_SERVICE });
  metrics.sleepDecodeRequestsTotal.inc();

  const queryType = opts.query_type ?? "sleep-stages";
  const headId = opts.head_id ?? SLEEP_DEFAULT_HEAD_ID;

  // Fetch or validate the query embedding.
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    // Reuse existing embedding — fetch from joint_embeddings_2312.
    const tEmbed = startTimer("sleep.decode.reuse_embedding");

    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new SleepDecodeError(
        "embedding_id not found or access denied",
        "EMBEDDING_NOT_FOUND",
      );
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.sleepEmbeddingReusedTotal.inc();
  } else if (opts.query_embedding) {
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    embedMs = undefined;
    metrics.sleepEmbeddingReembeddedTotal.inc();
  } else {
    throw new SleepDecodeError(
      "Either query_embedding or embedding_id must be provided",
      "INVALID_REQUEST",
    );
  }

  // Validate embedding dimension before inference.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new SleepDecodeError(
      `Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
      "DIMENSION_MISMATCH",
    );
  }

  // Run the ONNX probe.
  const tInfer = startTimer("sleep.decode.inference", { query_type: queryType });

  let probeAdapter: ONNXAdapter;
  try {
    probeAdapter = await ensureSleepProbe();
  } catch (e) {
    metrics.sleepDecodeErrorsTotal.inc({ error: "probe_load_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: SLEEP_SERVICE });
    log("error", "sleep.decode.probe_load_failed", {
      error: (e as Error).message,
      userId,
    });
    throw new SleepDecodeError(
      "Sleep probe model unavailable",
      "PROBE_UNAVAILABLE",
    );
  }

  // Predict: extract all 5 class logits and apply softmax.
  let logitValues: number[];
  try {
    const input = { kind: "features" as const, features: [queryEmbedding] };
    const result = await probeAdapter.predict(input);
    // ONNX model outputs 5 logits (class_0 through class_4)
    logitValues = [0, 1, 2, 3, 4].map((i) => result.values[`class_${i}`] ?? 0);
  } catch (e) {
    metrics.sleepDecodeErrorsTotal.inc({ error: "inference_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: SLEEP_SERVICE });
    log("error", "sleep.decode.inference_failed", {
      error: (e as Error).message,
      userId,
      queryType,
    });
    throw new SleepDecodeError("Inference failed", "INFERENCE_FAILED");
  }

  const inferMs = tInfer.end({ results: 1 });
  metrics.sleepDecodeLatencyMs.observe({ query_type: queryType }, inferMs);

  // Apply softmax to get per-stage probabilities.
  const probabilities = softmax(logitValues) as [number, number, number, number, number];
  const confidence = Math.max(...probabilities);
  const stageId = probabilities.indexOf(confidence);
  const stage: SleepStageLabel | "UNKNOWN" = (SLEEP_STAGES_5 as readonly string[])[stageId] as SleepStageLabel;

  // Confidence interval based on the gap between top-1 and top-2 probability.
  const sorted = [...probabilities].sort((a, b) => b - a);
  const margin = sorted[0] - sorted[1];
  const ciMargin = Math.max(SLEEP_DEFAULT_CI_MARGIN, 1 - margin); // wider CI when uncertain
  const ciLower = Math.max(0, confidence - ciMargin);
  const ciUpper = Math.min(1, confidence + ciMargin);

  const result: SleepResult = {
    stage_id: stageId,
    stage,
    probabilities,
    confidence,
    confidence_interval: [ciLower, ciUpper],
    metric: queryType,
  };

  const totalMs = tStart.end({
    query_type: queryType,
    embedding_reused: embeddingReused,
  });

  metrics.sleepStagePredictionsTotal.inc({}, 1);
  metrics.sleepConfidenceDistribution.observe({ query_type: queryType }, confidence);
  metrics.tier1ServiceLatencyMs.observe({ service: SLEEP_SERVICE }, totalMs);
  metrics.tier1AuditLogInsertsTotal.inc();

  // Build provenance (reuses M33/M34's buildServiceProvenance)
  const provenance = buildServiceProvenance({
    service: SLEEP_SERVICE,
    serviceVersion: SLEEP_VERSION,
    taskHeadId: headId,
    taskHeadVersion: SLEEP_STAGING_PROBE_JOINT_2312.version,
    taskHeadSha256: SLEEP_STAGING_PROBE_JOINT_2312.sha256,
    taskHeadDataset: "Sleep-EDF (PhysioNet 1.0.0)",
    taskHeadMetrics: SLEEP_STAGING_PROBE_JOINT_2312.training?.metrics ?? {
      acc_5class: 0.0,
      macro_f1: 0.0,
      kappa: 0.0,
    },
    experimentId: "m39-sleep-staging-probe",
  });

  return {
    service: SLEEP_SERVICE,
    model: JOINT_2312_MODEL_ID,
    head: headId,
    head_version: SLEEP_STAGING_PROBE_JOINT_2312.version,
    ...(embeddingId ? { embedding_id: embeddingId } : {}),
    provenance,
    results: [result],
    metadata: {
      embedding_reused: embeddingReused,
      probe_sha256: SLEEP_STAGING_PROBE_JOINT_2312.sha256,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      inference_ms: inferMs,
      total_ms: totalMs,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────
// Sleep Quality Decode (M40 — regression on Joint-2312)
// ─────────────────────────────────────────────────────────────────────

/**
 * Decode sleep quality (normalized [0, 1]) from a Joint-2312 embedding.
 *
 * Mirrors `decodeSleepState()` but for the regression task head. The ONNX probe
 * outputs a single scalar `class_0` representing predicted quality. The value
 * is clamped to [0, 1] and mapped to a quality band.
 */
export async function decodeSleepQuality(
  opts: SleepQualityDecodeRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<SleepQualityDecodeResponse> {
  const tStart = startTimer("sleep.decode.total");

  metrics.tier1ServiceRequestsTotal.inc({ service: SLEEP_SERVICE });
  metrics.sleepDecodeRequestsTotal.inc();

  const queryType = opts.query_type ?? "sleep-quality";
  const headId = opts.head_id ?? SLEEP_QUALITY_DEFAULT_HEAD_ID;

  // Fetch or validate the query embedding (same 3-branch pattern as staging).
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    const tEmbed = startTimer("sleep.decode.reuse_embedding");
    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new SleepDecodeError("embedding_id not found or access denied", "EMBEDDING_NOT_FOUND");
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.sleepEmbeddingReusedTotal.inc();
  } else if (opts.query_embedding) {
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    metrics.sleepEmbeddingReembeddedTotal.inc();
  } else {
    throw new SleepDecodeError(
      "Either query_embedding or embedding_id must be provided",
      "INVALID_REQUEST",
    );
  }

  // Validate embedding dimension.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new SleepDecodeError(
      `Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
      "DIMENSION_MISMATCH",
    );
  }

  // Run the ONNX quality probe.
  const tInfer = startTimer("sleep.decode.inference", { query_type: queryType });

  let probeAdapter: ONNXAdapter;
  try {
    probeAdapter = await ensureSleepQualityProbe();
  } catch {
    metrics.sleepDecodeErrorsTotal.inc({ error: "probe_load_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: SLEEP_SERVICE });
    log("error", "sleep.decode.quality_probe_load_failed", { userId });
    throw new SleepDecodeError("Sleep quality probe model unavailable", "PROBE_UNAVAILABLE");
  }

  // Predict: read the single regression output (class_0).
  let rawScore: number;
  try {
    const input = { kind: "features" as const, features: [queryEmbedding] };
    const result = await probeAdapter.predict(input);
    rawScore = result.values["class_0"] ?? 0;
  } catch (e) {
    metrics.sleepDecodeErrorsTotal.inc({ error: "inference_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: SLEEP_SERVICE });
    log("error", "sleep.decode.quality_inference_failed", {
      error: (e as Error).message,
      userId,
      queryType,
    });
    throw new SleepDecodeError("Inference failed", "INFERENCE_FAILED");
  }

  const inferMs = tInfer.end({ results: 1 });
  metrics.sleepDecodeLatencyMs.observe({ query_type: queryType }, inferMs);

  // Clamp to [0, 1] and derive quality band + confidence.
  const score = Math.max(SLEEP_QUALITY_MIN, Math.min(SLEEP_QUALITY_MAX, rawScore));
  const band: "poor" | "fair" | "good" | "excellent" =
    score < 0.4
      ? "poor"
      : score < 0.6
        ? "fair"
        : score < 0.8
          ? "good"
          : "excellent";

  // Confidence proxy: distance from nearest band boundary (higher = more confident).
  const distances = [0.0, 0.4, 0.6, 0.8, 1.0].map((b) => Math.abs(score - b));
  const confidence = 1 - Math.min(...distances);

  const result: SleepQualityResult = {
    score,
    band,
    confidence_interval: [Math.max(0, score - 0.1), Math.min(1, score + 0.1)],
    confidence,
    metric: "sleep-quality",
  };

  const totalMs = tStart.end({
    query_type: queryType,
    embedding_reused: embeddingReused,
  });

  metrics.sleepStagePredictionsTotal.inc();
  metrics.sleepConfidenceDistribution.observe({ query_type: queryType }, confidence);
  metrics.tier1ServiceLatencyMs.observe({ service: SLEEP_SERVICE }, totalMs);
  metrics.tier1AuditLogInsertsTotal.inc();

  const provenance = buildServiceProvenance({
    service: SLEEP_SERVICE,
    serviceVersion: SLEEP_VERSION,
    taskHeadId: headId,
    taskHeadVersion: SLEEP_QUALITY_PROBE_JOINT_2312.version,
    taskHeadSha256: SLEEP_QUALITY_PROBE_JOINT_2312.sha256,
    taskHeadDataset: "Sleep-EDF (PhysioNet 1.0.0)",
    taskHeadMetrics: SLEEP_QUALITY_PROBE_JOINT_2312.training?.metrics ??
      { r2: 0.0, rmse: 0.0, mae: 0.0, pearson_r: 0.0 },
    experimentId: "m40-sleep-quality-probe",
  });

  return {
    service: SLEEP_SERVICE,
    model: JOINT_2312_MODEL_ID,
    head: headId,
    head_version: SLEEP_QUALITY_PROBE_JOINT_2312.version,
    ...(embeddingId ? { embedding_id: embeddingId } : {}),
    provenance,
    results: [result],
    metadata: {
      embedding_reused: embeddingReused,
      probe_sha256: SLEEP_QUALITY_PROBE_JOINT_2312.sha256,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      inference_ms: inferMs,
      total_ms: totalMs,
    },
  };
}
