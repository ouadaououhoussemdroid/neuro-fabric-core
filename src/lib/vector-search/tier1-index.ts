/**
 * M31 — Tier-1 DownstreamVectorIndex.
 *
 * Extends {@link NeuralVectorIndex} to support downstream Tier-1 result
 * storage and search. Each Tier-1 service (Subject Identity, Cognitive State,
 * Anomaly Detection) gets its own result table, RPC pair, and dimension
 * contract — but reuses the exact same NeuralVectorIndex interface
 * (add, addAll, search, nearest, size) and the same pgvector-backed search
 * pattern (ANN ivfflat or brute-force exact RPC).
 *
 * The existing Tier-1/Tier-2 tables are NOT modified:
 *   - `embeddings`       (vector(32))    — untouched
 *   - `foundation_embeddings` (vector(200)) — untouched
 *   - `joint_embeddings` (vector(264))  — untouched
 *   - `joint_embeddings_2312` (vector(2312)) — untouched (read-only for downstream)
 *
 * This class simply parameterizes NeuralVectorIndex with service-specific
 * table/RPC names. Services store results (not embeddings) in their result
 * tables, which reference `joint_embeddings_2312(id)` via a foreign key.
 */
import type { NeuralVectorIndexOptions } from "./neural-index";
import { NeuralVectorIndex, DimensionMismatchError } from "./neural-index";
import { log } from "../logging";
import { metrics } from "../metrics";

export { DimensionMismatchError, VectorIndexError } from "./neural-index";

/** Service identifier used in RLS tags and provenance. */
export type Tier1ServiceName = "subject-identity" | "cognitive-intelligence" | "anomaly-detection";

/** Configuration for a Tier-1 downstream result index. */
export interface Tier1IndexOptions extends NeuralVectorIndexOptions {
  /** The downstream service this index belongs to. */
  service: Tier1ServiceName;
  /**
   * The upstream embedding table that result rows reference.
   * Default: "joint_embeddings_2312" (the frozen backbone for all Tier-1 services).
   */
  sourceTable?: string;
}

/**
 * Metadata stored alongside each downstream result row. Every result carries
 * its embedding provenance so callers can audit the exact artifacts that
 * produced a downstream prediction.
 */
export interface DownstreamResultMeta {
  /** FK to the source embedding (e.g. joint_embeddings_2312.id). */
  embedding_id: string;
  /** The service that produced this result. */
  service: Tier1ServiceName;
  /** The task-head model id that produced this result. */
  model_id: string;
  /** SHA-256 digests of the 4 embedding artifacts (CBRaMod, V2, PCA, EEGPT). */
  artifact_shas: {
    cbramod: string;
    v2: string;
    pca: string;
    eegpt: string;
  };
  /** Optional arbitrary metadata (subject_id, session_id, threshold, etc.). */
  [key: string]: unknown;
}

/**
 * pgvector-backed vector index for Tier-1 downstream results.
 *
 * Reuses {@link NeuralVectorIndex} under the hood — same interface, same
 * fallback to in-memory {@link VectorIndex} when no Supabase client is
 * configured. The service-specific configuration (table name, RPC names,
 * dimensions) is set at construction time.
 *
 * Each Tier-1 service creates its own instance with its result-table name:
 *
 * ```ts
 * const subjIdx = new DownstreamVectorIndex({
 *   service: "subject-identity",
 *   tableName: "subject_similarity_results",
 *   matchRpc: "match_subject_similarity",
 *   matchRpcExact: "match_subject_similarity_results_exact",
 *   modelId: "subject-identity-mahalanobis-v1",
 *   dimensions: 2312,  // similarity is computed in embedding space
 *   service: "subject-identity",
 * });
 * ```
 */
export class DownstreamVectorIndex<M = unknown> extends NeuralVectorIndex<M> {
  readonly service: Tier1ServiceName;
  readonly sourceTable: string;

  constructor(opts: Tier1IndexOptions) {
    super(opts);
    this.service = opts.service;
    this.sourceTable = opts.sourceTable ?? "joint_embeddings_2312";
  }

  /**
   * Tag a result row with full provenance before insertion.
   * Reuses the caller's embedding metadata to ensure the audit trail is
   * complete: which embedding + which artifacts + which head produced this.
   */
  provenanceMeta(meta: Omit<DownstreamResultMeta, "service">): DownstreamResultMeta {
    return {
      ...meta,
      service: this.service,
    };
  }
}
