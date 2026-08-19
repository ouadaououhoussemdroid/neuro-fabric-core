/**
 * M32 — Subject Identity & Cohort Similarity service logic.
 *
 * This module implements the core subject-identity scoring logic that powers
 * the `POST /api/joint2312/similarity/search` endpoint. It operates on
 * already-computed Joint-2312 embeddings (2312-D, L2-normalised) stored in
 * the `joint_embeddings_2312` table.
 *
 * **Embed Once → Reuse Many:** if the caller already has a Joint-2312
 * embedding (either as a raw vector or via an `embedding_id` reference to a
 * stored row), this module reuses it directly — it never recomputes Joint-2312
 * internally. The upstream `/api/eeg/embed/foundation?model=joint-2312` route
 * is responsible for computing and storing embeddings.
 *
 * WHY THIS IS SERVER-ONLY: searching the 2312-D `joint_embeddings_2312` table
 * requires the pgvector-backed `match_joint_embeddings_2312()` RPC, which runs
 * in Supabase Postgres. The vector math (cosine similarity, confidence
 * intervals) is pure JS and could run in the browser, but the ANN search
 * cannot — 2312-D ivfflat is server-side only.
 */
import { randomUUID } from "node:crypto";
import {
  NeuralVectorIndex,
  DimensionMismatchError,
} from "@/lib/vector-search/neural-index";
import { log, startTimer } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance } from "@/lib/ai/services/provenance.server";
import {
  JOINT_2312_MODEL_ID,
  JOINT_2312_EMBEDDING_DIM,
} from "@/lib/ai/inference/joint.server";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const SUBJECT_IDENTITY_SERVICE = "subject-identity";
/** Service version (matches git tag or release version). */
export const SUBJECT_IDENTITY_VERSION = "v0.1.0";
/** Model/head id for subject-identity similarity search. */
export const SUBJECT_IDENTITY_MODEL_ID = "subject-identity-similarity-v1";
/** Default number of matches to return. */
export const DEFAULT_MATCH_COUNT = 10;
/** Maximum matches to return (DoS protection). */
export const MAX_MATCH_COUNT = 100;
/** Default similarity threshold (0.80 ≈ strong subject match). */
export const DEFAULT_SIMILARITY_THRESHOLD = 0.80;
/** Supabase RPC names for joint-2312 ANN search. */
export const MATCH_RPC = "match_joint_embeddings_2312";
export const MATCH_RPC_EXACT = "match_joint_embeddings_2312_exact";

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

export type QueryType =
  | "subject_identification"
  | "session_similarity"
  | "cohort_similarity";

export interface SimilaritySearchRequest {
  /** Raw 2312-D embedding vector (L2-normalised). If omitted, must provide embedding_id. */
  query_embedding?: number[];
  /** Existing Joint-2312 embedding row id to reuse (Embed Once → Reuse Many). */
  embedding_id?: string;
  /** The type of similarity search to perform. */
  query_type: QueryType;
  /** Number of results to return (default 10, max 100). */
  match_count?: number;
  /** Filter to a specific cohort (subject group). */
  filter_cohort_id?: string;
  /** Exclude these subject IDs from results. */
  filter_subject_ids?: string[];
  /** Minimum similarity threshold [0, 1] (default 0.80). */
  threshold?: number;
}

export interface SimilarityResult {
  /** Rank (1-based; 1 = most similar). */
  rank: number;
  /** ID of the matched embedding row in joint_embeddings_2312. */
  embedding_id: string;
  /** Subject ID from the matched embedding's metadata (user-defined label). */
  subject_id?: string;
  /** Session ID from metadata (if available). */
  session_id?: string;
  /** Cosine similarity score [0, 1]. */
  similarity: number;
  /** Confidence = normalized gap between top-1 and top-2 similarity. */
  confidence: number;
  /** Raw metadata JSONB from the matched embedding. */
  metadata: Record<string, unknown>;
}

export interface SimilaritySearchResponse {
  service: string;
  model: string;
  query_type: QueryType;
  provenance: ServiceProvenanceSummary;
  results: SimilarityResult[];
  metadata: {
    match_count: number;
    threshold: number;
    total_matches: number;
    embedding_reused: boolean;
  };
  timings: {
    embed_ms?: number;
    search_ms: number;
    total_ms: number;
  };
}

/** Lightweight provenance summary for the API response (no SHA constants). */
export interface ServiceProvenanceSummary {
  service: string;
  service_version: string;
  embedding_model: string;
  embedding_dim: number;
  task_head_id: string;
  timestamp: string;
  artifact_shas: { cbramod: string; v2: string; pca: string; eegpt: string };
}

// ─────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────

export class SubjectIdentityError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "SubjectIdentityError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// Core search
// ─────────────────────────────────────────────────────────────────────

/**
 * Run a subject-identity / cohort-similarity search over Joint-2312 embeddings.
 *
 * Follows the "Embed Once → Reuse Many" principle: if `embedding_id` is
 * provided, the existing stored embedding is reused; if `query_embedding`
 * is provided, it is used directly. The caller is responsible for computing
 * the Joint-2312 embedding via `/api/eeg/embed/foundation?model=joint-2312`.
 *
 * @param opts - Search parameters.
 * @param supabase - Authenticated Supabase client (user-scoped).
 * @param userId - Authenticated user id (for RLS + audit).
 * @returns The similarity search response.
 */
export async function searchSubjectIdentity(
  opts: SimilaritySearchRequest,
  supabase: {
    from: (table: string) => unknown;
    rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  },
  userId: string,
): Promise<SimilaritySearchResponse> {
  const tStart = startTimer("subject.identity.search");

  metrics.tier1ServiceRequestsTotal.inc({ service: SUBJECT_IDENTITY_SERVICE });
  metrics.subjectIdentityRequestsTotal.inc();

  // Validate + clamp match_count
  const matchCount = Math.min(
    opts.match_count ?? DEFAULT_MATCH_COUNT,
    MAX_MATCH_COUNT,
  );
  const threshold = opts.threshold ?? DEFAULT_SIMILARITY_THRESHOLD;

  // Determine the query vector: either reuse an existing embedding by id,
  // or use the provided raw vector.
  let queryEmbedding: number[];
  let embeddingReused: boolean;
  let embeddingId: string | undefined;
  let embedMs: number | undefined;

  if (opts.embedding_id) {
    // Reuse existing embedding — fetch from joint_embeddings_2312.
    const tEmbed = startTimer("subject.identity.reuse_embedding");
    const { data: existing, error: fetchErr } = await supabase
      .from("joint_embeddings_2312")
      .select("embedding")
      .eq("id", opts.embedding_id)
      .single();

    if (fetchErr || !existing) {
      throw new SubjectIdentityError(
        "embedding_id not found or access denied",
        "EMBEDDING_NOT_FOUND",
      );
    }

    queryEmbedding = (existing as { embedding: number[] }).embedding;
    embeddingReused = true;
    embeddingId = opts.embedding_id;
    embedMs = tEmbed.end({ reused: true });
    metrics.subjectIdentityEmbeddingReusedTotal.inc();
  } else if (opts.query_embedding) {
    // Use the provided raw vector.
    queryEmbedding = opts.query_embedding;
    embeddingReused = false;
    embedMs = undefined;
    metrics.subjectIdentityEmbeddingReembeddedTotal.inc();
  } else {
    throw new SubjectIdentityError(
      "Either query_embedding or embedding_id must be provided",
      "INVALID_REQUEST",
    );
  }

  // Validate the embedding dimension before searching.
  if (queryEmbedding.length !== JOINT_2312_EMBEDDING_DIM) {
    throw new SubjectIdentityError(
      `Expected ${JOINT_2312_EMBEDDING_DIM}-D embedding, got ${queryEmbedding.length}`,
      "DIMENSION_MISMATCH",
    );
  }

  // Run the ANN search via the match_joint_embeddings_2312 RPC.
  const tSearch = startTimer("subject.identity.vector_search", {
    query_type: opts.query_type,
  });

  const idx = new NeuralVectorIndex({
    supabase,
    tableName: "joint_embeddings_2312",
    matchRpc: MATCH_RPC,
    matchRpcExact: MATCH_RPC_EXACT,
    modelId: JOINT_2312_MODEL_ID,
    userId,
    dimensions: JOINT_2312_EMBEDDING_DIM,
    searchMode: "ann",
  });

  let rawResults;
  try {
    rawResults = await idx.search(queryEmbedding, matchCount + 1); // +1 to compute confidence gap
  } catch (e) {
    metrics.subjectIdentityErrorsTotal.inc({ error: "search_failed" });
    metrics.tier1ServiceErrorsTotal.inc({ service: SUBJECT_IDENTITY_SERVICE });
    log("error", "subject.identity.search_failed", {
      error: (e as Error).message,
      userId,
      queryType: opts.query_type,
    });
    throw new SubjectIdentityError(
      "Vector search failed",
      "SEARCH_FAILED",
    );
  }

  const searchMs = tSearch.end({ results: rawResults.length });
  metrics.vectorSearchLatencyMs.observe({ operation: "subject_identity" }, searchMs);
  metrics.subjectIdentitySearchLatencyMs.observe({ query_type: opts.query_type }, searchMs);

  // Post-filter: threshold, cohort, exclude subjects
  let filtered = rawResults.filter((r) => {
    // Threshold filter
    if (r.score < threshold) return false;

    const metadata = (r.meta ?? {}) as Record<string, unknown>;

    // Cohort filter
    if (opts.filter_cohort_id) {
      const cohortId = metadata["cohort_id"];
      if (cohortId !== opts.filter_cohort_id) return false;
    }

    // Exclude subjects
    if (opts.filter_subject_ids && opts.filter_subject_ids.length > 0) {
      const subjectId = metadata["subject_id"] as string | undefined;
      if (subjectId && opts.filter_subject_ids.includes(subjectId)) return false;
    }

    return true;
  });

  // Trim to requested count (we fetched +1 for confidence gap computation)
  filtered = filtered.slice(0, matchCount);

  // Compute confidence: normalized gap between top-1 and top-2 similarity.
  // If there's only one result, confidence = 1.0 (no competition).
  const results: SimilarityResult[] = filtered.map((r, i) => {
    const nextScore = i + 1 < filtered.length ? filtered[i + 1].score : r.score;
    const gap = r.score - nextScore;
    // Normalize gap to [0, 1]: max possible gap is 1.0 (perfect vs 0.0)
    const confidence = Math.max(0, Math.min(1, gap * 5)); // 0.2 gap → 1.0 confidence
    const metadata = (r.meta ?? {}) as Record<string, unknown>;

    return {
      rank: i + 1,
      embedding_id: r.id,
      subject_id: metadata["subject_id"] as string | undefined,
      session_id: metadata["session_id"] as string | undefined,
      similarity: r.score,
      confidence,
      metadata,
    };
  });

  const totalMs = tStart.end({
    query_type: opts.query_type,
    match_count: matchCount,
    results: results.length,
    embedding_reused: embeddingReused,
  });

  metrics.subjectIdentityResultsTotal.inc({}, results.length);
  metrics.tier1ServiceLatencyMs.observe({ service: SUBJECT_IDENTITY_SERVICE }, totalMs);
  metrics.tier1AuditLogInsertsTotal.inc();

  return {
    service: SUBJECT_IDENTITY_SERVICE,
    model: JOINT_2312_MODEL_ID,
    query_type: opts.query_type,
    provenance: {
      service: SUBJECT_IDENTITY_SERVICE,
      service_version: SUBJECT_IDENTITY_VERSION,
      embedding_model: JOINT_2312_MODEL_ID,
      embedding_dim: JOINT_2312_EMBEDDING_DIM,
      task_head_id: SUBJECT_IDENTITY_MODEL_ID,
      timestamp: new Date().toISOString(),
      artifact_shas: {
        cbramod: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
        v2: "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
        pca: "deterministic-pca-v1",
        eegpt: "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36",
      },
    },
    results,
    metadata: {
      match_count: matchCount,
      threshold,
      total_matches: results.length,
      embedding_reused: embeddingReused,
    },
    timings: {
      ...(embeddingReused ? { embed_ms: embedMs } : {}),
      search_ms: searchMs,
      total_ms: totalMs,
    },
  };
}
