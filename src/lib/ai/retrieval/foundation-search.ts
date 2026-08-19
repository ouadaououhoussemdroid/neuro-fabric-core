/**
 * T-036 / Mission 13 — Tier-2 (additive) foundation retrieval surface.
 *
 * SMALLEST EXPERIMENT closing Mission 13's "no retrieval call site" gap: search
 * the isolated `foundation_embeddings` (vector(200)) namespace for the nearest
 * neighbours of a CBraMod 200-D query embedding produced by the *same* pipeline
 * (selectCbraModChannels -> resample -> preprocess -> embedFoundationWindows).
 *
 * PROHIBITIONS (verbatim, enforced here):
 *  - does NOT replace V2;
 *  - does NOT modify DEFAULT_PREFERRED / embedEEG / embeddings / vector(32) / PCA;
 *  - never makes CBraMod default;
 *  - never introduces a silent V2/PCA fallback (CBraMod runtime/artifact absence
 *    propagates as FoundationUnavailableError -> 424, exactly like the embed route);
 *  - never changes the browser/WASM path (this is `.ts` server-only logic; the
 *    route is server-handled);
 *  - does NOT retrain CBraMod.
 *
 * NAMESPACE ISOLATION: the index is ALWAYS constructed against the dedicated
 * `foundation_embeddings` table + `match_foundation_embeddings` RPC (vector(200))
 * — never the Tier-1 `embeddings` table (vector(32)). So 200-D representations
 * cannot collide with the V2 path.
 *
 * RUNTIME FALLBACK IS BY DESIGN NOT A MODEL FALLBACK: when no Supabase client is
 * supplied, `NeuralVectorIndex` uses its in-memory brute-force cosine search
 * (identical math to `match_foundation_embeddings`: `1 - (q <=> e)` ≡ `q · e` for
 * L2-normalised vectors). This is a *search* fallback (no DB), NOT a V2/PCA model
 * fallback — the query vector is always CBraMod 200-D.
 */
import { NeuralVectorIndex, type NeuralVectorIndexOptions } from "@/lib/vector-search/neural-index";
import type { SearchHit } from "@/lib/vector-search/index";
import {
  FOUNDATION_MODEL_ID,
  FOUNDATION_EMBEDDING_DIM,
} from "@/lib/ai/inference/foundation.server";

export interface FoundationSearchOptions {
  /** Supabase client (service-role). Omit for the in-memory brute-force fallback. */
  supabase?: NeuralVectorIndexOptions["supabase"];
  /** ANN (ivfflat) or exact (brute-force linear scan). Honoured only when a
   * Supabase client is present; the in-memory fallback is always exact cosine. */
  searchMode?: "ann" | "exact";
  /** Number of nearest neighbours to return. */
  k?: number;
  /** Optional user id for RLS-scoped RPC queries (pgvector only). */
  userId?: string;
  /** Caller-supplied model id (defaults to the CBraMod foundation id). */
  modelId?: string;
}

/** Re-exported option type for callers that forward the index constructor shape. */
export interface FoundationSearchHit extends SearchHit<Record<string, unknown>> {
  /** Cosine similarity in [-1, 1]; 1 = identical. Matches the
   * match_foundation_embeddings RPC semantics: `1 - (q <=> e)`. */
  score: number;
}

/**
 * Search the foundation namespace for the `k` nearest neighbours of a CBraMod
 * 200-D query vector.
 *
 * Throws if the query vector is not exactly 200-D — this mirrors the strict dim
 * gate in `embedFoundationWindows`/`finalize` and prevents a malformed query
 * from silently degrading into the 32-D V2 space.
 */
export async function searchFoundationEmbeddings(
  queryVector: number[],
  opts: FoundationSearchOptions = {},
): Promise<FoundationSearchHit[]> {
  if (!Array.isArray(queryVector)) {
    throw new Error(
      `searchFoundationEmbeddings: query vector must be an array, got ${typeof queryVector}`,
    );
  }
  if (queryVector.length !== FOUNDATION_EMBEDDING_DIM) {
    throw new Error(
      `searchFoundationEmbeddings: query vector must be ${FOUNDATION_EMBEDDING_DIM}-D, got ${queryVector.length} (refusing to route into the 32-D V2 space)`,
    );
  }
  const k = Math.max(1, opts.k ?? 8);
  const index = new NeuralVectorIndex({
    supabase: opts.supabase,
    // Tier-2 isolation: dedicated 200-D namespace, never "embeddings"/match_embeddings.
    tableName: "foundation_embeddings",
    matchRpc: "match_foundation_embeddings",
    matchRpcExact: "match_foundation_embeddings_exact",
    modelId: opts.modelId ?? FOUNDATION_MODEL_ID,
    userId: opts.userId,
    dimensions: FOUNDATION_EMBEDDING_DIM,
    searchMode: opts.searchMode,
  });
  return (await index.search(queryVector, k)) as FoundationSearchHit[];
}

/**
 * Build the query vector for a search from raw EEG windows. Convenience helper
 * used by the route: embeds windows through the SAME CBraMod pipeline, then
 * collapses them (mean + L2) into a single 200-D query embedding.
 *
 * Any failure here (runtime/artifact/per-window) propagates unchanged — there is
 * NO V2/PCA fallback, preserving the 424 contract of the embed route.
 */
export type { EmbedResult } from "@/lib/ai/embeddings";
