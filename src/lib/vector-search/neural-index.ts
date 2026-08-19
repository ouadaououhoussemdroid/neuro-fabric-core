/**
 * T-011 — pgvector-backed NeuralVectorIndex.
 *
 * Preserves the {@link VectorIndex} interface (add, addAll, search, nearest,
 * size) but routes storage and similarity search to a Supabase `embeddings`
 * table with `vector(32)` + ivfflat index (see migration
 * `20260711060000_pgvector_embeddings.sql`).
 *
 * When no Supabase client is provided (e.g. unit tests, local dev), it
 * transparently falls back to the in-memory brute-force {@link VectorIndex},
 * so callers that depend on the interface keep working without a database.
 */
import { VectorIndex, type IndexedVector, type SearchHit } from "./index";
import { log } from "../logging";
import { metrics } from "../metrics";

/** Search strategy: ANN (ivfflat) or exact (brute-force linear scan). */
export type SearchMode = "ann" | "exact";

/** Raised when a vector's dimension does not match the index contract. */
export class DimensionMismatchError extends Error {
  constructor(
    public readonly expected: number,
    public readonly actual: number,
    public readonly vectorId: string,
  ) {
    super(`DimensionMismatch: vector "${vectorId}" has dim ${actual}, expected ${expected}`);
    this.name = "DimensionMismatchError";
  }
}

/** Raised when a vector-store operation fails at the persistence layer. */
export class VectorIndexError extends Error {
  constructor(
    public readonly operation: string,
    message: string,
  ) {
    super(`VectorIndex[${operation}]: ${message}`);
    this.name = "VectorIndexError";
  }
}

export interface NeuralVectorIndexOptions {
  /** Supabase client with service-role or authenticated access. */
  supabase?: {
    from: (table: string) => unknown;
    rpc?: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
  };
  /** Table name to read/write (default: "embeddings"; Tier-2 uses "foundation_embeddings"). */
  tableName?: string;
  /** Model id to tag inserted embeddings with. */
  modelId?: string;
  /** User id for RLS-scoped queries. */
  userId?: string;
  /** Vector dimension (must match the migration's vector(N)). */
  dimensions?: number;
  /**
   * Search mode: "ann" (default) uses the pgvector ivfflat index for fast
   * approximate search; "exact" uses `match_embeddings_exact` RPC for a
   * precise linear scan. When no Supabase client is configured, the
   * in-memory {@link VectorIndex} is always used regardless of this option.
   */
  searchMode?: SearchMode;
  /** RPC name for ANN search (default: "match_embeddings"). Tier-2: "match_foundation_embeddings". */
  matchRpc?: string;
  /** RPC name for exact search (default: "match_embeddings_exact"). Tier-2: "match_foundation_embeddings_exact". */
  matchRpcExact?: string;
}

interface EmbeddingRow {
  id: string;
  embedding: number[];
  metadata: Record<string, unknown> | null;
}

interface SupabaseQueryBuilder {
  insert: (rows: unknown | unknown[]) => {
    select: () => Promise<{ data: unknown[] | null; error: unknown }>;
  };
  select: (columns?: string) => SupabaseSelectBuilder;
  delete: () => { eq: (col: string, val: unknown) => Promise<{ error: unknown }> };
}

interface SupabaseSelectBuilder {
  eq: (col: string, val: unknown) => SupabaseSelectBuilder;
  order: (col: string, opts?: unknown) => SupabaseSelectBuilder;
  limit: (n: number) => SupabaseSelectBuilder;
  rpc?: unknown;
}

interface SupabaseRpcClient {
  rpc: (fn: string, args: Record<string, unknown>) => Promise<{ data: unknown; error: unknown }>;
}

/**
 * pgvector-backed vector index preserving the {@link VectorIndex} interface.
 *
 * Uses a Supabase RPC (`match_embeddings`) for ANN search when available;
 * falls back to in-memory cosine when no client is configured.
 */
export class NeuralVectorIndex<M = unknown> {
  private readonly fallback = new VectorIndex<M>();
  private readonly supabase: NeuralVectorIndexOptions["supabase"];
  private readonly tableName: string;
  private readonly modelId: string;
  private readonly userId?: string;
  private readonly dimensions: number;
  private readonly searchMode: SearchMode;
  private readonly matchRpcName: string;
  private readonly matchRpcExactName: string;

  constructor(opts: NeuralVectorIndexOptions = {}) {
    this.supabase = opts.supabase;
    this.tableName = opts.tableName ?? "embeddings";
    this.modelId = opts.modelId ?? "unknown";
    this.userId = opts.userId;
    this.dimensions = opts.dimensions ?? 32;
    this.searchMode = opts.searchMode ?? "ann";
    this.matchRpcName = opts.matchRpc ?? "match_embeddings";
    this.matchRpcExactName = opts.matchRpcExact ?? "match_embeddings_exact";
  }

  /** Whether the index is backed by pgvector (true) or in-memory (false). */
  get isPersistent(): boolean {
    return this.supabase !== undefined;
  }

  async add(item: IndexedVector<M>): Promise<void> {
    if (!this.supabase) {
      this.fallback.add(item);
      return;
    }
    // Validate the dimension contract before hitting the DB. This prevents
    // silent corruption (wrong-dim vectors stored in a vector(N) column) and
    // surfaces the bug at the call site rather than via a swallowed error.
    if (item.vector.length !== this.dimensions) {
      metrics.vectorStoreErrorsTotal.inc({ operation: "add", error: "dimension_mismatch" });
      throw new DimensionMismatchError(this.dimensions, item.vector.length, item.id);
    }
    const row = {
      id: item.id,
      user_id: this.userId,
      model_id: this.modelId,
      embedding: item.vector,
      embedding_dim: this.dimensions,
      metadata: { ...((item.meta as object) ?? {}), _localId: item.id } as Record<string, unknown>,
    };
    const qb = (this.supabase as { from: (t: string) => SupabaseQueryBuilder }).from(
      this.tableName,
    );
    const { error } = await qb.insert(row).select();
    if (error) {
      // Propagate DB errors — do NOT silently fall back to in-memory.
      // A failed ANN write must be visible so callers can report
      // vector_indexed=false rather than masking the failure.
      metrics.vectorStoreErrorsTotal.inc({ operation: "add", error: "db_insert_failed" });
      log("error", "eeg.upload.vector_store_failed", {
        error: (error as { message?: string })?.message ?? String(error),
      });
      throw new VectorIndexError("add", (error as { message: string })?.message ?? String(error));
    }
  }

  async addAll(items: IndexedVector<M>[]): Promise<void> {
    for (const item of items) await this.add(item);
  }

  async search(query: number[], k = 8): Promise<SearchHit<M>[]> {
    if (!this.supabase) {
      return this.fallback.search(query, k);
    }
    const rpcClient = this.supabase as unknown as SupabaseRpcClient;
    if (typeof rpcClient.rpc !== "function") {
      return this.fallback.search(query, k);
    }
    // Select the exact RPC based on search mode. The RPC names default to the
    // Tier-1 `match_embeddings` / `match_embeddings_exact` but the Tier-2 route
    // passes `matchRpc` / `matchRpcExact` = `match_foundation_embeddings*`.
    const rpcName = this.searchMode === "exact" ? this.matchRpcExactName : this.matchRpcName;
    const { data, error } = await rpcClient.rpc(rpcName, {
      query_embedding: query,
      match_count: k,
      filter_model_id: this.modelId,
      filter_user_id: this.userId,
    });
    if (error || !data) {
      return this.fallback.search(query, k);
    }
    return (data as Array<{ id: string; similarity: number; metadata: M | null }>).map((row) => ({
      id: row.id,
      score: row.similarity,
      meta: row.metadata ?? undefined,
    }));
  }

  async nearest(query: number[]): Promise<SearchHit<M> | null> {
    const hits = await this.search(query, 1);
    return hits.length > 0 ? hits[0] : null;
  }

  size(): number {
    return this.fallback.size();
  }
}
