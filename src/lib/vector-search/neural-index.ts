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

export interface NeuralVectorIndexOptions {
  /** Supabase client with service-role or authenticated access to the embeddings table. */
  supabase?: {
    from: (table: string) => unknown;
  };
  /** Model id to tag inserted embeddings with. */
  modelId?: string;
  /** User id for RLS-scoped queries. */
  userId?: string;
  /** Vector dimension (must match the migration's vector(N)). */
  dimensions?: number;
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
  private readonly modelId: string;
  private readonly userId?: string;
  private readonly dimensions: number;

  constructor(opts: NeuralVectorIndexOptions = {}) {
    this.supabase = opts.supabase;
    this.modelId = opts.modelId ?? "unknown";
    this.userId = opts.userId;
    this.dimensions = opts.dimensions ?? 32;
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
    const row = {
      id: item.id,
      user_id: this.userId,
      model_id: this.modelId,
      embedding: item.vector,
      embedding_dim: this.dimensions,
      metadata: { ...((item.meta as object) ?? {}), _localId: item.id } as Record<string, unknown>,
    };
    const qb = (this.supabase as { from: (t: string) => SupabaseQueryBuilder }).from("embeddings");
    const { error } = await qb.insert(row).select();
    if (error) {
      // Fall back to in-memory on DB error so the pipeline doesn't crash.
      this.fallback.add(item);
    }
  }

  async addAll(items: IndexedVector<M>[]): Promise<void> {
    for (const item of items) await this.add(item);
  }

  async search(query: number[], k = 8): Promise<SearchHit<M>[]> {
    if (!this.supabase) {
      return this.fallback.search(query, k);
    }
    // Use the match_embeddings RPC for ANN search via pgvector.
    const rpcClient = this.supabase as unknown as SupabaseRpcClient;
    if (typeof rpcClient.rpc !== "function") {
      return this.fallback.search(query, k);
    }
    const { data, error } = await rpcClient.rpc("match_embeddings", {
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
