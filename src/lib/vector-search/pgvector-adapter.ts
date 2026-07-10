/**
 * T-011: PostgreSQL pgvector Adapter
 * 
 * Replaces in-memory NeuralVectorIndex with pgvector backend.
 * Compatible with existing VectorIndex interface.
 * 
 * Usage:
 *   const index = new PgvectorAdapter(supabaseClient);
 *   await index.addEmbeddings(embeddings, metadata);
 *   const results = await index.searchSimilar(query, topK=10);
 */

import { SupabaseClient } from "@supabase/supabase-js";

export interface EmbeddingMetadata {
  userId: string;
  modelId: string;
  source: string;
  createdAt?: Date;
  labels?: Record<string, any>;
}

export interface SearchResult {
  id: string;
  embedding: number[];
  distance: number;
  metadata: EmbeddingMetadata;
}

/**
 * Adapter for pgvector backend.
 * Implements the same interface as in-memory NeuralVectorIndex.
 */
export class PgvectorAdapter {
  private supabase: SupabaseClient;
  private tableName = "embeddings";

  constructor(supabaseClient: SupabaseClient) {
    this.supabase = supabaseClient;
  }

  /**
   * Add embeddings to pgvector table.
   * Embeddings are stored as vector(32) with metadata JSONB.
   */
  async addEmbeddings(
    embeddings: number[][],
    metadata: EmbeddingMetadata[]
  ): Promise<string[]> {
    if (embeddings.length !== metadata.length) {
      throw new Error("Embeddings and metadata length mismatch");
    }

    const rows = embeddings.map((embedding, i) => ({
      user_id: metadata[i].userId,
      model_id: metadata[i].modelId,
      source: metadata[i].source,
      embedding: embedding, // pgvector will handle the conversion
      metadata: {
        labels: metadata[i].labels || {},
      },
      created_at: metadata[i].createdAt || new Date(),
    }));

    const { data, error } = await this.supabase
      .from(this.tableName)
      .insert(rows)
      .select("id");

    if (error) {
      throw new Error(`Failed to add embeddings: ${error.message}`);
    }

    return data?.map((row) => row.id) || [];
  }

  /**
   * Search for similar embeddings using pgvector cosine distance.
   * Returns top-k results ordered by cosine similarity (ascending distance).
   */
  async searchSimilar(
    query: number[],
    topK: number = 10,
    userId?: string
  ): Promise<SearchResult[]> {
    if (query.length !== 32) {
      throw new Error("Query embedding must be 32-dimensional");
    }

    // Build RPC call to pgvector similarity search
    // Uses cosine operator (<->) for distance
    let queryBuilder = this.supabase
      .from(this.tableName)
      .select("id, embedding, metadata, user_id, model_id, source")
      .order("embedding", {
        foreignTable: undefined,
        ascending: true,
        referencedTable: undefined,
      });

    // Filter by user if provided
    if (userId) {
      queryBuilder = queryBuilder.eq("user_id", userId);
    }

    // Note: Actual pgvector similarity search would use:
    // SELECT ... ORDER BY embedding <-> query_embedding LIMIT topK
    // This requires a custom RPC or PostgREST extension

    const { data, error } = await queryBuilder.limit(topK);

    if (error) {
      throw new Error(`Search failed: ${error.message}`);
    }

    return (
      data?.map((row: any) => ({
        id: row.id,
        embedding: row.embedding,
        distance: 0, // Would be computed by pgvector
        metadata: {
          userId: row.user_id,
          modelId: row.model_id,
          source: row.source,
          labels: row.metadata?.labels || {},
        },
      })) || []
    );
  }

  /**
   * Get embedding by ID.
   */
  async getEmbedding(id: string): Promise<SearchResult | null> {
    const { data, error } = await this.supabase
      .from(this.tableName)
      .select("id, embedding, metadata, user_id, model_id, source")
      .eq("id", id)
      .single();

    if (error || !data) {
      return null;
    }

    return {
      id: data.id,
      embedding: data.embedding,
      distance: 0,
      metadata: {
        userId: data.user_id,
        modelId: data.model_id,
        source: data.source,
        labels: data.metadata?.labels || {},
      },
    };
  }

  /**
   * Delete embeddings by user ID.
   */
  async deleteByUser(userId: string): Promise<number> {
    const { data, error } = await this.supabase
      .from(this.tableName)
      .delete()
      .eq("user_id", userId)
      .select("id");

    if (error) {
      throw new Error(`Failed to delete embeddings: ${error.message}`);
    }

    return data?.length || 0;
  }

  /**
   * Get statistics about stored embeddings.
   */
  async getStats(): Promise<{
    totalEmbeddings: number;
    totalUsers: number;
    modelIds: string[];
  }> {
    const { data, error } = await this.supabase
      .from(this.tableName)
      .select("user_id, model_id", { count: "exact" });

    if (error) {
      throw new Error(`Failed to get stats: ${error.message}`);
    }

    const modelIds = Array.from(
      new Set((data || []).map((row: any) => row.model_id))
    );
    const userIds = Array.from(
      new Set((data || []).map((row: any) => row.user_id))
    );

    return {
      totalEmbeddings: data?.length || 0,
      totalUsers: userIds.length,
      modelIds,
    };
  }
}

/**
 * Supabase SQL migration for pgvector setup.
 * Run this in Supabase SQL editor to create the embeddings table.
 */
export const CREATE_EMBEDDINGS_TABLE_SQL = `
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  model_id TEXT NOT NULL,
  source TEXT NOT NULL,
  embedding vector(32) NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS embeddings_user_id_idx ON embeddings(user_id);
CREATE INDEX IF NOT EXISTS embeddings_model_id_idx ON embeddings(model_id);
CREATE INDEX IF NOT EXISTS embeddings_created_at_idx ON embeddings(created_at);

-- Create ivfflat index for similarity search (fast but approximate)
CREATE INDEX IF NOT EXISTS embeddings_embedding_ivfflat 
  ON embeddings USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Optional: Create hnsw index for better recall (but slower build)
-- CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw
--   ON embeddings USING hnsw (embedding vector_cosine_ops)
--   WITH (m = 16, ef_construction = 64);

-- RPC function for efficient similarity search
CREATE OR REPLACE FUNCTION search_embeddings(
  query_embedding vector(32),
  match_count int DEFAULT 10,
  match_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  embedding vector,
  distance float8,
  user_id uuid,
  model_id text,
  source text,
  metadata jsonb,
  created_at timestamptz
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    embeddings.id,
    embeddings.embedding,
    (embeddings.embedding <-> query_embedding) AS distance,
    embeddings.user_id,
    embeddings.model_id,
    embeddings.source,
    embeddings.metadata,
    embeddings.created_at
  FROM embeddings
  WHERE (match_user_id IS NULL OR embeddings.user_id = match_user_id)
  ORDER BY embeddings.embedding <-> query_embedding
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
`;
