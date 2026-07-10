/**
 * T-011: Database Migrations for pgvector
 * 
 * Applies schema migrations to set up pgvector embeddings table.
 * Run migrations before using PgvectorAdapter.
 */

import { SupabaseClient } from "@supabase/supabase-js";

export interface MigrationResult {
  success: boolean;
  message: string;
  executedAt: Date;
}

/**
 * Apply pgvector migrations to Supabase PostgreSQL.
 * Creates embeddings table, indexes, and RPC functions.
 */
export async function applyPgvectorMigrations(
  supabase: SupabaseClient
): Promise<MigrationResult[]> {
  const results: MigrationResult[] = [];

  // Migration 1: Enable pgvector extension
  try {
    await supabase.rpc("raw_sql", {
      sql: "CREATE EXTENSION IF NOT EXISTS vector;",
    });
    results.push({
      success: true,
      message: "✅ pgvector extension enabled",
      executedAt: new Date(),
    });
  } catch (error: any) {
    results.push({
      success: false,
      message: `❌ Failed to enable pgvector: ${error.message}`,
      executedAt: new Date(),
    });
  }

  // Migration 2: Create embeddings table
  try {
    const { error } = await supabase.from("embeddings").select("id").limit(1);

    if (error && error.code === "PGRST116") {
      // Table doesn't exist, create it
      await supabase.rpc("raw_sql", {
        sql: `
        CREATE TABLE embeddings (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID NOT NULL,
          model_id TEXT NOT NULL,
          source TEXT NOT NULL,
          embedding vector(32) NOT NULL,
          metadata JSONB DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ DEFAULT now(),
          updated_at TIMESTAMPTZ DEFAULT now()
        );
      `,
      });
      results.push({
        success: true,
        message: "✅ embeddings table created",
        executedAt: new Date(),
      });
    } else if (error) {
      throw error;
    } else {
      results.push({
        success: true,
        message: "✅ embeddings table already exists",
        executedAt: new Date(),
      });
    }
  } catch (error: any) {
    results.push({
      success: false,
      message: `❌ Failed to create embeddings table: ${error.message}`,
      executedAt: new Date(),
    });
  }

  // Migration 3: Create indexes
  try {
    await supabase.rpc("raw_sql", {
      sql: `
      CREATE INDEX IF NOT EXISTS embeddings_user_id_idx ON embeddings(user_id);
      CREATE INDEX IF NOT EXISTS embeddings_model_id_idx ON embeddings(model_id);
      CREATE INDEX IF NOT EXISTS embeddings_created_at_idx ON embeddings(created_at);
    `,
    });
    results.push({
      success: true,
      message: "✅ Indexes created",
      executedAt: new Date(),
    });
  } catch (error: any) {
    results.push({
      success: false,
      message: `⚠️  Failed to create indexes (may already exist): ${error.message}`,
      executedAt: new Date(),
    });
  }

  // Migration 4: Create ivfflat index for similarity search
  try {
    await supabase.rpc("raw_sql", {
      sql: `
      CREATE INDEX IF NOT EXISTS embeddings_embedding_ivfflat
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    `,
    });
    results.push({
      success: true,
      message: "✅ ivfflat index created (similarity search optimized)",
      executedAt: new Date(),
    });
  } catch (error: any) {
    results.push({
      success: false,
      message: `⚠️  Failed to create ivfflat index: ${error.message}`,
      executedAt: new Date(),
    });
  }

  // Migration 5: Create search RPC function
  try {
    await supabase.rpc("raw_sql", {
      sql: `
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
    `,
    });
    results.push({
      success: true,
      message: "✅ search_embeddings RPC function created",
      executedAt: new Date(),
    });
  } catch (error: any) {
    results.push({
      success: false,
      message: `⚠️  Failed to create RPC function: ${error.message}`,
      executedAt: new Date(),
    });
  }

  return results;
}

/**
 * Verify pgvector setup and report status.
 */
export async function verifyPgvectorSetup(
  supabase: SupabaseClient
): Promise<{
  pgvectorEnabled: boolean;
  embeddingsTableExists: boolean;
  embeddingCount: number;
  indexes: string[];
}> {
  let pgvectorEnabled = false;
  let embeddingsTableExists = false;
  let embeddingCount = 0;
  let indexes: string[] = [];

  // Check if embeddings table exists
  try {
    const { data, count } = await supabase
      .from("embeddings")
      .select("*", { count: "exact", head: true });
    embeddingsTableExists = true;
    embeddingCount = count || 0;
  } catch (error) {
    embeddingsTableExists = false;
  }

  // Check indexes (from information_schema)
  try {
    const { data } = await supabase.rpc("raw_sql", {
      sql: `
        SELECT indexname FROM pg_indexes WHERE tablename = 'embeddings';
      `,
    });
    indexes = (data || []).map((row: any) => row.indexname);
  } catch (error) {
    indexes = [];
  }

  // pgvector is enabled if we can create vector columns
  pgvectorEnabled = embeddingsTableExists && indexes.length > 0;

  return {
    pgvectorEnabled,
    embeddingsTableExists,
    embeddingCount,
    indexes,
  };
}
