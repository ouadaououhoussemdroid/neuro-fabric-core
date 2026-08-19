-- Tier-2 (Mission 12) SERVER-NATIVE CBraMod foundation embeddings.
--
-- Additive only. This migration creates a DEDICATED vector(200) namespace for
-- CBraMod's native 200-D server-side representation. It does NOT modify the
-- Tier-1 `embeddings` table (vector(32)), its `CHECK (vector_dims(embedding)=32)`,
-- the `match_embeddings` RPC, or any prior migration. V2 / DEFAULT_PREFERRED /
-- vector(32) contract are preserved byte-for-byte.
--
-- Mission 11 validated that CBraMod 200-D provides cross-session subject-identity
-- retrieval that V2-32-D cannot (Recall@5 Δ+0.31, Δ@10 +0.32, Bonferroni p<<0.05;
-- MI acc 0.275 >= chance 0.25). This table is the storage backing for that path.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.foundation_embeddings (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  -- The model that produced this embedding. Tagged explicitly so 200-D rows
  -- can never collide with the Tier-1 `embeddings` (vector(32)) namespace.
  model_id      TEXT        NOT NULL DEFAULT 'onnx-cbramod-foundation-200d',
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- CBraMod native output: 200-D mean-tokens pooled [1,19,5,200] -> [200].
  -- Hard-checked to 200 so a 32-D (or any other dim) vector is a DB error,
  -- mirroring the Tier-1 `CHECK (vector_dims(embedding) = 32)` contract.
  embedding     vector(200) NOT NULL,
  embedding_dim INT         NOT NULL DEFAULT 200,
  -- Provenance / source metadata: artifact sha256, sample_rate, window_samples,
  -- channels, input_format, subject/session/recording where applicable.
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT foundation_embeddings_dims CHECK (vector_dims(embedding) = 200)
);

-- IVFFlat index for approximate nearest-neighbour cosine search (200-D).
CREATE INDEX IF NOT EXISTS idx_foundation_embedding_ivfflat
  ON public.foundation_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Supporting indexes for filtered / model-versioned search.
CREATE INDEX IF NOT EXISTS idx_foundation_embeddings_model_id  ON public.foundation_embeddings (model_id);
CREATE INDEX IF NOT EXISTS idx_foundation_embeddings_user_id   ON public.foundation_embeddings (user_id);
CREATE INDEX IF NOT EXISTS idx_foundation_embeddings_created_at ON public.foundation_embeddings (created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.foundation_embeddings TO authenticated;
GRANT ALL ON TABLE public.foundation_embeddings TO service_role;

ALTER TABLE public.foundation_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own foundation embeddings"
  ON public.foundation_embeddings FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own foundation embeddings"
  ON public.foundation_embeddings FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own foundation embeddings"
  ON public.foundation_embeddings FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

-- Tier-2 cosine ANN RPC — vector(200) counterpart to `match_embeddings`.
-- Same shape/GRANTs as the Tier-1 RPC so callers can swap the table + RPC name.
CREATE OR REPLACE FUNCTION public.match_foundation_embeddings(
  query_embedding vector(200),
  match_count     INT      DEFAULT 10,
  filter_model_id TEXT     DEFAULT NULL,
  filter_user_id  UUID     DEFAULT NULL
)
RETURNS TABLE (
  id          UUID,
  similarity  FLOAT8,
  metadata    JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    f.id,
    1 - (f.embedding <=> query_embedding) AS similarity,
    f.metadata
  FROM public.foundation_embeddings f
  WHERE (filter_model_id IS NULL OR f.model_id = filter_model_id)
    AND (filter_user_id IS NULL OR f.user_id = filter_user_id)
  ORDER BY f.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_foundation_embeddings TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_foundation_embeddings TO service_role;

-- T-036: Exact (brute-force) cosine search over the 200-D foundation space.
-- Mirrors match_embeddings_exact: on L2-normalised embeddings, squared-Euclidean
-- (<\#>) is monotonic to cosine distance, so ordering is identical to a cosine
-- scan while avoiding an extra `vector_cosine_distance()` call per row.
create or replace function public.match_foundation_embeddings_exact(
  query_embedding vector(200),
  match_count     INT      DEFAULT 10,
  filter_model_id TEXT     DEFAULT NULL,
  filter_user_id  UUID     DEFAULT NULL
)
RETURNS TABLE (
  id          UUID,
  similarity  FLOAT8,
  metadata    JSONB
)
LANGUAGE plpgsql
SECURITY DEFINER
STABLE
AS $$
begin
  return query
  select
    f.id,
    1 - (f.embedding <#> query_embedding) as similarity,
    f.metadata
  from public.foundation_embeddings f
  where (filter_model_id is null or f.model_id = filter_model_id)
    and (filter_user_id is null or f.user_id = filter_user_id)
  order by f.embedding <#> query_embedding asc
  limit match_count;
end;
$$;

comment on function public.match_foundation_embeddings_exact is
  'Exact (brute-force) cosine similarity search over CBraMod 200-D foundation embeddings.';

REVOKE ALL ON FUNCTION public.match_foundation_embeddings_exact FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.match_foundation_embeddings_exact TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_foundation_embeddings_exact TO service_role;
