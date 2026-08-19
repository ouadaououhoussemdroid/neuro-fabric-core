-- M25 — Joint 264-D embedding storage (CBraMod-200 ⊕ EEGConformer-V2-32 ⊕ PCA-32).
--
-- Additive only. This migration creates a DEDICATED vector(264) namespace for the
-- block-weighted joint embedding [CBraMod-200×0.62 ⊕ V2-32×0.16 ⊕ PCA-32×0.22].
-- It does NOT modify the Tier-1 `embeddings` table (vector(32)), its CHECK contract,
-- the `match_embeddings` RPC, the Tier-2 `foundation_embeddings` table (vector(200)),
-- or any prior migration. V2 / DEFAULT_PREFERRED / vector(32) contract are preserved
-- byte-for-byte. M18 proved this joint space with R@5=0.7856 (p=4.5e-9).
--
-- Block weights are fixed (stable across all 50 LOSO folds in M18):
--   CBraMod = 0.62, V2 = 0.16, PCA = 0.22  (sum = 1.00)
-- Each block is L2-normalised before weighted concatenation; the final 264-D
-- vector is L2-normalised after concatenation.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.joint_embeddings (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  -- The fused model that produced this embedding. Tagged explicitly so 264-D
  -- rows can never collide with the Tier-1 `embeddings` (vector(32)) or the
  -- Tier-2 `foundation_embeddings` (vector(200)) namespace.
  model_id      TEXT        NOT NULL DEFAULT 'onnx-cbramod-joint-264',
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- Joint 264-D embedding: CBraMod-200 ⊕ V2-32 ⊕ PCA-32, block-weighted, L2-normalised.
  -- Hard-checked to 264 so a 200-D (or any other dim) vector is a DB error,
  -- mirroring the Tier-1 CHECK(vector_dims=32) and Tier-2 CHECK(vector_dims=200).
  embedding     vector(264) NOT NULL,
  embedding_dim INT         NOT NULL DEFAULT 264,
  -- Provenance / source metadata: artifact shas, block weights, window info,
  -- source channel counts, preprocessing params.
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT joint_embeddings_dims CHECK (vector_dims(embedding) = 264)
);

-- IVFFlat index for approximate nearest-neighbour cosine search (264-D).
CREATE INDEX IF NOT EXISTS idx_joint_embedding_ivfflat
  ON public.joint_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Supporting indexes for filtered / model-versioned search.
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_model_id  ON public.joint_embeddings (model_id);
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_user_id   ON public.joint_embeddings (user_id);
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_created_at ON public.joint_embeddings (created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.joint_embeddings TO authenticated;
GRANT ALL ON TABLE public.joint_embeddings TO service_role;

ALTER TABLE public.joint_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own joint embeddings"
  ON public.joint_embeddings FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own joint embeddings"
  ON public.joint_embeddings FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own joint embeddings"
  ON public.joint_embeddings FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

-- M25 cosine ANN RPC — vector(264) counterpart to `match_embeddings` and
-- `match_foundation_embeddings`. Same shape/GRANTs so callers can swap the
-- table + RPC name.
CREATE OR REPLACE FUNCTION public.match_joint_embeddings(
  query_embedding vector(264),
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
    j.id,
    1 - (j.embedding <=> query_embedding) AS similarity,
    j.metadata
  FROM public.joint_embeddings j
  WHERE (filter_model_id IS NULL OR j.model_id = filter_model_id)
    AND (filter_user_id IS NULL OR j.user_id = filter_user_id)
  ORDER BY j.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_joint_embeddings TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings TO service_role;

-- Exact (brute-force) cosine search over the 264-D joint space.
-- Mirrors match_foundation_embeddings_exact: on L2-normalised embeddings,
-- squared-Euclidean (<#>) is monotonic to cosine distance, so ordering is identical
-- to a cosine scan while avoiding an extra vector_cosine_distance() call per row.
create or replace function public.match_joint_embeddings_exact(
  query_embedding vector(264),
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
    j.id,
    1 - (j.embedding <#> query_embedding) as similarity,
    j.metadata
  from public.joint_embeddings j
  where (filter_model_id is null or j.model_id = filter_model_id)
    and (filter_user_id is null or j.user_id = filter_user_id)
  order by j.embedding <#> query_embedding asc
  limit match_count;
end;
$$;

comment on function public.match_joint_embeddings_exact is
  'Exact (brute-force) cosine similarity search over the 264-D joint [CBraMod-200⊕V2-32⊕PCA-32] embeddings.';

REVOKE ALL ON FUNCTION public.match_joint_embeddings_exact FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_exact TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_exact TO service_role;
