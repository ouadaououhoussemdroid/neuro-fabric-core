-- M28 — Joint-2312 4-block embedding storage (CBraMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048).
--
-- Additive only. This migration creates a DEDICATED vector(2312) namespace for the
-- M27-learned 4-block fused joint embedding [CBraMod-200×0.3062 ⊕ V2-32×0.1434 ⊕
-- PCA-32×0.1519 ⊕ EEGPT-2048×0.3985 → 2312-D].
--
-- It does NOT modify:
--   - Tier-1 `embeddings` table (vector(32))
--   - Tier-2 `foundation_embeddings` table (vector(200))
--   - M25 `joint_embeddings` table (vector(264))
--   - Any prior migration or RPC.
--
-- M27 validated Joint-2312: R@5=0.8527 (Δ=+0.0669 vs Joint-264, p=4.8e-28,
-- Cohen's d=0.704). Adding EEGPT as a 4th fusion block significantly improves
-- session-disjoint retrieval quality over the production Joint-264 baseline.
--
-- Block weights (M27 learned, stable across all 50 LOSO folds, CV < 0.5%):
--   CBraMod = 0.3062, V2 = 0.1434, PCA = 0.1519, EEGPT = 0.3985  (sum ≈ 1.00)
-- Each block is L2-normalised before weighted concatenation; the final 2312-D
-- vector is L2-normalised after concatenation.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.joint_embeddings_2312 (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  -- The fused model that produced this embedding. Tagged explicitly so 2312-D
  -- rows can never collide with the Tier-1 `embeddings` (vector(32)), the
  -- Tier-2 `foundation_embeddings` (vector(200)), or the M25 `joint_embeddings`
  -- (vector(264)) namespace.
  model_id      TEXT        NOT NULL DEFAULT 'onnx-cbramod-joint-2312',
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- Joint-2312 2312-D embedding: CBraMod-200 ⊕ V2-32 ⊕ PCA-32 ⊕ EEGPT-2048,
  -- block-weighted with M27-learned weights, L2-normalised.
  -- Hard-checked to 2312 so any other dimension is a DB error, mirroring the
  -- Tier-1 CHECK(vector_dims=32), Tier-2 CHECK(vector_dims=200), and
  -- M25 CHECK(vector_dims=264) contracts.
  embedding     vector(2312) NOT NULL,
  embedding_dim INT         NOT NULL DEFAULT 2312,
  -- Provenance / source metadata: artifact shas, block weights, window info,
  -- source channel counts, preprocessing params.
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT joint_embeddings_2312_dims CHECK (vector_dims(embedding) = 2312)
);

-- IVFFlat index for approximate nearest-neighbour cosine search (2312-D).
CREATE INDEX IF NOT EXISTS idx_joint_embedding_2312_ivfflat
  ON public.joint_embeddings_2312
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Supporting indexes for filtered / model-versioned search.
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_model_id  ON public.joint_embeddings_2312 (model_id);
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_user_id     ON public.joint_embeddings_2312 (user_id);
CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_created_at  ON public.joint_embeddings_2312 (created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.joint_embeddings_2312 TO authenticated;
GRANT ALL ON TABLE public.joint_embeddings_2312 TO service_role;

ALTER TABLE public.joint_embeddings_2312 ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own joint-2312 embeddings"
  ON public.joint_embeddings_2312 FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own joint-2312 embeddings"
  ON public.joint_embeddings_2312 FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own joint-2312 embeddings"
  ON public.joint_embeddings_2312 FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

-- M28 cosine ANN RPC — vector(2312) counterpart to `match_joint_embeddings` and
-- `match_foundation_embeddings`. Same shape/GRANTs so callers can swap the
-- table + RPC name.
CREATE OR REPLACE FUNCTION public.match_joint_embeddings_2312(
  query_embedding vector(2312),
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
  FROM public.joint_embeddings_2312 j
  WHERE (filter_model_id IS NULL OR j.model_id = filter_model_id)
    AND (filter_user_id IS NULL OR j.user_id = filter_user_id)
  ORDER BY j.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_2312 TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_2312 TO service_role;

-- Exact (brute-force) cosine search over the 2312-D joint-2312 space.
-- Mirrors match_joint_embeddings_exact: on L2-normalised embeddings,
-- squared-Euclidean (<#>) is monotonic to cosine distance, so ordering is identical
-- to a cosine scan while avoiding an extra vector_cosine_distance() call per row.
create or replace function public.match_joint_embeddings_2312_exact(
  query_embedding vector(2312),
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
  from public.joint_embeddings_2312 j
  where (filter_model_id is null or j.model_id = filter_model_id)
    and (filter_user_id is null or j.user_id = filter_user_id)
  order by j.embedding <#> query_embedding asc
  limit match_count;
end;
$$;

comment on function public.match_joint_embeddings_2312_exact is
  'Exact (brute-force) cosine similarity search over the 2312-D joint-2312 [CBraMod-200⊕V2-32⊕PCA-32⊕EEGPT-2048] embeddings.';

REVOKE ALL ON FUNCTION public.match_joint_embeddings_2312_exact FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_2312_exact TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_joint_embeddings_2312_exact TO service_role;
