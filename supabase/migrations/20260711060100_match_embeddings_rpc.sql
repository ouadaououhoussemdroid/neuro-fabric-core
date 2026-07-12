-- T-011 — pgvector match_embeddings RPC for ANN search.
-- Called by NeuralVectorIndex.search() to perform cosine similarity search
-- with optional model_id and user_id filtering.

CREATE OR REPLACE FUNCTION public.match_embeddings(
  query_embedding vector(32),
  match_count     INT     DEFAULT 10,
  filter_model_id TEXT    DEFAULT NULL,
  filter_user_id  UUID    DEFAULT NULL
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
    e.id,
    1 - (e.embedding <=> query_embedding) AS similarity,
    e.metadata
  FROM public.embeddings e
  WHERE (filter_model_id IS NULL OR e.model_id = filter_model_id)
    AND (filter_user_id IS NULL OR e.user_id = filter_user_id)
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_embeddings TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_embeddings TO service_role;
