-- T-011 — pgvector migration of NeuralVectorIndex.
-- Creates the `embeddings` table with vector(32) + ivfflat index, model-id tagged.
-- Preserves the existing VectorIndex TS interface; the backend switches from
-- in-memory brute-force to pgvector.

-- Enable the pgvector extension (idempotent; must be allowed by the Supabase plan).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.embeddings (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- The model that produced this embedding (e.g. "eegconformer-prod", "pca-legacy-v1").
  model_id      TEXT        NOT NULL,
  -- The embedding vector. vector(32) matches the EEGConformer embedding head;
  -- other dims are stored in embedding_dimensions for traceability.
  embedding     vector(32)  NOT NULL,
  embedding_dim INT         NOT NULL DEFAULT 32,
  -- Optional metadata payload (source file, subject, session, etc.).
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat index for approximate nearest-neighbour search.
-- lists = sqrt(rows) is the standard heuristic; 100 is a reasonable default
-- for the expected dataset size (< 100k embeddings). Rebuild with a higher
-- lists value once the table grows past that.
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_ivfflat
  ON public.embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Supporting indexes for filtered search.
CREATE INDEX IF NOT EXISTS idx_embeddings_user_id   ON public.embeddings (user_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_id  ON public.embeddings (model_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON public.embeddings (created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.embeddings TO authenticated;
GRANT ALL ON public.embeddings TO service_role;

ALTER TABLE public.embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own embeddings"
  ON public.embeddings FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own embeddings"
  ON public.embeddings FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own embeddings"
  ON public.embeddings FOR DELETE TO authenticated
  USING (auth.uid() = user_id);
