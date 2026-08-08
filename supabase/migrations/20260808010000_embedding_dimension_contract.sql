/**
 * T-011 / Fix 2 — Canonical embedding dimension contract.
 *
 * Enforces at the database level that every vector stored in `public.embeddings`
 * is exactly 32-dimensional, matching:
 *   - EEGConformer output head: embedding[0,32]
 *   - `embeddings.embedding vector(32)` (see 20260711060000_pgvector_embeddings.sql)
 *   - PCA legacy adapter (aligned from 64 → 32)
 *
 * Previously, the PCA adapter emitted 64-dim vectors while the column was
 * `vector(32)`; pgvector silently truncated or rejected writes depending on
 * the operator, and the `NeuralVectorIndex.add()` path swallowed the mismatch
 * error and fell back to in-memory storage. This migration adds an explicit
 * CHECK constraint so any dimension mismatch is a hard DB error that cannot
 * be silently swallowed by application code.
 */
-- Only add the constraint if it doesn't already exist (idempotent).
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'embeddings_dim_32'
      AND conrelid = 'public.embeddings'::regclass
  ) THEN
    ALTER TABLE public.embeddings
      ADD CONSTRAINT embeddings_dim_32 CHECK (vector_dims(embedding) = 32);
  END IF;
END $$;

COMMENT ON CONSTRAINT embeddings_dim_32 ON public.embeddings IS
  'Canonical embedding dimension = 32 (EEGConformer head; PCA legacy aligned from 64).';
