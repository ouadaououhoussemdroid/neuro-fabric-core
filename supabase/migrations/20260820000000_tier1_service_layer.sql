-- M32 — Tier-1 Service Layer database schema.
--
-- Additive only. This migration creates downstream result tables for the
-- three Tier-1 services (Subject Identity, Cognitive State, Anomaly Detection),
-- an audit-log table, and supporting indexes/RPCs.
--
-- It does NOT modify:
--   - `embeddings`           (vector(32))    — Tier-1 V2, untouched
--   - `foundation_embeddings` (vector(200))  — Tier-2 CBraMod, untouched
--   - `joint_embeddings`      (vector(264))  — M25 3-block, untouched
--   - `joint_embeddings_2312` (vector(2312)) — M28 4-block, untouched (read-only)
--   - Any prior migration or RPC.
--
-- Each result table has a foreign key to `joint_embeddings_2312(id)`, ensuring
-- Embed-Once-Reuse-Many: downstream results reference the exact Joint-2312
-- embedding that produced them, rather than recomputing.

CREATE EXTENSION IF NOT EXISTS vector;

-- =======================================================================
-- Subject Similarity Results (Subject Identity & Cohort Similarity Service)
-- =======================================================================

CREATE TABLE IF NOT EXISTS public.subject_similarity_results (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- FK to the Joint-2312 embedding that was the query source.
  embedding_id  UUID        NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  -- The service head used (e.g. "subject-identity-similarity-v1").
  model_id      TEXT        NOT NULL,
  -- Operation type: identify / session-similarity / cohort-similarity
  query_type    TEXT        NOT NULL CHECK (query_type IN (
    'subject_identification', 'session_similarity', 'cohort_similarity'
  )),
  -- Rank of this match (1 = top match, 2 = next, etc.).
  rank          INT,
  -- Cosine similarity score in [0, 1] (embeddings are L2-normalised).
  similarity    FLOAT8 CHECK (similarity >= 0 AND similarity <= 1),
  -- Confidence = normalized (top-1 − top-2) similarity gap.
  confidence    FLOAT8 CHECK (confidence >= 0 AND confidence <= 1),
  -- The matched subject's ID (user-defined label, not PII).
  matched_subject_id TEXT,
  -- Ground-truth label (for evaluation only; NULL in production).
  is_true_match BOOLEAN,
  -- Arbitrary metadata (query parameters, subject/session IDs, etc.).
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_subject_sim_results_user_id
  ON public.subject_similarity_results (user_id);
CREATE INDEX IF NOT EXISTS idx_subject_sim_results_embedding_id
  ON public.subject_similarity_results (embedding_id);
CREATE INDEX IF NOT EXISTS idx_subject_sim_results_query_type
  ON public.subject_similarity_results (query_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subject_sim_results_model_id
  ON public.subject_similarity_results (model_id);

ALTER TABLE public.subject_similarity_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own subject similarity results"
  ON public.subject_similarity_results FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own subject similarity results"
  ON public.subject_similarity_results FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own subject similarity results"
  ON public.subject_similarity_results FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

GRANT SELECT, INSERT, DELETE ON TABLE public.subject_similarity_results TO authenticated;
GRANT ALL ON TABLE public.subject_similarity_results TO service_role;

-- =======================================================================
-- Cognitive State Results (Cognitive State Intelligence Service)
-- =======================================================================

CREATE TABLE IF NOT EXISTS public.cognitive_state_results (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- FK to the Joint-2312 embedding used for this prediction.
  embedding_id  UUID        NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  -- The task-head model id (e.g. "cognitive-linear-v1").
  model_id      TEXT        NOT NULL,
  -- Soft-metric scores in [0, 1].
  attention     FLOAT8 CHECK (attention >= 0 AND attention <= 1),
  workload      FLOAT8 CHECK (workload >= 0 AND workload <= 1),
  arousal       FLOAT8 CHECK (arousal >= 0 AND arousal <= 1),
  -- Mean confidence across the 3 metrics.
  confidence    FLOAT8 CHECK (confidence >= 0 AND confidence <= 1),
  -- Confidence intervals as ARRAY[2] (lower, upper).
  attention_ci  FLOAT8[2],
  workload_ci   FLOAT8[2],
  arousal_ci    FLOAT8[2],
  model_version TEXT,
  -- Training dataset for the head (for provenance).
  dataset       TEXT,
  -- Arbitrary metadata (window index, epoch, etc.).
  metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cognitive_results_user_id
  ON public.cognitive_state_results (user_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_results_embedding_id
  ON public.cognitive_state_results (embedding_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_results_model_id
  ON public.cognitive_state_results (model_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_results_created_at
  ON public.cognitive_state_results (created_at DESC);

ALTER TABLE public.cognitive_state_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own cognitive results"
  ON public.cognitive_state_results FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own cognitive results"
  ON public.cognitive_state_results FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own cognitive results"
  ON public.cognitive_state_results FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

GRANT SELECT, INSERT, DELETE ON TABLE public.cognitive_state_results TO authenticated;
GRANT ALL ON TABLE public.cognitive_state_results TO service_role;

-- =======================================================================
-- Anomaly Detection Results (EEG Anomaly Detection Service)
-- =======================================================================

CREATE TABLE IF NOT EXISTS public.anomaly_detection_results (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  -- FK to the Joint-2312 embedding analysed.
  embedding_id  UUID        NOT NULL REFERENCES joint_embeddings_2312(id) ON DELETE CASCADE,
  -- The detection model/method id (e.g. "anomaly-mahalanobis-v1").
  model_id      TEXT        NOT NULL,
  -- Z-score (MAD-normalised Mahalanobis distance) in embedding space.
  anomaly_score FLOAT8,
  -- Raw Mahalanobis distance² before z-score normalisation.
  raw_distance  FLOAT8,
  -- Calibration threshold used for the z-score.
  threshold     FLOAT8,
  -- Whether this window was flagged as anomalous.
  is_anomaly    BOOLEAN,
  -- Per-block contribution to the anomaly score (for interpretability).
  block_contribution_cbramod FLOAT8,
  block_contribution_v2      FLOAT8,
  block_contribution_pca     FLOAT8,
  block_contribution_eegpt   FLOAT8,
  confidence    FLOAT8 CHECK (confidence >= 0 AND confidence <= 1),
  model_version TEXT,
  dataset       TEXT,
  metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_results_user_id
  ON public.anomaly_detection_results (user_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_results_embedding_id
  ON public.anomaly_detection_results (embedding_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_results_model_id
  ON public.anomaly_detection_results (model_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_results_is_anomaly
  ON public.anomaly_detection_results (is_anomaly, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_results_score
  ON public.anomaly_detection_results (anomaly_score);

ALTER TABLE public.anomaly_detection_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own anomaly results"
  ON public.anomaly_detection_results FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own anomaly results"
  ON public.anomaly_detection_results FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own anomaly results"
  ON public.anomaly_detection_results FOR DELETE TO authenticated
  USING (auth.uid() = user_id);

GRANT SELECT, INSERT, DELETE ON TABLE public.anomaly_detection_results TO authenticated;
GRANT ALL ON TABLE public.anomaly_detection_results TO service_role;

-- =======================================================================
-- Service Audit Log (all Tier-1 services)
-- =======================================================================

CREATE TABLE IF NOT EXISTS public.service_audit_log (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  service       TEXT        NOT NULL,
  action        TEXT        NOT NULL,
  resource_id   TEXT,
  model         TEXT,
  status        TEXT,  -- 'success' | 'error' | 'rate_limited'
  latency_ms    FLOAT8,
  error_type    TEXT,
  error_message TEXT,
  client_ip     TEXT,
  metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_service_audit_user_id
  ON public.service_audit_log (user_id);
CREATE INDEX IF NOT EXISTS idx_service_audit_service
  ON public.service_audit_log (service);
CREATE INDEX IF NOT EXISTS idx_service_audit_created_at
  ON public.service_audit_log (created_at DESC);

ALTER TABLE public.service_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own service audit log"
  ON public.service_audit_log FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

GRANT SELECT, INSERT ON TABLE public.service_audit_log TO authenticated;
GRANT ALL ON TABLE public.service_audit_log TO service_role;

-- =======================================================================
-- Subject metadata (for identification/enrollment)
-- =======================================================================

CREATE TABLE IF NOT EXISTS public.subject_metadata (
  id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  subject_id    TEXT        NOT NULL,
  label         TEXT,
  metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, subject_id)
);

CREATE INDEX IF NOT EXISTS idx_subject_metadata_user
  ON public.subject_metadata (user_id);
CREATE INDEX IF NOT EXISTS idx_subject_metadata_subject_id
  ON public.subject_metadata (user_id, subject_id);

ALTER TABLE public.subject_metadata ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own subject metadata"
  ON public.subject_metadata FOR ALL TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.subject_metadata TO authenticated;
GRANT ALL ON TABLE public.subject_metadata TO service_role;

-- =======================================================================
-- Index on joint_embeddings_2312 metadata for subject/session filtering
-- =======================================================================
-- Allows fast filtering of ANN RPC results by subject_id or session_id
-- in the metadata JSONB column.

CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_subject_id
  ON public.joint_embeddings_2312
  USING BTREE ((metadata->>'subject_id'));

CREATE INDEX IF NOT EXISTS idx_joint_embeddings_2312_session_id
  ON public.joint_embeddings_2312
  USING BTREE ((metadata->>'session_id'));

-- =======================================================================
-- RPC: match_subject_similarity
--
-- Search the subject_similarity_results table for similar results to a given
-- embedding. This is an ANN-style search but on the result space (not the
-- embedding space). Mirrors match_joint_embeddings_2312 shape.
-- =======================================================================

CREATE OR REPLACE FUNCTION public.match_subject_similarity(
  query_similarity  FLOAT8,
  match_count       INT      DEFAULT 10,
  filter_user_id    UUID     DEFAULT NULL,
  filter_query_type TEXT     DEFAULT NULL
)
RETURNS TABLE (
  id              UUID,
  embedding_id    UUID,
  rank            INT,
  similarity      FLOAT8,
  confidence      FLOAT8,
  matched_subject_id TEXT,
  metadata        JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    s.id,
    s.embedding_id,
    s.rank,
    s.similarity,
    s.confidence,
    s.matched_subject_id,
    s.metadata
  FROM public.subject_similarity_results s
  WHERE (filter_user_id IS NULL OR s.user_id = filter_user_id)
    AND (filter_query_type IS NULL OR s.query_type = filter_query_type)
    AND s.similarity >= query_similarity
  ORDER BY s.similarity DESC, s.created_at DESC
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_subject_similarity TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_subject_similarity TO service_role;

REVOKE ALL ON FUNCTION public.match_subject_similarity FROM PUBLIC;
