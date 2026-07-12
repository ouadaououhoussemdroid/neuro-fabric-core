-- T-019 — Dataset manifest + DVC-lite metadata table.
-- Tracks reproducibility metadata for every dataset the platform trains on or
-- evaluates against. Required by any audit-bearing benchmark (T-017, T-010).

CREATE TABLE IF NOT EXISTS public.datasets (
  id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID        REFERENCES auth.users(id) ON DELETE SET NULL,
  -- Human-readable name (e.g. "BCI-IV-2a").
  name              TEXT        NOT NULL UNIQUE,
  -- License identifier (e.g. "BSD-3-Clause", "CC-BY-4.0").
  license           TEXT        NOT NULL,
  -- SHA-256 of the raw dataset archive (for integrity verification).
  raw_sha256        TEXT,
  -- Source URL where the dataset was acquired.
  source_url        TEXT,
  -- Number of subjects in the dataset.
  n_subjects        INT,
  -- Number of channels.
  n_channels        INT,
  -- Sample rate in Hz.
  sample_rate       INT,
  -- Number of classes (for classification datasets).
  n_classes         INT,
  -- SHA-256 of the preprocessing pipeline output (reproducibility).
  preprocessing_sha256 TEXT,
  -- Free-form metadata (paradigm, recording type, etc.).
  metadata          JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_datasets_name        ON public.datasets (name);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id     ON public.datasets (user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_raw_sha256   ON public.datasets (raw_sha256);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.datasets TO authenticated;
GRANT ALL ON public.datasets TO service_role;

ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;

-- Datasets are publicly readable (they're reference metadata, not user data).
CREATE POLICY "Anyone can view datasets"
  ON public.datasets FOR SELECT TO authenticated USING (true);

CREATE POLICY "Users can insert datasets"
  ON public.datasets FOR INSERT TO authenticated
  WITH CHECK (true);

CREATE POLICY "Users can update datasets"
  ON public.datasets FOR UPDATE TO authenticated
  USING (true);

CREATE POLICY "Users can delete datasets"
  ON public.datasets FOR DELETE TO authenticated
  USING (true);
