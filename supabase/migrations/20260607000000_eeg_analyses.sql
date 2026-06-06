CREATE TABLE IF NOT EXISTS public.eeg_analyses (
  id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id              UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  file_name            TEXT        NOT NULL,
  file_size_bytes      INT         NOT NULL,
  sample_rate          INT         NOT NULL,
  num_channels         INT         NOT NULL,
  num_samples          INT         NOT NULL,
  embedding            FLOAT8[]    NOT NULL,
  embedding_dimensions INT         NOT NULL,
  embedding_model      TEXT        NOT NULL,
  attention            FLOAT8      NOT NULL CHECK (attention BETWEEN 0 AND 1),
  workload             FLOAT8      NOT NULL CHECK (workload  BETWEEN 0 AND 1),
  arousal              FLOAT8      NOT NULL CHECK (arousal   BETWEEN 0 AND 1),
  bandpass_low         FLOAT8,
  bandpass_high        FLOAT8,
  notch_frequency      INT         CHECK (notch_frequency IN (50, 60)),
  processing_time_ms   INT         NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_eeg_analyses_user_id    ON public.eeg_analyses (user_id);
CREATE INDEX idx_eeg_analyses_created_at ON public.eeg_analyses (created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.eeg_analyses TO authenticated;
GRANT ALL ON public.eeg_analyses TO service_role;

ALTER TABLE public.eeg_analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own analyses"
  ON public.eeg_analyses FOR SELECT TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own analyses"
  ON public.eeg_analyses FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own analyses"
  ON public.eeg_analyses FOR DELETE TO authenticated
  USING (auth.uid() = user_id);
