ALTER TABLE public.experiment_runs
  ADD COLUMN IF NOT EXISTS user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  ADD COLUMN IF NOT EXISTS analysis_id uuid REFERENCES public.eeg_analyses(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS completed_at timestamptz;

CREATE INDEX IF NOT EXISTS experiment_runs_user_idx ON public.experiment_runs(user_id);
CREATE INDEX IF NOT EXISTS experiment_runs_analysis_idx ON public.experiment_runs(analysis_id);