-- experiments
CREATE TABLE public.experiments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  description text,
  tags text[] NOT NULL DEFAULT '{}',
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.experiments TO authenticated;
GRANT ALL ON public.experiments TO service_role;
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users view own experiments" ON public.experiments
  FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users insert own experiments" ON public.experiments
  FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users update own experiments" ON public.experiments
  FOR UPDATE TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "Users delete own experiments" ON public.experiments
  FOR DELETE TO authenticated USING (auth.uid() = user_id);

CREATE INDEX experiments_user_created_idx ON public.experiments(user_id, created_at DESC);

-- experiment_runs
CREATE TABLE public.experiment_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_id uuid NOT NULL REFERENCES public.experiments(id) ON DELETE CASCADE,
  name text,
  status text NOT NULL DEFAULT 'pending',
  params jsonb NOT NULL DEFAULT '{}'::jsonb,
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  notes text,
  duration_ms integer,
  created_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.experiment_runs TO authenticated;
GRANT ALL ON public.experiment_runs TO service_role;
ALTER TABLE public.experiment_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users view own runs" ON public.experiment_runs
  FOR SELECT TO authenticated USING (
    EXISTS (SELECT 1 FROM public.experiments e WHERE e.id = experiment_id AND e.user_id = auth.uid())
  );
CREATE POLICY "Users insert own runs" ON public.experiment_runs
  FOR INSERT TO authenticated WITH CHECK (
    EXISTS (SELECT 1 FROM public.experiments e WHERE e.id = experiment_id AND e.user_id = auth.uid())
  );
CREATE POLICY "Users update own runs" ON public.experiment_runs
  FOR UPDATE TO authenticated USING (
    EXISTS (SELECT 1 FROM public.experiments e WHERE e.id = experiment_id AND e.user_id = auth.uid())
  );
CREATE POLICY "Users delete own runs" ON public.experiment_runs
  FOR DELETE TO authenticated USING (
    EXISTS (SELECT 1 FROM public.experiments e WHERE e.id = experiment_id AND e.user_id = auth.uid())
  );

CREATE INDEX experiment_runs_experiment_created_idx ON public.experiment_runs(experiment_id, created_at DESC);

-- updated_at trigger for experiments
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

CREATE TRIGGER experiments_set_updated_at
  BEFORE UPDATE ON public.experiments
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
