-- T-013 — Concept graph schema (subject → session → window → embedding → label).
--
-- Minimal property-graph schema on Postgres using `ltree` + the `embeddings`
-- FK (from T-011) to support provenance queries before adopting a dedicated
-- graph DB (Apache AGE reserved for the long-term scalable path).
--
-- Hierarchy:
--   subject → session → window → embedding → label
--
-- Each node carries an `ltree` path encoding its position in the hierarchy,
-- enabling recursive CTE provenance traversals without a full graph engine.

CREATE EXTENSION IF NOT EXISTS ltree;

-- --- Subject node ---
CREATE TABLE IF NOT EXISTS public.graph_subjects (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  subject_code TEXT       NOT NULL,        -- e.g. "B01" from BCI-IV-2a
  dataset     TEXT        NOT NULL,        -- e.g. "bci-iv-2a"
  metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  path        LTREE       NOT NULL DEFAULT text2ltree(''),  -- root: empty path
  UNIQUE (user_id, dataset, subject_code)
);

-- --- Session node ---
CREATE TABLE IF NOT EXISTS public.graph_sessions (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_id   UUID        NOT NULL REFERENCES public.graph_subjects(id) ON DELETE CASCADE,
  session_code TEXT        NOT NULL,        -- e.g. "T1" (session 1, training)
  metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  path         LTREE       NOT NULL DEFAULT text2ltree(''),
  UNIQUE (subject_id, session_code)
);

-- --- Window node ---
CREATE TABLE IF NOT EXISTS public.graph_windows (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      UUID        NOT NULL REFERENCES public.graph_sessions(id) ON DELETE CASCADE,
  embedding_id    UUID        REFERENCES public.embeddings(id) ON DELETE SET NULL,
  window_index    INT         NOT NULL,       -- index within the session
  start_sample    INT         NOT NULL,
  end_sample      INT         NOT NULL,
  sample_rate     INT         NOT NULL,
  label           TEXT,                       -- class label (e.g. "left_hand")
  metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  path            LTREE       NOT NULL DEFAULT text2ltree(''),
  UNIQUE (session_id, window_index)
);

-- --- Label node (normalised, shared across windows) ---
CREATE TABLE IF NOT EXISTS public.graph_labels (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset     TEXT        NOT NULL,
  label       TEXT        NOT NULL,
  description TEXT,
  UNIQUE (dataset, label)
);

-- --- Edge: window → label ---
CREATE TABLE IF NOT EXISTS public.graph_window_labels (
  window_id  UUID NOT NULL REFERENCES public.graph_windows(id) ON DELETE CASCADE,
  label_id   UUID NOT NULL REFERENCES public.graph_labels(id) ON DELETE CASCADE,
  PRIMARY KEY (window_id, label_id)
);

-- --- ltree path triggers ---
-- Populate path on insert: subject.path = subject_code,
-- session.path = subject.path.session_code, window.path = session.path.window_index.
CREATE OR REPLACE FUNCTION public._graph_subject_set_path()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.path := text2ltree(NEW.subject_code);
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public._graph_session_set_path()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  SELECT path FROM public.graph_subjects WHERE id = NEW.subject_id INTO NEW.path;
  IF NEW.path IS NULL THEN
    NEW.path := text2ltree('');
  END IF;
  NEW.path := NEW.path || text2ltree(NEW.session_code);
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public._graph_window_set_path()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  SELECT path FROM public.graph_sessions WHERE id = NEW.session_id INTO NEW.path;
  IF NEW.path IS NULL THEN
    NEW.path := text2ltree('');
  END IF;
  NEW.path := NEW.path || text2ltree(NEW.window_index::text);
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_graph_subjects_path ON public.graph_subjects;
CREATE TRIGGER trg_graph_subjects_path
  BEFORE INSERT ON public.graph_subjects
  FOR EACH ROW EXECUTE FUNCTION public._graph_subject_set_path();

DROP TRIGGER IF EXISTS trg_graph_sessions_path ON public.graph_sessions;
CREATE TRIGGER trg_graph_sessions_path
  BEFORE INSERT ON public.graph_sessions
  FOR EACH ROW EXECUTE FUNCTION public._graph_session_set_path();

DROP TRIGGER IF EXISTS trg_graph_windows_path ON public.graph_windows;
CREATE TRIGGER trg_graph_windows_path
  BEFORE INSERT ON public.graph_windows
  FOR EACH ROW EXECUTE FUNCTION public._graph_window_set_path();

-- --- Indexes for provenance queries ---
CREATE INDEX IF NOT EXISTS idx_graph_subjects_path    ON public.graph_subjects USING GIST (path);
CREATE INDEX IF NOT EXISTS idx_graph_sessions_path     ON public.graph_sessions USING GIST (path);
CREATE INDEX IF NOT EXISTS idx_graph_windows_path      ON public.graph_windows USING GIST (path);
CREATE INDEX IF NOT EXISTS idx_graph_windows_embedding  ON public.graph_windows (embedding_id);
CREATE INDEX IF NOT EXISTS idx_graph_windows_session    ON public.graph_windows (session_id);
CREATE INDEX IF NOT EXISTS idx_graph_window_labels_win  ON public.graph_window_labels (window_id);
CREATE INDEX IF NOT EXISTS idx_graph_window_labels_lab  ON public.graph_window_labels (label_id);

-- --- Recursive CTE: full provenance for an embedding ---
-- Given an embedding_id, returns the full chain: subject → session → window → labels.
CREATE OR REPLACE FUNCTION public.get_embedding_provenance(p_embedding_id UUID)
RETURNS TABLE (
  subject_code TEXT,
  dataset      TEXT,
  session_code TEXT,
  window_index INT,
  labels       TEXT[]
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    s.subject_code,
    s.dataset,
    ses.session_code,
    w.window_index,
    COALESCE(
      ARRAY_AGG(l.label) FILTER (WHERE l.label IS NOT NULL),
      ARRAY[]::TEXT[]
    ) AS labels
  FROM public.graph_windows w
  JOIN public.graph_sessions ses ON ses.id = w.session_id
  JOIN public.graph_subjects s ON s.id = ses.subject_id
  LEFT JOIN public.graph_window_labels wl ON wl.window_id = w.id
  LEFT JOIN public.graph_labels l ON l.id = wl.label_id
  WHERE w.embedding_id = p_embedding_id
  GROUP BY s.subject_code, s.dataset, ses.session_code, w.window_index;
$$;

-- --- Grants ---
GRANT SELECT, INSERT, UPDATE, DELETE ON
  public.graph_subjects, public.graph_sessions, public.graph_windows,
  public.graph_labels, public.graph_window_labels
  TO authenticated;
GRANT ALL ON
  public.graph_subjects, public.graph_sessions, public.graph_windows,
  public.graph_labels, public.graph_window_labels
  TO service_role;

ALTER TABLE public.graph_subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_windows   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_labels    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graph_window_labels ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own subjects"
  ON public.graph_subjects FOR SELECT TO authenticated
  USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own subjects"
  ON public.graph_subjects FOR INSERT TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own sessions"
  ON public.graph_sessions FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.graph_subjects s WHERE s.id = subject_id AND s.user_id = auth.uid()));
CREATE POLICY "Users can insert own sessions"
  ON public.graph_sessions FOR INSERT TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM public.graph_subjects s WHERE s.id = subject_id AND s.user_id = auth.uid()));

CREATE POLICY "Users can view own windows"
  ON public.graph_windows FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.graph_sessions ses
    JOIN public.graph_subjects s ON s.id = ses.subject_id
    WHERE ses.id = session_id AND s.user_id = auth.uid()
  ));
CREATE POLICY "Users can insert own windows"
  ON public.graph_windows FOR INSERT TO authenticated
  WITH CHECK (EXISTS (
    SELECT 1 FROM public.graph_sessions ses
    JOIN public.graph_subjects s ON s.id = ses.subject_id
    WHERE ses.id = session_id AND s.user_id = auth.uid()
  ));

CREATE POLICY "Users can view own window_labels"
  ON public.graph_window_labels FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.graph_windows w
    JOIN public.graph_sessions ses ON ses.id = w.session_id
    JOIN public.graph_subjects s ON s.id = ses.subject_id
    WHERE w.id = window_id AND s.user_id = auth.uid()
  ));

CREATE POLICY "Users can view labels"
  ON public.graph_labels FOR SELECT TO authenticated USING (true);
