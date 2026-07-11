-- rate_limits: durable, cross-isolate request rate limiting.
--
-- The upload API previously rate-limited requests with a plain in-memory
-- Map, which is per-isolate state. This app builds against nitro's
-- cloudflare-module preset (confirmed via the installed nitro version
-- matching @lovable.dev/vite-tanstack-config's default-preset threshold),
-- i.e. a distributed multi-isolate edge runtime, so an in-memory counter
-- is trivially bypassed by a client routed to a different isolate/PoP.
-- This table + function make the counter durable and shared.
CREATE TABLE public.rate_limits (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  window_start timestamptz NOT NULL DEFAULT now(),
  request_count integer NOT NULL DEFAULT 0
);

-- RLS is enabled with zero policies for authenticated/anon, i.e. deny-all
-- direct table access. The only sanctioned access path is the
-- SECURITY DEFINER function below, which trusts its p_user_id argument
-- because callers (see request-auth.ts) have already verified the
-- caller's JWT before invoking it.
ALTER TABLE public.rate_limits ENABLE ROW LEVEL SECURITY;
GRANT ALL ON public.rate_limits TO service_role;

CREATE OR REPLACE FUNCTION public.check_rate_limit(
  p_user_id uuid,
  p_max_requests integer,
  p_window_seconds integer
)
RETURNS TABLE(allowed boolean, retry_after_ms bigint)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_now timestamptz := now();
  v_window_start timestamptz;
  v_count integer;
BEGIN
  -- Single atomic UPSERT: the unique constraint on user_id row-locks
  -- concurrent requests for the same user, so this is race-free without
  -- an explicit application-level lock.
  INSERT INTO public.rate_limits AS rl (user_id, window_start, request_count)
  VALUES (p_user_id, v_now, 1)
  ON CONFLICT (user_id) DO UPDATE SET
    request_count = CASE
      WHEN rl.window_start <= v_now - make_interval(secs => p_window_seconds) THEN 1
      ELSE rl.request_count + 1
    END,
    window_start = CASE
      WHEN rl.window_start <= v_now - make_interval(secs => p_window_seconds) THEN v_now
      ELSE rl.window_start
    END
  RETURNING rl.window_start, rl.request_count INTO v_window_start, v_count;

  RETURN QUERY SELECT
    v_count <= p_max_requests,
    GREATEST(
      0,
      EXTRACT(EPOCH FROM (v_window_start + make_interval(secs => p_window_seconds) - v_now)) * 1000
    )::bigint;
END;
$$;

REVOKE ALL ON FUNCTION public.check_rate_limit(uuid, integer, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.check_rate_limit(uuid, integer, integer) TO authenticated;
