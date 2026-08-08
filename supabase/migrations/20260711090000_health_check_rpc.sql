/**
 * T-PR-001 — Lightweight health-check RPC.
 *
 * Called by the /api/health endpoint to verify database connectivity.
 * Returns a single row with value 1 — an atomic no-op that confirms the
 * database is reachable and queryable.
 */
create or replace function public.health_check()
returns table (ok boolean)
language plpgsql
security definer
volatile
as $$
begin
  return query select true as ok;
end;
$$;

comment on function public.health_check is
  'Lightweight health check used by the /api/health endpoint to verify DB connectivity.';

REVOKE ALL ON FUNCTION public.health_check FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.health_check TO authenticated;
