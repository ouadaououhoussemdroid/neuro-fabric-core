/**
 * T-012 — Exact (linear) vector search RPC.
 *
 * Companion to `match_embeddings` (which relies on the ivfflat index for
 * approximate nearest-neighbour search). This RPC performs an exact /brute-force
 * cosine-distance scan so that recall-SLO measurements can compare ANN recall
 * against a ground-truth baseline *at the database level* (rather than relying
 * solely on the in-process JS baseline).
 *
 * The distance is `1 - (embedding <#> query_embedding)` where `<#>` is the
 * pgvector squared-Euclidean operator. Because we store L2-normalised embeddings
 * (the upload pipeline normalises to unit norm), squared-Euclidean is
 * monotonically related to cosine distance:
 *   cos(a,b) = 1 - ||a-b||^2 / 2
 * so ordering by `<#>` ascending is identical to ordering by cosine distance
 * ascending. This avoids the cost of an extra `vector_cosine_distance()` call
 * per row while preserving correct ranking.
 *
 * The ivfflat index is *not* used here (PostgreSQL will plan a sequential scan),
 * making this an exact but O(n) query. For large tables you can cap the scan with
 * an optional `filter_user_id` or `filter_model_id`.
 */
create or replace function public.match_embeddings_exact(
  query_embedding vector(32),
  match_count int default 10,
  filter_model_id text default null,
  filter_user_id uuid default null
)
returns table (
  id uuid,
  similarity double precision,
  metadata jsonb
)
language plpgsql
security definer
stable
as $$
begin
  return query
  select
    e.id,
    1 - (e.embedding <#> query_embedding) as similarity,
    e.metadata
  from public.embeddings e
  where (filter_model_id is null or e.model_id = filter_model_id)
    and (filter_user_id is null or e.user_id = filter_user_id)
  order by e.embedding <#> query_embedding asc
  limit match_count;
end;
$$;

comment on function public.match_embeddings_exact is
  'Exact (brute-force) cosine similarity search over embeddings — no ivfflat index used.';

REVOKE ALL ON FUNCTION public.match_embeddings_exact FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.match_embeddings_exact TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_embeddings_exact TO service_role;
