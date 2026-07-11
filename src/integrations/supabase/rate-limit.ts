import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "./types";

export interface RateLimitResult {
  allowed: boolean;
  retryAfterMs: number;
}

/**
 * Durable, cross-isolate rate limiting via the `check_rate_limit` Postgres
 * function (see supabase/migrations/20260711050000_rate_limits.sql). This
 * app builds against nitro's cloudflare-module preset — a distributed
 * multi-isolate edge runtime — so a plain in-memory counter is bypassable
 * by a client routed to a different isolate. The atomic UPSERT in the SQL
 * function makes the count race-free across concurrent requests.
 */
export async function checkRateLimit(
  supabase: SupabaseClient<Database>,
  userId: string,
  maxRequests: number,
  windowSeconds: number,
): Promise<RateLimitResult> {
  const { data, error } = await supabase.rpc("check_rate_limit", {
    p_user_id: userId,
    p_max_requests: maxRequests,
    p_window_seconds: windowSeconds,
  });
  if (error || !data || data.length === 0) {
    throw new Error(error?.message ?? "rate limit check returned no rows");
  }
  const row = data[0];
  return { allowed: row.allowed, retryAfterMs: Number(row.retry_after_ms) };
}
