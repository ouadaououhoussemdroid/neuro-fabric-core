/**
 * T-010: Auth rate limiting middleware.
 *
 * Applies durable rate limiting to authentication endpoints (signin, signup,
 * password reset) to prevent brute-force and credential-stuffing attacks.
 *
 * Uses the same cross-isolate `check_rate_limit` Postgres function as the
 * embedding endpoints, but with different thresholds per auth action:
 *
 *   - signin: 5 attempts per 60s per IP+email
 *   - signup: 3 attempts per 60s per IP
 *   - password reset: 3 attempts per 60s per IP+email
 *
 * On the server, we identify the user by their email (pre-auth) or IP address.
 * Since auth rate limits apply before authentication (the user doesn't have
 * a Supabase user_id yet), we hash the email/IP to a deterministic pseudo-id
 * and use that as the rate-limit key.
 */
import { type SupabaseClient } from "@supabase/supabase-js";
import { createHash } from "node:crypto";
import type { Database } from "./types";

export interface AuthRateLimitResult {
  allowed: boolean;
  retryAfterMs: number;
}

/**
 * Rate limit thresholds for each auth action.
 * The key is used to derive a deterministic pseudo-user-id for the rate limiter.
 */
export const AUTH_RATE_LIMITS = {
  signin: { maxRequests: 5, windowSeconds: 60 },
  signup: { maxRequests: 3, windowSeconds: 60 },
  "password-reset": { maxRequests: 3, windowSeconds: 60 },
} as const;

/**
 * Derive a deterministic pseudo user_id (UUID-formatted) from an email or IP.
 * The rate_limits table uses auth.users(id) as the FK, but auth endpoints
 * don't have a user_id yet. We hash the identifier and format it as a UUID
 * so the check_rate_limit function's UPSERT works (it expects a UUID PK).
 */
export function deriveRateLimitId(identifier: string): string {
  const hash = createHash("sha256").update(identifier).digest("hex");
  // Format first 32 hex chars as a UUID v5-like string
  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20, 32)}`;
}

/**
 * Check rate limit for an auth action.
 * @param supabase - Authenticated Supabase client (service role)
 * @param action - Which auth action to rate limit
 * @param identifier - Email or IP address to rate-limit by
 */
export async function checkAuthRateLimit(
  supabase: SupabaseClient<Database>,
  action: keyof typeof AUTH_RATE_LIMITS,
  identifier: string,
): Promise<AuthRateLimitResult> {
  const { maxRequests, windowSeconds } = AUTH_RATE_LIMITS[action];
  const rateLimitId = deriveRateLimitId(`${action}:${identifier}`);

  const { data, error } = await supabase.rpc("check_rate_limit", {
    p_user_id: rateLimitId,
    p_max_requests: maxRequests,
    p_window_seconds: windowSeconds,
  });

  if (error || !data || data.length === 0) {
    throw new Error(error?.message ?? "rate limit check returned no rows");
  }

  const row = data[0];
  return {
    allowed: row.allowed,
    retryAfterMs: Number(row.retry_after_ms),
  };
}

/**
 * Rate-limit error thrown when an auth action is rate-limited.
 */
export class AuthRateLimitError extends Error {
  retryAfterMs: number;

  constructor(retryAfterMs: number) {
    super(`Rate limit exceeded. Try again in ${Math.ceil(retryAfterMs / 1000)} seconds.`);
    this.name = "AuthRateLimitError";
    this.retryAfterMs = retryAfterMs;
  }
}
