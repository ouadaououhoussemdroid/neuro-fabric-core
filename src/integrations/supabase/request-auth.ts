import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { requireServerEnv } from "@/lib/env.server";
import type { Database } from "./types";
import { getSessionFromCookie } from "./cookie-auth.server";

/**
 * Bearer-token verification for TanStack Start file-route HTTP handlers
 * (createFileRoute(...).server.handlers.*). These are plain HTTP handlers,
 * not TanStack "server functions" — `functionMiddleware` (see
 * auth-attacher.ts) does not apply to them, so each handler must verify
 * the token itself. This is the single shared implementation; route
 * handlers should call it instead of re-implementing auth inline.
 */
export class AuthError extends Error {
  status: number;
  constructor(message: string, status = 401) {
    super(message);
    this.status = status;
  }
}

export async function authenticateRequest(
  request: Request,
): Promise<{ userId: string; supabase: SupabaseClient<Database> }> {
  let SUPABASE_URL: string, SUPABASE_PUBLISHABLE_KEY: string;
  try {
    ({ SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY } = requireServerEnv([
      "SUPABASE_URL",
      "SUPABASE_PUBLISHABLE_KEY",
    ]));
  } catch (e) {
    throw new AuthError((e as Error).message, 500);
  }

  // T-005: Check both Bearer header (from auth-attacher) and HttpOnly cookie
  // (for browser navigations or when the middleware hasn't run).
  const authHeader = request.headers.get("authorization") ?? "";
  let token: string | null = null;

  if (authHeader.startsWith("Bearer ")) {
    token = authHeader.slice("Bearer ".length).trim();
    if (!token) token = null;
  }

  // Fallback: check the session cookie
  if (!token) {
    const session = getSessionFromCookie(request);
    if (session && session.access_token) {
      token = session.access_token;
    }
  }

  if (!token) {
    throw new AuthError("Unauthorized: missing Bearer token", 401);
  }

  const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
    global: { headers: { Authorization: `Bearer ${token}` } },
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { data: userData, error: userErr } = await supabase.auth.getUser(token);
  if (userErr || !userData?.user) {
    throw new AuthError("Unauthorized: invalid token", 401);
  }

  return { userId: userData.user.id, supabase };
}
