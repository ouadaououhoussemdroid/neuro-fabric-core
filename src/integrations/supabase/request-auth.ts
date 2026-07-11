import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "./types";

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
  const SUPABASE_URL = process.env.SUPABASE_URL;
  const SUPABASE_PUBLISHABLE_KEY = process.env.SUPABASE_PUBLISHABLE_KEY;
  if (!SUPABASE_URL || !SUPABASE_PUBLISHABLE_KEY) {
    throw new AuthError("Server misconfigured: Supabase env vars missing", 500);
  }

  const authHeader = request.headers.get("authorization") ?? "";
  if (!authHeader.startsWith("Bearer ")) {
    throw new AuthError("Unauthorized: missing Bearer token", 401);
  }
  const token = authHeader.slice("Bearer ".length).trim();
  if (!token) {
    throw new AuthError("Unauthorized: empty token", 401);
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
