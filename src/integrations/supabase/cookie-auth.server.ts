/**
 * T-005: Secure cookie-based authentication for Tier-1 services.
 *
 * Migrates from localStorage token storage to HttpOnly, Secure, SameSite=Strict
 * cookies, mitigating XSS-based token theft.
 *
 * Cookie spec:
 *   - Name: __session (HttpOnly, Secure in prod, SameSite=Strict)
 *   - Max-Age: 7 days (matches Supabase refresh token lifetime)
 *   - Path: / (available to all routes)
 *
 * The session object (access_token + refresh_token + user) is stored in
 * the cookie. On every request, the cookie is read server-side and used
 * to create an authenticated Supabase client instance. This keeps tokens
 * out of client-side JavaScript (unreachable by XSS).
 *
 * For browser-side usage, the `supabase` client in client.ts uses a
 * `createBrowserCookieStorage` adapter that stores the session in memory
 * only (no localStorage), while server-side routes read from the HttpOnly
 * cookie.
 */
import { type SupabaseClient, type AuthChangeEvent } from "@supabase/supabase-js";
import type { Database } from "./types";
import { requireServerEnv } from "@/lib/env.server";
import { createClient } from "@supabase/supabase-js";

export const SESSION_COOKIE_NAME = "__session";
export const SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

export interface SessionCookie {
  access_token: string;
  refresh_token: string;
  user: {
    id: string;
    email?: string;
    [key: string]: unknown;
  };
  expires_at: number;
}

/**
 * Set the session cookie on the server response. Call this from any
 * server route that performs authentication (signin, signup, refresh).
 */
export function setSessionCookie(
  headers: Headers,
  session: SessionCookie,
): void {
  const serialized = JSON.stringify(session);
  const encoded = Buffer.from(serialized).toString("base64");

  // Build cookie with security flags
  const cookieParts = [
    `${SESSION_COOKIE_NAME}=${encoded}`,
    `Path=/`,
    `Max-Age=${SESSION_COOKIE_MAX_AGE}`,
    `SameSite=Strict`,
    `HttpOnly`,
  ];

  // Secure flag only in production (HTTPS). In dev (HTTP), we omit it
  // so cookies still work on localhost.
  if (process.env.NODE_ENV === "production") {
    cookieParts.push("Secure");
  }

  headers.append("Set-Cookie", cookieParts.join("; "));
}

/**
 * Clear the session cookie (logout). Sets an expired cookie to remove it
 * from the browser.
 */
export function clearSessionCookie(headers: Headers): void {
  const cookieParts = [
    `${SESSION_COOKIE_NAME}=`,
    `Path=/`,
    `Max-Age=0`,
    `SameSite=Strict`,
    `HttpOnly`,
  ];

  if (process.env.NODE_ENV === "production") {
    cookieParts.push("Secure");
  }

  headers.append("Set-Cookie", cookieParts.join("; "));
}

/**
 * Parse the session cookie from a request's Cookie header.
 * Returns the decoded session object, or null if the cookie is absent/invalid.
 */
export function getSessionFromCookie(req: Request): SessionCookie | null {
  const cookieHeader = req.headers.get("cookie");
  if (!cookieHeader) return null;

  // Parse cookie string
  const cookies = cookieHeader.split(";").map((c) => c.trim());
  const sessionCookie = cookies.find((c) => c.startsWith(`${SESSION_COOKIE_NAME}=`));
  if (!sessionCookie) return null;

  const encoded = sessionCookie.slice(SESSION_COOKIE_NAME.length + 1);
  if (!encoded) return null;

  try {
    const decoded = Buffer.from(encoded, "base64").toString("utf-8");
    const session = JSON.parse(decoded) as SessionCookie;

    // Check expiry
    if (session.expires_at && session.expires_at < Date.now()) {
      return null;
    }

    return session;
  } catch {
    return null;
  }
}

/**
 * Create a server-side Supabase client authenticated with the session
 * from the cookie. Bypasses localStorage entirely.
 */
export function createAuthenticatedClient(req: Request): SupabaseClient<Database> | null {
  const session = getSessionFromCookie(req);
  if (!session) return null;

  const { SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY } = requireServerEnv([
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
  ]);

  return createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
    global: {
      headers: { Authorization: `Bearer ${session.access_token}` },
    },
    auth: {
      persistSession: false,
      autoRefreshToken: false,
      storage: undefined,
    },
  });
}

/**
 * Browser-side session change handler. Call this from the client to sync
 * session changes back to the server cookie via a server action.
 *
 * This replaces direct localStorage access with a server round-trip
 * that writes an HttpOnly cookie.
 */
export async function syncSessionToServer(
  event: AuthChangeEvent,
  session: SessionCookie | null,
): Promise<void> {
  if (typeof window === "undefined") return;

  try {
    const url = session
      ? `/api/auth/sync?event=${encodeURIComponent(event)}`
      : `/api/auth/sync?event=${encodeURIComponent(event)}`;

    const body = session ? JSON.stringify(session) : undefined;

    await fetch(url, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": body ? "application/json" : "application/octet-stream",
      },
      body,
    });
  } catch (e) {
    // Non-fatal — the next request will retry
    if (typeof console !== "undefined") {
      console.warn("[cookie-auth] Failed to sync session:", (e as Error).message);
    }
  }
}

/**
 * Check if the current session has expired and needs refresh.
 * Returns true if the session expires within the next 60 seconds (buffer).
 */
export function isSessionExpiringSoon(session: SessionCookie | null): boolean {
  if (!session?.expires_at) return true;
  return session.expires_at < Date.now() + 60_000;
}
