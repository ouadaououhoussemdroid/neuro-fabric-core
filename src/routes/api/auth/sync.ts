/**
 * T-005: Server endpoint to sync browser session to HttpOnly cookie.
 *
 * Called by `syncSessionToServer()` in cookie-auth.server.ts whenever the
 * Supabase client emits a session change event (SIGNED_IN, SIGNED_OUT, etc.).
 *
 * - POST /api/auth/sync?event=SIGNED_IN   → sets cookie with session body
 * - POST /api/auth/sync?event=SIGNED_OUT  → clears cookie
 *
 * The session JSON is sent in the POST body (not query params) for security.
 * The cookie is HttpOnly, Secure (in prod), SameSite=Strict — unreachable
 * from client-side JavaScript, mitigating XSS token theft.
 */
import { defineEventHandler, type EventHandler } from "h3";
import { setSessionCookie, clearSessionCookie, type SessionCookie } from "@/integrations/supabase/cookie-auth.server";
import { log } from "@/lib/logging";

export default defineEventHandler(async (event) => {
  const method = event.node.req.method;

  if (method !== "POST") {
    event.node.res.statusCode = 405;
    return { error: "Method not allowed" };
  }

  const eventParam = event.node.req.url?.split("?")[1];
  const params = new URLSearchParams(eventParam);
  const authEvent = params.get("event");

  if (!authEvent) {
    event.node.res.statusCode = 400;
    return { error: "Missing 'event' query parameter" };
  }

  const headers = event.node.res.getHeader("Set-Cookie");

  if (authEvent === "SIGNED_OUT") {
    clearSessionCookie(event.node.res);
    log("info", "auth.cookie_cleared", { event: authEvent });
    return { ok: true };
  }

  if (authEvent === "SIGNED_IN" || authEvent === "TOKEN_REFRESHED") {
    // Read session from POST body
    const body = await event.node.req
      .on("data", (chunk: Buffer) => chunk)
      .then(() => null)
      .catch(() => null);

    // Parse body from the raw Node request
    let session: SessionCookie | null = null;
    try {
      const chunks: Buffer[] = [];
      event.node.req.on("data", (chunk: Buffer) => chunks.push(chunk));
      await new Promise<void>((resolve, reject) => {
        event.node.req.on("end", resolve);
        event.node.req.on("error", reject);
      });
      const rawBody = Buffer.concat(chunks).toString("utf-8");
      if (rawBody) {
        session = JSON.parse(rawBody) as SessionCookie;
      }
    } catch (e) {
      log("warn", "auth.cookie_sync_parse_error", { error: (e as Error).message });
    }

    if (!session?.access_token) {
      event.node.res.statusCode = 400;
      return { error: "Missing session body" };
    }

    setSessionCookie(event.node.res, session);
    log("info", "auth.cookie_set", { event: authEvent, userId: session.user?.id });
    return { ok: true };
  }

  // TOKEN_REFRESHED without a new session still means the cookie is valid
  return { ok: true };
}) as EventHandler;
