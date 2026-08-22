import { createMiddleware } from "@tanstack/react-start";
import { supabase } from "./client";
import { syncSessionToServer, type SessionCookie } from "./cookie-auth.server";

// Must be registered as a global `functionMiddleware` in `src/start.ts`; otherwise
// the browser never attaches the bearer token to serverFn RPCs.
export const attachSupabaseAuth = createMiddleware({ type: "function" }).client(
  async ({ next }) => {
    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;

    // T-005: Sync session to HttpOnly cookie after any session change.
    // This ensures the server-side cookie stays in sync with the browser session.
    if (data.session) {
      const cookieSession: SessionCookie = {
        access_token: data.session.access_token,
        refresh_token: data.session.refresh_token,
        user: {
          id: data.session.user.id,
          email: data.session.user.email,
        },
        expires_at: data.session.expires_at ?? 0,
      };
      syncSessionToServer("SIGNED_IN", cookieSession);
    }

    return next({
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  },
);
