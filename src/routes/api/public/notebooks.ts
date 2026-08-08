/**
 * T-026 — Notebook portal API route.
 *
 * Serves the notebook manifest at `/api/public/notebooks`. The manifest is
 * built by `scripts/build_notebook_portal.py` and written to
 * `public/research/notebooks/manifest.json`. Individual notebooks are
 * served as static HTML at `/research/notebooks/:id.html`.
 */
import { createFileRoute } from "@tanstack/react-router";
import { handleCors, getCorsHeadersForResponse } from "@/middleware/cors";
import { applySecurityHeaders } from "@/middleware/security";

export const Route = createFileRoute("/api/public/notebooks")({
  server: {
    handlers: {
      GET: async ({ request }) => {
        // CORS pre-flight / origin check.
        const corsResponse = handleCors(request);
        if (corsResponse) return corsResponse;

        const origin = request.headers.get("origin");
        try {
          const response = await fetch("/research/notebooks/manifest.json");
          if (!response.ok) {
            return json(
              { error: "Notebook manifest not found. Run scripts/build_notebook_portal.py." },
              404,
              origin,
            );
          }
          const manifest = await response.json();
          return json(manifest, 200, origin);
        } catch (err) {
          return json(
            { error: `Failed to load notebook manifest: ${(err as Error).message}` },
            500,
            origin,
          );
        }
      },
    },
  },
});

function json(body: unknown, status = 200, origin: string | null = null): Response {
  const corsHeaders = getCorsHeadersForResponse(origin);
  return applySecurityHeaders(
    new Response(JSON.stringify(body), {
      status,
      headers: {
        "content-type": "application/json",
        ...corsHeaders,
      },
    }),
  );
}
