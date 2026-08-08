/**
 * Controlled CORS middleware.
 *
 * Only allows requests from origins configured in the `CORS_ALLOWED_ORIGINS`
 * environment variable (comma-separated). Never uses a wildcard (`*`).
 * Handles OPTIONS preflight requests with appropriate caching headers.
 */

/** Parse comma-separated origins from the environment variable. */
function getAllowedOrigins(): string[] {
  const raw = process.env.CORS_ALLOWED_ORIGINS ?? "";
  return raw
    .split(",")
    .map((o) => o.trim())
    .filter(Boolean);
}

/** Determine whether the request Origin should be allowed. */
export function getCorsOrigin(requestOrigin: string | null): string | null {
  if (!requestOrigin) return null;
  const allowed = getAllowedOrigins();
  if (allowed.includes(requestOrigin)) {
    return requestOrigin;
  }
  return null;
}

/** CORS headers for a given allowed origin (or null if rejected). */
export function getCorsHeaders(requestOrigin: string | null): Record<string, string> | null {
  const origin = getCorsOrigin(requestOrigin);
  if (!origin) return null;
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
    "Access-Control-Max-Age": "86400", // 24 hours
  };
}

/**
 * Handle CORS for a request. Returns a Response for OPTIONS (preflight),
 * or null to let the request proceed normally.
 */
export function handleCors(request: Request): Response | null {
  const origin = request.headers.get("origin");
  const headers = getCorsHeaders(origin);

  // Always respond to OPTIONS preflight — even for rejected origins
  // (return 403 so the browser knows the origin is not allowed).
  if (request.method === "OPTIONS") {
    if (headers) {
      return new Response(null, { status: 204, headers });
    }
    return new Response(null, {
      status: 403,
      headers: { Vary: "Origin" },
    });
  }

  // For actual requests, if the origin is rejected, return a 403 response.
  if (origin && !headers) {
    return new Response(JSON.stringify({ error: "CORS: origin not allowed" }), {
      status: 403,
      headers: {
        "content-type": "application/json",
        Vary: "Origin",
      },
    });
  }

  // Origin is allowed (or no origin header — same-origin request).
  // Return null to let the handler proceed, with CORS headers to merge.
  return null;
}

/** Get CORS headers to merge into an existing response. */
export function getCorsHeadersForResponse(requestOrigin: string | null): Record<string, string> {
  const headers = getCorsHeaders(requestOrigin);
  return headers ?? { Vary: "Origin" };
}
