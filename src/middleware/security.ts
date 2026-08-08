/**
 * Production security headers middleware.
 *
 * Applies OWASP-recommended security headers to every HTTP response.
 * CSP is scoped to allow same-origin resources plus the self-hosted ONNX
 * Runtime Web WASM bundle under /ort/.
 */

/** Security headers to apply to all responses. */
export const SECURITY_HEADERS = {
  "Content-Security-Policy":
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; " +
    "style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: blob:; " +
    "connect-src 'self'; " +
    // Allow WebSocket connections for LSL streaming and model loading.
    "wss://self; " +
    "font-src 'self'; " +
    "object-src 'none'; " +
    "base-uri 'self'; " +
    "frame-ancestors 'none'; " +
    "form-action 'self'; " +
    "frame-src 'self';",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
  "X-Frame-Options": "DENY",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy":
    "accelerometer=(), " +
    "camera=(), " +
    "geolocation=(), " +
    "gyroscope=(), " +
    "microphone=(), " +
    "payment=(), " +
    "usb=()",
} as const;

/**
 * Apply security headers to a Response.
 *
 * Call this in each route handler, or wrap it around your fetch handler.
 * Returns a new Response with all security headers merged.
 */
export function applySecurityHeaders(
  response: Response,
  extra: Record<string, string> = {},
): Response {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(SECURITY_HEADERS)) {
    headers.set(key, value);
  }
  for (const [key, value] of Object.entries(extra)) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

/**
 * Create a Response with security headers already applied.
 * Convenience wrapper for routes that build responses from scratch.
 */
export function secureJson(body: unknown, init: ResponseInit = {}): Response {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...init.headers,
    },
  });
  return applySecurityHeaders(response);
}
