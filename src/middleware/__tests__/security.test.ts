import { describe, it, expect } from "vitest";
import { SECURITY_HEADERS, applySecurityHeaders, secureJson } from "../security";

describe("Security headers middleware", () => {
  it("defines all OWASP-recommended headers", () => {
    const keys = Object.keys(SECURITY_HEADERS);
    expect(keys).toContain("Content-Security-Policy");
    expect(keys).toContain("Strict-Transport-Security");
    expect(keys).toContain("X-Frame-Options");
    expect(keys).toContain("X-Content-Type-Options");
    expect(keys).toContain("Referrer-Policy");
    expect(keys).toContain("Permissions-Policy");
  });

  it("CSP allows same-origin (covers /ort/) and wasm-unsafe-eval", () => {
    const csp = SECURITY_HEADERS["Content-Security-Policy"];
    expect(csp).toContain("'self'");
    // /ort/ is same-origin, so 'self' in connect-src covers WASM fetching.
    // No wildcard origins allowed.
    expect(csp).toContain("'wasm-unsafe-eval'");
  });

  it("HSTS is set for 1 year with subdomains and preload", () => {
    const hsts = SECURITY_HEADERS["Strict-Transport-Security"];
    expect(hsts).toContain("max-age=31536000");
    expect(hsts).toContain("includeSubDomains");
    expect(hsts).toContain("preload");
  });

  it("X-Frame-Options is DENY (not SAMEORIGIN)", () => {
    expect(SECURITY_HEADERS["X-Frame-Options"]).toBe("DENY");
  });

  it("X-Content-Type-Options is nosniff", () => {
    expect(SECURITY_HEADERS["X-Content-Type-Options"]).toBe("nosniff");
  });

  it("Permissions-Policy blocks sensitive APIs", () => {
    const pp = SECURITY_HEADERS["Permissions-Policy"];
    expect(pp).toContain("camera=()");
    expect(pp).toContain("microphone=()");
    expect(pp).toContain("geolocation=()");
  });

  it("applySecurityHeaders preserves status and body", async () => {
    const original = new Response("hello", { status: 200 });
    const response = applySecurityHeaders(original);
    expect(response.status).toBe(200);
    expect(await response.text()).toBe("hello");
    expect(response.headers.get("X-Frame-Options")).toBe("DENY");
    expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
  });

  it("applySecurityHeaders merges extra headers", () => {
    const original = new Response("test");
    const response = applySecurityHeaders(original, {
      "X-Custom": "value",
      "content-type": "application/json",
    });
    expect(response.headers.get("X-Custom")).toBe("value");
    expect(response.headers.get("content-type")).toBe("application/json");
  });

  it("secureJson sets content-type and all security headers", async () => {
    const response = secureJson({ ok: true }, { status: 200 });
    expect(response.headers.get("content-type")).toBe("application/json");
    expect(response.headers.get("X-Frame-Options")).toBe("DENY");
    expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(response.headers.get("Content-Security-Policy")).toBeTruthy();
    const body = await response.json();
    expect(body.ok).toBe(true);
  });

  it("CSP does not use wildcard origin", () => {
    const csp = SECURITY_HEADERS["Content-Security-Policy"];
    // The CSP should never have `default-src *` or similar wildcard origins.
    expect(csp).not.toMatch(/default-src\s+\*/);
    expect(csp).not.toMatch(/script-src\s+\*\s/);
  });
});
