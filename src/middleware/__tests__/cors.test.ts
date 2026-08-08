import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { getCorsOrigin, getCorsHeaders, handleCors, getCorsHeadersForResponse } from "../cors";

describe("CORS middleware", () => {
  const ORIGIN_A = "https://app.neuro-fabric.com";
  const ORIGIN_B = "https://neuro-fabric.vercel.app";
  const ORIGIN_BLOCKED = "https://evil-site.com";

  beforeEach(() => {
    // Set allowed origins for testing.
    process.env.CORS_ALLOWED_ORIGINS = `${ORIGIN_A},${ORIGIN_B}`;
  });

  afterEach(() => {
    delete process.env.CORS_ALLOWED_ORIGINS;
  });

  describe("getCorsOrigin", () => {
    it("returns the origin when it is in the allowed list", () => {
      expect(getCorsOrigin(ORIGIN_A)).toBe(ORIGIN_A);
      expect(getCorsOrigin(ORIGIN_B)).toBe(ORIGIN_B);
    });

    it("returns null when the origin is not allowed", () => {
      expect(getCorsOrigin(ORIGIN_BLOCKED)).toBeNull();
    });

    it("returns null when origin is null", () => {
      expect(getCorsOrigin(null)).toBeNull();
    });
  });

  describe("getCorsHeaders", () => {
    it("returns headers with allowed origin for permitted requests", () => {
      const headers = getCorsHeaders(ORIGIN_A);
      expect(headers).not.toBeNull();
      expect(headers!["Access-Control-Allow-Origin"]).toBe(ORIGIN_A);
      expect(headers!["Access-Control-Allow-Credentials"]).toBe("true");
      expect(headers!["Access-Control-Allow-Methods"]).toContain("POST");
      expect(headers!["Access-Control-Allow-Headers"]).toContain("Authorization");
    });

    it("returns null for disallowed origins", () => {
      expect(getCorsHeaders(ORIGIN_BLOCKED)).toBeNull();
    });

    it("returns null when CORS_ALLOWED_ORIGINS is empty", () => {
      delete process.env.CORS_ALLOWED_ORIGINS;
      expect(getCorsHeaders(ORIGIN_A)).toBeNull();
    });
  });

  describe("handleCors", () => {
    it("returns 204 for OPTIONS preflight with allowed origin", () => {
      const req = new Request("http://localhost/api/health", {
        method: "OPTIONS",
        headers: { origin: ORIGIN_A },
      });
      const res = handleCors(req);
      expect(res).not.toBeNull();
      expect(res!.status).toBe(204);
      expect(res!.headers.get("Access-Control-Allow-Origin")).toBe(ORIGIN_A);
    });

    it("returns 403 for OPTIONS preflight with disallowed origin", () => {
      const req = new Request("http://localhost/api/health", {
        method: "OPTIONS",
        headers: { origin: ORIGIN_BLOCKED },
      });
      const res = handleCors(req);
      expect(res).not.toBeNull();
      expect(res!.status).toBe(403);
    });

    it("returns null for same-origin GET requests (let handler proceed)", () => {
      const req = new Request("http://localhost/api/health", {
        method: "GET",
      });
      const res = handleCors(req);
      expect(res).toBeNull();
    });

    it("returns 403 for non-OPTIONS requests from blocked origins", () => {
      const req = new Request("http://localhost/api/health", {
        method: "GET",
        headers: { origin: ORIGIN_BLOCKED },
      });
      const res = handleCors(req);
      expect(res).not.toBeNull();
      expect(res!.status).toBe(403);
      expect(res!.headers.get("content-type")).toBe("application/json");
    });

    it("returns null for GET requests from allowed origins", () => {
      const req = new Request("http://localhost/api/health", {
        method: "GET",
        headers: { origin: ORIGIN_A },
      });
      const res = handleCors(req);
      expect(res).toBeNull();
    });
  });

  describe("getCorsHeadersForResponse", () => {
    it("returns CORS headers for allowed origins", () => {
      const headers = getCorsHeadersForResponse(ORIGIN_A);
      expect(headers["Access-Control-Allow-Origin"]).toBe(ORIGIN_A);
    });

    it("returns Vary header only for disallowed origins", () => {
      const headers = getCorsHeadersForResponse(ORIGIN_BLOCKED);
      expect(headers["Vary"]).toBe("Origin");
      expect(headers["Access-Control-Allow-Origin"]).toBeUndefined();
    });
  });
});
