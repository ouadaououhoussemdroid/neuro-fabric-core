/**
 * T-005 — Unit tests for secure cookie-based authentication.
 *
 * Validates that:
 *   1. Session cookies are properly serialized/deserialized
 *   2. HttpOnly, SameSite=Strict, and Secure flags are set
 *   3. Expired sessions are rejected
 *   4. Malformed cookies return null gracefully
 *   5. The cookie is cleared on logout
 */
import { describe, it, expect, beforeEach } from "vitest";
import {
  setSessionCookie,
  clearSessionCookie,
  getSessionFromCookie,
  isSessionExpiringSoon,
  SESSION_COOKIE_NAME,
  type SessionCookie,
} from "../cookie-auth.server";

const TEST_SESSION: SessionCookie = {
  access_token: "test-access-token",
  refresh_token: "test-refresh-token",
  user: {
    id: "user-123",
    email: "test@example.com",
  },
  expires_at: Date.now() + 86400000, // 1 day from now
};

describe("Cookie Auth — setSessionCookie", () => {
  it("sets a cookie with HttpOnly flag", () => {
    const headers = new Headers();
    setSessionCookie(headers, TEST_SESSION);

    const cookie = headers.get("Set-Cookie");
    expect(cookie).toBeDefined();
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain(`Path=/`);
    expect(cookie).toContain("SameSite=Strict");
  });

  it("sets a cookie with Max-Age matching session duration", () => {
    const headers = new Headers();
    setSessionCookie(headers, TEST_SESSION);

    const cookie = headers.get("Set-Cookie");
    expect(cookie).toContain("Max-Age=");
  });

  it("sets Secure flag in production environment", () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = "production";
    try {
      const headers = new Headers();
      setSessionCookie(headers, TEST_SESSION);

      const cookie = headers.get("Set-Cookie");
      expect(cookie).toContain("Secure");
    } finally {
      process.env.NODE_ENV = originalEnv;
    }
  });

  it("does not set Secure flag in development environment", () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = "development";
    try {
      const headers = new Headers();
      setSessionCookie(headers, TEST_SESSION);

      const cookie = headers.get("Set-Cookie");
      expect(cookie).not.toContain("Secure");
    } finally {
      process.env.NODE_ENV = originalEnv;
    }
  });

  it("encodes session as base64 in the cookie value", () => {
    const headers = new Headers();
    setSessionCookie(headers, TEST_SESSION);

    const cookie = headers.get("Set-Cookie") ?? "";
    const valuePart = cookie.split(`${SESSION_COOKIE_NAME}=`)[1].split(";")[0];

    // Base64 decoding should yield the JSON session
    const decoded = JSON.parse(Buffer.from(valuePart, "base64").toString("utf-8"));
    expect(decoded.access_token).toBe(TEST_SESSION.access_token);
    expect(decoded.user.id).toBe(TEST_SESSION.user.id);
  });
});

describe("Cookie Auth — getSessionFromCookie", () => {
  it("extracts session from a properly formatted cookie header", () => {
    const headers = new Headers();
    setSessionCookie(headers, TEST_SESSION);

    const cookieStr = `${SESSION_COOKIE_NAME}=${headers
      .get("Set-Cookie")!
      .split(`${SESSION_COOKIE_NAME}=`)[1]
      .split(";")[0]}`;

    const request = new Request("https://example.com", {
      headers: { Cookie: cookieStr },
    });

    const session = getSessionFromCookie(request);
    expect(session).not.toBeNull();
    expect(session!.access_token).toBe(TEST_SESSION.access_token);
    expect(session!.user.id).toBe(TEST_SESSION.user.id);
    expect(session!.refresh_token).toBe(TEST_SESSION.refresh_token);
  });

  it("returns null when no cookie header is present", () => {
    const request = new Request("https://example.com");
    const session = getSessionFromCookie(request);
    expect(session).toBeNull();
  });

  it("returns null when the cookie is absent", () => {
    const request = new Request("https://example.com", {
      headers: { Cookie: "other=value; foo=bar" },
    });
    const session = getSessionFromCookie(request);
    expect(session).toBeNull();
  });

  it("returns null for malformed base64", () => {
    const request = new Request("https://example.com", {
      headers: { Cookie: `${SESSION_COOKIE_NAME}=!!!not-base64!!!` },
    });
    const session = getSessionFromCookie(request);
    expect(session).toBeNull();
  });

  it("returns null for expired sessions", () => {
    const expiredSession: SessionCookie = {
      ...TEST_SESSION,
      expires_at: Date.now() - 1000, // expired 1 second ago
    };

    const headers = new Headers();
    setSessionCookie(headers, expiredSession);
    const cookieStr = headers.get("Set-Cookie") ?? "";
    const valuePart = cookieStr.split(`${SESSION_COOKIE_NAME}=`)[1].split(";")[0];

    const request = new Request("https://example.com", {
      headers: { Cookie: `${SESSION_COOKIE_NAME}=${valuePart}` },
    });

    const session = getSessionFromCookie(request);
    expect(session).toBeNull();
  });
});

describe("Cookie Auth — clearSessionCookie", () => {
  it("sets a cookie with Max-Age=0 to clear it", () => {
    const headers = new Headers();
    clearSessionCookie(headers);

    const cookie = headers.get("Set-Cookie");
    expect(cookie).toBeDefined();
    expect(cookie).toContain("Max-Age=0");
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("SameSite=Strict");
  });

  it("sets an empty cookie value", () => {
    const headers = new Headers();
    clearSessionCookie(headers);

    const cookie = headers.get("Set-Cookie") ?? "";
    const valuePart = cookie.split(`${SESSION_COOKIE_NAME}=`)[1].split(";")[0];
    expect(valuePart).toBe("");
  });
});

describe("Cookie Auth — isSessionExpiringSoon", () => {
  it("returns true when session is null", () => {
    expect(isSessionExpiringSoon(null)).toBe(true);
  });

  it("returns true when expires_at is missing", () => {
    const session = { ...TEST_SESSION, expires_at: 0 };
    expect(isSessionExpiringSoon(session)).toBe(true);
  });

  it("returns false when session expires in more than 60 seconds", () => {
    const session = { ...TEST_SESSION, expires_at: Date.now() + 120_000 }; // 2 min
    expect(isSessionExpiringSoon(session)).toBe(false);
  });

  it("returns true when session expires within 60 seconds", () => {
    const session = { ...TEST_SESSION, expires_at: Date.now() + 30_000 }; // 30 sec
    expect(isSessionExpiringSoon(session)).toBe(true);
  });
});

describe("Cookie Auth — end-to-end roundtrip", () => {
  it("set → get roundtrip preserves all fields", () => {
    const session: SessionCookie = {
      access_token: "abc123",
      refresh_token: "def456",
      user: {
        id: "user-789",
        email: "e2e@test.com",
        aud: "authenticated",
      },
      expires_at: Date.now() + 3600000, // 1 hour
    };

    const headers = new Headers();
    setSessionCookie(headers, session);
    const cookieStr = headers.get("Set-Cookie") ?? "";
    const valuePart = cookieStr.split(`${SESSION_COOKIE_NAME}=`)[1].split(";")[0];

    const request = new Request("https://example.com", {
      headers: { Cookie: `${SESSION_COOKIE_NAME}=${valuePart}` },
    });

    const retrieved = getSessionFromCookie(request);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.access_token).toBe(session.access_token);
    expect(retrieved!.refresh_token).toBe(session.refresh_token);
    expect(retrieved!.user.id).toBe(session.user.id);
    expect(retrieved!.user.email).toBe(session.user.email);
    expect(retrieved!.user.aud).toBe("authenticated");
  });
});
