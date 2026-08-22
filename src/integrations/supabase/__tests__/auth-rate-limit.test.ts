/**
 * T-010 — Unit tests for auth rate limiting.
 *
 * Validates that:
 *   1. Rate limit IDs are derived deterministically from emails/IPs
 *   2. Rate limit thresholds are correctly configured per action
 *   3. The check_auth_rate_limit function correctly calls check_rate_limit RPC
 *   4. AuthRateLimitError carries the correct retry-after value
 */
import { describe, it, expect, vi } from "vitest";
import {
  AUTH_RATE_LIMITS,
  deriveRateLimitId,
  checkAuthRateLimit,
  AuthRateLimitError,
} from "../auth-rate-limit";

// ─── Mock Supabase client ─────────────────────────────────────────────────────────────────
function createMockSupabase(rpcImpl: (...args: unknown[]) => Promise<{ data: unknown; error: unknown }>) {
  return {
    rpc: vi.fn(rpcImpl),
  } as unknown as Parameters<typeof checkAuthRateLimit>[0];
}

describe("Auth Rate Limiting — deriveRateLimitId", () => {
  it("produces a deterministic UUID from the same input", () => {
    const id1 = deriveRateLimitId("user@example.com");
    const id2 = deriveRateLimitId("user@example.com");
    expect(id1).toBe(id2);
  });

  it("produces different UUIDs for different inputs", () => {
    const id1 = deriveRateLimitId("user1@example.com");
    const id2 = deriveRateLimitId("user2@example.com");
    expect(id1).not.toBe(id2);
  });

  it("produces a UUID-formatted string", () => {
    const id = deriveRateLimitId("test@example.com");
    expect(id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("includes the action prefix in the derivation", () => {
    const signinId = deriveRateLimitId("signin:user@example.com");
    const signupId = deriveRateLimitId("signup:user@example.com");
    expect(signinId).not.toBe(signupId);
  });
});

describe("Auth Rate Limiting — AUTH_RATE_LIMITS", () => {
  it("has stricter limits for signup than signin", () => {
    expect(AUTH_RATE_LIMITS.signup.maxRequests).toBeLessThanOrEqual(AUTH_RATE_LIMITS.signin.maxRequests);
  });

  it("has reasonable limits for all auth actions", () => {
    expect(AUTH_RATE_LIMITS.signin).toEqual({ maxRequests: 5, windowSeconds: 60 });
    expect(AUTH_RATE_LIMITS.signup).toEqual({ maxRequests: 3, windowSeconds: 60 });
    expect(AUTH_RATE_LIMITS["password-reset"]).toEqual({ maxRequests: 3, windowSeconds: 60 });
  });

  it("all windows are 60 seconds", () => {
    for (const config of Object.values(AUTH_RATE_LIMITS)) {
      expect(config.windowSeconds).toBe(60);
    }
  });
});

describe("Auth Rate Limiting — checkAuthRateLimit", () => {
  it("calls check_rate_limit RPC with correct parameters for signin", async () => {
    const rpc = vi.fn(async () => ({
      data: [{ allowed: true, retry_after_ms: 0 }],
      error: null,
    }));
    const client = { rpc } as unknown as Parameters<typeof checkAuthRateLimit>[0];

    const result = await checkAuthRateLimit(client, "signin", "user@example.com");

    expect(rpc).toHaveBeenCalledWith("check_rate_limit", {
      p_user_id: expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/),
      p_max_requests: 5,
      p_window_seconds: 60,
    });
    expect(result.allowed).toBe(true);
    expect(result.retryAfterMs).toBe(0);
  });

  it("calls check_rate_limit RPC with correct parameters for signup", async () => {
    const rpc = vi.fn(async () => ({
      data: [{ allowed: true, retry_after_ms: 0 }],
      error: null,
    }));
    const client = { rpc } as unknown as Parameters<typeof checkAuthRateLimit>[0];

    await checkAuthRateLimit(client, "signup", "user@example.com");

    expect(rpc).toHaveBeenCalledWith("check_rate_limit", {
      p_user_id: expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/),
      p_max_requests: 3,
      p_window_seconds: 60,
    });
  });

  it("returns allowed=false with retry_after_ms when rate-limited", async () => {
    const client = createMockSupabase(async () => ({
      data: [{ allowed: false, retry_after_ms: 35000 }],
      error: null,
    }));

    const result = await checkAuthRateLimit(client, "signin", "user@example.com");

    expect(result.allowed).toBe(false);
    expect(result.retryAfterMs).toBe(35000);
  });

  it("throws when RPC returns an error", async () => {
    const client = createMockSupabase(async () => ({
      data: null,
      error: { message: "connection refused" },
    }));

    await expect(
      checkAuthRateLimit(client, "signin", "user@example.com"),
    ).rejects.toThrow("connection refused");
  });

  it("throws when RPC returns no data", async () => {
    const client = createMockSupabase(async () => ({
      data: [],
      error: null,
    }));

    await expect(
      checkAuthRateLimit(client, "signin", "user@example.com"),
    ).rejects.toThrow("no rows");
  });

  it("uses IP address as identifier when email is not available", async () => {
    const client = createMockSupabase(async () => ({
      data: [{ allowed: true, retry_after_ms: 0 }],
      error: null,
    }));

    // IP-based rate limiting for signup (email not yet provided)
    await checkAuthRateLimit(client, "signup", "192.168.1.100");

    // Should have been called with a hash of "signup:192.168.1.100"
    const call = (client as unknown as { rpc: ReturnType<typeof vi.fn> }).rpc.mock.calls[0];
    expect(call[0]).toBe("check_rate_limit");
    expect(call[1].p_max_requests).toBe(3);
  });
});

describe("Auth Rate Limiting — AuthRateLimitError", () => {
  it("creates an error with retry-after value", () => {
    const error = new AuthRateLimitError(42000);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("AuthRateLimitError");
    expect(error.retryAfterMs).toBe(42000);
    expect(error.message).toContain("42");
  });

  it("formats message in seconds", () => {
    const error = new AuthRateLimitError(5000);
    expect(error.message).toContain("5");
  });

  it("can be thrown and caught", () => {
    expect(() => {
      throw new AuthRateLimitError(10000);
    }).toThrow(AuthRateLimitError);
  });
});
