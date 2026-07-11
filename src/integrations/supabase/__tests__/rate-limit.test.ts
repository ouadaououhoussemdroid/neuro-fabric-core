import { describe, it, expect, vi } from "vitest";
import { checkRateLimit } from "../rate-limit";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "../types";

function fakeClient(rpcImpl: (...args: unknown[]) => Promise<{ data: unknown; error: unknown }>) {
  return { rpc: vi.fn(rpcImpl) } as unknown as SupabaseClient<Database>;
}

describe("checkRateLimit", () => {
  it("calls the check_rate_limit RPC with the expected arguments", async () => {
    const rpc = vi.fn(async () => ({
      data: [{ allowed: true, retry_after_ms: 0 }],
      error: null,
    }));
    const client = { rpc } as unknown as SupabaseClient<Database>;

    await checkRateLimit(client, "user-1", 20, 60);

    expect(rpc).toHaveBeenCalledWith("check_rate_limit", {
      p_user_id: "user-1",
      p_max_requests: 20,
      p_window_seconds: 60,
    });
  });

  it("returns allowed=true with the RPC's reported values under the limit", async () => {
    const client = fakeClient(async () => ({
      data: [{ allowed: true, retry_after_ms: 0 }],
      error: null,
    }));
    const result = await checkRateLimit(client, "user-1", 20, 60);
    expect(result).toEqual({ allowed: true, retryAfterMs: 0 });
  });

  it("returns allowed=false with a positive retryAfterMs when over the limit", async () => {
    const client = fakeClient(async () => ({
      data: [{ allowed: false, retry_after_ms: 42000 }],
      error: null,
    }));
    const result = await checkRateLimit(client, "user-1", 20, 60);
    expect(result).toEqual({ allowed: false, retryAfterMs: 42000 });
  });

  it("throws when the RPC returns an error", async () => {
    const client = fakeClient(async () => ({
      data: null,
      error: { message: "connection refused" },
    }));
    await expect(checkRateLimit(client, "user-1", 20, 60)).rejects.toThrow("connection refused");
  });

  it("throws when the RPC returns no rows", async () => {
    const client = fakeClient(async () => ({ data: [], error: null }));
    await expect(checkRateLimit(client, "user-1", 20, 60)).rejects.toThrow(
      "rate limit check returned no rows",
    );
  });
});
