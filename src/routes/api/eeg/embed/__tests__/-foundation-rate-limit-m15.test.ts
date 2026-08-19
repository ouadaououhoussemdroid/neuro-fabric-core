/**
 * Mission 15 Phase 2 — Production-like Rate Limiting through the REAL route.
 *
 * Extends Mission 14 Gate 2 by exercising the rate-limit boundary through the
 * actual POST /api/eeg/embed/foundation handler (not psql), with real GoTrue
 * JWT auth + real check_rate_limit RPC via PostgREST.
 *
 * Verifies: 20 accepted → 21st 429, retry_after_ms metadata, per-user isolation,
 * concurrent bypass blocked (atomic UPSERT), nonexistent-user isolation, no
 * V2/PCA fallback on 429.
 *
 * Only embedFoundationWindows is mocked (heavy ONNX — Phase 3/4). Auth, rate
 * limit, and vector-store are all REAL.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

// ── 1. Point at the LOCAL Supabase stack ──────────────────────────────────
const _tokens = JSON.parse(
  readFileSync("reports/m15_jwt_test_tokens.json", "utf-8"),
) as {
  apiUrl: string;
  anonKey: string;
  serviceRoleKey: string;
  userA: { id: string; jwt: string };
  userB: { id: string; jwt: string };
  expiredJWT: string;
  invalidJWT: string;
};
process.env.SUPABASE_URL = _tokens.apiUrl;
process.env.SUPABASE_PUBLISHABLE_KEY = _tokens.anonKey;
process.env.SUPABASE_SERVICE_ROLE_KEY = _tokens.serviceRoleKey;

// ── 2. Mock ONLY embedFoundationWindows ──────────────────────────────────
const mockEmbedFoundationWindows = vi.fn();
vi.mock("@/lib/ai/inference/foundation.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/foundation.server")>();
  return {
    ...actual,
    embedFoundationWindows: (...args: unknown[]) =>
      mockEmbedFoundationWindows(...args),
  };
});

// ── 3. Mock embedEEG to assert no V2 fallback ──────────────────────────────
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// ── 4. Real imports ───────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../foundation");
const { FOUNDATION_EMBEDDING_DIM, FOUNDATION_MODEL_ID } = await import(
  "@/lib/ai/inference/foundation.server"
);

// ── 5. Helpers ────────────────────────────────────────────────────────────
const CBraMod19 = [
  "FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
  "F7","F8","T7","T8","P7","P8","FZ","CZ","PZ",
];

function csvFile(name = "signal.csv", rows = 2000): File {
  const lines = [CBraMod19.join(",")];
  for (let r = 0; r < rows; r++) {
    lines.push(
      CBraMod19.map((_, c) => (Math.sin((r + c) * 0.01) * 0.3).toFixed(4)).join(","),
    );
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

function makeRequest(jwt?: string): Request {
  const form = new FormData();
  form.set("file", csvFile("signal.csv", 2000));
  form.set("sampleRate", "250");
  const headers: Record<string, string> = {};
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers,
  });
}

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;
function callFoundation(request: Request) {
  const handlers = Route.options.server!.handlers as unknown as {
    POST: PostHandler;
  };
  return handlers.POST({ request, context: {} });
}

async function resetDB() {
  await admin
    .from("foundation_embeddings")
    .delete()
    .neq("id", "00000000-0000-0000-0000-000000000000");
  await admin
    .from("rate_limits")
    .delete()
    .neq("user_id", "00000000-0000-0000-0000-000000000000");
}

function fakeEmbedding() {
  const v = Array.from({ length: FOUNDATION_EMBEDDING_DIM }, (_, i) =>
    (i % 17 + 1) / 42,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return {
    vector: v.map((x) => x / norm),
    dim: FOUNDATION_EMBEDDING_DIM,
    modelId: FOUNDATION_MODEL_ID,
    durationMs: 1,
    fellBack: false,
    normalized: true,
  };
}

// ── 6. Tests ──────────────────────────────────────────────────────────────

describe("M15 Phase 2: Production-like Rate Limiting (real route + real RPC)", () => {
  beforeEach(async () => {
    await resetDB();
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockResolvedValue([fakeEmbedding()]);
    embedEEGMock.mockReset();
  });

  it("accepts 20 within budget, rejects 21st with 429 + retry_after_ms", async () => {
    const statuses: number[] = [];

    // First 20 — all should be 200
    for (let i = 0; i < 20; i++) {
      const res = await callFoundation(makeRequest(_tokens.userA.jwt));
      statuses.push(res.status);
    }
    expect(statuses).toHaveLength(20);
    expect(statuses.every((s) => s === 200)).toBe(true);
    expect(statuses.filter((s) => s === 200)).toHaveLength(20);

    // Clear mock call count AFTER the 20 accepted requests so we can assert
    // the rejected request did NOT invoke the embed path
    mockEmbedFoundationWindows.mockClear();
    embedEEGMock.mockClear();

    // 21st — should be 429
    const res21 = await callFoundation(makeRequest(_tokens.userA.jwt));
    expect(res21.status).toBe(429);
    const body21 = await res21.json();
    expect(body21.retry_after_ms).toBeTypeOf("number");
    expect(body21.retry_after_ms).toBeGreaterThan(0);
    // No V2/PCA fallback on rate-limit rejection
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  }, 60_000);

  it("per-user isolation: user B budget independent from user A", async () => {
    // Exhaust user A's budget
    for (let i = 0; i < 20; i++) {
      const res = await callFoundation(makeRequest(_tokens.userA.jwt));
      expect(res.status).toBe(200);
    }
    // 21st for user A → 429
    const resA21 = await callFoundation(makeRequest(_tokens.userA.jwt));
    expect(resA21.status).toBe(429);

    // User B should still have a full budget
    const resB = await callFoundation(makeRequest(_tokens.userB.jwt));
    expect(resB.status).toBe(200);

    // User B can do 20 more (total 20 for B)
    let bCount = 0;
    for (let i = 0; i < 19; i++) {
        const res = await callFoundation(makeRequest(_tokens.userB.jwt));
      expect(res.status).toBe(200);
      bCount++;
    }
    expect(bCount).toBe(19); // 1 + 19 = 20 total

    // 21st for user B → 429
    const resB21 = await callFoundation(makeRequest(_tokens.userB.jwt));
    expect(resB21.status).toBe(429);
  }, 120_000);

  it("concurrent bypass blocked: 21 simultaneous requests → only 20 succeed", async () => {
    // Fire 21 requests simultaneously for user A
    const promises = Array.from({ length: 21 }, (_, i) =>
      callFoundation(
        new Request("http://localhost/api/eeg/embed/foundation", {
          method: "POST",
          body: (() => {
            const form = new FormData();
            form.set("file", csvFile(`concurrent-${i}.csv`, 2000));
            form.set("sampleRate", "250");
            return form;
          })(),
          headers: { authorization: `Bearer ${_tokens.userA.jwt}` },
        }),
      ),
    );

    const results = await Promise.all(promises);
    const statuses = results.map((r) => r.status);
    const okCount = statuses.filter((s) => s === 200).length;
    const limitedCount = statuses.filter((s) => s === 429).length;

    // Exactly 20 should succeed, exactly 1 should be rate-limited
    expect(okCount).toBe(20);
    expect(limitedCount).toBe(1);
  }, 60_000);

  it("nonexistent/invalid-user isolation: invalid JWTs get 401, not 429", async () => {
    // Make 22 requests with invalid JWT — all should be 401 (auth fails before rate-limit)
    const statuses: number[] = [];
    for (let i = 0; i < 22; i++) {
      const res = await callFoundation(makeRequest(_tokens.invalidJWT));
      statuses.push(res.status);
    }
    expect(statuses.every((s) => s === 401)).toBe(true);

    // User A's budget should be untouched (invalid JWTs didn't consume it)
    const resA = await callFoundation(makeRequest(_tokens.userA.jwt));
    expect(resA.status).toBe(200); // First request for user A should succeed
  }, 60_000);

  it("no V2/PCA fallback on 429 (embedEEG never called)", async () => {
    // Exhaust user A's budget
    for (let i = 0; i < 20; i++) {
      const res = await callFoundation(makeRequest(_tokens.userA.jwt));
      expect(res.status).toBe(200);
    }
    // Clear mock call count AFTER the 20 accepted requests
    mockEmbedFoundationWindows.mockClear();
    embedEEGMock.mockClear();

    // 21st — 429
    const res = await callFoundation(makeRequest(_tokens.userA.jwt));
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error).toMatch(/rate limit/i);
    // The rejected request must NOT invoke embed or V2 fallback
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  }, 60_000);
}, 120_000);
