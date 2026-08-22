/**
 * M42 Phase 2a — Production-like Rate Limiting for sleep staging route.
 *
 * Extends M15 Phase 2 to POST /api/joint2312/sleep/decode. Exercises the real
 * check_rate_limit RPC via PostgREST (real local Supabase stack), with real JWT
 * auth. Only decodeSleepState is mocked (heavy ONNX inference excluded).
 *
 * Verifies: 20 within budget → 200, 21st → 429 with retry_after_ms, per-user
 * isolation, concurrent bypass blocked (atomic UPSERT), invalid JWTs get 401 not
 * 429.
 *
 * Same 20 req/min budget as all Tier-2/Tier-3 routes.
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
  invalidJWT: string;
};
process.env.SUPABASE_URL = _tokens.apiUrl;
process.env.SUPABASE_PUBLISHABLE_KEY = _tokens.anonKey;
process.env.SUPABASE_SERVICE_ROLE_KEY = _tokens.serviceRoleKey;

// ── 2. Mock ONLY decodeSleepState ──────────────────────────────────────────
const mockDecodeSleepState = vi.fn();
vi.mock("@/lib/ai/inference/sleep.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/sleep.server")>();
  return {
    ...actual,
    decodeSleepState: (...args: unknown[]) => mockDecodeSleepState(...args),
  };
});

// ── 3. Real imports ────────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../decode");
const { JOINT_2312_EMBEDDING_DIM } = await import("@/lib/ai/inference/joint.server");

// ── 4. Helpers ─────────────────────────────────────────────────────────────

function fakeJoint2312Embedding(seed = 0): number[] {
  const v = Array.from({ length: JOINT_2312_EMBEDDING_DIM }, (_, i) =>
    Math.sin((i + seed) * 0.01) * 0.5,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

function jsonBody(embedding: number[]): string {
  return JSON.stringify({ query_embedding: embedding, query_type: "sleep-stages" });
}

type PostHandler = (ctx: {
  request: Request;
  context: unknown;
}) => Promise<Response>;

function callSleepDecode(body: string, jwt?: string): Promise<Response> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({
    request: new Request("http://localhost:3000/api/joint2312/sleep/decode", {
      method: "POST",
      body,
      headers,
    }),
    context: {},
  });
}

async function resetDB() {
  await admin.from("joint_embeddings_2312").delete().neq("id", "00000000-0000-0000-0000-000000000000");
  await admin.from("rate_limits").delete().neq("user_id", "00000000-0000-0000-0000-000000000000");
}

// ── 5. Tests ────────────────────────────────────────────────────────────────

describe("M42 Phase 2a: Production-like Rate Limiting — sleep staging route", () => {
  beforeEach(async () => {
    await resetDB();
    mockDecodeSleepState.mockReset();
    mockDecodeSleepState.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-staging-v1",
      head_version: "0.1.0",
      provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-staging-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: [{ stage_id: 3, stage: "N3", probabilities: [0.1, 0.1, 0.1, 0.6, 0.1], confidence: 0.6, confidence_interval: [0.52, 0.68], metric: "sleep-stages" }],
      metadata: { embedding_reused: false, probe_sha256: "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6" },
      timings: { inference_ms: 0.52, total_ms: 1.0 },
    });
  });

  it("accepts 20 within budget, rejects 21st with 429 + retry_after_ms", async () => {
    const statuses: number[] = [];

    for (let i = 0; i < 20; i++) {
      const res = await callSleepDecode(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      statuses.push(res.status);
    }
    expect(statuses).toHaveLength(20);
    expect(statuses.every((s) => s === 200)).toBe(true);

    mockDecodeSleepState.mockClear();

    const res21 = await callSleepDecode(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(res21.status).toBe(429);
    const body21 = await res21.json();
    expect(body21.retry_after_ms).toBeTypeOf("number");
    expect(body21.retry_after_ms).toBeGreaterThan(0);
    expect(mockDecodeSleepState).not.toHaveBeenCalled(); // No inference on rate-limited request
  }, 60_000);

  it("per-user isolation: user B budget independent from user A", async () => {
    // Exhaust user A's budget
    for (let i = 0; i < 20; i++) {
      const res = await callSleepDecode(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      expect(res.status).toBe(200);
    }
    const resA21 = await callSleepDecode(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(resA21.status).toBe(429);

    // User B should still have a full budget
    const resB = await callSleepDecode(jsonBody(fakeJoint2312Embedding(200)), _tokens.userB.jwt);
    expect(resB.status).toBe(200);

    for (let i = 0; i < 19; i++) {
      const res = await callSleepDecode(jsonBody(fakeJoint2312Embedding(201 + i)), _tokens.userB.jwt);
      expect(res.status).toBe(200);
    }

    // 21st for user B → 429
    const resB21 = await callSleepDecode(jsonBody(fakeJoint2312Embedding(300)), _tokens.userB.jwt);
    expect(resB21.status).toBe(429);
  }, 120_000);

  it("concurrent bypass blocked: 21 simultaneous requests → only 20 succeed", async () => {
    const promises = Array.from({ length: 21 }, (_, i) =>
      callSleepDecode(
        jsonBody(fakeJoint2312Embedding(i)),
        _tokens.userA.jwt,
      ),
    );

    const results = await Promise.all(promises);
    const statuses = results.map((r) => r.status);
    const okCount = statuses.filter((s) => s === 200).length;
    const limitedCount = statuses.filter((s) => s === 429).length;

    expect(okCount).toBe(20);
    expect(limitedCount).toBe(1);
  }, 60_000);

  it("nonexistent/invalid-user isolation: invalid JWTs get 401, not 429", async () => {
    // Make 22 requests with invalid JWT — all should be 401 (auth fails before rate-limit)
    const statuses: number[] = [];
    for (let i = 0; i < 22; i++) {
      const res = await callSleepDecode(jsonBody(fakeJoint2312Embedding(i)), _tokens.invalidJWT);
      statuses.push(res.status);
    }
    expect(statuses.every((s) => s === 401)).toBe(true);

    // User A's budget should be untouched
    const resA = await callSleepDecode(jsonBody(fakeJoint2312Embedding(500)), _tokens.userA.jwt);
    expect(resA.status).toBe(200);
  }, 60_000);
}, 120_000);
