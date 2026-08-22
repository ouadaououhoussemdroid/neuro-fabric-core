/**
 * M42 Phase 2b — Production-like Rate Limiting for sleep quality route.
 *
 * Extends M15 Phase 2 to POST /api/joint2312/sleep/quality. Same real Supabase
 * stack, real check_rate_limit RPC, real JWT auth. Only decodeSleepQuality mocked.
 *
 * Verifies: 20 within budget → 200, 21st → 429 with retry_after_ms, per-user
 * isolation, concurrent bypass blocked, invalid JWTs get 401 not 429.
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

// ── 2. Mock ONLY decodeSleepQuality ─────────────────────────────────────────
const mockDecodeSleepQuality = vi.fn();
vi.mock("@/lib/ai/inference/sleep.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/sleep.server")>();
  return {
    ...actual,
    decodeSleepQuality: (...args: unknown[]) => mockDecodeSleepQuality(...args),
  };
});

// ── 3. Real imports ────────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../quality");
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
  return JSON.stringify({ query_embedding: embedding, query_type: "sleep-quality" });
}

type PostHandler = (ctx: {
  request: Request;
  context: unknown;
}) => Promise<Response>;

function callSleepQuality(body: string, jwt?: string): Promise<Response> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({
    request: new Request("http://localhost:3000/api/joint2312/sleep/quality", {
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

describe("M42 Phase 2b: Production-like Rate Limiting — sleep quality route", () => {
  beforeEach(async () => {
    await resetDB();
    mockDecodeSleepQuality.mockReset();
    mockDecodeSleepQuality.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-quality-v1",
      head_version: "0.1.0",
      provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-quality-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: [{ score: 0.75, band: "good", confidence_interval: [0.65, 0.85], confidence: 0.8, metric: "sleep-quality" }],
      metadata: { embedding_reused: false, probe_sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f" },
      timings: { inference_ms: 0.42, total_ms: 1.0 },
    });
  });

  it("accepts 20 within budget, rejects 21st with 429 + retry_after_ms", async () => {
    const statuses: number[] = [];

    for (let i = 0; i < 20; i++) {
      const res = await callSleepQuality(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      statuses.push(res.status);
    }
    expect(statuses).toHaveLength(20);
    expect(statuses.every((s) => s === 200)).toBe(true);

    mockDecodeSleepQuality.mockClear();

    const res21 = await callSleepQuality(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(res21.status).toBe(429);
    const body21 = await res21.json();
    expect(body21.retry_after_ms).toBeTypeOf("number");
    expect(body21.retry_after_ms).toBeGreaterThan(0);
    expect(mockDecodeSleepQuality).not.toHaveBeenCalled();
  }, 60_000);

  it("per-user isolation: user B budget independent from user A", async () => {
    for (let i = 0; i < 20; i++) {
      const res = await callSleepQuality(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      expect(res.status).toBe(200);
    }
    const resA21 = await callSleepQuality(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(resA21.status).toBe(429);

    // User B still full budget
    const resB = await callSleepQuality(jsonBody(fakeJoint2312Embedding(200)), _tokens.userB.jwt);
    expect(resB.status).toBe(200);

    for (let i = 0; i < 19; i++) {
      const res = await callSleepQuality(jsonBody(fakeJoint2312Embedding(201 + i)), _tokens.userB.jwt);
      expect(res.status).toBe(200);
    }

    const resB21 = await callSleepQuality(jsonBody(fakeJoint2312Embedding(300)), _tokens.userB.jwt);
    expect(resB21.status).toBe(429);
  }, 120_000);

  it("concurrent bypass blocked: 21 simultaneous → only 20 succeed", async () => {
    const promises = Array.from({ length: 21 }, (_, i) =>
      callSleepQuality(
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

  it("invalid JWTs get 401, not 429 — budget untouched", async () => {
    const statuses: number[] = [];
    for (let i = 0; i < 22; i++) {
      const res = await callSleepQuality(jsonBody(fakeJoint2312Embedding(i)), _tokens.invalidJWT);
      statuses.push(res.status);
    }
    expect(statuses.every((s) => s === 401)).toBe(true);

    const resA = await callSleepQuality(jsonBody(fakeJoint2312Embedding(500)), _tokens.userA.jwt);
    expect(resA.status).toBe(200);
  }, 60_000);
}, 120_000);
