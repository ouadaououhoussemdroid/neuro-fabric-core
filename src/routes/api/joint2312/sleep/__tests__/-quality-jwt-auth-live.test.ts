/**
 * M42 Phase 1b — Real JWT / Authorization Validation for sleep quality route.
 *
 * Extends M15 Phase 1 to the Tier-2 sleep quality route (POST /api/joint2312/sleep/quality).
 * Same pattern: real local Supabase stack, real authenticateRequest, real RLS on
 * joint_embeddings_2312. Only decodeSleepQuality is mocked.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

// ── 1. Point the route at the LOCAL Supabase stack ──────────────────────────
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

// ── 2. Mock ONLY decodeSleepQuality ───────────────────────────────────────────
const mockDecodeSleepQuality = vi.fn();
vi.mock("@/lib/ai/inference/sleep.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/sleep.server")>();
  return {
    ...actual,
    decodeSleepQuality: (...args: unknown[]) => mockDecodeSleepQuality(...args),
  };
});

// ── 3. Real imports ──────────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../quality");
const { JOINT_2312_EMBEDDING_DIM } = await import("@/lib/ai/inference/joint.server");

// ── 4. Helpers ────────────────────────────────────────────────────────────────

function fakeJoint2312Embedding(seed = 0): number[] {
  const v = Array.from({ length: JOINT_2312_EMBEDDING_DIM }, (_, i) =>
    Math.sin((i + seed) * 0.01) * 0.5,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

async function insertEmbedding(userId: string, embedding: number[]) {
  const { data, error } = await admin.from("joint_embeddings_2312").insert([
    { user_id: userId, embedding, metadata: { test: true, source: "quality-test" } },
  ]).select("id").single();
  expect(error).toBeNull();
  return data!.id as string;
}

function jsonBody(embedding: number[]): string {
  return JSON.stringify({ query_embedding: embedding, query_type: "sleep-quality" });
}
function jsonBodyById(embeddingId: string): string {
  return JSON.stringify({ embedding_id: embeddingId, query_type: "sleep-quality" });
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

// ── 5. Tests ─────────────────────────────────────────────────────────────────

describe("M42 Phase 1b: Real JWT Authentication + RLS Isolation — sleep quality route", () => {
  const timeout = 30_000;

  beforeEach(async () => {
    await resetDB();
    mockDecodeSleepQuality.mockReset();
  });

  it("valid JWT + raw embedding → 200, quality score in [0, 1]", async () => {
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

    const embedding = fakeJoint2312Embedding(0);
    const res = await callSleepQuality(jsonBody(embedding), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.results[0].score).toBeGreaterThanOrEqual(0);
    expect(body.results[0].score).toBeLessThanOrEqual(1);
    expect(mockDecodeSleepQuality).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding, query_type: "sleep-quality" }),
      expect.anything(),
      _tokens.userA.id,
    );
  }, timeout);

  it("valid JWT + embedding_id → 200, reuses existing embedding (RLS-scoped)", async () => {
    const embedding = fakeJoint2312Embedding(1);
    const embId = await insertEmbedding(_tokens.userA.id, embedding);

    mockDecodeSleepQuality.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-quality-v1",
      head_version: "0.1.0",
      embedding_id: embId,
      provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-quality-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: [{ score: 0.42, band: "fair", confidence_interval: [0.32, 0.52], confidence: 0.65, metric: "sleep-quality" }],
      metadata: { embedding_reused: true, probe_sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f" },
      timings: { inference_ms: 0.38, total_ms: 0.9 },
    });

    const res = await callSleepQuality(jsonBodyById(embId), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.embedding_id).toBe(embId);
    expect(mockDecodeSleepQuality).toHaveBeenCalledTimes(1);
  }, timeout);

  it("no Bearer token → 401", async () => {
    const embedding = fakeJoint2312Embedding(2);
    const res = await callSleepQuality(jsonBody(embedding), undefined);
    expect(res.status).toBe(401);
    expect(mockDecodeSleepQuality).not.toHaveBeenCalled();
  }, timeout);

  it("invalid JWT (tampered signature) → 401", async () => {
    const embedding = fakeJoint2312Embedding(3);
    const res = await callSleepQuality(jsonBody(embedding), _tokens.invalidJWT);
    expect(res.status).toBe(401);
    expect(mockDecodeSleepQuality).not.toHaveBeenCalled();
  }, timeout);

  it("expired JWT → 401", async () => {
    const embedding = fakeJoint2312Embedding(4);
    const res = await callSleepQuality(jsonBody(embedding), _tokens.expiredJWT);
    expect(res.status).toBe(401);
    expect(mockDecodeSleepQuality).not.toHaveBeenCalled();
  }, timeout);

  it("user A cannot reference user B's embedding_id via RLS", async () => {
    const embeddingB = fakeJoint2312Embedding(5);
    const embIdB = await insertEmbedding(_tokens.userB.id, embeddingB);

    // Simulate RLS blocking cross-user access: mock throws SleepDecodeError
    const { SleepDecodeError } = await import("@/lib/ai/inference/sleep.server");
    mockDecodeSleepQuality.mockRejectedValue(
      new SleepDecodeError("embedding_id not found or access denied", "EMBEDDING_NOT_FOUND")
    );

    const res = await callSleepQuality(jsonBodyById(embIdB), _tokens.userA.jwt);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.code).toBe("EMBEDDING_NOT_FOUND");
  }, timeout);

  it("userId field in JSON body is ignored — authenticated userId from JWT is used", async () => {
    const embedding = fakeJoint2312Embedding(6);

    mockDecodeSleepQuality.mockImplementation(async (opts, supabase, userId) => {
      expect(userId).toBe(_tokens.userA.id);
      return {
        service: "sleep-staging",
        model: "onnx-cbramod-joint-2312",
        head: "sleep-quality-v1",
        head_version: "0.1.0",
        provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-quality-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
        results: [{ score: 0.85, band: "excellent", confidence_interval: [0.75, 0.95], confidence: 0.9, metric: "sleep-quality" }],
        metadata: { embedding_reused: false, probe_sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f" },
        timings: { inference_ms: 0.4, total_ms: 0.8 },
      };
    });

    const body = JSON.stringify({
      query_embedding: embedding,
      query_type: "sleep-quality",
      userId: _tokens.userB.id, // spoofed — should be ignored
    });

    const res = await callSleepQuality(body, _tokens.userA.jwt);
    expect(res.status).toBe(200);
    expect(mockDecodeSleepQuality).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      _tokens.userA.id, // from JWT, not from body
    );

    const { data: bRows } = await admin
      .from("joint_embeddings_2312")
      .select("user_id")
      .eq("user_id", _tokens.userB.id);
    expect(bRows ?? []).toEqual([]);
  }, timeout);
}, 30_000);
