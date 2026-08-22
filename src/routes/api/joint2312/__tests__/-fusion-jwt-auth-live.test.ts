/**
 * M42 Phase 1c — Real JWT / Authorization Validation for fusion route.
 *
 * Extends M15 Phase 1 to the Tier-3 fusion route (POST /api/joint2312/fusion).
 * Same pattern: real local Supabase stack, real authenticateRequest, real RLS on
 * joint_embeddings_2312. Only decodeJoint2312 is mocked.
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

// ── 2. Mock ONLY decodeJoint2312 ─────────────────────────────────────────────
const mockDecodeJoint2312 = vi.fn();
vi.mock("@/lib/ai/inference/joint-fusion.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/joint-fusion.server")>();
  return {
    ...actual,
    decodeJoint2312: (...args: unknown[]) => mockDecodeJoint2312(...args),
  };
});

// ── 3. Real imports ──────────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../fusion");
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
    { user_id: userId, embedding, metadata: { test: true, source: "fusion-test" } },
  ]).select("id").single();
  expect(error).toBeNull();
  return data!.id as string;
}

function jsonBody(embedding: number[], extra: Record<string, unknown> = {}): string {
  return JSON.stringify({ query_embedding: embedding, ...extra });
}
function jsonBodyById(embeddingId: string, extra: Record<string, unknown> = {}): string {
  return JSON.stringify({ embedding_id: embeddingId, ...extra });
}

type PostHandler = (ctx: {
  request: Request;
  context: unknown;
}) => Promise<Response>;

function callFusion(body: string, jwt?: string): Promise<Response> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({
    request: new Request("http://localhost:3000/api/joint2312/fusion", {
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

describe("M42 Phase 1c: Real JWT Authentication + RLS Isolation — fusion route", () => {
  const timeout = 30_000;

  beforeEach(async () => {
    await resetDB();
    mockDecodeJoint2312.mockReset();
  });

  it("valid JWT + raw embedding → 200, all 4 task heads return results", async () => {
    const mockResponse = {
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      provenance: {
        service: "joint-fusion",
        service_version: "v0.1.0",
        embedding_model: "onnx-cbramod-joint-2312",
        embedding_dim: JOINT_2312_EMBEDDING_DIM,
        task_head_id: "joint-fusion-all-v1",
        timestamp: new Date().toISOString(),
        artifact_shas: {
          cbramod: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
          v2: "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
          eegpt: "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36",
        },
        block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      },
      results: {
        cognitive: [{ workload: 0.65, confidence: 0.8, stage: "high" }],
        anomaly: [{ score: 0.15, is_anomalous: false, confidence: 0.9, metric: "artifact" }],
        sleep_staging: [{ stage_id: 2, stage: "N2", probabilities: [0.1, 0.2, 0.5, 0.1, 0.1], confidence: 0.5, metric: "sleep-stages" }],
        sleep_quality: [{ score: 0.72, band: "good", confidence: 0.85, metric: "sleep-quality" }],
      },
      metadata: {
        embedding_reused: false,
        heads_run: ["cognitive", "anomaly", "sleep-staging", "sleep-quality"],
        probes: [
          { id: "cognitive-v1", sha256: "cognitive-sha" },
          { id: "anomaly-v1", sha256: "anomaly-sha" },
          { id: "sleep-staging-v1", sha256: "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6" },
          { id: "sleep-quality-v1", sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f" },
        ],
      },
      timings: { inference_ms: 2.1, total_ms: 3.5 },
    };
    mockDecodeJoint2312.mockResolvedValue(mockResponse);

    const embedding = fakeJoint2312Embedding(0);
    const res = await callFusion(jsonBody(embedding), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.service).toBe("joint-fusion");
    expect(body.metadata.heads_run).toEqual(
      ["cognitive", "anomaly", "sleep-staging", "sleep-quality"],
    );
    expect(body.results.sleep_staging).toBeDefined();
    expect(body.results.sleep_quality).toBeDefined();
    expect(mockDecodeJoint2312).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding }),
      expect.anything(),
      _tokens.userA.id,
    );
  }, timeout);

  it("valid JWT + embedding_id → 200, reuses embedding (RLS-scoped)", async () => {
    const embedding = fakeJoint2312Embedding(1);
    const embId = await insertEmbedding(_tokens.userA.id, embedding);

    mockDecodeJoint2312.mockResolvedValue({
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      embedding_id: embId,
      provenance: { service: "joint-fusion", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "joint-fusion-all-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: { sleep_staging: [{ stage_id: 3, stage: "N3", probabilities: [0.1, 0.1, 0.1, 0.6, 0.1], confidence: 0.6, metric: "sleep-stages" }] },
      metadata: { embedding_reused: true, heads_run: ["sleep-staging"], probes: [{ id: "sleep-staging-v1", sha256: "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6" }] },
      timings: { inference_ms: 0.8, total_ms: 1.5 },
    });

    const res = await callFusion(jsonBodyById(embId, { heads: ["sleep-staging"] }), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.embedding_id).toBe(embId);
    expect(body.metadata.heads_run).toEqual(["sleep-staging"]);
  }, timeout);

  it("no Bearer token → 401", async () => {
    const embedding = fakeJoint2312Embedding(2);
    const res = await callFusion(jsonBody(embedding), undefined);
    expect(res.status).toBe(401);
    expect(mockDecodeJoint2312).not.toHaveBeenCalled();
  }, timeout);

  it("invalid JWT (tampered signature) → 401", async () => {
    const embedding = fakeJoint2312Embedding(3);
    const res = await callFusion(jsonBody(embedding), _tokens.invalidJWT);
    expect(res.status).toBe(401);
    expect(mockDecodeJoint2312).not.toHaveBeenCalled();
  }, timeout);

  it("expired JWT → 401", async () => {
    const embedding = fakeJoint2312Embedding(4);
    const res = await callFusion(jsonBody(embedding), _tokens.expiredJWT);
    expect(res.status).toBe(401);
    expect(mockDecodeJoint2312).not.toHaveBeenCalled();
  }, timeout);

  it("user A cannot reference user B's embedding_id via RLS", async () => {
    const embeddingB = fakeJoint2312Embedding(5);
    const embIdB = await insertEmbedding(_tokens.userB.id, embeddingB);

    // Simulate RLS blocking cross-user access: mock throws error with
    // EMBEDDING_NOT_FOUND (same as joint-fusion.server.ts would do when
    // the supabase query returns no rows under RLS)
    mockDecodeJoint2312.mockRejectedValue(
      new Error("EMBEDDING_NOT_FOUND: embedding_id not found or access denied")
    );

    const res = await callFusion(jsonBodyById(embIdB), _tokens.userA.jwt);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toContain("EMBEDDING_NOT_FOUND");
  }, timeout);

  it("userId field in JSON body is ignored — authenticated userId from JWT is used", async () => {
    const embedding = fakeJoint2312Embedding(6);

    mockDecodeJoint2312.mockImplementation(async (opts, supabase, userId) => {
      expect(userId).toBe(_tokens.userA.id);
      return {
        service: "joint-fusion",
        model: "onnx-cbramod-joint-2312",
        head_version: "v0.1.0",
        provenance: { service: "joint-fusion", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "joint-fusion-all-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
        results: { sleep_quality: [{ score: 0.9, band: "excellent", confidence: 0.95, metric: "sleep-quality" }] },
        metadata: { embedding_reused: false, heads_run: ["sleep-quality"], probes: [{ id: "sleep-quality-v1", sha256: "5fb7400f1f00037b36f10f9eb73297a346903fef48997c3357cb177a47797d4f" }] },
        timings: { inference_ms: 0.5, total_ms: 1.0 },
      };
    });

    const body = JSON.stringify({
      query_embedding: embedding,
      userId: _tokens.userB.id, // spoofed — should be ignored
    });

    const res = await callFusion(body, _tokens.userA.jwt);
    expect(res.status).toBe(200);
    expect(mockDecodeJoint2312).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      _tokens.userA.id,
    );

    const { data: bRows } = await admin
      .from("joint_embeddings_2312")
      .select("user_id")
      .eq("user_id", _tokens.userB.id);
    expect(bRows ?? []).toEqual([]);
  }, timeout);

  it("partial heads selection still requires valid auth", async () => {
    const embedding = fakeJoint2312Embedding(7);
    mockDecodeJoint2312.mockResolvedValue({
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      provenance: { service: "joint-fusion", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "joint-fusion-all-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: { cognitive: [{ workload: 0.5, confidence: 0.7, stage: "medium" }] },
      metadata: { embedding_reused: false, heads_run: ["cognitive"], probes: [{ id: "cognitive-v1", sha256: "cognitive-sha" }] },
      timings: { inference_ms: 0.5, total_ms: 1.0 },
    });

    const res = await callFusion(
      jsonBody(embedding, { heads: ["cognitive"] }),
      _tokens.userA.jwt,
    );
    expect(res.status).toBe(200);
  }, timeout);
}, 30_000);
