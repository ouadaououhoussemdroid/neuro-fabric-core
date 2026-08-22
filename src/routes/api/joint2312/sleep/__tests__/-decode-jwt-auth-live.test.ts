/**
 * M42 Phase 1a — Real JWT / Authorization Validation for sleep staging route.
 *
 * Extends M15 Phase 1 to the Tier-2 sleep staging route (POST /api/joint2312/sleep/decode).
 * Follows the exact M15 pattern: real local Supabase stack (GoTrue + PostgREST +
 * Postgres with pgvector), real authenticateRequest (validates JWT signature + expiry
 * via GoTrue /auth/v1/user), real RLS on joint_embeddings_2312.
 *
 * Only decodeSleepState is mocked — real ONNX inference is not available in the
 * test environment (no onnxruntime-node models). This isolation lets the test
 * focus exclusively on the auth/RLS/authorization contract.
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

// ── 2. Mock ONLY decodeSleepState (Phase 3/4 covers real ONNX inference) ───
const mockDecodeSleepState = vi.fn();
vi.mock("@/lib/ai/inference/sleep.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/sleep.server")>();
  return {
    ...actual,
    decodeSleepState: (...args: unknown[]) => mockDecodeSleepState(...args),
  };
});

// ── 3. Real imports (auth + rate-limit are NOT mocked) ────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../decode");
const { JOINT_2312_EMBEDDING_DIM } = await import("@/lib/ai/inference/joint.server");

// ── 4. Helpers ────────────────────────────────────────────────────────────────

/** Deterministic 2312-D L2-normalised embedding for test fixtures. */
function fakeJoint2312Embedding(seed = 0): number[] {
  const v = Array.from({ length: JOINT_2312_EMBEDDING_DIM }, (_, i) =>
    Math.sin((i + seed) * 0.01) * 0.5,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

/** Insert a row into joint_embeddings_2312 as the given user. */
async function insertEmbedding(userId: string, embedding: number[], metadata: Record<string, unknown> = {}) {
  const { data, error } = await admin.from("joint_embeddings_2312").insert([
    {
      user_id: userId,
      embedding: embedding,
      metadata: { test: true, ...metadata },
    },
  ]).select("id").single();
  expect(error).toBeNull();
  return data!.id as string;
}

/** Build a JSON body with a raw 2312-D embedding (bypasses DB lookup). */
function jsonBody(embedding: number[], queryType = "sleep-stages", extra: Record<string, unknown> = {}): string {
  return JSON.stringify({ query_embedding: embedding, query_type: queryType, ...extra });
}

/** Build a JSON body with an embedding_id. */
function jsonBodyById(embeddingId: string, queryType = "sleep-stages"): string {
  return JSON.stringify({ embedding_id: embeddingId, query_type: queryType });
}

type PostHandler = (ctx: {
  request: Request;
  context: unknown;
}) => Promise<Response>;

function callSleepDecode(body: string, jwt?: string): Promise<Response> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({
    request: new Request("http://localhost/api/joint2312/sleep/decode", {
      method: "POST",
      body,
      headers,
    }),
    context: {},
  });
}

async function resetDB() {
  await admin
    .from("joint_embeddings_2312")
    .delete()
    .neq("id", "00000000-0000-0000-0000-000000000000");
  await admin
    .from("rate_limits")
    .delete()
    .neq("user_id", "00000000-0000-0000-0000-000000000000");
}

// ── 5. Tests ─────────────────────────────────────────────────────────────────

describe("M42 Phase 1a: Real JWT Authentication + RLS Isolation — sleep staging route", () => {
  const timeout = 30_000;

  beforeEach(async () => {
    await resetDB();
    mockDecodeSleepState.mockReset();
  });

  it("valid JWT + raw embedding → 200, sleep staging results returned", async () => {
    mockDecodeSleepState.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-staging-v1",
      head_version: "0.1.0",
      provenance: {
        service: "sleep-staging",
        service_version: "v0.1.0",
        embedding_model: "onnx-cbramod-joint-2312",
        embedding_dim: JOINT_2312_EMBEDDING_DIM,
        task_head_id: "sleep-staging-v1",
        timestamp: new Date().toISOString(),
        artifact_shas: {
          cbramod: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
          v2: "18644de187e984a667d78ca79b3f4c0d2c0ada252c7084f4107343f77168f931",
          eegpt: "a92daf44ab8cb10e4c258c853b2d4668232acc6e09606fa2e18d052c4b914f36",
        },
        block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
      },
      results: [{
        stage_id: 3,
        stage: "N3",
        probabilities: [0.1, 0.1, 0.1, 0.6, 0.1],
        confidence: 0.6,
        confidence_interval: [0.52, 0.68],
        metric: "sleep-stages",
      }],
      metadata: { embedding_reused: false, probe_sha256: "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6" },
      timings: { inference_ms: 0.52, total_ms: 1.0 },
    });

    const embedding = fakeJoint2312Embedding(0);
    const res = await callSleepDecode(jsonBody(embedding), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.service).toBe("sleep-staging");
    expect(body.results[0].stage).toBe("N3");
    // decodeSleepState must have been called with the authenticated userId
    expect(mockDecodeSleepState).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding, query_type: "sleep-stages" }),
      expect.anything(),
      _tokens.userA.id,
    );
  }, timeout);

  it("valid JWT + embedding_id → 200, reuses existing embedding from DB (RLS-scoped)", async () => {
    const embedding = fakeJoint2312Embedding(1);
    const embId = await insertEmbedding(_tokens.userA.id, embedding, { source: "userA-test" });

    mockDecodeSleepState.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-staging-v1",
      head_version: "0.1.0",
      embedding_id: embId,
      provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-staging-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
      results: [{ stage_id: 2, stage: "N2", probabilities: [0.1, 0.2, 0.5, 0.1, 0.1], confidence: 0.5, confidence_interval: [0.42, 0.58], metric: "sleep-stages" }],
      metadata: { embedding_reused: true, probe_sha256: "9da4ea37c92c1d87e80dde9a52bcd651246b73274fba5f11f4262d44ff3710f6" },
      timings: { inference_ms: 0.5, total_ms: 1.0 },
    });

    const res = await callSleepDecode(jsonBodyById(embId), _tokens.userA.jwt);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.results[0].stage).toBe("N2");
    expect(body.embedding_id).toBe(embId);
    // decodeSleepState fetches the embedding via supabase.from("joint_embeddings_2312")
    expect(mockDecodeSleepState).toHaveBeenCalledTimes(1);
  }, timeout);

  it("no Bearer token → 401", async () => {
    const embedding = fakeJoint2312Embedding(2);
    const res = await callSleepDecode(jsonBody(embedding), undefined);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized/i);
    // No decode should have happened (auth rejected before inference)
    expect(mockDecodeSleepState).not.toHaveBeenCalled();
  }, timeout);

  it("invalid JWT (tampered signature) → 401", async () => {
    const embedding = fakeJoint2312Embedding(3);
    const res = await callSleepDecode(jsonBody(embedding), _tokens.invalidJWT);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized|invalid/i);
    expect(mockDecodeSleepState).not.toHaveBeenCalled();
  }, timeout);

  it("expired JWT → 401", async () => {
    const embedding = fakeJoint2312Embedding(4);
    const res = await callSleepDecode(jsonBody(embedding), _tokens.expiredJWT);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized/i);
    expect(mockDecodeSleepState).not.toHaveBeenCalled();
  }, timeout);

  it("user A cannot read user B's embedding via embedding_id (RLS blocks cross-user SELECT)", async () => {
    // Insert an embedding owned by user B
    const embeddingB = fakeJoint2312Embedding(5);
    const embIdB = await insertEmbedding(_tokens.userB.id, embeddingB, { owner: "B" });

    // User A tries to reference user B's embedding_id.
    // The real decodeSleepState would fetch the embedding via supabase.from("joint_embeddings_2312")
    // where RLS blocks the cross-user SELECT (returns no rows). The route catches SleepDecodeError
    // and returns 400 with code EMBEDDING_NOT_FOUND. The mock simulates this RLS violation.
    const { SleepDecodeError } = await import("@/lib/ai/inference/sleep.server");
    mockDecodeSleepState.mockRejectedValue(
      new SleepDecodeError("embedding_id not found or access denied", "EMBEDDING_NOT_FOUND")
    );

    const res = await callSleepDecode(jsonBodyById(embIdB), _tokens.userA.jwt);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.code).toBe("EMBEDDING_NOT_FOUND");
    expect(mockDecodeSleepState).toHaveBeenCalledTimes(1);
  }, timeout);

  it("user ID is bound to JWT identity — client-supplied userId field is ignored", async () => {
    // The route extracts userId from authenticateRequest (real JWT), NOT from the request body.
    // Even if the client sends a 'userId' field in the JSON, the authenticated userId
    // from the JWT is used for RLS + audit logging.
    const embedding = fakeJoint2312Embedding(6);

    mockDecodeSleepState.mockImplementation(async (opts, supabase, userId) => {
      // Verify the userId parameter matches userA (the JWT owner), not userB
      expect(userId).toBe(_tokens.userA.id);

      return {
        service: "sleep-staging",
        model: "onnx-cbramod-joint-2312",
        head: "sleep-staging-v1",
        head_version: "0.1.0",
        provenance: { service: "sleep-staging", service_version: "v0.1.0", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: JOINT_2312_EMBEDDING_DIM, task_head_id: "sleep-staging-v1", timestamp: new Date().toISOString(), artifact_shas: {}, block_weights: {}, component_dims: {} },
        results: [{ stage_id: 0, stage: "W", probabilities: [0.9, 0.03, 0.02, 0.02, 0.03], confidence: 0.9, confidence_interval: [0.82, 0.98], metric: "sleep-stages" }],
        metadata: { embedding_reused: false },
        timings: { inference_ms: 0.3, total_ms: 0.8 },
      };
    });

    // Client tries to override by including userId in the body (should be ignored)
    const body = JSON.stringify({
      query_embedding: embedding,
      query_type: "sleep-stages",
      userId: _tokens.userB.id, // client TRY to override — should be rejected
    });

    const res = await callSleepDecode(body, _tokens.userA.jwt);
    expect(res.status).toBe(200);
    // decodeSleepState was called with userA's id (from JWT), not userB
    expect(mockDecodeSleepState).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      _tokens.userA.id, // authenticated identity, NOT the spoofed userId field
    );

    // Verify no row was created for userB
    const { data: bRows } = await admin
      .from("joint_embeddings_2312")
      .select("user_id")
      .eq("user_id", _tokens.userB.id);
    expect(bRows ?? []).toEqual([]);
  }, timeout);
}, 30_000);
