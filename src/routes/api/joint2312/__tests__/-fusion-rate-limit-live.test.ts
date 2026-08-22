/**
 * M42 Phase 2c — Production-like Rate Limiting for fusion route.
 *
 * Extends M15 Phase 2 to POST /api/joint2312/fusion. Same real Supabase stack,
 * real check_rate_limit RPC, real JWT auth. Only decodeJoint2312 mocked.
 *
 * Verifies: 20 within budget → 200, 21st → 429 with retry_after_ms, per-user
 * isolation, concurrent bypass blocked (atomic UPSERT), invalid JWTs get 401 not
 * 429.
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

// ── 2. Mock ONLY decodeJoint2312 ───────────────────────────────────────────
const mockDecodeJoint2312 = vi.fn();
vi.mock("@/lib/ai/inference/joint-fusion.server", async (importActual) => {
  const actual =
    await importActual<typeof import("@/lib/ai/inference/joint-fusion.server")>();
  return {
    ...actual,
    decodeJoint2312: (...args: unknown[]) => mockDecodeJoint2312(...args),
  };
});

// ── 3. Real imports ────────────────────────────────────────────────────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../fusion");
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
  return JSON.stringify({ query_embedding: embedding });
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

// ── 5. Tests ────────────────────────────────────────────────────────────────

describe("M42 Phase 2c: Production-like Rate Limiting — fusion route", () => {
  beforeEach(async () => {
    await resetDB();
    mockDecodeJoint2312.mockReset();
    mockDecodeJoint2312.mockResolvedValue({
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
    });
  });

  it("accepts 20 within budget, rejects 21st with 429 + retry_after_ms", async () => {
    const statuses: number[] = [];

    for (let i = 0; i < 20; i++) {
      const res = await callFusion(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      statuses.push(res.status);
    }
    expect(statuses).toHaveLength(20);
    expect(statuses.every((s) => s === 200)).toBe(true);

    mockDecodeJoint2312.mockClear();

    const res21 = await callFusion(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(res21.status).toBe(429);
    const body21 = await res21.json();
    expect(body21.retry_after_ms).toBeTypeOf("number");
    expect(body21.retry_after_ms).toBeGreaterThan(0);
    expect(mockDecodeJoint2312).not.toHaveBeenCalled();
  }, 60_000);

  it("per-user isolation: user B budget independent from user A", async () => {
    for (let i = 0; i < 20; i++) {
      const res = await callFusion(jsonBody(fakeJoint2312Embedding(i)), _tokens.userA.jwt);
      expect(res.status).toBe(200);
    }
    const resA21 = await callFusion(jsonBody(fakeJoint2312Embedding(100)), _tokens.userA.jwt);
    expect(resA21.status).toBe(429);

    const resB = await callFusion(jsonBody(fakeJoint2312Embedding(200)), _tokens.userB.jwt);
    expect(resB.status).toBe(200);

    for (let i = 0; i < 19; i++) {
      const res = await callFusion(jsonBody(fakeJoint2312Embedding(201 + i)), _tokens.userB.jwt);
      expect(res.status).toBe(200);
    }

    const resB21 = await callFusion(jsonBody(fakeJoint2312Embedding(300)), _tokens.userB.jwt);
    expect(resB21.status).toBe(429);
  }, 120_000);

  it("concurrent bypass blocked: 21 simultaneous → only 20 succeed", async () => {
    const promises = Array.from({ length: 21 }, (_, i) =>
      callFusion(
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
      const res = await callFusion(jsonBody(fakeJoint2312Embedding(i)), _tokens.invalidJWT);
      statuses.push(res.status);
    }
    expect(statuses.every((s) => s === 401)).toBe(true);

    const resA = await callFusion(jsonBody(fakeJoint2312Embedding(500)), _tokens.userA.jwt);
    expect(resA.status).toBe(200);
  }, 60_000);
}, 120_000);
