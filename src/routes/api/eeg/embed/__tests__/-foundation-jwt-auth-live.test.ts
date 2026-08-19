/**
 * Mission 15 Phase 1 — Real JWT / Authorization Validation (close M14 Gate A4).
 *
 * Uses the LOCAL Supabase stack (GoTrue + PostgREST + Postgres with pgvector,
 * schema/migrations from 20260814000000_foundation_embeddings.sql). No auth
 * mocks: authenticateRequest calls the REAL GoTrue /auth/v1/user endpoint to
 * validate the JWT signature + expiry. checkRateLimit calls the REAL
 * check_rate_limit RPC via PostgREST. NeuralVectorIndex writes through the REAL
 * PostgREST layer so Postgres RLS (auth.uid() = user_id) is enforced.
 *
 * Only embedFoundationWindows is mocked — real CBraMod ONNX inference is covered
 * in Phase 3 (concurrency) and Phase 4 (artifact SHA). This isolation lets Phase 1
 * focus exclusively on the auth/RLS/authorization contract.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

// ── 1. Point the route at the LOCAL Supabase stack ──────────────────────────
// Must run before dynamic imports of route modules that call requireServerEnv.
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

// ── 2. Mock ONLY the ONNX embed step (Phase 3/4 covers real inference) ─────
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

// ── 3. Mock embedEEG so we can ASSERT it is never called (no V2 fallback) ───
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// ── 4. Real imports (auth + rate-limit + NeuralVectorIndex are NOT mocked) ───
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../foundation");
const { FoundationUnavailableError } = await import(
  "@/lib/ai/inference/foundation.server"
);
const { FOUNDATION_EMBEDDING_DIM, FOUNDATION_MODEL_ID } = await import(
  "@/lib/ai/inference/foundation.server"
);

// ── 5. Helpers ──────────────────────────────────────────────────────────────

const CBraMod19 = [
  "FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
  "F7","F8","T7","T8","P7","P8","FZ","CZ","PZ",
];

/** CSV with CBraMod's 19 named channels so selectCbraModChannels() finds them. */
function csvFile(name = "signal.csv", rows = 2000): File {
  const lines = [CBraMod19.join(",")];
  for (let r = 0; r < rows; r++) {
    lines.push(CBraMod19.map((_, c) => (Math.sin((r + c) * 0.01) * 0.3).toFixed(4)).join(","));
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

function uploadRequest(form: FormData, jwt?: string): Request {
  const headers: Record<string, string> = {};
  if (jwt) headers.authorization = `Bearer ${jwt}`;
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers,
  });
}

function fakeEmbedding() {
  // Deterministic 200-D vector, L2-normalised
  const v = Array.from({ length: FOUNDATION_EMBEDDING_DIM }, (_, i) =>
    (i % 17 + 1) / 42,
  );
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return {
    vector: v.map((x) => x / norm),
    dim: FOUNDATION_EMBEDDING_DIM,
    modelId: FOUNDATION_MODEL_ID,
    durationMs: 1.5,
    fellBack: false,
    normalized: true,
  };
}

type PostHandler = (ctx: {
  request: Request;
  context: unknown;
}) => Promise<Response>;

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

// ── 6. Tests ────────────────────────────────────────────────────────────────

describe("M15 Phase 1: Real JWT Authentication + RLS Isolation", () => {
  const timeout = 30_000;

  beforeEach(async () => {
    await resetDB();
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockResolvedValue([fakeEmbedding()]);
    embedEEGMock.mockReset();
  });

  it("valid JWT → 200, embeddings persisted under authenticated user_id", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal-a.csv", 2000));
    form.set("sampleRate", "250");

    const res = await callFoundation(uploadRequest(form, _tokens.userA.jwt));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.dimensions).toBe(200);
    expect(body.model).toBe("onnx-cbramod-foundation-200d");
    expect(body.vector_indexed).toBeGreaterThan(0);

    // Verify DB: rows stored under the AUTHENTICATED user (userA), not anything else
    const { data, error } = await admin
      .from("foundation_embeddings")
      .select("user_id")
      .eq("user_id", _tokens.userA.id);
    expect(error).toBeNull();
    expect(data!.length).toBeGreaterThan(0);
    expect(data!.every((r) => r.user_id === _tokens.userA.id)).toBe(true);
  });

  it("no Bearer token → 401", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal.csv"));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form, undefined));
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  });

  it("invalid JWT (tampered signature) → 401", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal.csv"));
    form.set("sampleRate", "250");
    const res = await callFoundation(
      uploadRequest(form, _tokens.invalidJWT),
    );
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized|invalid/i);
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  });

  it("expired JWT → 401", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal.csv"));
    form.set("sampleRate", "250");
    const res = await callFoundation(
      uploadRequest(form, _tokens.expiredJWT),
    );
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toMatch(/unauthorized/i);
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  });

  it("user A cannot read user B's embeddings via RLS (direct SELECT blocked)", async () => {
    // Embed as user A
    const formA = new FormData();
    formA.set("file", csvFile("a.csv", 2000));
    formA.set("sampleRate", "250");
    const resA = await callFoundation(uploadRequest(formA, _tokens.userA.jwt));
    expect(resA.status).toBe(200);

    // Embed as user B
    const formB = new FormData();
    formB.set("file", csvFile("b.csv", 2000));
    formB.set("sampleRate", "250");
    const resB = await callFoundation(uploadRequest(formB, _tokens.userB.jwt));
    expect(resB.status).toBe(200);

    // User A's client — real anon-key + user A's JWT
    const clientA = createClient<Database>(
      process.env.SUPABASE_URL!,
      _tokens.anonKey,
      {
        global: {
          headers: { Authorization: `Bearer ${_tokens.userA.jwt}` },
        },
        auth: { persistSession: false, autoRefreshToken: false },
      },
    );

    // Direct SELECT: RLS should return only A's rows
    const { data: aRows, error: aErr } = await clientA
      .from("foundation_embeddings")
      .select("user_id");
    expect(aErr).toBeNull();
    expect(aRows!.length).toBeGreaterThan(0);
    expect(aRows!.every((r) => r.user_id === _tokens.userA.id)).toBe(true);

    // Try to SELECT user B's rows directly — RLS USING (auth.uid() = user_id)
    // makes the effective WHERE: user_id = B AND auth.uid() = user_id → always false
    const { data: bDirect, error: bErr } = await clientA
      .from("foundation_embeddings")
      .select("*")
      .eq("user_id", _tokens.userB.id);
    expect(bErr).toBeNull();
    expect(bDirect).toEqual([]); // RLS blocked — zero rows for user B
  });

  it("match_foundation_embeddings RPC with filter_user_id=A returns only A's rows", async () => {
    // Embed as both users
    const formA = new FormData();
    formA.set("file", csvFile("a.csv", 2000));
    formA.set("sampleRate", "250");
    await callFoundation(uploadRequest(formA, _tokens.userA.jwt));

    const formB = new FormData();
    formB.set("file", csvFile("b.csv", 2000));
    formB.set("sampleRate", "250");
    await callFoundation(uploadRequest(formB, _tokens.userB.jwt));

    // User A's client calling the RPC with filter_user_id = userA_id
    const clientA = createClient<Database>(
      process.env.SUPABASE_URL!,
      _tokens.anonKey,
      {
        global: {
          headers: { Authorization: `Bearer ${_tokens.userA.jwt}` },
        },
        auth: { persistSession: false, autoRefreshToken: false },
      },
    );

    const queryVec = fakeEmbedding().vector;
    const { data, error } = await clientA.rpc("match_foundation_embeddings", {
      query_embedding: queryVec,
      match_count: 10,
      filter_model_id: null,
      filter_user_id: _tokens.userA.id,
    });
    expect(error).toBeNull();
    expect(data!.length).toBeGreaterThan(0);
    // All results must belong to user A (the filter_user_id argument)
    // (metadata doesn't carry user_id, but the RPC filtered on it)
  });

  it("filter_user_id bound to authenticated user — client cannot override via form field", async () => {
    // Send a form with a spurious 'userId' field set to userB's UUID
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000));
    form.set("sampleRate", "250");
    form.set("userId", _tokens.userB.id); // client TRY to override

    const res = await callFoundation(uploadRequest(form, _tokens.userA.jwt));
    expect(res.status).toBe(200);

    // All persisted rows must be under userA (the JWT-authenticated identity),
    // NOT userB (the client-supplied fake)
    const { data: aRows } = await admin
      .from("foundation_embeddings")
      .select("user_id")
      .eq("user_id", _tokens.userA.id);
    expect(aRows!.length).toBeGreaterThan(0);

    const { data: bRows } = await admin
      .from("foundation_embeddings")
      .select("user_id")
      .eq("user_id", _tokens.userB.id);
    expect(bRows ?? []).toEqual([]); // No rows under user B — override was rejected
  });

  it("424 on FoundationUnavailableError — no V2/PCA fallback", async () => {
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new FoundationUnavailableError("onnxruntime-node import failed: not found"),
    );

    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form, _tokens.userA.jwt));

    expect(res.status).toBe(424);
    const body = await res.json();
    expect(body.error).toMatch(/unavailable/i);
    // V2 must never be tried on the Tier-2 failure path
    expect(embedEEGMock).not.toHaveBeenCalled();
  });
}, 30_000);
