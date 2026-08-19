/**
 * Mission 15 Phase 5 — Full API Contract Validation (close M14 Gate A4 remainder).
 *
 * Exercises the REAL route handler (POST /api/eeg/embed/foundation) through real
 * JWT auth (local GoTrue) + real rate-limit RPC (check_rate_limit), with
 * embedFoundationWindows mocked only to control 408/500 error injection.
 * embedEEG (V2) is mocked so we can ASSERT no fallback on any error path.
 *
 * Validates every HTTP status code the route can produce:
 *   200  – valid request, 200-D vector, DB persisted
 *   400  – non-multipart, missing file, missing sampleRate
 *   401  – no Bearer, invalid JWT, expired JWT
 *   408  – processing timeout (mocked timeout-exceeded error)
 *   413  – file exceeds 50 MB cap
 *   415  – unsupported file type
 *   422  – malformed CSV (parse error), signal too short for segmentation
 *   424  – FoundationUnavailableError (runtime/artifact)
 *   429  – rate limit exceeded
 *   500  – unhandled internal error
 *
 * On ALL non-200 paths: embedEEG (V2/PCA) must NEVER be called.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { createClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

// ── 1. Point at LOCAL Supabase stack ───────────────────────────────────────────
const _tokens = JSON.parse(readFileSync("reports/m15_jwt_test_tokens.json", "utf-8")) as {
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

// ── 2. Mock embedFoundationWindows (control 408/500 injection) + embedEEG ─────
const mockEmbedFoundationWindows = vi.fn();
vi.mock("@/lib/ai/inference/foundation.server", async (importActual) => {
  const actual = await importActual<typeof import("@/lib/ai/inference/foundation.server")>();
  return {
    ...actual,
    embedFoundationWindows: (...args: unknown[]) => mockEmbedFoundationWindows(...args),
  };
});

const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// ── 3. Real imports (auth, rate-limit, NeuralVectorIndex are NOT mocked) ─────────
const admin = createClient<Database>(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
);
const { Route } = await import("../foundation");
const { FoundationUnavailableError } = await import("@/lib/ai/inference/foundation.server");
const { FOUNDATION_EMBEDDING_DIM, FOUNDATION_MODEL_ID } =
  await import("@/lib/ai/inference/foundation.server");

// ── 4. Helpers ─────────────────────────────────────────────────────────────────
const CBraMod19 = [
  "FP1",
  "FP2",
  "F3",
  "F4",
  "C3",
  "C4",
  "P3",
  "P4",
  "O1",
  "O2",
  "F7",
  "F8",
  "T7",
  "T8",
  "P7",
  "P8",
  "FZ",
  "CZ",
  "PZ",
];

function csvFile(name = "signal.csv", rows = 2000, seed = 0): File {
  const lines = [CBraMod19.join(",")];
  for (let r = 0; r < rows; r++) {
    lines.push(
      CBraMod19.map((_, c) => (Math.sin((r + c + seed) * 0.01 + seed * 0.7) * 0.3).toFixed(4)).join(
        ",",
      ),
    );
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

/** Build a request with a valid JWT (user A). */
function authedRequest(form: FormData): Request {
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers: { authorization: `Bearer ${_tokens.userA.jwt}` },
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
  await admin.from("rate_limits").delete().neq("user_id", "00000000-0000-0000-0000-000000000000");
}

function fakeEmbedding() {
  const v = Array.from({ length: FOUNDATION_EMBEDDING_DIM }, (_, i) => ((i % 17) + 1) / 42);
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

// ── 6. Tests ───────────────────────────────────────────────────────────────────
describe("M15 Phase 5: Full API Contract — all status codes (real route + real auth)", () => {
  beforeEach(async () => {
    await resetDB();
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockResolvedValue([fakeEmbedding()]);
    embedEEGMock.mockReset();
  });

  // ── 200 ─────────────────────────────────────────────────────────────────────
  it("200: valid CSV → 200 with 200-D vector, provenance, timings, DB persisted", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000, 0));
    form.set("sampleRate", "250");

    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(200);

    const body = await res.json();
    expect(body.dimensions).toBe(200);
    expect(body.model).toBe(FOUNDATION_MODEL_ID);
    expect(body.windows).toBeGreaterThanOrEqual(1);
    expect(body.embeddings.length).toBe(body.windows);
    expect(body.vector_indexed).toBeGreaterThan(0);
    expect(body.provenance.artifact_id).toBe("cbramod-encoder");
    expect(body.provenance.sha256).toBe(
      "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    );
    expect(body.timings.total_ms).toBeTypeOf("number");
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 30_000);

  // ── 400 ─────────────────────────────────────────────────────────────────────
  it("400: non-multipart content-type rejected before auth", async () => {
    const res = await callFoundation(
      new Request("http://localhost/api/eeg/embed/foundation", {
        method: "POST",
        body: new Blob(["x"]),
        headers: { authorization: `Bearer ${_tokens.userA.jwt}` },
      }),
    );
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toMatch(/multipart/i);
  }, 15_000);

  it("400: missing 'file' field", async () => {
    const form = new FormData();
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toMatch(/file/i);
  }, 15_000);

  it("400: CSV without sampleRate rejected (sampleRate required for CSV)", async () => {
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000, 0));
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toMatch(/sampleRate/i);
  }, 15_000);

  // ── 401 ─────────────────────────────────────────────────────────────────────
  it("401: no Bearer token", async () => {
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "250");
    const res = await callFoundation(
      new Request("http://localhost/api/eeg/embed/foundation", {
        method: "POST",
        body: form,
      }),
    );
    expect(res.status).toBe(401);
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  it("401: invalid JWT (tampered signature)", async () => {
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "250");
    const res = await callFoundation(
      new Request("http://localhost/api/eeg/embed/foundation", {
        method: "POST",
        body: form,
        headers: { authorization: `Bearer ${_tokens.invalidJWT}` },
      }),
    );
    expect(res.status).toBe(401);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  it("401: expired JWT", async () => {
    const form = new FormData();
    form.set("file", csvFile());
    form.set("sampleRate", "250");
    const res = await callFoundation(
      new Request("http://localhost/api/eeg/embed/foundation", {
        method: "POST",
        body: form,
        headers: { authorization: `Bearer ${_tokens.expiredJWT}` },
      }),
    );
    expect(res.status).toBe(401);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  // ── 408 ─────────────────────────────────────────────────────────────────────
  it("408: processing timeout mapped correctly (no V2 fallback on timeout)", async () => {
    // Inject a timeout-exceeded error from the embed step to exercise the route's
    // 408 mapping (foundation.ts L118-121). The real timeout is 120s; we simulate
    // it by rejecting embedFoundationWindows with a timeout error.
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new Error("Foundation processing timeout exceeded (120000ms)"),
    );

    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000, 0));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));

    expect(res.status).toBe(408);
    const body = await res.json();
    expect(body.error).toMatch(/timeout/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 30_000);

  // ── 413 ─────────────────────────────────────────────────────────────────────
  it("413: file exceeds 50 MB cap", async () => {
    // 51 MB — exceeds MAX_FILE_BYTES (50 * 1024 * 1024) at foundation.ts L40.
    const big = new File([new Uint8Array(51 * 1024 * 1024)], "big.csv", {
      type: "text/csv",
    });
    const form = new FormData();
    form.set("file", big);
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(413);
    const body = await res.json();
    expect(body.error).toMatch(/too large/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  // ── 415 ─────────────────────────────────────────────────────────────────────
  it("415: unsupported file extension (.png)", async () => {
    const form = new FormData();
    form.set("file", new File(["x"], "signal.png", { type: "image/png" }));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(415);
    const body = await res.json();
    expect(body.error).toMatch(/unsupported/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  it("415: unsupported file extension (.wav)", async () => {
    const form = new FormData();
    form.set("file", new File(["x"], "signal.wav", { type: "audio/wav" }));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(415);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  // ── 422 ─────────────────────────────────────────────────────────────────────
  it("422: signal too short for CBraMod window segmentation (< 1000 samples)", async () => {
    // 100 rows @ 250 Hz = 0.4 s — too short for a 4 s window (1000 samples).
    const form = new FormData();
    form.set("file", csvFile("short.csv", 100, 0));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(422);
    const body = await res.json();
    expect(body.error).toMatch(/too short/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  it("422: malformed CSV (garbage that fails to parse)", async () => {
    // Non-numeric garbage in every cell — parseCSV forward-fills NaNs but a
    // completely garbage file with a text header that's not a channel list will
    // still parse. We instead omit the channel header entirely by using a
    // single-column garbage file which still parses but won't have 19 channels.
    const form = new FormData();
    form.set("file", new File(["garbage\nnotanumber\n"], "bad.csv", { type: "text/csv" }));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(422);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 15_000);

  // ── 424 ─────────────────────────────────────────────────────────────────────
  it("424: FoundationUnavailableError → 424, never V2 fallback", async () => {
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new FoundationUnavailableError("onnxruntime-node import failed: not found"),
    );

    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000, 0));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));

    expect(res.status).toBe(424);
    const body = await res.json();
    expect(body.error).toMatch(/unavailable/i);
    expect(body.detail).toMatch(/onnxruntime-node/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 30_000);

  // ── 429 ─────────────────────────────────────────────────────────────────────
  it("429: rate limit exceeded → 429 with retry_after_ms, no V2 fallback", async () => {
    // Burn user A's budget (20 requests).
    for (let i = 0; i < 20; i++) {
      const form = new FormData();
      form.set("file", csvFile(`rl-${i}.csv`, 2000, i));
      form.set("sampleRate", "250");
      const res = await callFoundation(authedRequest(form));
      expect(res.status).toBe(200);
    }

    // Clear mock call history AFTER the 20 accepted requests — the rejected
    // request must NOT invoke the embed path (same pattern as Phase 2 test).
    mockEmbedFoundationWindows.mockClear();
    embedEEGMock.mockClear();

    // 21st — 429.
    const form = new FormData();
    form.set("file", csvFile("over-limit.csv", 2000, 21));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.error).toMatch(/rate limit/i);
    expect(body.retry_after_ms).toBeTypeOf("number");
    expect(body.retry_after_ms).toBeGreaterThan(0);
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  }, 60_000);

  // ── 500 ─────────────────────────────────────────────────────────────────────
  it("500: unhandled internal error → 500, never V2 fallback", async () => {
    // Inject a generic (non-timeout, non-FoundationUnavailableError) error.
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(new Error("Unexpected CUDA OOM in ONNX"));

    const form = new FormData();
    form.set("file", csvFile("signal.csv", 2000, 0));
    form.set("sampleRate", "250");
    const res = await callFoundation(authedRequest(form));

    expect(res.status).toBe(500);
    const body = await res.json();
    expect(body.error).toMatch(/error occurred/i);
    // The generic error message is NOT leaked to the client (T-PR-003 sanitisation).
    expect(body.error).not.toMatch(/CUDA OOM/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  }, 30_000);
});
