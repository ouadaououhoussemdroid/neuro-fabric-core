/**
 * T-036 / Mission 13 — route test for the additive Tier-2 foundation SEARCH endpoint.
 *
 * Mirrors `-foundation.test.ts` conventions: auth + rate-limit + the EEG pipeline
 * are mocked; BUT the search surface (`searchFoundationEmbeddings` -> real
 * `NeuralVectorIndex` with no Supabase client) is NOT mocked, so the route is
 * proven to wire the 200-D query into the ISOLATED `foundation_embeddings`
 * namespace (vector(200)) via the real call site. With no client the index uses
 * its in-memory brute-force cosine fallback (same metric as the RPC), so the
 * retrieval call site is exercised without a database.
 *
 * Prohibitions asserted explicitly: never calls embedEEG / V2 / PCA; never falls
 * back to V2 on unavailability (424); refuses non-200-D queries.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// --- Auth + rate-limit (identical mocking pattern to -foundation.test.ts) ---
const mockAuthenticateRequest = vi.fn();
const mockCheckRateLimit = vi.fn();
vi.mock("@/integrations/supabase/request-auth", async (importActual) => {
  const actual = await importActual<typeof import("@/integrations/supabase/request-auth")>();
  return {
    ...actual,
    authenticateRequest: (...args: unknown[]) => mockAuthenticateRequest(...args),
  };
});
vi.mock("@/integrations/supabase/rate-limit", () => ({
  checkRateLimit: (...args: unknown[]) => mockCheckRateLimit(...args),
}));

// --- Foundation service: mock ONLY the heavy embed step (real ONNX forward).
// searchFoundationEmbeddings + NeuralVectorIndex are NOT mocked, so the route's
// wiring into the foundation namespace runs for real (in-memory fallback). ---
const mockEmbedFoundationWindows = vi.fn();
vi.mock("@/lib/ai/inference/foundation.server", () => {
  class FoundationUnavailableError extends Error {
    constructor(public reason: string) {
      super(`CBraMod foundation runtime unavailable: ${reason}`);
      this.name = "FoundationUnavailableError";
    }
  }
  const unit = () => {
    const v = Array.from({ length: 200 }, () => Math.random() * 2 - 1);
    let s = 0;
    for (const x of v) s += x * x;
    const n = Math.sqrt(s) || 1;
    return v.map((x) => x / n);
  };
  return {
    FOUNDATION_MODEL_ID: "onnx-cbramod-foundation-200d",
    FOUNDATION_EMBEDDING_DIM: 200,
    FOUNDATION_SAMPLE_RATE_HZ: 250,
    FOUNDATION_ARTIFACT_ID: "cbramod-encoder",
    FoundationUnavailableError,
    foundationRuntime: vi.fn(),
    foundationProvenance: vi.fn().mockReturnValue({
      artifact_id: "cbramod-encoder",
      sha256: "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
      size: 22018587,
      url: "/models/cbramod-encoder.onnx",
      sample_rate_hz: 250,
      window_samples: 1000,
      channels: 19,
      embedding_dim: 200,
      bandpass_hz: [4, 38],
      output_pooling: "mean-tokens",
      normalization: "zscore per channel + mean-tokens L2",
      runtime: "onnxruntime-node cpu",
    }),
    embedFoundationWindows: (...args: unknown[]) => mockEmbedFoundationWindows(...args),
    resetFoundationAdapter: vi.fn(),
  };
});

// --- Prevent any Tier-1 leakage: embedEEG must NEVER be imported/called. ---
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// --- EEG pipeline stubs (channel select / resample / preprocess). ---
vi.mock("@/lib/eeg/channels", () => ({
  CBRAMOD_CHANNELS_19: [
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
  ],
  selectCbraModChannels: (signal: {
    channels: string[];
    data: number[][];
    sampleRate: number;
  }) => ({
    channels: signal.channels.slice(0, 19),
    data: signal.data.slice(0, 19),
    sampleRate: signal.sampleRate,
    meta: { source_channels: signal.channels.length },
  }),
}));
vi.mock("@/lib/eeg/preprocessing/resample", () => ({
  resampleSignal: (signal: unknown) => signal,
}));
vi.mock("@/lib/eeg/preprocessing", () => ({
  preprocess: () => ({
    signal: { channels: [], data: [], sampleRate: 250 },
    windows: [
      {
        data: Array.from({ length: 19 }, () => Array.from({ length: 1000 }, () => 0)),
        sampleRate: 250,
        start: 0,
        end: 1000,
      },
    ],
    report: { channels: 19, samples: 1000, sampleRate: 250, steps: [], totalDurationMs: 0 },
  }),
}));

const { Route } = await import("../foundation-search");
const { AuthError } = await import("@/integrations/supabase/request-auth");
const { FoundationUnavailableError } = await import("@/lib/ai/inference/foundation.server");

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;

function callSearch(request: Request) {
  const handlers = Route.options.server!.handlers as unknown as { POST: PostHandler };
  return handlers.POST({ request, context: {} });
}

function fakeSupabase() {
  return {
    from: () => ({
      insert: () => ({ select: async () => ({ data: [{ id: "x" }], error: null }) }),
    }),
  };
}

function unit200(): number[] {
  const v = Array.from({ length: 200 }, () => Math.random() * 2 - 1);
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

describe("POST /api/eeg/embed/foundation-search (Tier-2, additive)", () => {
  beforeEach(() => {
    mockAuthenticateRequest.mockReset();
    mockCheckRateLimit.mockReset();
    mockCheckRateLimit.mockResolvedValue({ allowed: true, retryAfterMs: 0 });
    mockEmbedFoundationWindows.mockReset();
    embedEEGMock.mockClear();
  });

  it("Path A (JSON 200-D query) searches foundation_embeddings and never calls embedEEG", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "user-1", supabase: fakeSupabase() });
    const q = unit200();

    const res = await callSearch(
      new Request("http://localhost/api/eeg/embed/foundation-search", {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer test-token" },
        body: JSON.stringify({ vector: q, k: 5 }),
      }),
    );

    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.model).toBe("onnx-cbramod-foundation-200d");
    expect(body.dimensions).toBe(200); // NOT the Tier-1 32-D
    expect(body.namespace).toBe("foundation_embeddings");
    // query echoed back (L2, 200-D).
    expect(Array.isArray(body.query_vector)).toBe(true);
    expect((body.query_vector as number[]).length).toBe(200);
    // With an empty in-memory index (no DB), 0 hits — but the real call site ran.
    expect(body.results).toEqual([]);
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(mockEmbedFoundationWindows).not.toHaveBeenCalled();
  });

  it("refuses a 32-D query with 422 (never routes into the V2/32-D space)", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "user-1", supabase: fakeSupabase() });
    const res = await callSearch(
      new Request("http://localhost/api/eeg/embed/foundation-search", {
        method: "POST",
        headers: { "content-type": "application/json", authorization: "Bearer test-token" },
        body: JSON.stringify({ vector: Array(32).fill(0.1) }),
      }),
    );
    expect(res.status).toBe(422);
    const body = await res.json();
    expect((body as { error: string }).error).toMatch(/200-D/);
    expect((body as { expected_dim: number }).expected_dim).toBe(200);
    expect(embedEEGMock).not.toHaveBeenCalled();
  });

  it("Path B (CSV) embeds through CBraMod then searches the foundation namespace", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "user-1", supabase: fakeSupabase() });
    // embedFoundationWindows returns real 200-D unit vectors (mocked heavy step).
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: unit200(),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 1,
        fellBack: false,
        normalized: true,
      },
    ]);

    const lines: string[] = [];
    for (let r = 0; r < 1200; r++) {
      lines.push(
        Array.from({ length: 19 }, (_, c) => Math.sin((r + c) * 0.1).toFixed(4)).join(","),
      );
    }
    const file = new File([lines.join("\n")], "signal.csv", { type: "text/csv" });

    const fd = new FormData();
    fd.set("sampleRate", "250");
    fd.set("file", file);

    const res = await callSearch(
      new Request("http://localhost/api/eeg/embed/foundation-search", {
        method: "POST",
        headers: { authorization: "Bearer test-token" },
        body: fd,
      }),
    );

    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body.model).toBe("onnx-cbramod-foundation-200d");
    expect(body.dimensions).toBe(200);
    expect(body.namespace).toBe("foundation_embeddings");
    // The query vector must be exactly 200-D (CBraMod), proved via the embed step.
    expect((body.query_vector as number[]).length).toBe(200);
    // embedFoundationWindows was invoked (the CBraMod path), never embedEEG.
    expect(mockEmbedFoundationWindows).toHaveBeenCalledTimes(1);
    expect(embedEEGMock).not.toHaveBeenCalled();
  });

  it("returns 424 (no V2 fallback) when the foundation runtime is unavailable", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "user-1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new FoundationUnavailableError("onnxruntime-node import failed: not found"),
    );

    const lines: string[] = [];
    for (let r = 0; r < 1200; r++) {
      lines.push(
        Array.from({ length: 19 }, (_, c) => Math.sin((r + c) * 0.1).toFixed(4)).join(","),
      );
    }
    const file = new File([lines.join("\n")], "signal.csv", { type: "text/csv" });

    const fd = new FormData();
    fd.set("sampleRate", "250");
    fd.set("file", file);

    const res = await callSearch(
      new Request("http://localhost/api/eeg/embed/foundation-search", {
        method: "POST",
        headers: { authorization: "Bearer test-token" },
        body: fd,
      }),
    );

    expect(res.status).toBe(424);
    const body = await res.json();
    expect((body as { error: string }).error).toMatch(/unavailable/i);
    expect(embedEEGMock).not.toHaveBeenCalled();
  });

  it("returns 401 when authenticateRequest rejects with AuthError", async () => {
    mockAuthenticateRequest.mockRejectedValue(
      new AuthError("Unauthorized: missing Bearer token", 401),
    );
    const res = await callSearch(
      new Request("http://localhost/api/eeg/embed/foundation-search", { method: "POST" }),
    );
    expect(res.status).toBe(401);
  });
});
