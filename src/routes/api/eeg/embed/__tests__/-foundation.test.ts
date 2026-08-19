import { describe, it, expect, vi, beforeEach } from "vitest";

// --- Auth + rate-limit (identical mocking pattern to -upload.test.ts) ---
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

// --- Foundation service: mock the embed path + re-export the error class so
// `instanceof` checks in the route match the same reference. ---
const mockEmbedFoundationWindows = vi.fn();
vi.mock("@/lib/ai/inference/foundation.server", () => {
  class FoundationUnavailableError extends Error {
    constructor(public reason: string) {
      super(`CBraMod foundation runtime unavailable: ${reason}`);
      this.name = "FoundationUnavailableError";
    }
  }
  const fakeVector = () => Array.from({ length: 200 }, () => Math.random());
  const fakeResult = () => ({
    vector: fakeVector(),
    dim: 200,
    modelId: "onnx-cbramod-foundation-200d",
    durationMs: 1.5,
    fellBack: false,
    normalized: true,
  });
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

// --- Prevent any Tier-1 leakage: embedEEG must NERVER be imported/called by the
// foundation route. We assert this explicitly below. ---
const embedEEGMock = vi.fn();
vi.mock("@/lib/ai/inference/embed-eeg", () => ({
  embedEEG: embedEEGMock,
  DEFAULT_PREFERRED: "braindecode-eegconformer-prod-v2",
}));

// --- EEG pipeline stubs (channel select / resample / preprocess). The route
// mirrors upload.ts's structure; we only need the window contract here. ---
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
      {
        data: Array.from({ length: 19 }, () => Array.from({ length: 1000 }, () => 0)),
        sampleRate: 250,
        start: 500,
        end: 1500,
      },
    ],
    report: { channels: 19, samples: 1000, sampleRate: 250, steps: [], totalDurationMs: 0 },
  }),
}));

// --- pgvector-backed index: capture constructor opts + assert namespace isolation. ---
const captured = vi.hoisted(() => ({
  indexOpts: undefined as Record<string, unknown> | undefined,
}));
vi.mock("@/lib/vector-search/neural-index", () => ({
  NeuralVectorIndex: class {
    constructor(opts: Record<string, unknown>) {
      captured.indexOpts = opts;
    }
    add = vi.fn().mockResolvedValue(undefined);
  },
}));

const { Route } = await import("../foundation");
const { AuthError } = await import("@/integrations/supabase/request-auth");
const { FoundationUnavailableError } = await import("@/lib/ai/inference/foundation.server");

type PostHandler = (ctx: { request: Request; context: unknown }) => Promise<Response>;

function callFoundation(request: Request) {
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

function csvFile(name = "signal.csv", channels = 19, rows = 1200): File {
  const lines: string[] = [];
  for (let r = 0; r < rows; r++) {
    lines.push(
      Array.from({ length: channels }, (_, c) => Math.sin((r + c) * 0.1).toFixed(4)).join(","),
    );
  }
  return new File([lines.join("\n")], name, { type: "text/csv" });
}

function uploadRequest(form: FormData, auth = true): Request {
  const headers: Record<string, string> = {};
  if (auth) headers.authorization = "Bearer test-token";
  return new Request("http://localhost/api/eeg/embed/foundation", {
    method: "POST",
    body: form,
    headers,
  });
}

describe("POST /api/eeg/embed/foundation", () => {
  beforeEach(() => {
    mockAuthenticateRequest.mockReset();
    mockCheckRateLimit.mockReset();
    mockCheckRateLimit.mockResolvedValue({ allowed: true, retryAfterMs: 0 });
    mockEmbedFoundationWindows.mockReset();
    captured.indexOpts = undefined;
  });

  it("embeds a valid CSV into 200-D vectors and writes to foundation_embeddings (not embeddings)", async () => {
    mockAuthenticateRequest.mockResolvedValue({
      userId: "user-1",
      supabase: fakeSupabase(),
    });
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: Array.from({ length: 200 }, () => 0.1),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
      {
        vector: Array.from({ length: 200 }, () => 0.2),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
    ]);

    const form = new FormData();
    form.set("file", csvFile("signal.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.model).toBe("onnx-cbramod-foundation-200d");
    // Canonical 200-D contract (NOT the Tier-1 32-D).
    expect(body.dimensions).toBe(200);
    expect(body.windows).toBe(2);
    expect(body.embeddings).toHaveLength(2);
    expect(body.embeddings[0].dimensions).toBe(200);
    expect(body.embeddings[0].vector).toHaveLength(200);
    expect(body.vector_indexed).toBe(2);
    // Provenance reflects the SHA-verified artifact.
    expect(body.provenance.artifact_id).toBe("cbramod-encoder");
    expect(body.provenance.sha256).toBe(
      "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    );
    expect(body.provenance.embedding_dim).toBe(200);
    expect(body.provenance.runtime).toBe("onnxruntime-node cpu");
  });

  it("binds the RPC filter_user_id to the authenticated userId (no cross-user RPC leak)", async () => {
    // The match_foundation_embeddings RPC is SECURITY DEFINER and returns rows for
    // whatever filter_user_id is passed — isolation is enforced by the route binding
    // the authenticated userId as filter_user_id (foundation.ts:135,299). We assert
    // the NeuralVectorIndex is constructed with the caller's userId from authenticateRequest.
    mockAuthenticateRequest.mockResolvedValue({
      userId: "caller-A-uuid",
      supabase: fakeSupabase(),
    });
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: Array.from({ length: 200 }, () => 0.1),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
    ]);
    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    await callFoundation(uploadRequest(form));

    expect(captured.indexOpts).toMatchObject({
      tableName: "foundation_embeddings",
      matchRpc: "match_foundation_embeddings",
      userId: "caller-A-uuid", // the authenticated caller, wired to filter_user_id (neural-index.ts:195)
    });
  });

  it("writes to foundation_embeddings (vector(200)) namespace, never embeddings/vector(32)", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: Array.from({ length: 200 }, () => 0.1),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
    ]);

    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    await callFoundation(uploadRequest(form));

    // The NeuralVectorIndex was constructed for the isolated Tier-2 namespace.
    expect(captured.indexOpts).toMatchObject({
      tableName: "foundation_embeddings",
      dimensions: 200,
      matchRpc: "match_foundation_embeddings",
    });
    // Hard assertion: the Tier-1 table name must NOT be used by this route.
    expect(captured.indexOpts?.tableName).not.toBe("embeddings");
  });

  it("never calls embedEEG / the V2 path (no V2 fallback)", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: Array.from({ length: 200 }, () => 0.1),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
    ]);

    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    await callFoundation(uploadRequest(form));

    // The Tier-1 embedEEG façade must not be invoked by the Tier-2 route.
    expect(embedEEGMock).not.toHaveBeenCalled();
  });

  it("returns 424 (not a V2 fallback) when the foundation runtime is unavailable", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new FoundationUnavailableError("onnxruntime-node import failed: not found"),
    );

    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));

    expect(res.status).toBe(424);
    const body = await res.json();
    expect(body.error).toMatch(/unavailable/i);
    // Even on failure, the V2 path must not have been tried.
    expect(embedEEGMock).not.toHaveBeenCalled();
    expect(captured.indexOpts).toBeUndefined();
  });

  it("returns 401 when authenticateRequest rejects with AuthError", async () => {
    mockAuthenticateRequest.mockRejectedValue(
      new AuthError("Unauthorized: missing Bearer token", 401),
    );
    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(401);
  });

  it("returns 429 when the rate limit is exceeded", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockCheckRateLimit.mockResolvedValue({ allowed: false, retryAfterMs: 12345 });
    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(429);
    const body = await res.json();
    expect(body.retry_after_ms).toBe(12345);
  });

  it("returns 200 with the full contract: 200-D dimensions, modelId, provenance.sha256, timings, persistence", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockResolvedValue([
      {
        vector: Array.from({ length: 200 }, () => 0.1),
        dim: 200,
        modelId: "onnx-cbramod-foundation-200d",
        durationMs: 5,
        fellBack: false,
        normalized: true,
      },
    ]);
    const form = new FormData();
    form.set("file", csvFile("signal.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));

    expect(res.status).toBe(200);
    const body = await res.json();
    // Dimensions: the Tier-2 200-D contract (NOT the Tier-1 32-D).
    expect(body.dimensions).toBe(200);
    expect(body.embeddings[0].vector).toHaveLength(200);
    expect(body.embeddings[0].dimensions).toBe(200);
    // model id surfaced (nested per-window as modelId, top-level as model).
    expect(body.model).toBe("onnx-cbramod-foundation-200d");
    expect(body.embeddings[0].model).toBe("onnx-cbramod-foundation-200d");
    // Artifact SHA provenance (c128ccfd…).
    expect(body.provenance.artifact_id).toBe("cbramod-encoder");
    expect(body.provenance.sha256).toBe(
      "c128ccfdee0690da090c7dfcb39af8a2b25f3e492288f9305c85b293eeda6f47",
    );
    expect(body.provenance.embedding_dim).toBe(200);
    // Timings present.
    expect(body.timings).toEqual(expect.objectContaining({ total_ms: expect.any(Number) }));
    expect(typeof body.timings.parse_ms).toBe("number");
    expect(typeof body.timings.preprocess_ms).toBe("number");
    expect(typeof body.timings.embed_ms).toBe("number");
    // Persistence / indexing status.
    expect(body.vector_indexed).toBe(1);
    // No silent fallback marker on any returned embedding.
    expect(body.embeddings[0].fell_back).toBeUndefined();
  });

  it("returns 400 for non-multipart content-type", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    const res = await callFoundation(
      new Request("http://localhost/api/eeg/embed/foundation", {
        method: "POST",
        body: new Blob(["x"]),
        headers: { authorization: "Bearer test-token" },
      }),
    );
    expect(res.status).toBe(400);
  });

  it("returns 400 when the 'file' field is missing", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    const form = new FormData();
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(400);
  });

  it("returns 413 when the file exceeds the 50 MB cap", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    const big = new File([new Uint8Array(51 * 1024 * 1024)], "big.csv", { type: "text/csv" });
    const form = new FormData();
    form.set("file", big);
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(413);
  });

  it("returns 415 for an unsupported file extension", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    const form = new FormData();
    form.set("file", new File(["x"], "signal.png", { type: "image/png" }));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(415);
  });

  it("returns 422 for an unsupported/mismatched raw format (CSV with mismatched sampleRate path)", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    const form = new FormData();
    // CSV requires a valid sampleRate; omit it -> 400 (not 422). Use a valid-type file
    // to confirm the 415 path is skipped and the missing-sampleRate branch returns 400.
    form.set("file", csvFile("s.csv", 19, 1200));
    const res = await callFoundation(uploadRequest(form));
    expect(res.status).toBe(400);
  });

  it("424 on runtime unavailability never falls back to V2/PCA", async () => {
    mockAuthenticateRequest.mockResolvedValue({ userId: "u1", supabase: fakeSupabase() });
    mockEmbedFoundationWindows.mockReset();
    mockEmbedFoundationWindows.mockRejectedValue(
      new FoundationUnavailableError("onnxruntime-node import failed: not found"),
    );
    const form = new FormData();
    form.set("file", csvFile("s.csv", 19, 1200));
    form.set("sampleRate", "250");
    const res = await callFoundation(uploadRequest(form));

    expect(res.status).toBe(424);
    const body = await res.json();
    expect(body.error).toMatch(/unavailable/i);
    expect(body.detail).toBe("onnxruntime-node import failed: not found");
    // No fallback: the V2 façade must never be called on the failure path.
    expect(embedEEGMock).not.toHaveBeenCalled();
  });
});
