/**
 * M39 — API route tests for POST /api/joint2312/sleep/decode.
 *
 * Mirrors the M33/M34 route tests: validates auth, rate limiting,
 * input validation, provenance, and error handling. Uses mock auth + mock
 * decode service to test the route layer in isolation.
 */
import { describe, it, expect, beforeEach, vi, type Mock } from "vitest";

// ─── Mock ONNX adapter + provenance + auth modules ──────────────────────────

const { mockBuildServiceProvenance, mockAuthenticateRequest, mockCheckRateLimit, mockDecodeSleepState } =
  vi.hoisted(() => ({
    mockBuildServiceProvenance: vi.fn(),
    mockAuthenticateRequest: vi.fn(),
    mockCheckRateLimit: vi.fn(),
    mockDecodeSleepState: vi.fn(),
  }));

vi.mock("@/lib/ai/adapters/onnx-adapter", () => {
  return {
    ONNXAdapter: class {
      readonly descriptor: { id: string };
      constructor(opts: { id: string; [key: string]: unknown }) {
        this.descriptor = { id: opts.id };
      }
      predict = vi.fn();
      load = vi.fn().mockResolvedValue(undefined);
      isLoaded = vi.fn().mockReturnValue(true);
      unload = vi.fn().mockResolvedValue(undefined);
      setModel = vi.fn();
    },
  };
});

vi.mock("@/lib/ai/services/provenance.server", () => ({
  buildServiceProvenance: (...args: unknown[]) => mockBuildServiceProvenance(...args),
}));

vi.mock("@/integrations/supabase/request-auth", () => ({
  AuthError: class AuthError extends Error {
    constructor(message: string, public status: number) {
      super(message);
      this.name = "AuthError";
    }
  },
  authenticateRequest: (...args: unknown[]) => mockAuthenticateRequest(...args),
}));

vi.mock("@/integrations/supabase/rate-limit", () => ({
  checkRateLimit: (...args: unknown[]) => mockCheckRateLimit(...args),
}));

vi.mock("@/middleware/cors", () => ({
  handleCors: vi.fn(() => null),
  getCorsHeadersForResponse: vi.fn(() => ({})),
}));

vi.mock("@/middleware/security", () => ({
  applySecurityHeaders: vi.fn((res: Response) => res),
}));

vi.mock("@/lib/logging", () => ({
  log: vi.fn(),
  startTimer: vi.fn(() => ({ end: vi.fn().mockReturnValue(0.1) })),
}));

vi.mock("@/lib/metrics", () => ({
  metrics: {
    tier1ServiceRequestsTotal: { inc: vi.fn() },
    tier1ServiceErrorsTotal: { inc: vi.fn() },
    tier1ServiceLatencyMs: { observe: vi.fn() },
    tier1AuditLogInsertsTotal: { inc: vi.fn() },
    sleepDecodeRequestsTotal: { inc: vi.fn() },
    sleepDecodeErrorsTotal: { inc: vi.fn() },
    sleepDecodeLatencyMs: { observe: vi.fn() },
    sleepStagePredictionsTotal: { inc: vi.fn() },
    sleepConfidenceDistribution: { observe: vi.fn() },
    sleepEmbeddingReusedTotal: { inc: vi.fn() },
    sleepEmbeddingReembeddedTotal: { inc: vi.fn() },
    cognitiveDecodeRequestsTotal: { inc: vi.fn() },
    cognitiveDecodeErrorsTotal: { inc: vi.fn() },
    cognitiveDecodeLatencyMs: { observe: vi.fn() },
    cognitiveWorkloadPredictionsTotal: { inc: vi.fn() },
    cognitiveConfidenceDistribution: { observe: vi.fn() },
    cognitiveEmbeddingReusedTotal: { inc: vi.fn() },
    cognitiveEmbeddingReembeddedTotal: { inc: vi.fn() },
    anomalyDetectRequestsTotal: { inc: vi.fn() },
    anomalyDetectErrorsTotal: { inc: vi.fn() },
    anomalyDetectLatencyMs: { observe: vi.fn() },
    anomalyScoresTotal: { inc: vi.fn() },
    anomalyConfidenceDistribution: { observe: vi.fn() },
    anomalyEmbeddingReusedTotal: { inc: vi.fn() },
    anomalyEmbeddingReembeddedTotal: { inc: vi.fn() },
  },
}));

// Mock the sleep server to isolate route-layer behavior
class MockSleepDecodeError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = "SleepDecodeError";
  }
}

vi.mock("@/lib/ai/inference/sleep.server", () => ({
  decodeSleepState: (...args: unknown[]) => mockDecodeSleepState(...args),
  SleepDecodeError: MockSleepDecodeError,
  SLEEP_TIMEOUT_MS: 10_000,
  SLEEP_SERVICE: "sleep-staging",
  SLEEP_VERSION: "v0.1.0",
  SLEEP_DEFAULT_HEAD_ID: "sleep-staging-v1",
  SLEEP_DEFAULT_CI_MARGIN: 0.08,
  SLEEP_STAGES_5: ["W", "N1", "N2", "N3", "REM"],
}));

describe("POST /api/joint2312/sleep/decode route", () => {
  let mockRequest: Partial<Request>;
  let mockSupabase: unknown;

  beforeEach(() => {
    vi.clearAllMocks();

    mockSupabase = { from: vi.fn(), rpc: vi.fn() };
    mockRequest = {
      headers: {
        get: vi.fn((k: string) => {
          if (k === "content-type") return "application/json";
          if (k === "origin") return "http://localhost:3000";
          return null;
        }),
      },
      json: vi.fn(),
    };

    // Default: authenticated, rate-limited OK
    mockAuthenticateRequest.mockResolvedValue({
      userId: "test-user-id",
      supabase: mockSupabase,
    });
    mockCheckRateLimit.mockResolvedValue({
      allowed: true,
      retryAfterMs: 0,
    });

    // Default decode success
    mockDecodeSleepState.mockResolvedValue({
      service: "sleep-staging",
      model: "onnx-cbramod-joint-2312",
      head: "sleep-staging-v1",
      head_version: "0.1.0",
      provenance: {
        service: "sleep-staging",
        service_version: "v0.1.0",
        embedding_model: "onnx-cbramod-joint-2312",
        embedding_dim: 2312,
        task_head_id: "sleep-staging-v1",
        timestamp: new Date().toISOString(),
        artifact_shas: {
          cbramod: "c128ccfd...",
          v2: "18644de1...",
          pca: "deterministic-pca-v1",
          eegpt: "a92daf44...",
        },
        block_weights: { cbramod: 0.3062, v2: 0.1434, pca: 0.1519, eegpt: 0.3985 },
        component_dims: { cbramod: 200, v2: 32, pca: 32, eegpt: 2048 },
      },
      results: [{
        stage_id: 3,
        stage: "N3",
        probabilities: [0.1, 0.1, 0.1, 0.6, 0.1],
        confidence: 0.6,
        confidence_interval: [0.52, 0.68],
        metric: "sleep-stages",
      }],
      metadata: { embedding_reused: false },
      timings: { inference_ms: 0.52, total_ms: 1.0 },
    });
  });

  // Helper to call the route handler
  async function callRoute(request: Partial<Request>) {
    const mod = await import("../decode");
    const handlers = mod.Route.options.server!.handlers as unknown as {
      POST: (ctx: { request: Request; context: unknown }) => Promise<Response>;
    };
    return handlers.POST({ request: request as Request, context: {} });
  }

  it("returns 400 when content-type is not JSON", async () => {
    (mockRequest.headers as Record<string, unknown>).get = vi.fn((k: string) => {
      if (k === "content-type") return "text/plain";
      return null;
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("expected application/json");
  });

  it("returns 400 when body is invalid JSON", async () => {
    mockRequest.json = vi.fn().mockRejectedValue(new Error("Invalid JSON"));

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toBe("Invalid JSON body");
  });

  it("returns 400 when query_type is invalid", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      query_type: "invalid",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("query_type");
  });

  it("returns 400 when neither embedding_id nor query_embedding provided", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toBe("Either embedding_id or query_embedding must be provided");
  });

  it("returns 400 when query_embedding dimension is wrong", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: [0.1, 0.2, 0.3],
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("2312-dimensional");
  });

  it("returns 200 with sleep staging results for valid request", async () => {
    const embedding = new Array(2312).fill(0.1);
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: embedding,
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(200);
    const body = await result.json();
    expect(body.service).toBe("sleep-staging");
    expect(body.head).toBe("sleep-staging-v1");
    expect(body.results[0].stage).toBe("N3");
    expect(body.results[0].stage_id).toBe(3);
    expect(body.results[0].probabilities).toHaveLength(5);
  });

  it("returns 401 on AuthError", async () => {
    const { AuthError } = await import("@/integrations/supabase/request-auth");
    mockAuthenticateRequest.mockReset();
    mockAuthenticateRequest.mockRejectedValue(
      new (AuthError as unknown as new (msg: string, status: number) => Error)(
        "Unauthorized",
        401,
      ),
    );
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(401);
  });

  it("returns 429 when rate limited", async () => {
    mockCheckRateLimit.mockResolvedValue({
      allowed: false,
      retryAfterMs: 30000,
    });
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(429);
    const body = await result.json();
    expect(body.retry_after_ms).toBe(30000);
  });

  it("returns 400 on SleepDecodeError", async () => {
    mockDecodeSleepState.mockRejectedValue(
      new MockSleepDecodeError(
        "embedding_id not found or access denied",
        "EMBEDDING_NOT_FOUND",
      ),
    );
    mockRequest.json = vi.fn().mockResolvedValue({
      embedding_id: "nonexistent-id",
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.code).toBe("EMBEDDING_NOT_FOUND");
  });

  it("returns 500 on unknown error", async () => {
    mockDecodeSleepState.mockRejectedValue(new Error("Unexpected failure"));
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      query_type: "sleep-stages",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(500);
    const body = await result.json();
    expect(body.error).toBe("An error occurred during sleep staging.");
  });

  it("returns 408 on timeout", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      query_type: "sleep-stages",
    });
    // Make decode hang indefinitely
    mockDecodeSleepState.mockImplementation(() => new Promise(() => {}));

    // Use fake timers to trigger the 10s timeout without waiting
    vi.useFakeTimers();

    const callPromise = callRoute(mockRequest);
    vi.advanceTimersByTime(10001);
    await vi.runAllTimersAsync();

    const result = await callPromise;
    expect(result.status).toBe(408);
    const body = await result.json();
    expect(body.error).toContain("timeout");

    vi.useRealTimers();
  }, 10000);
});
