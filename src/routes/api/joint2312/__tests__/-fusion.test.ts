/**
 * M41 — API route tests for POST /api/joint2312/fusion.
 *
 * Tests the multi-task fusion route: validates auth, rate limiting, input
 * validation, embedding resolution, provenance, partial heads, and error
 * handling. Uses mock auth + mock decodeJoint2312 to test the route layer
 * in isolation.
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

// ─── Mock ONNX adapter + downstream dependencies ─────────────────────────────

const mockPredict = vi.fn();

vi.mock("@/lib/ai/adapters/onnx-adapter", () => {
  return {
    ONNXAdapter: class {
      readonly descriptor: { id: string };
      constructor(opts: { id: string; [key: string]: unknown }) {
        this.descriptor = { id: opts.id };
      }
      predict = mockPredict;
      load = vi.fn().mockResolvedValue(undefined);
      isLoaded = vi.fn().mockReturnValue(true);
      unload = vi.fn().mockResolvedValue(undefined);
      setModel = vi.fn();
    },
  };
});

const mockBuildServiceProvenance = vi.fn();

vi.mock("@/lib/ai/services/provenance.server", () => ({
  buildServiceProvenance: (...args: unknown[]) => mockBuildServiceProvenance(...args),
}));

const mockAuthenticateRequest = vi.fn();
const mockCheckRateLimit = vi.fn();
const mockDecodeJoint2312 = vi.fn();

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
  },
}));

// Mock decodeJoint2312 to isolate route-layer behavior
vi.mock("@/lib/ai/inference/joint-fusion.server", () => ({
  decodeJoint2312: (...args: unknown[]) => mockDecodeJoint2312(...args),
  JOINT_FUSION_SERVICE: "joint-fusion",
  JOINT_FUSION_VERSION: "v0.1.0",
  JOINT_FUSION_TIMEOUT_MS: 10_000,
}));

// ─── Tests ─────────────────────────────────────────────────────────────────

describe("POST /api/joint2312/fusion route", () => {
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

    // Default: authenticated, rate-limit OK
    mockAuthenticateRequest.mockResolvedValue({
      userId: "test-user-id",
      supabase: mockSupabase,
    });
    mockCheckRateLimit.mockResolvedValue({
      allowed: true,
      retryAfterMs: 0,
    });

    // Default fusion decode success (all 4 heads)
    mockDecodeJoint2312.mockResolvedValue({
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      provenance: {
        service: "joint-fusion",
        service_version: "v0.1.0",
        embedding_model: "onnx-cbramod-joint-2312",
        embedding_dim: 2312,
        task_head_id: "joint-fusion-all-v1",
        task_head_version: "0.1.0",
        task_head_sha256: "multi-probe-fusion",
        experiment_id: "m41-multi-task-fusion",
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
      results: {
        cognitive: [{ score: 0.73, confidence_interval: [0.65, 0.81], confidence: 0.84, metric: "workload" }],
        anomaly: [{ score: 0.15, is_anomalous: false, confidence: 0.90, confidence_interval: [0.10, 0.20], metric: "artifact" }],
        sleep_staging: [{ stage_id: 3, stage: "N3", probabilities: [0.05, 0.10, 0.20, 0.45, 0.20], confidence: 0.45, confidence_interval: [0.37, 0.53], metric: "sleep-stages" }],
        sleep_quality: [{ score: 0.75, band: "good", confidence_interval: [0.65, 0.85], confidence: 0.80, metric: "sleep-quality" }],
      },
      metadata: {
        embedding_reused: false,
        heads_run: ["cognitive", "anomaly", "sleep-staging", "sleep-quality"],
        probes: [
          { id: "cognitive-linear-v1", sha256: "abc123" },
          { id: "anomaly-mahalanobis-v1", sha256: "def456" },
          { id: "sleep-staging-v1", sha256: "9da4ea37" },
          { id: "sleep-quality-v1", sha256: "5fb7400f" },
        ],
      },
      timings: { inference_ms: 1.2, total_ms: 2.0 },
    });
  });

  // Helper to call the route handler
  async function callRoute(request: Partial<Request>) {
    const mod = await import("../fusion");
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

  it("returns 400 when neither embedding_id nor query_embedding provided", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      heads: ["cognitive"],
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toBe("Either embedding_id or query_embedding must be provided");
  });

  it("returns 400 when query_embedding dimension is wrong", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: [0.1, 0.2, 0.3],
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("2312-dimensional");
  });

  it("returns 400 when heads contains invalid value", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      heads: ["cognitive", "invalid-head"],
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("Invalid head");
  });

  it("returns 200 with full fusion results for valid query_embedding request", async () => {
    const embedding = new Array(2312).fill(0.1);
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: embedding,
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(200);
    const body = await result.json();
    expect(body.service).toBe("joint-fusion");
    expect(body.model).toBe("onnx-cbramod-joint-2312");
    expect(body.results.cognitive).toBeDefined();
    expect(body.results.anomaly).toBeDefined();
    expect(body.results.sleep_staging).toBeDefined();
    expect(body.results.sleep_quality).toBeDefined();
    expect(body.metadata.heads_run).toEqual(["cognitive", "anomaly", "sleep-staging", "sleep-quality"]);
  });

  it("returns 200 with embedding_id echoed in response", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      embedding_id: "existing-embedding-id",
    });
    mockDecodeJoint2312.mockResolvedValueOnce({
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      embedding_id: "existing-embedding-id",
      provenance: { service: "joint-fusion", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: 2312 },
      results: {},
      metadata: { embedding_reused: true, heads_run: [], probes: [] },
      timings: { embed_ms: 0.5, inference_ms: 0.3, total_ms: 1.0 },
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(200);
    const body = await result.json();
    expect(body.embedding_id).toBe("existing-embedding-id");
    expect(body.metadata.embedding_reused).toBe(true);
    expect(body.timings.embed_ms).toBeDefined();
  });

  it("returns 200 with partial heads when only some requested", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
      heads: ["sleep-staging", "sleep-quality"],
    });
    mockDecodeJoint2312.mockResolvedValueOnce({
      service: "joint-fusion",
      model: "onnx-cbramod-joint-2312",
      head_version: "v0.1.0",
      provenance: { service: "joint-fusion", embedding_model: "onnx-cbramod-joint-2312", embedding_dim: 2312 },
      results: {
        sleep_staging: [{ stage_id: 3, stage: "N3", probabilities: [0.05, 0.10, 0.20, 0.45, 0.20], confidence: 0.45, confidence_interval: [0.37, 0.53], metric: "sleep-stages" }],
        sleep_quality: [{ score: 0.75, band: "good", confidence_interval: [0.65, 0.85], confidence: 0.80, metric: "sleep-quality" }],
      },
      metadata: { embedding_reused: false, heads_run: ["sleep-staging", "sleep-quality"], probes: [] },
      timings: { inference_ms: 0.8, total_ms: 1.5 },
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(200);
    const body = await result.json();
    expect(body.metadata.heads_run).toEqual(["sleep-staging", "sleep-quality"]);
  });

  it("returns 401 on AuthError", async () => {
    const { AuthError } = await import("@/integrations/supabase/request-auth");
    mockAuthenticateRequest.mockReset();
    mockAuthenticateRequest.mockRejectedValue(
      new (AuthError as unknown as new (msg: string, status: number) => Error)("Unauthorized", 401),
    );
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(401);
  });

  it("returns 401 on unexpected auth error", async () => {
    mockAuthenticateRequest.mockReset();
    mockAuthenticateRequest.mockRejectedValue(new Error("Supabase down"));
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(401);
    const body = await result.json();
    expect(body.error).toBe("Authentication failed.");
  });

  it("returns 429 when rate limited", async () => {
    mockCheckRateLimit.mockResolvedValue({
      allowed: false,
      retryAfterMs: 30000,
    });
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(429);
    const body = await result.json();
    expect(body.retry_after_ms).toBe(30000);
  });

  it("returns 400 on INVALID_REQUEST decode error", async () => {
    // The fusion server throws plain Errors with code prefixed in the message.
    // The route catches them and returns 400 with the error message.
    mockDecodeJoint2312.mockRejectedValueOnce(
      new Error("INVALID_REQUEST: Either query_embedding or embedding_id must be provided"),
    );
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("INVALID_REQUEST");
    expect(body.code).toBe("DECODE_ERROR");
  });

  it("returns 400 on EMBEDDING_NOT_FOUND decode error", async () => {
    mockDecodeJoint2312.mockRejectedValueOnce(
      new Error("EMBEDDING_NOT_FOUND: embedding_id not found or access denied"),
    );
    mockRequest.json = vi.fn().mockResolvedValue({
      embedding_id: "nonexistent",
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toContain("EMBEDDING_NOT_FOUND");
    expect(body.code).toBe("DECODE_ERROR");
  });

  it("returns 400 on unknown decode error (non-structured Error)", async () => {
    // The fusion route catches all decode errors as 400, not 500, because
    // decodeJoint2312 throws plain Errors (not typed domain errors).
    mockDecodeJoint2312.mockRejectedValueOnce(new Error("Unexpected internal failure"));
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(400);
    const body = await result.json();
    expect(body.error).toBe("Unexpected internal failure");
    expect(body.code).toBe("DECODE_ERROR");
  });

  it("returns 408 on timeout", async () => {
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: new Array(2312).fill(0.1),
    });
    // Make decode hang indefinitely
    mockDecodeJoint2312.mockImplementation(() => new Promise(() => {}));

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

  it("includes unified provenance with shared embed artifacts", async () => {
    const embedding = new Array(2312).fill(0.1);
    mockRequest.json = vi.fn().mockResolvedValue({
      query_embedding: embedding,
    });

    const result = await callRoute(mockRequest);
    expect(result.status).toBe(200);
    const body = await result.json();
    expect(body.provenance).toBeDefined();
    expect(body.provenance.service).toBe("joint-fusion");
    expect(body.provenance.embedding_model).toBe("onnx-cbramod-joint-2312");
    expect(body.provenance.embedding_dim).toBe(2312);
  });
});
