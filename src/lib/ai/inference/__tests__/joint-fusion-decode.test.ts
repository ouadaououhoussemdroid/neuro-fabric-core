/**
 * M41 — Unit tests for the Joint-2312 Multi-Task Fusion service.
 *
 * Tests `decodeJoint2312()` — the embed-once-reuse-many fusion function that
 * resolves a single Joint-2312 embedding and dispatches it to all 4 Tier-1+Tier-2
 * task probes (cognitive, anomaly, sleep-staging, sleep-quality) in parallel via
 * Promise.all. All downstream probes are mocked to isolate the fusion logic.
 *
 * Mirrors the M39/M40 service test patterns (mock ONNX adapter + mock Supabase).
 */
import { describe, it, expect, beforeEach, vi } from "vitest";

// ─── Mock all downstream probe modules ──────────────────────────────────────
// The fusion server calls decodeCognitiveState, detectAnomalies, decodeSleepState,
// and decodeSleepQuality. We mock these to test the fusion orchestration only.

const mockDecodeCognitiveState = vi.fn();
const mockDetectAnomalies = vi.fn();
const mockDecodeSleepState = vi.fn();
const mockDecodeSleepQuality = vi.fn();
const mockBuildServiceProvenance = vi.fn();

vi.mock("@/lib/ai/inference/cognitive.server", () => ({
  decodeCognitiveState: (...args: unknown[]) => mockDecodeCognitiveState(...args),
  COGNITIVE_SERVICE: "cognitive-intelligence",
}));

vi.mock("@/lib/ai/inference/anomaly.server", () => ({
  detectAnomalies: (...args: unknown[]) => mockDetectAnomalies(...args),
  ANOMALY_SERVICE: "anomaly-detection",
}));

vi.mock("@/lib/ai/inference/sleep.server", () => ({
  decodeSleepState: (...args: unknown[]) => mockDecodeSleepState(...args),
  decodeSleepQuality: (...args: unknown[]) => mockDecodeSleepQuality(...args),
  SLEEP_SERVICE: "sleep-staging",
  SLEEP_TIMEOUT_MS: 10_000,
  SLEEP_DEFAULT_HEAD_ID: "sleep-staging-v1",
  SLEEP_QUALITY_DEFAULT_HEAD_ID: "sleep-quality-v1",
  SLEEP_QUALITY_MIN: 0,
  SLEEP_QUALITY_MAX: 1,
}));

vi.mock("@/lib/ai/services/provenance.server", () => ({
  buildServiceProvenance: (...args: unknown[]) => mockBuildServiceProvenance(...args),
}));

vi.mock("@/lib/logging", () => ({
  log: vi.fn(),
  startTimer: vi.fn(() => ({ end: vi.fn().mockReturnValue(0.5) })),
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

// ─── Import after mocks ─────────────────────────────────────────────────────

import {
  decodeJoint2312,
  type JointFusionRequest,
  JOINT_FUSION_SERVICE,
  JOINT_FUSION_VERSION,
  JOINT_FUSION_TIMEOUT_MS,
} from "../joint-fusion.server";
import { JOINT_2312_EMBEDDING_DIM } from "../joint.server";

// ─── Mock Supabase client ───────────────────────────────────────────────────

function createMockSupabase(existingEmbedding?: number[]) {
  const embedding = existingEmbedding ??
    new Array(JOINT_2312_EMBEDDING_DIM).fill(0).map(() => Math.random());
  return {
    from: vi.fn((table: string) => ({
      select: vi.fn(() => ({
        eq: vi.fn(() => ({
          single: vi.fn().mockResolvedValue({
            data: { embedding },
            error: null,
          }),
        })),
      })),
    })),
    rpc: vi.fn((fn: string, args: Record<string, unknown>) =>
      Promise.resolve({ data: [], error: null }),
    ),
  };
}

// ─── Shared mock results ────────────────────────────────────────────────────

const MOCK_COGNITIVE_RESULT = {
  service: "cognitive-intelligence",
  model: "onnx-cbramod-joint-2312",
  head: "cognitive-linear-v1",
  head_version: "0.1.0",
  provenance: { service: "cognitive-intelligence", task_head_id: "cognitive-linear-v1" },
  results: [{ score: 0.73, confidence_interval: [0.65, 0.81], confidence: 0.84, metric: "workload" }],
  metadata: { embedding_reused: false, probe_sha256: "abc123" },
  timings: { inference_ms: 0.5, total_ms: 1.0 },
};

const MOCK_ANOMALY_RESULT = {
  service: "anomaly-detection",
  model: "onnx-cbramod-joint-2312",
  head: "anomaly-mahalanobis-v1",
  head_version: "0.1.0",
  provenance: { service: "anomaly-detection", task_head_id: "anomaly-mahalanobis-v1" },
  results: [{ score: 0.15, is_anomalous: false, confidence: 0.90, confidence_interval: [0.10, 0.20], metric: "artifact" }],
  metadata: { embedding_reused: false, probe_sha256: "def456" },
  timings: { inference_ms: 0.3, total_ms: 0.8 },
};

const MOCK_SLEEP_STAGING_RESULT = {
  service: "sleep-staging",
  model: "onnx-cbramod-joint-2312",
  head: "sleep-staging-v1",
  head_version: "0.1.0",
  provenance: { service: "sleep-staging", task_head_id: "sleep-staging-v1" },
  results: [{ stage_id: 3, stage: "N3", probabilities: [0.05, 0.10, 0.20, 0.45, 0.20], confidence: 0.45, confidence_interval: [0.37, 0.53], metric: "sleep-stages" }],
  metadata: { embedding_reused: false, probe_sha256: "9da4ea37" },
  timings: { inference_ms: 0.3, total_ms: 0.8 },
};

const MOCK_SLEEP_QUALITY_RESULT = {
  service: "sleep-staging",
  model: "onnx-cbramod-joint-2312",
  head: "sleep-quality-v1",
  head_version: "0.1.0",
  provenance: { service: "sleep-staging", task_head_id: "sleep-quality-v1" },
  results: [{ score: 0.75, band: "good", confidence_interval: [0.65, 0.85], confidence: 0.80, metric: "sleep-quality" }],
  metadata: { embedding_reused: false, probe_sha256: "5fb7400f" },
  timings: { inference_ms: 0.4, total_ms: 0.9 },
};

// ─── Tests ──────────────────────────────────────────────────────────────────

describe("decodeJoint2312 — Multi-Task Fusion Service", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock for buildServiceProvenance (used by fusion for its own provenance)
    mockBuildServiceProvenance.mockReturnValue({
      service: JOINT_FUSION_SERVICE,
      service_version: JOINT_FUSION_VERSION,
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
    });

    // Default: all 4 probe mocks succeed
    mockDecodeCognitiveState.mockResolvedValue(MOCK_COGNITIVE_RESULT);
    mockDetectAnomalies.mockResolvedValue(MOCK_ANOMALY_RESULT);
    mockDecodeSleepState.mockResolvedValue(MOCK_SLEEP_STAGING_RESULT);
    mockDecodeSleepQuality.mockResolvedValue(MOCK_SLEEP_QUALITY_RESULT);
  });

  it("decodes all 4 task heads from a raw 2312-D query_embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.service).toBe(JOINT_FUSION_SERVICE);
    expect(result.model).toBe("onnx-cbramod-joint-2312");
    expect(result.results.cognitive).toBeDefined();
    expect(result.results.anomaly).toBeDefined();
    expect(result.results.sleep_staging).toBeDefined();
    expect(result.results.sleep_quality).toBeDefined();
  });

  it("runs all 4 probes in parallel via Promise.all", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    await decodeJoint2312(req, supabase, "test-user");

    expect(mockDecodeCognitiveState).toHaveBeenCalledTimes(1);
    expect(mockDetectAnomalies).toHaveBeenCalledTimes(1);
    expect(mockDecodeSleepState).toHaveBeenCalledTimes(1);
    expect(mockDecodeSleepQuality).toHaveBeenCalledTimes(1);
  });

  it("passes the same embedding to all 4 probe functions", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    await decodeJoint2312(req, supabase, "test-user");

    // Each probe should receive the same query_embedding
    const cognitiveArg = mockDecodeCognitiveState.mock.calls[0][0] as { query_embedding: number[] };
    const anomalyArg = mockDetectAnomalies.mock.calls[0][0] as { query_embedding: number[] };
    const stagingArg = mockDecodeSleepState.mock.calls[0][0] as { query_embedding: number[] };
    const qualityArg = mockDecodeSleepQuality.mock.calls[0][0] as { query_embedding: number[] };

    expect(cognitiveArg.query_embedding).toEqual(embedding);
    expect(anomalyArg.query_embedding).toEqual(embedding);
    expect(stagingArg.query_embedding).toEqual(embedding);
    expect(qualityArg.query_embedding).toEqual(embedding);
  });

  it("decodes from embedding_id with embed-once-reuse", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.02);
    const supabase = createMockSupabase(embedding);
    const req: JointFusionRequest = { embedding_id: "test-embedding-id" };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(true);
    expect(result.embedding_id).toBe("test-embedding-id");
    expect(result.timings.embed_ms).toBeDefined();
  });

  it("sets embedding_reused=false for raw embedding", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.metadata.embedding_reused).toBe(false);
    expect(result.timings.embed_ms).toBeUndefined();
  });

  it("throws INVALID_REQUEST when neither embedding provided", async () => {
    const supabase = createMockSupabase();
    const req: JointFusionRequest = {};

    await expect(decodeJoint2312(req, supabase, "test-user")).rejects.toThrow(
      "INVALID_REQUEST",
    );
  });

  it("throws DIMENSION_MISMATCH for wrong embedding dimension", async () => {
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: [0.1, 0.2, 0.3] };

    await expect(decodeJoint2312(req, supabase, "test-user")).rejects.toThrow(
      "DIMENSION_MISMATCH",
    );
  });

  it("throws EMBEDDING_NOT_FOUND when embedding_id doesn't exist", async () => {
    const supabase = {
      from: vi.fn((table: string) => ({
        select: vi.fn(() => ({
          eq: vi.fn(() => ({
            single: vi.fn().mockResolvedValue({
              data: null,
              error: new Error("Not found"),
            }),
          })),
        })),
      })),
      rpc: vi.fn(),
    };

    const req: JointFusionRequest = { embedding_id: "nonexistent" };

    await expect(decodeJoint2312(req, supabase, "test-user")).rejects.toThrow(
      "EMBEDDING_NOT_FOUND",
    );
  });

  it("supports partial head selection", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = {
      query_embedding: embedding,
      heads: ["sleep-staging", "sleep-quality"],
    };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.metadata.heads_run).toEqual(["sleep-staging", "sleep-quality"]);
    expect(mockDecodeCognitiveState).not.toHaveBeenCalled();
    expect(mockDetectAnomalies).not.toHaveBeenCalled();
    expect(mockDecodeSleepState).toHaveBeenCalledTimes(1);
    expect(mockDecodeSleepQuality).toHaveBeenCalledTimes(1);
  });

  it("includes probe SHAs in metadata.probes", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.metadata.probes.length).toBe(4);
    const probeIds = result.metadata.probes.map((p) => p.id);
    expect(probeIds).toContain("cognitive-linear-v1");
    expect(probeIds).toContain("anomaly-mahalanobis-v1");
    expect(probeIds).toContain("sleep-staging-v1");
    expect(probeIds).toContain("sleep-quality-v1");
  });

  it("collects probe SHA256 in probes array", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    const shaSet = result.metadata.probes.map((p) => p.sha256);
    expect(shaSet).toContain("abc123");
    expect(shaSet).toContain("def456");
    expect(shaSet).toContain("9da4ea37");
    expect(shaSet).toContain("5fb7400f");
  });

  it("handles partial probe failure gracefully — failed head omitted", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    // Make cognitive fail
    mockDecodeCognitiveState.mockRejectedValueOnce(new Error("Cognitive probe failed"));

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.results.cognitive).toBeUndefined();
    expect(result.results.anomaly).toBeDefined();
    expect(result.results.sleep_staging).toBeDefined();
    expect(result.results.sleep_quality).toBeDefined();
    expect(result.metadata.heads_run).not.toContain("cognitive");
    expect(result.metadata.heads_run).toContain("anomaly");
  });

  it("returns unified provenance with shared embed provenance", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.provenance).toBeDefined();
    expect(result.provenance.service).toBe(JOINT_FUSION_SERVICE);
    expect(result.provenance.embedding_model).toBe("onnx-cbramod-joint-2312");
    expect(result.provenance.embedding_dim).toBe(2312);
    expect(result.provenance.artifact_shas).toHaveProperty("cbramod");
    expect(result.provenance.artifact_shas).toHaveProperty("v2");
    expect(result.provenance.artifact_shas).toHaveProperty("pca");
    expect(result.provenance.artifact_shas).toHaveProperty("eegpt");
    expect(result.provenance.block_weights.cbramod).toBeCloseTo(0.3062, 3);
    expect(result.provenance.block_weights.eegpt).toBeCloseTo(0.3985, 3);
  });

  it("includes timings in response", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.timings.inference_ms).toBeDefined();
    expect(result.timings.total_ms).toBeDefined();
    expect(result.timings.embed_ms).toBeUndefined(); // not reused
  });

  it("returns full response shape with all required top-level fields", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result).toHaveProperty("service");
    expect(result).toHaveProperty("model");
    expect(result).toHaveProperty("head_version");
    expect(result).toHaveProperty("provenance");
    expect(result).toHaveProperty("results");
    expect(result).toHaveProperty("metadata");
    expect(result).toHaveProperty("timings");
    expect(result.metadata).toHaveProperty("embedding_reused");
    expect(result.metadata).toHaveProperty("heads_run");
    expect(result.metadata).toHaveProperty("probes");
  });

  it("defaults heads to all 4 when heads param is omitted", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.metadata.heads_run).toEqual(["cognitive", "anomaly", "sleep-staging", "sleep-quality"]);
  });

  it("passes userId to all probe functions", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    await decodeJoint2312(req, supabase, "user-abc-123");

    expect(mockDecodeCognitiveState).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding }),
      supabase,
      "user-abc-123",
    );
    expect(mockDetectAnomalies).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding }),
      supabase,
      "user-abc-123",
    );
    expect(mockDecodeSleepState).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding }),
      supabase,
      "user-abc-123",
    );
    expect(mockDecodeSleepQuality).toHaveBeenCalledWith(
      expect.objectContaining({ query_embedding: embedding }),
      supabase,
      "user-abc-123",
    );
  });

  it("handles all 4 probes failing gracefully", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    mockDecodeCognitiveState.mockRejectedValueOnce(new Error("Cognitive failed"));
    mockDetectAnomalies.mockRejectedValueOnce(new Error("Anomaly failed"));
    mockDecodeSleepState.mockRejectedValueOnce(new Error("Staging failed"));
    mockDecodeSleepQuality.mockRejectedValueOnce(new Error("Quality failed"));

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.results.cognitive).toBeUndefined();
    expect(result.results.anomaly).toBeUndefined();
    expect(result.results.sleep_staging).toBeUndefined();
    expect(result.results.sleep_quality).toBeUndefined();
    expect(result.metadata.heads_run).toEqual([]);
    expect(result.metadata.probes).toEqual([]);
  });

  it("sets head_version to JOINT_FUSION_VERSION", async () => {
    const embedding = new Array(JOINT_2312_EMBEDDING_DIM).fill(0.01);
    const supabase = createMockSupabase();
    const req: JointFusionRequest = { query_embedding: embedding };

    const result = await decodeJoint2312(req, supabase, "test-user");

    expect(result.head_version).toBe(JOINT_FUSION_VERSION);
  });
});
