/**
 * M33 — Tests for the Cognitive task-head registry.
 *
 * Tests that the cognitive workload probe is properly registered in the
 * shared TaskHeadRegistry, with correct dimensions, SHAs, and metrics.
 */
import { describe, it, expect, beforeAll } from "vitest";
import {
  registerTaskHead,
  getTaskHead,
  hasTaskHead,
  getTaskHeadsByService,
  getDefaultTaskHead,
  type TaskHeadDescriptor,
} from "../registry";
import {
  COGNITIVE_HEADS,
  COGNITIVE_LINEAR_PROBE_JOINT_2312,
  COGNITIVE_LINEAR_PROBE_V2_32,
  registerCognitiveHeads,
  getDefaultCognitiveHead,
} from "../cognitive.registry";

describe("Cognitive TaskHeadRegistry", () => {
  beforeAll(() => {
    registerCognitiveHeads();
  });

  it("registers the cognitive-linear-v1 head with correct id", () => {
    expect(hasTaskHead(COGNITIVE_LINEAR_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("cognitive-linear-v1");
    expect(head).toBeDefined();
    expect(head?.service).toBe("cognitive-intelligence");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1); // workload prediction
  });

  it("cognitive-linear-v1 uses server inference (onnxruntime-node)", () => {
    const head = getTaskHead("cognitive-linear-v1");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe("ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32");
  });

  it("cognitive-linear-v1 training metrics match M33 results", () => {
    const head = getTaskHead("cognitive-linear-v1");
    expect(head?.training.metrics).toBeDefined();
    expect(head?.training.metrics?.r2).toBeCloseTo(0.7348, 2);
    expect(head?.training.metrics?.pearson_r).toBeCloseTo(0.8874, 2);
    expect(head?.training.dataset).toContain("EEGMMIDB");
    expect(head?.experimentId).toBe("m33-cognitive-workload-probe");
  });

  it("registers V2-32 browser fallback head", () => {
    expect(hasTaskHead("cognitive-linear-v2-32d")).toBe(true);
    const head = getTaskHead("cognitive-linear-v2-32d");
    expect(head?.inputDim).toBe(32);
    expect(head?.outputDim).toBe(1);
    expect(head?.inferenceTarget).toBe("both");
    expect(head?.service).toBe("cognitive-intelligence");
  });

  it("COGNITIVE_HEADS contains all registered probes", () => {
    expect(COGNITIVE_HEADS.length).toBe(3);
    expect(COGNITIVE_HEADS.map((h) => h.id)).toContain("cognitive-linear-v1");
    expect(COGNITIVE_HEADS.map((h) => h.id)).toContain("cognitive-linear-v2-32d");
  });

  it("getTaskHeadsByService returns all cognitive heads", () => {
    const heads = getTaskHeadsByService("cognitive-intelligence");
    expect(heads.length).toBeGreaterThanOrEqual(2);
    expect(heads.every((h) => h.service === "cognitive-intelligence")).toBe(true);
  });

  it("getDefaultCognitiveHead returns server head by default", () => {
    const head = getDefaultCognitiveHead("server");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(2312);
    expect(head?.inferenceTarget).toBe("server");
  });

  it("getDefaultCognitiveHead browser returns V2-32 head", () => {
    const head = getDefaultCognitiveHead("browser");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(32);
    expect(head?.inferenceTarget).toBe("both");
  });

  it("all cognitive heads have correct block weight provenance", () => {
    for (const head of COGNITIVE_HEADS) {
      expect(head.training.dataset).toBeTruthy();
      expect(head.training.protocol).toBeTruthy();
      expect(head.training.metrics).toBeDefined();
    }
  });

  it("registerCognitiveHeads is idempotent", () => {
    registerCognitiveHeads();
    registerCognitiveHeads();
    // Should still have the same head
    const head = getTaskHead("cognitive-linear-v1");
    expect(head).toBeDefined();
    expect(head?.version).toBe("0.1.0");
  });
});
