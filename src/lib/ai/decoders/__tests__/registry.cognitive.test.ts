/**
 * M33-Scientific-Reboot — Tests for the Cognitive task-head registry.
 *
 * Tests that the cognitive classification probe is properly registered in the
 * shared TaskHeadRegistry, with correct dimensions, SHAs, scientific status,
 * and reclassified previous metrics.
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

  it("registers the cognitive-linear-v2 head with correct id", () => {
    expect(hasTaskHead(COGNITIVE_LINEAR_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("cognitive-linear-v2");
    expect(head).toBeDefined();
    expect(head?.service).toBe("cognitive-intelligence");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(4); // 4-class MI classification
  });

  it("cognitive-linear-v2 uses server inference (onnxruntime-node)", () => {
    const head = getTaskHead("cognitive-linear-v2");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe("ab8bc6389d98a9461fc7f0f4fea47c3cd9860595c305879351ad0cf6592a6b32");
  });

  it("cognitive-linear-v2 training metrics reflect GENUINE labels (not proxy R²)", () => {
    const head = getTaskHead("cognitive-linear-v2");
    expect(head?.training.metrics).toBeDefined();
    // V2: accuracy-based on genuine MI labels, NOT R²=0.7348 from proxy
    expect(head?.training.metrics?.accuracy).toBeCloseTo(0.32, 1);
    expect(head?.training.metrics?.macro_f1).toBeCloseTo(0.30, 1);
    expect(head?.training.dataset).toContain("EEGMMIDB");
    expect(head?.training.protocol).not.toContain("proxy");
    expect(head?.experimentId).toBe("m33-scientific-reboot");
  });

  it("cognitive-linear-v2 has scientificStatus SCIENTIFICALLY_VALIDATED", () => {
    const head = getTaskHead("cognitive-linear-v2");
    expect(head?.scientificStatus).toBe("SCIENTIFICALLY_VALIDATED");
    expect(head?.previousMetrics?.status).toBe("INVALID");
    expect(head?.previousMetrics?.reason).toContain("proxy");
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
    expect(COGNITIVE_HEADS.map((h) => h.id)).toContain("cognitive-linear-v2");
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

  it("all cognitive heads have scientific classification", () => {
    for (const head of COGNITIVE_HEADS) {
      expect(head.scientificStatus).toBeDefined();
      expect(head.scientificStatus).toMatch(
        /SCIENTIFICALLY_VALIDATED|ENGINEERING_VALIDATED|EXPERIMENTAL|PROXY_DEMONSTRATION|BLOCKED/,
      );
    }
  });

  it("registerCognitiveHeads is idempotent", () => {
    registerCognitiveHeads();
    registerCognitiveHeads();
    const head = getTaskHead("cognitive-linear-v2");
    expect(head).toBeDefined();
    expect(head?.version).toBe("0.2.0");
  });
});
