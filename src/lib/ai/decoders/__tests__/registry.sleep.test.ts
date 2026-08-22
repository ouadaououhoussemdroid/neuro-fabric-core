/**
 * M39 + M40 — Tests for the Sleep task-head registry.
 *
 * Tests that the sleep staging and sleep quality probes are properly registered
 * in the shared TaskHeadRegistry, with correct dimensions, SHAs, and metrics.
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
  SLEEP_HEADS,
  SLEEP_STAGING_PROBE_JOINT_2312,
  SLEEP_STAGING_PROBE_V2_32,
  SLEEP_QUALITY_PROBE_JOINT_2312,
  SLEEP_QUALITY_PROBE_V2_32,
  registerSleepHeads,
  getDefaultSleepHead,
  getDefaultSleepQualityHead,
} from "../sleep.registry";

describe("Sleep Staging TaskHeadRegistry", () => {
  beforeAll(() => {
    registerSleepHeads();
  });

  it("registers the sleep-staging-v1 head with correct id", () => {
    expect(hasTaskHead(SLEEP_STAGING_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("sleep-staging-v1");
    expect(head).toBeDefined();
    expect(head?.service).toBe("sleep-staging");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(5); // 5 sleep stages
  });

  it("sleep-staging-v1 uses server inference (onnxruntime-node)", () => {
    const head = getTaskHead("sleep-staging-v1");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe(
      "33dde2d3801e74cce6ed33e0e83ec072df62ede9e3ca9c0187ba39f0d7673cff",
    );
  });

  it("sleep-staging-v1 has correct artifact URI", () => {
    const head = getTaskHead("sleep-staging-v1");
    expect(head?.artifactUri).toBe("/models/sleep/staging-probe-joint2312-v1.onnx");
  });

  it("sleep-staging-v1 training metadata references Sleep-EDF", () => {
    const head = getTaskHead("sleep-staging-v1");
    expect(head?.training.dataset).toContain("Sleep-EDF");
    expect(head?.training.dataset).toContain("PhysioNet");
    expect(head?.experimentId).toBe("m39-sleep-staging-probe");
  });

  it("registers V2-32 browser fallback head", () => {
    expect(hasTaskHead("sleep-staging-v2-32d")).toBe(true);
    const head = getTaskHead("sleep-staging-v2-32d");
    expect(head?.inputDim).toBe(32);
    expect(head?.outputDim).toBe(5);
    expect(head?.inferenceTarget).toBe("both");
    expect(head?.service).toBe("sleep-staging");
  });

  it("SLEEP_HEADS contains all registered probes (M39 + M40)", () => {
    expect(SLEEP_HEADS.length).toBe(4);
    expect(SLEEP_HEADS.map((h) => h.id)).toContain("sleep-staging-v1");
    expect(SLEEP_HEADS.map((h) => h.id)).toContain("sleep-staging-v2-32d");
    expect(SLEEP_HEADS.map((h) => h.id)).toContain("sleep-quality-v1");
    expect(SLEEP_HEADS.map((h) => h.id)).toContain("sleep-quality-v2-32d");
  });

  it("all sleep heads have correct training metadata populated", () => {
    for (const head of SLEEP_HEADS) {
      expect(head.training.dataset).toBeTruthy();
      expect(head.training.protocol).toBeTruthy();
      expect(head.training.metrics).toBeDefined();
    }
  });

  it("getTaskHeadsByService returns all sleep heads", () => {
    const heads = getTaskHeadsByService("sleep-staging");
    expect(heads.length).toBe(4);
    expect(heads.every((h) => h.service === "sleep-staging")).toBe(true);
  });

  it("getDefaultSleepHead returns staging server head by default", () => {
    const head = getDefaultSleepHead("server");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(2312);
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.id).toBe("sleep-staging-v1");
  });

  it("getDefaultSleepHead browser returns V2-32 staging head", () => {
    const head = getDefaultSleepHead("both");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(32);
    expect(head?.inferenceTarget).toBe("both");
  });

  it("registerSleepHeads is idempotent", () => {
    registerSleepHeads();
    registerSleepHeads();
    const head = getTaskHead("sleep-staging-v1");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(2312);
  });

  it("getDefaultTaskHead returns sleep head when service matches", () => {
    registerSleepHeads();
    const head = getDefaultTaskHead("sleep-staging");
    expect(head).toBeDefined();
    expect(head?.id).toBe("sleep-staging-v1");
  });

  // ─── M40: Sleep Quality registry tests ─────────────────────────────

  it("registers sleep-quality-v1 head with correct id and dimensions", () => {
    expect(hasTaskHead(SLEEP_QUALITY_PROBE_JOINT_2312.id)).toBe(true);
    const head = getTaskHead("sleep-quality-v1");
    expect(head).toBeDefined();
    expect(head?.service).toBe("sleep-staging");
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1); // regression (quality score)
  });

  it("sleep-quality-v1 uses server inference (onnxruntime-node)", () => {
    const head = getTaskHead("sleep-quality-v1");
    expect(head?.inferenceTarget).toBe("server");
    expect(head?.sha256).toBe(
      "e41ed5282d77aa3b401b587aa3fdbb375ed46b480a71e6f8d9a471efe82ccdfd",
    );
  });

  it("sleep-quality-v1 has correct artifact URI", () => {
    const head = getTaskHead("sleep-quality-v1");
    expect(head?.artifactUri).toBe("/models/sleep/quality-probe-joint2312-v1.onnx");
  });

  it("sleep-quality-v1 training metadata references Sleep-EDF", () => {
    const head = getTaskHead("sleep-quality-v1");
    expect(head?.training.dataset).toContain("Sleep-EDF");
    expect(head?.experimentId).toBe("m40-sleep-quality-probe");
  });

  it("registers sleep-quality-v2-32d browser fallback head", () => {
    expect(hasTaskHead("sleep-quality-v2-32d")).toBe(true);
    const head = getTaskHead("sleep-quality-v2-32d");
    expect(head?.inputDim).toBe(32);
    expect(head?.outputDim).toBe(1);
    expect(head?.inferenceTarget).toBe("both");
    expect(head?.service).toBe("sleep-staging");
  });

  it("getDefaultSleepQualityHead returns server quality head by default", () => {
    const head = getDefaultSleepQualityHead("server");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(2312);
    expect(head?.outputDim).toBe(1);
    expect(head?.id).toBe("sleep-quality-v1");
  });

  it("getDefaultSleepQualityHead browser returns V2-32 quality head", () => {
    const head = getDefaultSleepQualityHead("both");
    expect(head).toBeDefined();
    expect(head?.inputDim).toBe(32);
    expect(head?.inferenceTarget).toBe("both");
  });
});
