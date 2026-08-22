/**
 * M49 — Tests for the browser-side FederatedClient (federated-learning.client.ts).
 *
 * Tests the client-side linear probe SGD training, weight delta computation,
 * dimension validation, and FederatedClient lifecycle — all without network I/O.
 * The production client.ts module imports from federated-learning.server.ts
 * (which re-exports constants), but the training math and validation are
 * pure browser-side functions.
 */
import { describe, it, expect, beforeEach } from "vitest";

import {
  FederatedClient,
  TASK_DIMENSIONS_BROWSER,
  FEDERATED_SERVICE,
  FEDERATED_VERSION,
  MAX_CLIENT_L2_NORM,
  DEFAULT_CLIENT_EPOCHS,
  DP_EPSILON,
  DP_DELTA,
  type TrainingSample,
  type ClientModel,
  type LocalTrainingResult,
} from "../federated-learning.browser";

/** Generate deterministic synthetic V2-32 embeddings. */
function makeSyntheticEmbedding(seed = 0): number[] {
  const v = Array.from({ length: 32 }, (_, i) => Math.sin((i + seed) * 0.1) * 0.5);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

/** Generate synthetic training samples for a task. */
function makeSamples(task: string, count: number, seed = 0): TrainingSample[] {
  const dims = TASK_DIMENSIONS_BROWSER[task as keyof typeof TASK_DIMENSIONS_BROWSER];
  const samples: TrainingSample[] = [];
  for (let i = 0; i < count; i++) {
    samples.push({
      embedding: makeSyntheticEmbedding(seed + i),
      label: i % dims.output,
    });
  }
  return samples;
}

/** Generate random weights matching task dimensions. */
function makeRandomWeights(task: string): { weights: number[][]; bias: number[] } {
  const dims = TASK_DIMENSIONS_BROWSER[task as keyof typeof TASK_DIMENSIONS_BROWSER];
  const weights = Array.from({ length: dims.output }, () =>
    Array.from({ length: dims.input }, () => (Math.random() - 0.5) * 0.01),
  );
  const bias = new Array(dims.output).fill(0);
  return { weights, bias };
}

describe("federated-learning.client (M49)", () => {
  describe("module constants", () => {
    it("exports correct service constants", () => {
      expect(FEDERATED_SERVICE).toBe("federated-brain-learning");
      expect(FEDERATED_VERSION).toBe("v0.1.0");
    });

    it("has correct DP parameters", () => {
      expect(DP_EPSILON).toBe(2.0);
      expect(DP_DELTA).toBe(1e-5);
    });

    it("has correct default client epochs", () => {
      expect(DEFAULT_CLIENT_EPOCHS).toBe(3);
    });

    it("has correct max L2 norm", () => {
      expect(MAX_CLIENT_L2_NORM).toBe(1.0);
    });

    it("defines browser task dimensions for all 4 tasks", () => {
      expect(TASK_DIMENSIONS_BROWSER["sleep-staging"]).toEqual({ input: 32, output: 5 });
      expect(TASK_DIMENSIONS_BROWSER["sleep-quality"]).toEqual({ input: 32, output: 1 });
      expect(TASK_DIMENSIONS_BROWSER["cognitive-workload"]).toEqual({ input: 32, output: 1 });
      expect(TASK_DIMENSIONS_BROWSER["anomaly-detection"]).toEqual({ input: 32, output: 1 });
    });
  });

  describe("FederatedClient lifecycle", () => {
    it("constructs with correct config", () => {
      const client = new FederatedClient({
        clientId: "test-001",
        authToken: "test-jwt",
        enableDP: false,
      });
      expect(client.isInitialized()).toBe(false);
      expect(client.getModel()).toBeNull();
      expect(client.getLastTrainingResult()).toBeNull();
    });

    it("train() throws before init (NOT_INITIALIZED)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      await expect(
        client.train(makeSamples("sleep-staging", 10), { epochs: 1 }),
      ).rejects.toThrow("init");
    });

    it("validateUpdate() throws before train()", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      await expect(client.validateUpdate()).rejects.toThrow("train");
    });

    it("submitRound() throws before train()", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      await expect(client.submitRound()).rejects.toThrow("train");
    });
  });

  describe("client-side model state injection", () => {
    it("accepts manually injected model state for training", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const dims = TASK_DIMENSIONS_BROWSER["sleep-staging"];

      // Inject model state (simulating server fetch)
      const weights = Array.from({ length: dims.output }, () =>
        Array.from({ length: dims.input }, () => (Math.random() - 0.5) * 0.01),
      );
      const bias = new Array(dims.output).fill(0);

      (client as any).model = { weights, bias, task: "sleep-staging", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      expect(client.isInitialized()).toBe(true);

      const result = await client.train(makeSamples("sleep-staging", 20, 42), {
        epochs: 1,
        learningRate: 0.01,
      });

      expect(result.clientId).toBe("test-001");
      expect(result.task).toBe("sleep-staging");
      expect(result.sampleCount).toBe(20);
      expect(result.epochs).toBe(1);
      expect(Number.isFinite(result.loss)).toBe(true);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
      expect(result.accuracy).toBeLessThanOrEqual(1);
      expect(result.weightDelta).toHaveLength(dims.output);
      expect(result.weightDelta[0]).toHaveLength(dims.input);
      expect(result.biasDelta).toHaveLength(dims.output);
    });
  });

  describe("weight delta computation", () => {
    it("weight delta = trained_weights - initial_weights (zero when no training)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const dims = TASK_DIMENSIONS_BROWSER["sleep-staging"];

      const weights = Array.from({ length: dims.output }, () =>
        Array.from({ length: dims.input }, () => 0.1),
      );
      const bias = [0.1, 0.1, 0.1, 0.1, 0.1];

      (client as any).model = { weights, bias, task: "sleep-staging", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      // No actual training — the delta should be small (LR × grad on samples)
      const result = await client.train(makeSamples("sleep-staging", 5), {
        epochs: 1,
        learningRate: 0.0,
      });

      // With LR=0, delta should be all zeros
      for (let o = 0; o < dims.output; o++) {
        for (let i = 0; i < dims.input; i++) {
          expect(result.weightDelta[o][i]).toBeCloseTo(0, 10);
        }
        expect(result.biasDelta[o]).toBeCloseTo(0, 10);
      }
    });

    it("weight delta is non-zero after training with LR > 0", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const dims = TASK_DIMENSIONS_BROWSER["sleep-staging"];

      const { weights, bias } = makeRandomWeights("sleep-staging");
      (client as any).model = { weights, bias, task: "sleep-staging", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      const result = await client.train(makeSamples("sleep-staging", 30, 5), {
        epochs: 3,
        learningRate: 0.01,
      });

      // At least some weights should have changed
      let hasChange = false;
      for (let o = 0; o < dims.output && !hasChange; o++) {
        for (let i = 0; i < dims.input && !hasChange; i++) {
          if (Math.abs(result.weightDelta[o][i]) > 1e-10) {
            hasChange = true;
          }
        }
      }
      expect(hasChange).toBe(true);
    });

    it("weight delta is finite (no NaN/Infinity)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const { weights, bias } = makeRandomWeights("sleep-quality");
      (client as any).model = { weights, bias, task: "sleep-quality", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      const result = await client.train(makeSamples("sleep-quality", 50), {
        epochs: 5,
        learningRate: 0.1,
      });

      const allValues = [...result.weightDelta.flat(), ...result.biasDelta];
      for (const v of allValues) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });
  });

  describe("multi-task support", () => {
    it("training works for sleep-quality (regression, output=1)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const { weights, bias } = makeRandomWeights("sleep-quality");
      (client as any).model = { weights, bias, task: "sleep-quality", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      const result = await client.train(
        Array.from({ length: 20 }, (_, i) => ({
          embedding: makeSyntheticEmbedding(i),
          label: 0.3 + (i % 5) * 0.1, // scalar regression labels
        })),
        { epochs: 2 },
      );

      expect(result.weightDelta).toHaveLength(1); // output=1
      expect(result.weightDelta[0]).toHaveLength(32);
      expect(result.biasDelta).toHaveLength(1);
    });

    it("training works for cognitive-workload (regression, output=1)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const { weights, bias } = makeRandomWeights("cognitive-workload");
      (client as any).model = { weights, bias, task: "cognitive-workload", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      const result = await client.train(makeSamples("cognitive-workload", 15), {
        epochs: 1,
      });

      expect(result.weightDelta).toHaveLength(1);
      expect(result.biasDelta).toHaveLength(1);
    });

    it("training works for anomaly-detection (regression, output=1)", async () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const { weights, bias } = makeRandomWeights("anomaly-detection");
      (client as any).model = { weights, bias, task: "anomaly-detection", round: 0 };
      (client as any).initialWeights = weights.map((w: number[]) => [...w]);
      (client as any).initialBias = [...bias];

      const result = await client.train(makeSamples("anomaly-detection", 15), {
        epochs: 1,
      });

      expect(result.weightDelta).toHaveLength(1);
      expect(result.biasDelta).toHaveLength(1);
    });
  });

  describe("getAcceleratorStatus (brain-flag integration)", () => {
    it("returns well-formed status object", () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const status = client.getAcceleratorStatus();

      expect(status).toHaveProperty("webnn");
      expect(status).toHaveProperty("webgpu");
      expect(status).toHaveProperty("wasm");
      expect(status).toHaveProperty("active");
      expect(status.wasm).toBe(true);
      expect(Array.isArray(status.active)).toBe(true);
      expect(status.active).toContain("wasm");
      expect(typeof status.webnn).toBe("boolean");
      expect(typeof status.webgpu).toBe("boolean");
    });

    it("priority chain ends with WASM", () => {
      const client = new FederatedClient({ clientId: "test-001" });
      const status = client.getAcceleratorStatus();
      // WASM is always the last (fallback) entry
      expect(status.active[status.active.length - 1]).toBe("wasm");
    });
  });
});
