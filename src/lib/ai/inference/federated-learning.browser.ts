/**
 * M49 — Federated Brain Learning (Browser Client)
 *
 * Browser-side counterpart to `federated-learning.server.ts`. Implements the
 * client-side training loop that:
 *   1. Fetches global model weights via GET /api/eeg/federated/model/:task
 *   2. Runs local SGD training on V2-32 embeddings using lightweight linear probes
 *      (the same browser-compatible decoders from sleep.browser.ts / cognitive.browser.ts / anomaly.browser.ts)
 *   3. Sends weight deltas to POST /api/eeg/federated/validate then POST /api/eeg/federated/round
 *   4. Re-fetches updated global weights from the server after each round
 *
 * NO raw EEG leaves the browser — only weight deltas (gradients) are transmitted.
 *
 * This module is browser-safe: it does NOT import any `.server.ts` files.
 * All constants, types, and the error class are defined locally (mirroring
 * the server-side values). Server communication is via `fetch()` only.
 *
 * The client integrates brain-flag.ts feature detection to use WebNN/WebGPU
 * execution providers when available for local probe training (future enhancement
 * — currently SGD runs on CPU as the linear probe math is trivial).
 */
import { log } from "@/lib/logging";
import {
  BROWSER_SLEEP_INPUT_DIM,
  BROWSER_SLEEP_OUTPUT_DIM,
  BROWSER_SLEEP_STAGES,
} from "../decoders/sleep.browser";
import {
  BROWSER_COGNITIVE_INPUT_DIM,
  BROWSER_COGNITIVE_OUTPUT_DIM,
} from "../decoders/cognitive.browser";
import {
  BROWSER_ANOMALY_INPUT_DIM,
  BROWSER_ANOMALY_OUTPUT_DIM,
} from "../decoders/anomaly.browser";

// ─────────────────────────────────────────────────────────────────────
// Constants (mirroring federated-learning.server.ts — browser-safe copy)
// ─────────────────────────────────────────────────────────────────────

/** Service id for provenance tracking. */
export const FEDERATED_SERVICE = "federated-brain-learning";
/** Service version. */
export const FEDERATED_VERSION = "v0.1.0";
/** Maximum L2 norm for client update clipping (prevents poisoning). */
export const MAX_CLIENT_L2_NORM = 1.0;
/** Default number of clients per federated round. */
export const DEFAULT_CLIENT_TARGET = 10;
/** Default learning rate for global model (FedAvg momentum). */
export const GLOBAL_LEARNING_RATE = 0.1;
/** Differential privacy epsilon (lower = more noise). */
export const DP_EPSILON = 2.0;
/** Differential privacy delta. */
export const DP_DELTA = 1e-5;
/** Default client epochs (local training rounds per client). */
export const DEFAULT_CLIENT_EPOCHS = 3;
/** Client connection timeout (ms). */
export const CLIENT_TIMEOUT_MS = 30_000;

/** Supported task head types for federated learning. */
export type FederatedTask =
  | "sleep-staging"
  | "sleep-quality"
  | "cognitive-workload"
  | "anomaly-detection";

/** Input/output dimensions per task (browser-safe copy of TASK_DIMENSIONS). */
export const TASK_DIMENSIONS_BROWSER: Record<FederatedTask, { input: number; output: number }> = {
  "sleep-staging": { input: BROWSER_SLEEP_INPUT_DIM, output: BROWSER_SLEEP_OUTPUT_DIM },
  "sleep-quality": { input: BROWSER_SLEEP_INPUT_DIM, output: 1 },
  "cognitive-workload": { input: BROWSER_COGNITIVE_INPUT_DIM, output: BROWSER_COGNITIVE_OUTPUT_DIM },
  "anomaly-detection": { input: BROWSER_ANOMALY_INPUT_DIM, output: BROWSER_ANOMALY_OUTPUT_DIM },
};

// ─────────────────────────────────────────────────────────────────────
// Types (mirrors federated-learning.server.ts — browser-safe copy)
// ─────────────────────────────────────────────────────────────────────

/** A client's weight delta submission. */
export interface ClientUpdate {
  clientId: string;
  task: FederatedTask;
  weightDelta: number[][];
  biasDelta: number[];
  sampleCount: number;
  loss: number;
  accuracy: number;
  epochs: number;
}

/** Result of a federated learning round (server response shape). */
export interface FederatedRoundResult {
  round: number;
  task: FederatedTask;
  participantCount: number;
  globalWeightDelta: number[][];
  globalBiasDelta: number[];
  meanLoss: number;
  meanAccuracy: number;
  convergence: number;
  totalSamples: number;
  durationMs: number;
}

/** Global model weights response from GET /api/eeg/federated/model/:task. */
export interface GlobalModelResponse {
  task: string;
  round: number;
  weights: number[][];
  bias: number[];
}

/** Validation response from POST /api/eeg/federated/validate. */
export interface ValidationResponse {
  valid: boolean;
  reason?: string;
}

/** Training sample: V2-32 embedding + optional label. */
export interface TrainingSample {
  embedding: number[];
  label?: number | number[];
}

/** Options for local training. */
export interface LocalTrainingOptions {
  epochs?: number;
  learningRate?: number;
  batchSize?: number;
  verbose?: boolean;
}

/** Client-side model weights for a task head. */
export interface ClientModel {
  weights: number[][];
  bias: number[];
  task: FederatedTask;
  round: number;
}

/** Result of local training. */
export interface LocalTrainingResult {
  clientId: string;
  task: FederatedTask;
  weightDelta: number[][];
  biasDelta: number[];
  sampleCount: number;
  loss: number;
  accuracy: number;
  epochs: number;
}

/** Configuration for the browser federated learning client. */
export interface FederatedClientConfig {
  apiBaseUrl?: string;
  clientId: string;
  authToken?: string;
  enableDP?: boolean;
}

/**
 * Error thrown when federated learning client operations fail.
 * Mirrors FederatedLearningError from federated-learning.server.ts.
 */
export class FederatedLearningError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = "FederatedLearningError";
  }
}

// ─────────────────────────────────────────────────────────────────────
// Local SGD training (pure browser-side math)
// ─────────────────────────────────────────────────────────────────────

/**
 * Apply a linear probe (weights + bias) to a V2-32 embedding.
 * Returns raw logits (pre-activation). This is the forward pass of the
 * browser-compatible linear probe used by sleep.browser.ts, cognitive.browser.ts,
 * and anomaly.browser.ts.
 */
function forwardProbe(weights: number[][], bias: number[], embedding: number[]): number[] {
  const output = new Array(weights.length).fill(0);
  for (let o = 0; o < weights.length; o++) {
    const w = weights[o];
    let sum = bias[o] ?? 0;
    for (let i = 0; i < w.length; i++) {
      sum += w[i] * embedding[i];
    }
    output[o] = sum;
  }
  return output;
}

/**
 * Stable softmax for converting logits to probabilities.
 */
function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exp = logits.map((l) => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((e) => e / sum);
}

/**
 * Compute cross-entropy loss for classification.
 */
function crossEntropyLoss(probs: number[], label: number): number {
  const p = Math.max(probs[label] ?? 1e-12, 1e-12);
  return -Math.log(p);
}

/**
 * Compute MSE loss for regression.
 */
function mseLoss(pred: number, target: number): number {
  const diff = pred - target;
  return diff * diff;
}

/**
 * Run a single epoch of SGD training on a batch of samples.
 * Updates weights in-place and returns the accumulated loss + correct count.
 */
function trainBatchSGD(
  weights: number[][],
  bias: number[],
  samples: TrainingSample[],
  task: FederatedTask,
  learningRate: number,
): { loss: number; correct: number; total: number } {
  let totalLoss = 0;
  let correct = 0;
  const dims = TASK_DIMENSIONS_BROWSER[task];
  const inputDim = dims.input;
  const outputDim = dims.output;

  for (const sample of samples) {
    const emb = sample.embedding;
    const label = sample.label;

    // Forward pass
    const logits = forwardProbe(weights, bias, emb);
    let loss: number;

    if (outputDim === 1 || (typeof label === "number" && !Number.isInteger(label))) {
      // Regression: MSE loss
      const pred = 1 / (1 + Math.exp(-logits[0])); // sigmoid for [0, 1]
      const target = label as number;
      loss = mseLoss(pred, target);
      totalLoss += loss;

      // Gradient: dL/dpred = 2*(pred - target)
      // dL/dlogit = 2*(pred - target) * pred * (1 - pred)
      const gradLogit = 2 * (pred - target) * pred * (1 - pred);

      // Backward pass: update weights + bias
      for (let i = 0; i < inputDim; i++) {
        weights[0][i] -= learningRate * gradLogit * emb[i];
      }
      bias[0] -= learningRate * gradLogit;

      // Accuracy: prediction within 0.25 of target
      if (Math.abs(pred - target) < 0.25) correct++;
    } else {
      // Classification: cross-entropy + softmax
      const probs = softmax(logits);
      const classIdx = label as number;
      loss = crossEntropyLoss(probs, classIdx);
      totalLoss += loss;

      // Gradient: dL/dlogit[o] = probs[o] - (o === label ? 1 : 0)
      for (let o = 0; o < outputDim; o++) {
        const gradLogit = probs[o] - (o === classIdx ? 1 : 0);
        for (let i = 0; i < inputDim; i++) {
          weights[o][i] -= learningRate * gradLogit * emb[i];
        }
        bias[o] -= learningRate * gradLogit;
      }

      // Accuracy: predicted class == true class
      const predClass = probs.indexOf(Math.max(...probs));
      if (predClass === classIdx) correct++;
    }
  }

  // Clamp weights to prevent explosion
  for (const row of weights) {
    for (let i = 0; i < row.length; i++) {
      row[i] = Math.max(-10, Math.min(10, row[i]));
    }
  }

  return { loss: totalLoss, correct, total: samples.length };
}

/**
 * Chunk an array into mini-batches.
 */
function chunk<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + Math.min(size, arr.length - i)));
  }
  return chunks;
}

/**
 * Deep-copy a weight matrix.
 */
function copyWeights(weights: number[][]): number[][] {
  return weights.map((row) => [...row]);
}

/**
 * Deep-copy a bias vector.
 */
function copyBias(bias: number[]): number[] {
  return [...bias];
}

// ─────────────────────────────────────────────────────────────────────
// FederatedClient — browser-side federated learning client
// ─────────────────────────────────────────────────────────────────────

/**
 * Browser-side federated learning client.
 *
 * Usage:
 *   const client = new FederatedClient({
 *     clientId: "browser-001",
 *     authToken: "jwt...",
 *     enableDP: true,
 *   });
 *   await client.init("sleep-staging");
 *   await client.train([{ embedding: [...32 values], label: 2 }], { epochs: 3 });
 *   const validation = await client.validateUpdate();
 *   const result = await client.submitRound();
 *
 * The client:
 *   - Fetches global weights only (GET /api/eeg/federated/model/:task)
 *   - Trains SGD on local V2-32 embeddings (pure CPU math, no ONNX needed)
 *   - Sends only weight deltas (never raw EEG)
 *   - Integrates brain-flag.ts for EP capability awareness
 */
export class FederatedClient {
  private config: Required<Omit<FederatedClientConfig, "authToken">> & {
    authToken?: string;
  };
  private model: ClientModel | null = null;
  private initialWeights: number[][] | null = null;
  private initialBias: number[] | null = null;
  private lastTrainingResult: LocalTrainingResult | null = null;
  private trainingSamples: TrainingSample[] = [];
  private headers: Record<string, string>;

  /**
   * Create a FederatedClient.
   *
   * @param config Client configuration
   */
  constructor(config: FederatedClientConfig) {
    this.config = {
      apiBaseUrl: config.apiBaseUrl ?? "/api/eeg/federated",
      clientId: config.clientId,
      enableDP: config.enableDP ?? true,
      authToken: config.authToken,
    };
    this.headers = {
      "content-type": "application/json",
      ...(this.config.authToken
        ? { Authorization: `Bearer ${this.config.authToken}` }
        : {}),
    };
  }

  /**
   * Initialize the client by fetching global model weights from the server.
   * Must be called before train().
   *
   * @param task - Federated task type
   */
  async init(task: FederatedTask): Promise<void> {
    const url = `${this.config.apiBaseUrl}/model/${task}`;
    log("info", "fl.client.init", { task, clientId: this.config.clientId, url });

    const response = await fetch(url, {
      method: "GET",
      headers: this.headers,
    });

    if (!response.ok) {
      throw new FederatedLearningError(
        `Failed to fetch global model weights: ${response.status} ${response.statusText}`,
        "INIT_FAILED",
      );
    }

    const data = (await response.json()) as GlobalModelResponse;

    this.validateWeights(data.weights, data.bias, task as FederatedTask);

    this.model = {
      weights: data.weights,
      bias: data.bias,
      task: task as FederatedTask,
      round: data.round,
    };
    this.initialWeights = copyWeights(data.weights);
    this.initialBias = copyBias(data.bias);
    this.lastTrainingResult = null;

    log("info", "fl.client.init.success", {
      task,
      round: data.round,
      weightShape: `${data.weights.length}×${data.weights[0]?.length ?? 0}`,
    });
  }

  /**
   * Validate that weights and bias match the expected dimensions for the task.
   */
  private validateWeights(weights: number[][], bias: number[], task: FederatedTask): void {
    const dims = TASK_DIMENSIONS_BROWSER[task];
    if (!dims) {
      throw new FederatedLearningError(
        `Unknown task: ${task}`,
        "INVALID_TASK",
      );
    }
    if (weights.length !== dims.output) {
      throw new FederatedLearningError(
        `Weight rows: expected ${dims.output}, got ${weights.length}`,
        "INVALID_WEIGHTS",
      );
    }
    if (weights[0]?.length !== dims.input) {
      throw new FederatedLearningError(
        `Weight cols: expected ${dims.input}, got ${weights[0]?.length}`,
        "INVALID_WEIGHTS",
      );
    }
    if (bias.length !== dims.output) {
      throw new FederatedLearningError(
        `Bias length: expected ${dims.output}, got ${bias.length}`,
        "INVALID_BIAS",
      );
    }
    // Check for NaN/Infinity
    const allValues = [...weights.flat(), ...bias];
    if (allValues.some((v) => !Number.isFinite(v))) {
      throw new FederatedLearningError(
        "Model weights contain NaN or Infinity",
        "INVALID_WEIGHTS",
      );
    }
  }

  /**
   * Train the local probe on provided samples using SGD.
   *
   * @param samples - Array of { embedding: number[32], label: number | number[] }
   * @param opts - Training options (epochs, learning rate, batch size)
   * @returns Training result with loss, accuracy, and epoch count
   */
  async train(
    samples: TrainingSample[],
    opts: LocalTrainingOptions = {},
  ): Promise<LocalTrainingResult> {
    if (!this.model) {
      throw new FederatedLearningError(
        "Client not initialized. Call init() first.",
        "NOT_INITIALIZED",
      );
    }

    const { task } = this.model;
    const epochs = opts.epochs ?? DEFAULT_CLIENT_EPOCHS;
    const learningRate = opts.learningRate ?? 0.01;
    const batchSize = opts.batchSize ?? 16;
    const verbose = opts.verbose ?? false;

    // Validate samples
    const dims = TASK_DIMENSIONS_BROWSER[task];
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].embedding.length !== dims.input) {
        throw new FederatedLearningError(
          `Sample ${i}: expected ${dims.input}-D embedding, got ${samples[i].embedding.length}`,
          "INVALID_SAMPLE",
        );
      }
    }

    // Store samples for delta computation
    this.trainingSamples = [...samples];

    // Work on copies — we compute delta, not direct updates
    const weights = copyWeights(this.model.weights);
    const bias = copyBias(this.model.bias);

    let totalLoss = 0;
    let totalCorrect = 0;
    let totalSamples = 0;

    const t0 = performance.now();

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle samples each epoch
      const shuffled = this.shuffleArray(samples);
      const batches = chunk(shuffled, batchSize);

      for (const batch of batches) {
        const result = trainBatchSGD(weights, bias, batch, task, learningRate);
        totalLoss += result.loss;
        totalCorrect += result.correct;
        totalSamples += result.total;

        if (verbose && epoch % 5 === 0) {
          log("debug", "fl.client.epoch", {
            epoch,
            batchLoss: result.loss / result.total,
            batchCorrect: result.correct,
            batchTotal: result.total,
          });
        }
      }
    }

    // Compute weight delta (updated - initial)
    const weightDelta: number[][] = [];
    const biasDelta: number[] = [];

    for (let o = 0; o < weights.length; o++) {
      weightDelta.push([]);
      for (let i = 0; i < weights[o].length; i++) {
        weightDelta[o].push(weights[o][i] - this.initialWeights![o][i]);
      }
      biasDelta.push(bias[o] - this.initialBias![o]);
    }

    const meanLoss = totalLoss / Math.max(totalSamples, 1);
    const accuracy = totalCorrect / Math.max(totalSamples, 1);
    const durationMs = +(performance.now() - t0).toFixed(2);

    this.lastTrainingResult = {
      clientId: this.config.clientId,
      task,
      weightDelta,
      biasDelta,
      sampleCount: samples.length,
      loss: meanLoss,
      accuracy,
      epochs,
    };

    log("info", "fl.client.train.complete", {
      task,
      epochs,
      loss: meanLoss,
      accuracy,
      durationMs,
      weightDeltaNorm: this.l2Norm(weightDelta, biasDelta),
    });

    return this.lastTrainingResult;
  }

  /**
   * Validate the computed weight delta via the server's /validate endpoint.
   * The server checks dimensions, NaN/Infinity, and other safety constraints.
   *
   * @returns Server validation result
   */
  async validateUpdate(): Promise<ValidationResponse> {
    if (!this.lastTrainingResult) {
      throw new FederatedLearningError(
        "No trained update to validate. Call train() first.",
        "NOT_INITIALIZED",
      );
    }

    const update: ClientUpdate = {
      clientId: this.lastTrainingResult.clientId,
      task: this.lastTrainingResult.task,
      weightDelta: this.lastTrainingResult.weightDelta,
      biasDelta: this.lastTrainingResult.biasDelta,
      sampleCount: this.lastTrainingResult.sampleCount,
      loss: this.lastTrainingResult.loss,
      accuracy: this.lastTrainingResult.accuracy,
      epochs: this.lastTrainingResult.epochs,
    };

    const response = await fetch(`${this.config.apiBaseUrl}/validate`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(update),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({})) as { error?: string };
      throw new FederatedLearningError(
        `Validation endpoint error: ${response.status} ${err.error ?? response.statusText}`,
        "VALIDATION_FAILED",
      );
    }

    const data = (await response.json()) as ValidationResponse;
    return data;
  }

  /**
   * Submit the trained weight delta to the server for federated aggregation.
   *
   * The server will:
   *   1. Clip the L2 norm to MAX_CLIENT_L2_NORM (1.0)
   *   2. Apply differential privacy noise (if enableDP)
   *   3. Aggregate via FedAvg (sample-weighted average)
   *   4. Update the global model
   *   5. Return the round result
   *
   * After submission, the client re-fetches the updated global weights
   * via GET /api/eeg/federated/model/:task for the next round.
   *
   * @param options - Round options (clientTarget, enableDP)
   * @returns Federated round result
   */
  async submitRound(options: {
    clientTarget?: number;
    enableDP?: boolean;
    minSamples?: number;
  } = {}): Promise<FederatedRoundResult> {
    if (!this.lastTrainingResult) {
      throw new FederatedLearningError(
        "No trained update to submit. Call train() first.",
        "NOT_INITIALIZED",
      );
    }

    const update: ClientUpdate = {
      clientId: this.lastTrainingResult.clientId,
      task: this.lastTrainingResult.task,
      weightDelta: this.lastTrainingResult.weightDelta,
      biasDelta: this.lastTrainingResult.biasDelta,
      sampleCount: this.lastTrainingResult.sampleCount,
      loss: this.lastTrainingResult.loss,
      accuracy: this.lastTrainingResult.accuracy,
      epochs: this.lastTrainingResult.epochs,
    };

    const body = {
      task: this.lastTrainingResult.task,
      updates: [update],
      options: {
        client_target: options.clientTarget ?? DEFAULT_CLIENT_TARGET,
        enable_dp: options.enableDP ?? this.config.enableDP,
        min_samples: options.minSamples,
      },
    };

    log("info", "fl.client.submit", {
      task: this.lastTrainingResult.task,
      clientId: this.config.clientId,
      enableDP: body.options.enable_dp,
    });

    const response = await fetch(`${this.config.apiBaseUrl}/round`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({})) as { error?: string };
      throw new FederatedLearningError(
        `Federated round failed: ${response.status} ${err.error ?? response.statusText}`,
        "SUBMIT_FAILED",
      );
    }

    const result = (await response.json()) as {
      service: string;
      version: string;
      round: number;
      task: string;
      participant_count: number;
      total_samples: number;
      mean_loss: number;
      mean_accuracy: number;
      convergence: number;
      duration_ms: number;
      provenance: unknown;
      global_weight_delta?: number[][];
      global_bias_delta?: number[];
    };

    // Re-fetch updated global weights from the server for the next round
    const updatedModel = await this.fetchGlobalModel(this.model!.task);
    this.model = {
      weights: updatedModel.weights,
      bias: updatedModel.bias,
      task: this.model!.task,
      round: updatedModel.round,
    };
    this.initialWeights = copyWeights(updatedModel.weights);
    this.initialBias = copyBias(updatedModel.bias);
    this.lastTrainingResult = null;
    this.trainingSamples = [];

    log("info", "fl.client.submit.success", {
      round: result.round,
      convergence: result.convergence,
      meanLoss: result.mean_loss,
      meanAccuracy: result.mean_accuracy,
    });

    return {
      round: result.round,
      task: result.task as FederatedTask,
      participantCount: result.participant_count,
      globalWeightDelta: result.global_weight_delta ?? [],
      globalBiasDelta: result.global_bias_delta ?? [],
      meanLoss: result.mean_loss,
      meanAccuracy: result.mean_accuracy,
      convergence: result.convergence,
      totalSamples: result.total_samples,
      durationMs: result.duration_ms,
    };
  }

  /**
   * Fetch the latest global model weights from the server.
   */
  private async fetchGlobalModel(task: FederatedTask): Promise<GlobalModelResponse> {
    const url = `${this.config.apiBaseUrl}/model/${task}`;
    const response = await fetch(url, {
      method: "GET",
      headers: this.headers,
    });

    if (!response.ok) {
      throw new FederatedLearningError(
        `Failed to re-fetch global model weights: ${response.status}`,
        "FETCH_FAILED",
      );
    }

    return (await response.json()) as GlobalModelResponse;
  }

  /**
   * Compute the L2 norm of a weight delta for logging.
   */
  private l2Norm(weights: number[][], bias: number[]): number {
    const all = [...weights.flat(), ...bias];
    return Math.sqrt(all.reduce((s, v) => s + v * v, 0));
  }

  /**
   * Fisher-Yates shuffle.
   */
  private shuffleArray<T>(arr: T[]): T[] {
    const shuffled = [...arr];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]] as [T, T];
    }
    return shuffled;
  }

  /**
   * Get the current client model state (for inspection/debugging).
   */
  getModel(): ClientModel | null {
    return this.model;
  }

  /**
   * Get the last training result.
   */
  getLastTrainingResult(): LocalTrainingResult | null {
    return this.lastTrainingResult;
  }

  /**
   * Check if the client is initialized.
   */
  isInitialized(): boolean {
    return this.model !== null;
  }

  /**
   * Get the brain-flag accelerator status for diagnostic purposes.
   * This mirrors the brain-flag.ts priority chain: WebNN > WebGPU > WASM.
   */
  getAcceleratorStatus(): { webnn: boolean; webgpu: boolean; wasm: boolean; active: string[] } {
    try {
      const webnn = typeof navigator !== "undefined" && "ml" in navigator;
      const webgpu = typeof navigator !== "undefined" && "gpu" in navigator;
      return {
        webnn,
        webgpu,
        wasm: true,
        active: this.getActiveProviders(webnn, webgpu),
      };
    } catch {
      return {
        webnn: false,
        webgpu: false,
        wasm: true,
        active: ["wasm"],
      };
    }
  }

  /**
   * Get active execution providers (priority chain: WebNN > WebGPU > WASM).
   */
  private getActiveProviders(webnnAvailable: boolean, webgpuAvailable: boolean): string[] {
    if (typeof navigator === "undefined") return ["wasm"];

    // Check brain-flag runtime toggles — in the browser, these are managed
    // by brain-flag.ts module scope via VITE_WEBNN / VITE_ORT_WEBGPU env vars.
    // We can't directly import the runtime toggle state here (it's in a
    // different module scope), so we report capability-only.
    if (webnnAvailable) return ["webnn", "webgpu", "wasm"];
    if (webgpuAvailable) return ["webgpu", "wasm"];
    return ["wasm"];
  }

  /**
   * Run a full federated learning cycle: train + validate + submit.
   * This is the convenience method that orchestrates the entire client-side
   * federated learning flow.
   *
   * @param samples - Training samples (V2-32 embeddings + labels)
   * @param trainingOpts - Local SGD options
   * @param submitOpts - Server round options
   * @returns Federated round result
   */
  async trainAndSubmit(
    samples: TrainingSample[],
    trainingOpts: LocalTrainingOptions = {},
    submitOpts: {
      clientTarget?: number;
      enableDP?: boolean;
      minSamples?: number;
    } = {},
  ): Promise<FederatedRoundResult> {
    // 1. Train locally
    await this.train(samples, trainingOpts);

    // 2. Validate the update
    const validation = await this.validateUpdate();
    if (!validation.valid) {
      throw new FederatedLearningError(
        `Client update validation failed: ${validation.reason ?? "unknown reason"}`,
        "VALIDATION_FAILED",
      );
    }

    // 3. Submit for aggregation
    return this.submitRound(submitOpts);
  }
}
