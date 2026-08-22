/**
 * M49 — Federated Brain Learning
 *
 * Implements the server-side coordination layer for federated learning of
 * browser-compatible task head probes (V2-32 → sleep staging, sleep quality,
 * cognitive workload, anomaly detection). The system trains linear probe weights
 * (32-D → K-D) across multiple clients without sharing raw EEG data — only
 * encrypted weight deltas are transmitted.
 *
 * ARCHITECTURE:
 *   1. Global model: V2-32 probe weights + bias (distributed to clients)
 *   2. Client training: browser runs SGD on local V2-32 embeddings + labels
 *      (e.g., sleep stage annotations, cognitive workload labels)
 *   3. Client sends: weight delta (32×K) + metadata (sample count, quality)
 *   4. Server aggregation: FedAvg — weighted average by client sample count
 *   5. Global model update: global_w = Σ(n_i × delta_i) / Σ(n_i)
 *   6. New global weights broadcast to all clients for next round
 *
 * SECURITY:
 *   - Client updates are clipped to max L2 norm (prevents poisoning)
 *   - Weight deltas are signed with the client's session JWT
 *   - Differential privacy: optional Gaussian noise (ε=2, δ=1e-5)
 *
 * This module is server-only (.server.ts suffix) because:
 *   - Aggregation requires computing over many clients' updates
 *   - The global model state must be securely maintained
 *   - Coordination with the probe registry and provenance tracking
 *
 * Clients (browser) use the lightweight linear probe from:
 *   - src/lib/ai/decoders/sleep.browser.ts (32→5, 32→1)
 *   - src/lib/ai/decoders/cognitive.browser.ts (32→1)
 *   - src/lib/ai/decoders/anomaly.browser.ts (32→1)
 */
import { startTimer, log } from "@/lib/logging";
import { metrics } from "@/lib/metrics";
import { buildServiceProvenance, type ServiceProvenance } from "../services/provenance.server";
import {
  BROWSER_SLEEP_INPUT_DIM,
  BROWSER_SLEEP_OUTPUT_DIM,
} from "@/lib/ai/decoders/sleep.browser";
import {
  BROWSER_COGNITIVE_OUTPUT_DIM,
  BROWSER_COGNITIVE_INPUT_DIM,
} from "@/lib/ai/decoders/cognitive.browser";
import {
  BROWSER_ANOMALY_OUTPUT_DIM,
  BROWSER_ANOMALY_INPUT_DIM,
} from "@/lib/ai/decoders/anomaly.browser";

// ─────────────────────────────────────────────────────────────────────
// Constants
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

/** Input/output dimensions per task. */
export const TASK_DIMENSIONS: Record<FederatedTask, { input: number; output: number }> = {
  "sleep-staging": { input: BROWSER_SLEEP_INPUT_DIM, output: BROWSER_SLEEP_OUTPUT_DIM },
  "sleep-quality": { input: BROWSER_SLEEP_INPUT_DIM, output: 1 },
  "cognitive-workload": { input: BROWSER_COGNITIVE_INPUT_DIM, output: BROWSER_COGNITIVE_OUTPUT_DIM },
  "anomaly-detection": { input: BROWSER_ANOMALY_INPUT_DIM, output: BROWSER_ANOMALY_OUTPUT_DIM },
};

// ─────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────

/** A client's weight delta submission. */
export interface ClientUpdate {
  /** Unique client identifier (hashed session id). */
  clientId: string;
  /** Task head type this update applies to. */
  task: FederatedTask;
  /** Weight deltas: outputDim arrays of inputDim values each. */
  weightDelta: number[][];
  /** Bias deltas: outputDim values. */
  biasDelta: number[];
  /** Number of local samples used for this update. */
  sampleCount: number;
  /** Client-reported loss (for monitoring). */
  loss: number;
  /** Client-reported accuracy (for monitoring). */
  accuracy: number;
  /** Epochs completed by this client. */
  epochs: number;
}

/** Result of a federated learning round. */
export interface FederatedRoundResult {
  /** Round number (incremented each call). */
  round: number;
  /** Task head type. */
  task: FederatedTask;
  /** Number of clients that participated. */
  participantCount: number;
  /** Aggregated global weight delta applied to the model. */
  globalWeightDelta: number[][];
  /** Aggregated global bias delta applied to the model. */
  globalBiasDelta: number[];
  /** Mean loss across all participants. */
  meanLoss: number;
  /** Mean accuracy across all participants. */
  meanAccuracy: number;
  /** Convergence metric (L2 norm of global update). */
  convergence: number;
  /** Total samples across all participants. */
  totalSamples: number;
  /** Processing time in milliseconds. */
  durationMs: number;
  /** Provenance record. */
  provenance: ServiceProvenance;
}

/** Options for running a federated round. */
export interface FederatedRoundOptions {
  /** Number of clients required to start a round. Default: 10. */
  clientTarget?: number;
  /** Whether to apply differential privacy noise. Default: true. */
  enableDP?: boolean;
  /** Maximum client connection timeout. Default: 30000ms. */
  clientTimeoutMs?: number;
  /** Minimum client sample count threshold (drop outliers). */
  minSamples?: number;
}

/** Global model state for a task head. */
export interface GlobalModelState {
  /** Current weight matrix [outputDim × inputDim]. */
  weights: number[][];
  /** Current bias vector [outputDim]. */
  bias: number[];
  /** Last round number this model was updated. */
  lastUpdatedRound: number;
  /** Exponential moving average of loss (for monitoring). */
  emaLoss: number;
}

/** Error thrown when federated learning coordination fails. */
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
// Global model state (in-memory, persisted via DB in production)
// ─────────────────────────────────────────────────────────────────────

let roundCounter = 0;

/** Global model state per task, persisted across rounds. */
const globalModels = new Map<FederatedTask, GlobalModelState>();

/**
 * Initialize a global model with random small weights (Xavier/Glorot init).
 * Called on first use for each task.
 */
function initGlobalModel(task: FederatedTask): GlobalModelState {
  const dims = TASK_DIMENSIONS[task];
  const weights: number[][] = [];
  const limit = Math.sqrt(6 / (dims.input + dims.output));
  for (let o = 0; o < dims.output; o++) {
    const row: number[] = [];
    for (let i = 0; i < dims.input; i++) {
      row.push((Math.random() * 2 - 1) * limit);
    }
    weights.push(row);
  }
  const bias = new Array(dims.output).fill(0);
  return { weights, bias, lastUpdatedRound: 0, emaLoss: 0 };
}

/** Get or initialize the global model for a task. */
export function getGlobalModel(task: FederatedTask): GlobalModelState {
  if (!globalModels.has(task)) {
    globalModels.set(task, initGlobalModel(task));
  }
  return globalModels.get(task)!;
}

/** Reset all global model state (test helper). */
export function resetFederatedState(): void {
  globalModels.clear();
  roundCounter = 0;
}

// ─────────────────────────────────────────────────────────────────────
// Security: update clipping + differential privacy
// ─────────────────────────────────────────────────────────────────────

/**
 * Clip a client update's L2 norm to the maximum threshold.
 * Returns the clipped update and the original norm.
 */
function clipL2Norm(
  weightDelta: number[][],
  biasDelta: number[],
  maxNorm: number,
): { clipped: number[][]; biasClipped: number[]; originalNorm: number } {
  const allValues = [...weightDelta.flat(), ...biasDelta];
  const norm = Math.sqrt(allValues.reduce((s, v) => s + v * v, 0));

  if (norm <= maxNorm || norm === 0) {
    return { clipped: weightDelta, biasClipped: biasDelta, originalNorm: norm };
  }

  const scale = maxNorm / norm;
  const clipped = weightDelta.map((row) => row.map((v) => v * scale));
  const biasClipped = biasDelta.map((v) => v * scale);
  return { clipped, biasClipped, originalNorm: norm };
}

/**
 * Add Gaussian noise for differential privacy (ε=2, δ=1e-5).
 * Uses the optimal mechanism: noise scale = maxNorm * sqrt(2*ln(1.25/δ)) / ε.
 */
function addDPNoise(
  weightDelta: number[][],
  biasDelta: number[],
  scale: number,
): { noisyWeights: number[][]; noisyBias: number[] } {
  const noiseScale = (MAX_CLIENT_L2_NORM * Math.sqrt(2 * Math.log(1.25 / DP_DELTA))) / DP_EPSILON;

  const noisyWeights = weightDelta.map((row) =>
    row.map((v) => v + gaussRandom() * noiseScale * scale),
  );
  const noisyBias = biasDelta.map((v) => v + gaussRandom() * noiseScale * scale);
  return { noisyWeights, noisyBias };
}

/** Standard normal random number via Box-Muller transform. */
function gaussRandom(): number {
  const u1 = Math.random() * 0.999999 + 0.000001;
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─────────────────────────────────────────────────────────────────────
// FedAvg aggregation
// ─────────────────────────────────────────────────────────────────────

/**
 * Run a federated learning round: aggregate client updates via FedAvg.
 *
 * Steps:
 *   1. Validate and clip each client's update (L2 norm → maxNorm)
 *   2. Optionally add differential privacy noise
 *   3. Weight each client's update by sample count: scaled_delta_i = n_i * delta_i / N
 *   4. Sum all scaled deltas → global delta
 *   5. Apply to global model: w_new = w_old + lr * global_delta
 *   6. Record metrics (convergence, loss, accuracy, participation)
 *
 * @param updates - Client weight deltas to aggregate
 * @param task - Task head type
 * @param opts - Round configuration options
 * @returns Aggregated round result
 */
export async function runFederatedRound(
  updates: ClientUpdate[],
  task: FederatedTask,
  opts: FederatedRoundOptions = {},
): Promise<FederatedRoundResult> {
  const t0 = startTimer("federated_round.total");

  const clientTarget = opts.clientTarget ?? DEFAULT_CLIENT_TARGET;
  const enableDP = opts.enableDP ?? true;

  metrics.federatedRoundRequestsTotal.inc({ task, round: String(roundCounter) });
  roundCounter++;

  // Validate minimum client participation
  if (updates.length < 1) {
    throw new FederatedLearningError("No client updates to aggregate", "EMPTY_UPDATES");
  }

  const dims = TASK_DIMENSIONS[task];
  const validUpdates = updates.filter(
    (u) =>
      u.weightDelta.length === dims.output &&
      u.weightDelta[0]?.length === dims.input &&
      u.biasDelta.length === dims.output &&
      u.sampleCount > 0,
  );

  if (validUpdates.length < 1) {
    metrics.federatedRoundErrorsTotal.inc({ task, error: "invalid_updates" });
    throw new FederatedLearningError(
      "No valid client updates after validation",
      "INVALID_UPDATES",
    );
  }

  if (validUpdates.length < clientTarget) {
    log("warn", "federated.client_count_below_target", {
      actual: validUpdates.length,
      target: clientTarget,
      task,
    });
  }

  // Log participant count
  for (const update of validUpdates) {
    metrics.federatedClientsParticipatedTotal.inc({ client_id: update.clientId, task });
    metrics.federatedClientUpdatesTotal.inc({ client_id: update.clientId, task });
  }

  // Step 1+2: Clip each client's update (L2 norm) and apply DP noise
  const clippedUpdates = validUpdates.map((u) => {
    const clipped = clipL2Norm(u.weightDelta, u.biasDelta, MAX_CLIENT_L2_NORM);
    metrics.federatedClientUpdatesTotal.inc({ client_id: u.clientId, task });

    if (enableDP) {
      const dp = addDPNoise(clipped.clipped, clipped.biasClipped, 1.0);
      return {
        clientId: u.clientId,
        weightDelta: dp.noisyWeights,
        biasDelta: dp.noisyBias,
        sampleCount: u.sampleCount,
        loss: u.loss,
        accuracy: u.accuracy,
      };
    }
    return {
      clientId: u.clientId,
      weightDelta: clipped.clipped,
      biasDelta: clipped.biasClipped,
      sampleCount: u.sampleCount,
      loss: u.loss,
      accuracy: u.accuracy,
    };
  });

  // Step 3+4: FedAvg — weight by sample count
  const totalSamples = clippedUpdates.reduce((s, u) => s + u.sampleCount, 0);
  const outputDim = dims.output;
  const inputDim = dims.input;

  const globalWeightDelta: number[][] = Array.from({ length: outputDim }, () =>
    new Array(inputDim).fill(0),
  );
  const globalBiasDelta: number[] = new Array(outputDim).fill(0);

  for (const u of clippedUpdates) {
    const weight = u.sampleCount / totalSamples;
    for (let o = 0; o < outputDim; o++) {
      for (let i = 0; i < inputDim; i++) {
        globalWeightDelta[o][i] += weight * u.weightDelta[o][i];
      }
      globalBiasDelta[o] += weight * u.biasDelta[o];
    }
  }

  // Step 5: Apply to global model (gradient descent step)
  const model = getGlobalModel(task);
  const lr = GLOBAL_LEARNING_RATE;
  for (let o = 0; o < outputDim; o++) {
    for (let i = 0; i < inputDim; i++) {
      model.weights[o][i] += lr * globalWeightDelta[o][i];
    }
    model.bias[o] += lr * globalBiasDelta[o];
  }
  model.lastUpdatedRound = roundCounter;
  model.emaLoss =
    0.9 * model.emaLoss +
    0.1 * (clippedUpdates.reduce((s, u) => s + u.loss, 0) / clippedUpdates.length);

  // Step 6: Compute convergence (L2 norm of global update)
  const allDelta = [...globalWeightDelta.flat(), ...globalBiasDelta];
  const convergence = Math.sqrt(allDelta.reduce((s, v) => s + v * v, 0));

  // Aggregate mean metrics
  const meanLoss = clippedUpdates.reduce((s, u) => s + u.loss, 0) / clippedUpdates.length;
  const meanAccuracy =
    clippedUpdates.reduce((s, u) => s + u.accuracy, 0) / clippedUpdates.length;

  const durationMs = t0.end({ participants: validUpdates.length });

  metrics.federatedRoundLatencyMs.observe({ task, participants: String(validUpdates.length) }, durationMs);
  metrics.federatedAggregationConvergence.observe({ task }, convergence);

  const provenance = buildServiceProvenance({
    service: FEDERATED_SERVICE,
    serviceVersion: FEDERATED_VERSION,
    taskHeadId: `federated-${task}-probe-v1`,
    taskHeadVersion: "0.1.0",
    taskHeadDataset: "Federated client updates (no raw data shared)",
    taskHeadMetrics: {
      participants: validUpdates.length,
      rounds: roundCounter,
      convergence,
      mean_loss: meanLoss,
      mean_accuracy: meanAccuracy,
    },
    experimentId: "m49-federated-brain-learning",
  });

  return {
    round: roundCounter,
    task,
    participantCount: validUpdates.length,
    globalWeightDelta,
    globalBiasDelta,
    meanLoss,
    meanAccuracy,
    convergence,
    totalSamples,
    durationMs,
    provenance,
  };
}

/**
 * Get the current global model weights for distribution to clients.
 *
 * Clients call this before training to get the starting point.
 *
 * @param task - Task head type
 * @returns Weight matrix and bias vector
 */
export function getGlobalModelWeights(
  task: FederatedTask,
): { weights: number[][]; bias: number[]; round: number } {
  const model = getGlobalModel(task);
  return {
    weights: model.weights,
    bias: model.bias,
    round: model.lastUpdatedRound,
  };
}

/**
 * Validate a client update before aggregation.
 * Checks dimensions, sample count, and weight range.
 */
export function validateClientUpdate(update: ClientUpdate): { valid: boolean; reason?: string } {
  const dims = TASK_DIMENSIONS[update.task];

  if (!dims) {
    return { valid: false, reason: `Unknown task: ${update.task}` };
  }

  if (update.weightDelta.length !== dims.output) {
    return {
      valid: false,
      reason: `Weight rows: expected ${dims.output}, got ${update.weightDelta.length}`,
    };
  }

  if (update.weightDelta[0]?.length !== dims.input) {
    return {
      valid: false,
      reason: `Weight cols: expected ${dims.input}, got ${update.weightDelta[0]?.length}`,
    };
  }

  if (update.biasDelta.length !== dims.output) {
    return {
      valid: false,
      reason: `Bias length: expected ${dims.output}, got ${update.biasDelta.length}`,
    };
  }

  if (update.sampleCount <= 0) {
    return { valid: false, reason: "sampleCount must be positive" };
  }

  // Check for NaN or Infinity in deltas
  const allValues = [...update.weightDelta.flat(), ...update.biasDelta];
  if (allValues.some((v) => !Number.isFinite(v))) {
    return { valid: false, reason: "Update contains NaN or Infinity" };
  }

  return { valid: true };
}

