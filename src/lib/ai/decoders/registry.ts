/**
 * M31 Shared Service Layer — TaskHeadRegistry.
 *
 * A generic, reusable registry for downstream NeuroAI task heads (linear
 * probes, small MLPs, statistical detectors). Each registered head declares
 * its input/output dimensions, provenance metadata, and whether it runs on
 * the server (requires onnxruntime-node) or browser (WASM-compatible).
 *
 * Services like Subject Identity, Cognitive State, and Anomaly Detection all
 * register their heads here. This module is browser-safe (no `.server.ts`
 * suffix; no onnxruntime import) so the registry can be read from both the
 * browser and the server.
 *
 * Pattern mirrors `src/lib/ai/models/registry.ts` — a module-level Map with
 * `register`, `get`, `has`, and `list` operations.
 */

/** Artifact SHA-256 digest format (64 hex chars). */
export type SHA256 = string;

/** Where a task head's model runs. */
export type InferenceTarget = "server" | "browser" | "both";

/** Describes a single task-head version. */
export interface TaskHeadDescriptor {
  /** Unique identifier: `${service}-${head-name}-${version}`. */
  id: string;
  /** Human-readable name. */
  name: string;
  /** Version tag (semver or git SHA). */
  version: string;
  /** Which service this head belongs to (e.g. "subject-identity", "cognitive-intelligence", "anomaly-detection"). */
  service: string;
  /** Input dimension — must match the upstream embedding space (e.g. 2312 for Joint-2312). */
  inputDim: number;
  /** Output dimension — task-specific. */
  outputDim: number;
  /** Where the model runs. */
  inferenceTarget: InferenceTarget;
  /** SHA-256 of the head artifact (ONNX or checkpoint). */
  sha256: SHA256;
  /** Optional URL to the artifact file. */
  artifactUri?: string;
  /** Optional size in bytes. */
  artifactSize?: number;
  /** Training metadata: dataset, protocol, held-out metrics. */
  training: {
    dataset: string;
    protocol: string;
    metrics: Record<string, number>;
  };
  /** Optional evaluation/validation results. */
  validation?: Record<string, unknown>;
}

/**
 * Register a task head descriptor. Idempotent — calling `registerTaskHead`
 * twice with the same `id` replaces the prior entry.
 */
export function registerTaskHead(descriptor: TaskHeadDescriptor): void {
  taskHeadRegistry.set(descriptor.id, descriptor);
}

/**
 * Look up a registered task head by id. Returns `undefined` if not found.
 */
export function getTaskHead(id: string): TaskHeadDescriptor | undefined {
  return taskHeadRegistry.get(id);
}

/** Returns `true` if a task head with the given id is registered. */
export function hasTaskHead(id: string): boolean {
  return taskHeadRegistry.has(id);
}

/** Returns all registered task-head descriptors. */
export function listTaskHeads(): TaskHeadDescriptor[] {
  return Array.from(taskHeadRegistry.values());
}

/**
 * Returns all task heads for a given service (e.g. "subject-identity").
 */
export function getTaskHeadsByService(service: string): TaskHeadDescriptor[] {
  return Array.from(taskHeadRegistry.values()).filter((h) => h.service === service);
}

/**
 * Returns the default (first-registered) task head for a service.
 * Used when the caller doesn't specify a particular head version.
 */
export function getDefaultTaskHead(service: string): TaskHeadDescriptor | undefined {
  return getTaskHeadsByService(service)[0];
}

/**
 * Deterministic service identifier — used for logging, metrics, and
 * audit-log tagging. Format: `neurofabric-{service}@v{version}`.
 */
export function serviceIdentity(service: string, version: string): string {
  return `neurofabric-${service}@v${version}`;
}

// ─────────────────────────────────────────────────────────────────────────
// Module-level registry (mirrors models/registry.ts)
// ─────────────────────────────────────────────────────────────────────────

const taskHeadRegistry = new Map<string, TaskHeadDescriptor>();
