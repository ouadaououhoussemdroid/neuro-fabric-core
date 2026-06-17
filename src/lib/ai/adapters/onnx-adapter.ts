/**
 * Real ONNX Runtime Web adapter.
 *
 * Loads an ONNX model from a URL, ArrayBuffer, or Uint8Array via
 * `onnxruntime-web`, manages session lifecycle, and maps ModelInput into the
 * session's first input tensor. Returns the first output as an EmbeddingOutput
 * or PredictionOutput.
 *
 * Designed to be drop-in: when load() or embed() fails (no WASM, no model,
 * bad shape) the higher-level facade catches and falls back to the PCA
 * adapter, preserving backward compatibility.
 */
import type { EEGModelAdapter } from "./types";
import type {
  EmbeddingOutput,
  ModelDescriptor,
  ModelInput,
  PredictionOutput,
} from "../types";
import { bandPowerFeatures } from "../../embeddings/features";
import { segment } from "../../eeg/preprocessing/segment";

// Minimal structural typing of the ort surface we depend on. Keeps tests
// trivial to mock without pulling the full type graph.
export interface OrtTensorLike {
  data: Float32Array | ArrayLike<number>;
  dims: readonly number[];
}
export interface OrtSessionLike {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
  release?(): Promise<void>;
}
export interface OrtRuntime {
  InferenceSession: {
    create(
      path: string | ArrayBuffer | Uint8Array,
      options?: Record<string, unknown>,
    ): Promise<OrtSessionLike>;
  };
  Tensor: new (
    type: "float32",
    data: Float32Array,
    dims: readonly number[],
  ) => OrtTensorLike;
  env?: { wasm?: { numThreads?: number; simd?: boolean } };
}

export type ONNXInputShape =
  /** [1, features] — pre-extracted band-power feature vector. */
  | { kind: "features"; dim: number }
  /** [1, channels, samples] — raw window. */
  | { kind: "raw"; channels: number; samples: number };

export interface ONNXAdapterOptions {
  id: string;
  name: string;
  version: string;
  description: string;
  /** URL, path, or in-memory bytes for the .onnx artifact. */
  artifact: string | ArrayBuffer | Uint8Array;
  task: "embedding" | "classification" | "regression";
  inputShape: ONNXInputShape;
  /** Override the session input name (defaults to session.inputNames[0]). */
  inputName?: string;
  /** Override the output name (defaults to session.outputNames[0]). */
  outputName?: string;
  channels?: number;
  sampleRate?: number;
  windowSamples?: number;
  embeddingDim?: number;
  numClasses?: number;
  isExperimental?: boolean;
  /** Pluggable runtime — defaults to dynamic import of onnxruntime-web. */
  runtime?: () => Promise<OrtRuntime>;
  /** Execution providers passed to InferenceSession.create. */
  executionProviders?: Array<"wasm" | "webgpu" | "webgl" | "cpu">;
}

/** Lazily resolves onnxruntime-web; isolated so tests can stub it. */
async function defaultRuntime(): Promise<OrtRuntime> {
  const mod = (await import("onnxruntime-web")) as unknown as OrtRuntime;
  return mod;
}

/**
 * Best-effort capability probe. Returns true when onnxruntime-web can be
 * imported in the current environment. Cheap to call; result is memoised.
 */
let capabilityProbe: Promise<boolean> | null = null;
export function isONNXRuntimeAvailable(
  runtime: () => Promise<OrtRuntime> = defaultRuntime,
): Promise<boolean> {
  if (!capabilityProbe) {
    capabilityProbe = runtime()
      .then((rt) => !!rt && !!rt.InferenceSession && !!rt.Tensor)
      .catch(() => false);
  }
  return capabilityProbe;
}

/** Reset memoised capability probe (test helper). */
export function __resetONNXCapabilityProbe(): void {
  capabilityProbe = null;
}

export class ONNXAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor;
  private session: OrtSessionLike | null = null;
  private runtime: OrtRuntime | null = null;
  private readonly opts: ONNXAdapterOptions;

  constructor(opts: ONNXAdapterOptions) {
    this.opts = opts;
    this.descriptor = {
      id: opts.id,
      kind: "onnx",
      name: opts.name,
      version: opts.version,
      description: opts.description,
      isExperimental: opts.isExperimental ?? true,
      artifactUri: typeof opts.artifact === "string" ? opts.artifact : undefined,
      capabilities: {
        task: opts.task,
        channels: opts.channels ?? null,
        sampleRate: opts.sampleRate ?? null,
        windowSamples: opts.windowSamples ?? null,
        embeddingDim: opts.embeddingDim,
        numClasses: opts.numClasses,
        runtime: "wasm",
        implemented: true,
      },
      createdAt: new Date().toISOString().slice(0, 10),
    };
  }

  async load(): Promise<void> {
    if (this.session) return;
    const resolve = this.opts.runtime ?? defaultRuntime;
    this.runtime = await resolve();
    if (!this.runtime?.InferenceSession || !this.runtime?.Tensor) {
      throw new Error(`ONNXAdapter "${this.descriptor.id}": runtime unavailable`);
    }
    this.session = await this.runtime.InferenceSession.create(this.opts.artifact, {
      executionProviders: this.opts.executionProviders ?? ["wasm"],
    });
  }

  async unload(): Promise<void> {
    try {
      await this.session?.release?.();
    } finally {
      this.session = null;
    }
  }

  isLoaded(): boolean {
    return this.session !== null;
  }

  /** Build the [1, ...] Float32Array tensor input for this model. */
  private buildTensor(input: ModelInput): OrtTensorLike {
    if (!this.runtime) throw new Error("ONNXAdapter: runtime not loaded");
    const shape = this.opts.inputShape;
    if (shape.kind === "features") {
      const feats = featureVectorFromInput(input);
      if (feats.length !== shape.dim) {
        throw new Error(
          `ONNXAdapter "${this.descriptor.id}": expected ${shape.dim} features, got ${feats.length}`,
        );
      }
      return new this.runtime.Tensor("float32", Float32Array.from(feats), [1, shape.dim]);
    }
    // raw [1, C, T]
    const win = firstWindowFromInput(input);
    if (win.length !== shape.channels) {
      throw new Error(
        `ONNXAdapter "${this.descriptor.id}": expected ${shape.channels} channels, got ${win.length}`,
      );
    }
    const samples = win[0]?.length ?? 0;
    if (samples !== shape.samples) {
      throw new Error(
        `ONNXAdapter "${this.descriptor.id}": expected ${shape.samples} samples, got ${samples}`,
      );
    }
    const buf = new Float32Array(shape.channels * shape.samples);
    for (let c = 0; c < shape.channels; c++) {
      buf.set(win[c], c * shape.samples);
    }
    return new this.runtime.Tensor("float32", buf, [1, shape.channels, shape.samples]);
  }

  private async runOnce(input: ModelInput): Promise<{ vector: number[]; durationMs: number }> {
    if (!this.session) throw new Error(`ONNXAdapter "${this.descriptor.id}": not loaded`);
    const inputName = this.opts.inputName ?? this.session.inputNames[0];
    const outputName = this.opts.outputName ?? this.session.outputNames[0];
    const t0 = performance.now();
    const feeds = { [inputName]: this.buildTensor(input) };
    const out = await this.session.run(feeds);
    const tensor = out[outputName];
    if (!tensor) throw new Error(`ONNXAdapter: output "${outputName}" missing`);
    const vector = Array.from(tensor.data as ArrayLike<number>, Number);
    return { vector, durationMs: +(performance.now() - t0).toFixed(2) };
  }

  async embed(input: ModelInput): Promise<EmbeddingOutput> {
    const { vector, durationMs } = await this.runOnce(input);
    return { vector, dim: vector.length, modelId: this.descriptor.id, durationMs };
  }

  async predict(input: ModelInput): Promise<PredictionOutput> {
    const { vector, durationMs } = await this.runOnce(input);
    const values: Record<string, number> = {};
    vector.forEach((v, i) => (values[`class_${i}`] = v));
    return { values, modelId: this.descriptor.id, durationMs };
  }
}

// ---------- input helpers ----------

function firstWindowFromInput(input: ModelInput): number[][] {
  if (input.kind === "windows") {
    if (input.windows.length === 0) throw new Error("ONNXAdapter: empty windows");
    return input.windows[0].data;
  }
  if (input.kind === "signal") {
    const w = segment(input.signal.data, input.signal.sampleRate, 2, 0.5);
    if (w.length === 0) throw new Error("ONNXAdapter: signal too short to window");
    return w[0].data;
  }
  throw new Error("ONNXAdapter: feature input cannot be used as raw window");
}

function featureVectorFromInput(input: ModelInput): number[] {
  if (input.kind === "features") {
    if (input.features.length === 0) throw new Error("ONNXAdapter: empty features");
    return input.features[0];
  }
  const windows =
    input.kind === "windows"
      ? input.windows
      : segment(input.signal.data, input.signal.sampleRate, 2, 0.5);
  if (windows.length === 0) throw new Error("ONNXAdapter: no windows for features");
  // Mean-pool across windows for a single feature vector.
  const feats = windows.map(bandPowerFeatures);
  const dim = feats[0].length;
  const pooled = new Array<number>(dim).fill(0);
  for (const f of feats) for (let i = 0; i < dim; i++) pooled[i] += f[i];
  for (let i = 0; i < dim; i++) pooled[i] /= feats.length;
  return pooled;
}