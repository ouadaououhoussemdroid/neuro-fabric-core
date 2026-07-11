/**
 * Braindecode adapter.
 *
 * Braindecode (https://braindecode.org) is a PyTorch-based library of EEG
 * decoding architectures (ShallowFBCSPNet, Deep4Net, EEGNetv4, EEGConformer,
 * EEGITNet, …). It cannot run as JS — it requires a Python+PyTorch runtime.
 * In the browser that means Pyodide + a PyTorch-on-Pyodide wheel, and on the
 * server it means a Python service. Either way, the adapter delegates the
 * actual forward pass to a `BraindecodeBridge` so we can:
 *
 *   1. Inject a real Pyodide-backed bridge in production once the wheel is
 *      available in the user's environment.
 *   2. Inject a fake bridge in tests / SSR.
 *   3. Fail cleanly when no bridge is available so the embed() facade can
 *      fall back to ONNX → PCA.
 *
 * The adapter validates input shape against the registered model's expected
 * channel/sample contract before forwarding, so callers get a precise error
 * instead of an opaque Python traceback.
 */
import type { EEGModelAdapter } from "./types";
import type { EmbeddingOutput, ModelDescriptor, ModelInput, PredictionOutput } from "../types";
import { segment } from "../../eeg/preprocessing/segment";

/**
 * Catalogue of Braindecode architectures we expose through the adapter.
 * Numbers come from the Braindecode reference implementations; treat them
 * as defaults that callers can override per ModelArtifact.
 */
export const BRAINDECODE_MODELS = {
  ShallowFBCSPNet: {
    name: "ShallowFBCSPNet",
    description: "Shallow ConvNet for motor imagery (Schirrmeister et al. 2017).",
    defaultSampleRate: 250,
    defaultWindowSamples: 1125, // 4.5 s @ 250 Hz, BCI-IV-2a default
    embeddingDim: 40,
    task: "classification",
  },
  Deep4Net: {
    name: "Deep4Net",
    description: "Deep ConvNet (Schirrmeister et al. 2017).",
    defaultSampleRate: 250,
    defaultWindowSamples: 1125,
    embeddingDim: 200,
    task: "classification",
  },
  EEGNetv4: {
    name: "EEGNetv4",
    description: "Compact CNN (Lawhern et al. 2018).",
    defaultSampleRate: 128,
    defaultWindowSamples: 256,
    embeddingDim: 16,
    task: "classification",
  },
  EEGConformer: {
    name: "EEGConformer",
    description: "Conv + Transformer hybrid (Song et al. 2022).",
    defaultSampleRate: 250,
    defaultWindowSamples: 1000,
    embeddingDim: 32,
    task: "classification",
  },
} as const satisfies Record<
  string,
  {
    name: string;
    description: string;
    defaultSampleRate: number;
    defaultWindowSamples: number;
    embeddingDim: number;
    task: "classification" | "embedding" | "regression";
  }
>;

export type BraindecodeModelName = keyof typeof BRAINDECODE_MODELS;

/**
 * Runtime bridge contract. Implementations forward an [C, T] window through
 * a Braindecode model and return both penultimate-layer features (embedding)
 * and final logits (if classification head present).
 */
export interface BraindecodeForwardResult {
  embedding: number[];
  logits?: number[];
}
export interface BraindecodeBridge {
  /** Whether this bridge can run a model in the current environment. */
  isAvailable(): Promise<boolean>;
  /** Prepare runtime + model weights. Idempotent. */
  load(opts: {
    architecture: BraindecodeModelName;
    channels: number;
    sampleRate: number;
    windowSamples: number;
    weightsUri?: string;
  }): Promise<void>;
  forward(window: number[][]): Promise<BraindecodeForwardResult>;
  unload(): Promise<void>;
}

/**
 * Default bridge that returns `isAvailable() === false`. A real
 * Pyodide+torch bridge can be injected by calling `setBraindecodeBridge()`.
 */
let activeBridgeFactory: (() => BraindecodeBridge) | null = null;
export function setBraindecodeBridge(factory: (() => BraindecodeBridge) | null): void {
  activeBridgeFactory = factory;
}
function defaultBridge(): BraindecodeBridge {
  return {
    async isAvailable() {
      return false;
    },
    async load() {
      throw new Error(
        "Braindecode bridge unavailable: no Pyodide+PyTorch runtime registered. " +
          "Call setBraindecodeBridge() with a real bridge to enable.",
      );
    },
    async forward() {
      throw new Error("Braindecode bridge unavailable");
    },
    async unload() {},
  };
}

export interface BraindecodeAdapterOptions {
  id: string;
  architecture: BraindecodeModelName;
  channels: number;
  sampleRate?: number;
  windowSamples?: number;
  weightsUri?: string;
  version?: string;
  bridge?: () => BraindecodeBridge;
  numClasses?: number;
  isExperimental?: boolean;
}

export class BraindecodeAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor;
  private bridge: BraindecodeBridge | null = null;
  private loaded = false;
  private readonly opts: BraindecodeAdapterOptions;
  private readonly spec: (typeof BRAINDECODE_MODELS)[BraindecodeModelName];

  constructor(opts: BraindecodeAdapterOptions) {
    this.opts = opts;
    this.spec = BRAINDECODE_MODELS[opts.architecture];
    const sampleRate = opts.sampleRate ?? this.spec.defaultSampleRate;
    const windowSamples = opts.windowSamples ?? this.spec.defaultWindowSamples;
    this.descriptor = {
      id: opts.id,
      kind: "braindecode",
      name: `Braindecode • ${this.spec.name}`,
      version: opts.version ?? "0.1.0",
      description: this.spec.description,
      isExperimental: opts.isExperimental ?? true,
      artifactUri: opts.weightsUri,
      capabilities: {
        task: this.spec.task,
        channels: opts.channels,
        sampleRate,
        windowSamples,
        embeddingDim: this.spec.embeddingDim,
        numClasses: opts.numClasses,
        runtime: "pyodide",
        implemented: true,
      },
      createdAt: "2026-06-17",
    };
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async load(): Promise<void> {
    if (this.loaded) return;
    const factory = this.opts.bridge ?? activeBridgeFactory ?? defaultBridge;
    this.bridge = factory();
    const available = await this.bridge.isAvailable();
    if (!available) {
      throw new Error(
        `Braindecode adapter "${this.descriptor.id}": runtime not available in this environment`,
      );
    }
    await this.bridge.load({
      architecture: this.opts.architecture,
      channels: this.opts.channels,
      sampleRate: this.descriptor.capabilities.sampleRate!,
      windowSamples: this.descriptor.capabilities.windowSamples!,
      weightsUri: this.opts.weightsUri,
    });
    this.loaded = true;
  }

  async unload(): Promise<void> {
    try {
      await this.bridge?.unload();
    } finally {
      this.bridge = null;
      this.loaded = false;
    }
  }

  /** Validate + extract the first [C, T] window from a ModelInput. */
  private prepareWindow(input: ModelInput): number[][] {
    const expectedC = this.opts.channels;
    const expectedT = this.descriptor.capabilities.windowSamples!;
    let win: number[][];
    if (input.kind === "windows") {
      if (input.windows.length === 0) throw new Error("Braindecode: empty windows");
      win = input.windows[0].data;
    } else if (input.kind === "signal") {
      const winSec = expectedT / this.descriptor.capabilities.sampleRate!;
      const segs = segment(input.signal.data, input.signal.sampleRate, winSec, 0.5);
      if (segs.length === 0) throw new Error("Braindecode: signal too short to window");
      win = segs[0].data;
    } else {
      throw new Error("Braindecode: feature input not supported (raw windows required)");
    }
    if (win.length !== expectedC) {
      throw new Error(
        `Braindecode "${this.descriptor.id}": expected ${expectedC} channels, got ${win.length}`,
      );
    }
    const T = win[0]?.length ?? 0;
    if (T !== expectedT) {
      throw new Error(
        `Braindecode "${this.descriptor.id}": expected ${expectedT} samples, got ${T}`,
      );
    }
    return win;
  }

  async embed(input: ModelInput): Promise<EmbeddingOutput> {
    if (!this.bridge || !this.loaded) {
      throw new Error(`Braindecode "${this.descriptor.id}": not loaded`);
    }
    const win = this.prepareWindow(input);
    const t0 = performance.now();
    const { embedding } = await this.bridge.forward(win);
    return {
      vector: embedding,
      dim: embedding.length,
      modelId: this.descriptor.id,
      durationMs: +(performance.now() - t0).toFixed(2),
    };
  }

  async predict(input: ModelInput): Promise<PredictionOutput> {
    if (!this.bridge || !this.loaded) {
      throw new Error(`Braindecode "${this.descriptor.id}": not loaded`);
    }
    const win = this.prepareWindow(input);
    const t0 = performance.now();
    const { logits, embedding } = await this.bridge.forward(win);
    const values: Record<string, number> = {};
    const out = logits ?? embedding;
    out.forEach((v, i) => (values[`class_${i}`] = v));
    return {
      values,
      modelId: this.descriptor.id,
      durationMs: +(performance.now() - t0).toFixed(2),
    };
  }
}

/** Capability probe: true if a Braindecode bridge has been registered and is available. */
export async function isBraindecodeAvailable(
  factory: (() => BraindecodeBridge) | null = activeBridgeFactory,
): Promise<boolean> {
  if (!factory) return false;
  try {
    return await factory().isAvailable();
  } catch {
    return false;
  }
}
