/**
 * ONNX adapter stub. A concrete implementation will load onnxruntime-web
 * (WASM / WebGPU EP), instantiate an InferenceSession from `artifactUri`, and
 * map ModelInput → input tensor → output tensor → EmbeddingOutput /
 * PredictionOutput.
 *
 * Kept as a stub so the registry can expose the slot today without shipping
 * the runtime.
 */
import type { EEGModelAdapter } from "./types";
import {
  NotImplementedError,
  type EmbeddingOutput,
  type ModelDescriptor,
  type ModelInput,
  type PredictionOutput,
} from "../types";

export interface ONNXAdapterOptions {
  id: string;
  name: string;
  version: string;
  description: string;
  artifactUri: string;
  task: "embedding" | "classification" | "regression";
  channels?: number;
  sampleRate?: number;
  windowSamples?: number;
  embeddingDim?: number;
  numClasses?: number;
  isExperimental?: boolean;
}

export class ONNXAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor;
  private loaded = false;

  constructor(opts: ONNXAdapterOptions) {
    this.descriptor = {
      id: opts.id,
      kind: "onnx",
      name: opts.name,
      version: opts.version,
      description: opts.description,
      isExperimental: opts.isExperimental ?? true,
      artifactUri: opts.artifactUri,
      capabilities: {
        task: opts.task,
        channels: opts.channels ?? null,
        sampleRate: opts.sampleRate ?? null,
        windowSamples: opts.windowSamples ?? null,
        embeddingDim: opts.embeddingDim,
        numClasses: opts.numClasses,
        runtime: "wasm",
        implemented: false,
      },
      createdAt: new Date().toISOString().slice(0, 10),
    };
  }

  async load(): Promise<void> {
    throw new NotImplementedError(this.descriptor.id, "onnxruntime-web not yet wired");
  }

  async unload(): Promise<void> {
    this.loaded = false;
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  async embed(_input: ModelInput): Promise<EmbeddingOutput> {
    throw new NotImplementedError(this.descriptor.id);
  }

  async predict(_input: ModelInput): Promise<PredictionOutput> {
    throw new NotImplementedError(this.descriptor.id);
  }
}
