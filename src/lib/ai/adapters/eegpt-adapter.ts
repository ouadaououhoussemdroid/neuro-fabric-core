/**
 * EEGPT (foundation-model) adapter stub. The eventual runtime will be either
 * a server-hosted inference endpoint or a quantized ONNX/WebGPU export.
 * Capability surface is `foundation` — embeddings and (later) downstream
 * heads will both route through this adapter.
 */
import type { EEGModelAdapter } from "./types";
import {
  NotImplementedError,
  type EmbeddingOutput,
  type ModelDescriptor,
  type ModelInput,
  type PredictionOutput,
} from "../types";

export class EEGPTAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor = {
    id: "eegpt-placeholder",
    kind: "eegpt",
    name: "EEGPT Foundation Model (Planned)",
    version: "0.0.0",
    description:
      "Placeholder for an EEGPT-class foundation model. Will expose a high-dimensional embedding and downstream heads. Not yet implemented.",
    isExperimental: true,
    capabilities: {
      task: "foundation",
      channels: null,
      sampleRate: 256,
      windowSamples: 1024,
      embeddingDim: 512,
      runtime: "server",
      implemented: false,
    },
    createdAt: "2026-06-17",
  };

  async load(): Promise<void> {
    throw new NotImplementedError(this.descriptor.id);
  }
  async unload(): Promise<void> {}
  isLoaded(): boolean {
    return false;
  }
  async embed(_input: ModelInput): Promise<EmbeddingOutput> {
    throw new NotImplementedError(this.descriptor.id);
  }
  async predict(_input: ModelInput): Promise<PredictionOutput> {
    throw new NotImplementedError(this.descriptor.id);
  }
}
