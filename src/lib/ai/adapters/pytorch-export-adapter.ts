/**
 * PyTorch-export adapter stub. Intended target: TorchScript / ExecuTorch /
 * TorchDynamo exports converted to ONNX or a WASM runtime. Until that
 * pipeline lands, this slot is registered but throws on load().
 */
import type { EEGModelAdapter } from "./types";
import {
  NotImplementedError,
  type EmbeddingOutput,
  type ModelDescriptor,
  type ModelInput,
  type PredictionOutput,
} from "../types";

export class PyTorchExportAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor = {
    id: "pytorch-export-placeholder",
    kind: "pytorch-export",
    name: "PyTorch Export (Planned)",
    version: "0.0.0",
    description:
      "Placeholder for PyTorch model exports (TorchScript → ONNX or ExecuTorch). Not yet implemented.",
    isExperimental: true,
    capabilities: {
      task: "embedding",
      channels: null,
      sampleRate: null,
      windowSamples: null,
      runtime: "wasm",
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
