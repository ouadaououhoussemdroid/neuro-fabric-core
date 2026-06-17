/**
 * Braindecode adapter stub. Concrete implementation will bridge through
 * Pyodide (see src/hooks/use-pyodide.ts) to run Braindecode model surfaces.
 */
import type { EEGModelAdapter } from "./types";
import {
  NotImplementedError,
  type EmbeddingOutput,
  type ModelDescriptor,
  type ModelInput,
  type PredictionOutput,
} from "../types";

export class BraindecodeAdapter implements EEGModelAdapter {
  readonly descriptor: ModelDescriptor = {
    id: "braindecode-placeholder",
    kind: "braindecode",
    name: "Braindecode (Planned, Pyodide)",
    version: "0.0.0",
    description:
      "Placeholder for Braindecode models executed in-browser via Pyodide. Not yet implemented.",
    isExperimental: true,
    capabilities: {
      task: "classification",
      channels: null,
      sampleRate: null,
      windowSamples: null,
      runtime: "pyodide",
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
