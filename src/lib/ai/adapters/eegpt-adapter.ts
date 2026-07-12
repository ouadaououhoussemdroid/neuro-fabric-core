/**
 * EEGPT adapter — scheduled, not implemented.
 *
 * T-016 (honesty pass): per the alignment audit (C2), EEGPT integration is
 * not yet actionable because no public, license-clear weight checkpoint
 * exists for an EEGPT-class foundation model. Rather than ship a placeholder
 * that implies readiness, this adapter now:
 *
 *   1. Explicitly declares `implemented: false` in its descriptor.
 *   2. Throws `NotImplementedError` on every method (unchanged behaviour).
 *   3. Documents the concrete unblock conditions so a future contributor
 *      knows exactly what's needed to make it real.
 *
 * The adapter remains registered so the model list shows EEGPT as
 * "scheduled" — but the embed facade will never route to it because
 * `load()` throws, and `implemented: false` lets the UI distinguish it
 * from production-ready models.
 *
 * Unblock conditions (all three required):
 *   - A publicly distributable EEGPT checkpoint (weights) with a clear
 *     license (BSD-3 / MIT / Apache-2.0 preferred).
 *   - An ONNX export path verified against the reference implementation
 *     (PyTorch↔ORT cosine > 0.999, matching the EEGConformer contract).
 *   - A runtime decision: server-side inference (recommended for 512-D) or
 *     quantized WebGPU ONNX (if < 50 MB post-quantisation).
 *
 * Until then, callers should use `braindecode-eegconformer-prod` for
 * foundation-model embeddings.
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
    name: "EEGPT Foundation Model (Scheduled)",
    version: "0.0.0",
    description:
      "Scheduled (not implemented): an EEGPT-class foundation model for high-dimensional embeddings. " +
      "Blocked on a public, license-clear weight checkpoint. See T-016. " +
      "Use braindecode-eegconformer-prod for foundation-model embeddings in the meantime.",
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
