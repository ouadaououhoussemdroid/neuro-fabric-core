/**
 * Shared types for the AI foundation layer.
 *
 * These types are intentionally runtime-agnostic so the same surface can host
 * a PCA wrapper, an ONNX session, a PyTorch export served via WASM, a
 * Braindecode model, or an EEGPT foundation model.
 */
import type { EEGSignal, EEGWindow } from "../eeg/types";

export type ModelKind =
  | "pca"
  | "linear-ae"
  | "onnx"
  | "pytorch-export"
  | "braindecode"
  | "eegpt";

export type ModelTask =
  | "embedding"
  | "classification"
  | "regression"
  | "reconstruction"
  | "foundation";

export type ModelRuntime =
  | "js"        // pure JS / WebAssembly-free
  | "wasm"      // WebAssembly (ONNX Runtime Web, tract, ggml, ...)
  | "webgpu"    // WebGPU accelerated
  | "pyodide"   // Python in browser (Braindecode, MNE)
  | "server";   // server-side inference

export interface ModelCapabilities {
  task: ModelTask;
  /** Input channel count expectation; null = adaptive */
  channels: number | null;
  /** Required sample rate, Hz; null = adaptive */
  sampleRate: number | null;
  /** Window length in samples; null = adaptive / whole-signal */
  windowSamples: number | null;
  /** Output embedding dim (for embedding tasks) */
  embeddingDim?: number;
  /** Number of output classes (for classification tasks) */
  numClasses?: number;
  runtime: ModelRuntime;
  /** True when the adapter has a real implementation backing it. */
  implemented: boolean;
}

export interface ModelDescriptor {
  id: string;
  kind: ModelKind;
  name: string;
  version: string;
  description: string;
  isExperimental: boolean;
  capabilities: ModelCapabilities;
  /** Optional URI to weights/artifact (resolved by the adapter). */
  artifactUri?: string;
  createdAt: string;
}

export interface EmbeddingOutput {
  vector: number[];
  dim: number;
  modelId: string;
  durationMs: number;
}

export interface PredictionOutput<T = Record<string, number>> {
  values: T;
  modelId: string;
  durationMs: number;
}

export type ModelInput =
  | { kind: "signal"; signal: EEGSignal }
  | { kind: "windows"; windows: EEGWindow[] }
  | { kind: "features"; features: number[][] };

export class NotImplementedError extends Error {
  constructor(modelId: string, detail?: string) {
    super(`Model adapter "${modelId}" is not yet implemented${detail ? `: ${detail}` : ""}`);
    this.name = "NotImplementedError";
  }
}
