/**
 * Model artifact system. An "artifact" is everything required to instantiate
 * an adapter: identity, metadata, input/output schema, and a resolver for the
 * underlying bytes (URL, ArrayBuffer, or in-memory bytes).
 *
 * Artifacts are pure metadata containers — they do NOT load the model. The
 * registry uses them to mint adapter factories; the adapter does the actual
 * load when `load()` is called.
 */
import type { ModelKind, ModelTask, ModelRuntime } from "../types";

export interface ArtifactInputSchema {
  /** "features" = [1, dim] band-power vector. "raw" = [1, C, T] window. */
  kind: "features" | "raw";
  dim?: number;
  channels?: number;
  samples?: number;
  sampleRate?: number;
}

export interface ArtifactOutputSchema {
  embeddingDim?: number;
  numClasses?: number;
  /** Whether outputs are L2-normalised by the model itself. */
  normalized?: boolean;
}

export type ArtifactSource =
  | { kind: "url"; url: string }
  | { kind: "bytes"; bytes: Uint8Array | ArrayBuffer }
  | { kind: "inline"; description: string };

export interface ModelArtifact {
  id: string;
  kind: ModelKind;
  task: ModelTask;
  runtime: ModelRuntime;
  name: string;
  version: string;
  description: string;
  input: ArtifactInputSchema;
  output: ArtifactOutputSchema;
  source: ArtifactSource;
  /** Free-form provenance: training corpus, license, citation, sha. */
  provenance?: Record<string, string>;
  isExperimental?: boolean;
  createdAt?: string;
}

const artifacts = new Map<string, ModelArtifact>();

export function registerArtifact(a: ModelArtifact): void {
  artifacts.set(a.id, a);
}
export function listArtifacts(): ModelArtifact[] {
  return Array.from(artifacts.values());
}
export function getArtifact(id: string): ModelArtifact | undefined {
  return artifacts.get(id);
}
export function unregisterArtifact(id: string): boolean {
  return artifacts.delete(id);
}

/** Built-in artifact describing the legacy PCA pipeline. */
registerArtifact({
  id: "pca-legacy-v1",
  kind: "linear-ae",
  task: "embedding",
  runtime: "js",
  name: "PCA / Linear Autoencoder (Legacy)",
  version: "1.0.0",
  description: "Closed-form linear AE over band-power features. Fallback embedder.",
  input: { kind: "features" },
  output: { embeddingDim: 64, normalized: false },
  source: { kind: "inline", description: "in-process JS" },
  isExperimental: false,
  createdAt: "2026-05-25",
});

registerArtifact({
  id: "braindecode-eegnetv4-default",
  kind: "braindecode",
  task: "classification",
  runtime: "pyodide",
  name: "Braindecode • EEGNetv4 (default)",
  version: "0.1.0",
  description:
    "EEGNetv4 over a 22-channel 2 s @ 128 Hz window. Requires a Braindecode bridge.",
  input: { kind: "raw", channels: 22, samples: 256, sampleRate: 128 },
  output: { embeddingDim: 16, normalized: false },
  source: { kind: "inline", description: "Pyodide+PyTorch bridge (deferred)" },
  provenance: {
    paper: "Lawhern et al. 2018 (EEGNet)",
    library: "braindecode>=0.8",
  },
  isExperimental: true,
  createdAt: "2026-06-17",
});

/**
 * Selected production EEG foundation model: Braindecode EEGConformer
 * exported to ONNX. See docs/audits/2026-06-17_braindecode-model-selection.md
 * for the selection rationale and docs/braindecode-deployment-guide.md for
 * the artefact preparation workflow.
 */
registerArtifact({
  id: "braindecode-eegconformer-prod",
  kind: "braindecode",
  task: "embedding",
  runtime: "wasm",
  name: "Braindecode • EEGConformer (production, ONNX)",
  version: "1.0.0",
  description:
    "EEGConformer (Conv + Transformer) over a 22-channel 4 s @ 250 Hz window, " +
    "exported to ONNX and executed via onnxruntime-web. 32-D attention-pooled embedding.",
  input: { kind: "raw", channels: 22, samples: 1000, sampleRate: 250 },
  output: { embeddingDim: 32, normalized: false },
  source: { kind: "inline", description: "operator-provided ONNX file via registerBraindecodeEEGConformer()" },
  provenance: {
    paper: "Song et al. 2022 (EEG Conformer)",
    library: "braindecode>=0.8",
    exporter: "torch.onnx.export, opset 17",
  },
  isExperimental: false,
  createdAt: "2026-06-17",
});