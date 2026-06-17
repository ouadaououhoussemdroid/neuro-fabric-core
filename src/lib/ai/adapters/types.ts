import type {
  EmbeddingOutput,
  ModelDescriptor,
  ModelInput,
  PredictionOutput,
} from "../types";

/**
 * Runtime contract every model adapter implements. Adapters wrap a concrete
 * runtime (PCA, ONNX session, Pyodide bridge, server RPC) behind a stable
 * interface so the inference engine can route uniformly.
 */
export interface EEGModelAdapter {
  readonly descriptor: ModelDescriptor;

  /** Lazy weight/runtime load. Idempotent. */
  load(): Promise<void>;

  /** Free runtime resources (sessions, tensors). Idempotent. */
  unload(): Promise<void>;

  /** Whether load() has succeeded. */
  isLoaded(): boolean;

  /** Embedding-task adapters implement this. */
  embed?(input: ModelInput): Promise<EmbeddingOutput>;

  /** Classification/regression adapters implement this. */
  predict?(input: ModelInput): Promise<PredictionOutput>;
}

export type AdapterFactory = () => EEGModelAdapter;
