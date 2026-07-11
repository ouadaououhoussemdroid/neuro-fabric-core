/**
 * Public surface of the AI foundation layer. Existing pipelines continue to
 * import from `src/lib/embeddings` and `src/lib/model-registry`; this module
 * is the forward-looking entry point.
 */
export * from "./types";
export * as adapters from "./adapters";
export * as models from "./models";
export * as embeddings from "./embeddings";
export * as inference from "./inference";
export * as artifacts from "./artifacts";
export * as validation from "./validation";
export * as benchmark from "./benchmark";
export * as vectorBridge from "./vector-bridge";
