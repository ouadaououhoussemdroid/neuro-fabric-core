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