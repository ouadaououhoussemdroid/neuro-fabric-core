/**
 * Foundation-model embeddings facade. Routes through the registered adapter
 * for `modelId`, defaulting to the PCA legacy adapter so existing callers see
 * no behavior change.
 */
import { createAdapter, DEFAULT_EMBEDDER_ID } from "../models/registry";
import type { EmbeddingOutput, ModelInput } from "../types";

export interface EmbedOptions {
  modelId?: string;
}

export async function embed(
  input: ModelInput,
  opts: EmbedOptions = {},
): Promise<EmbeddingOutput> {
  const id = opts.modelId ?? DEFAULT_EMBEDDER_ID;
  const adapter = createAdapter(id);
  if (!adapter.embed) {
    throw new Error(`Adapter "${id}" does not support embeddings`);
  }
  await adapter.load();
  try {
    return await adapter.embed(input);
  } finally {
    await adapter.unload();
  }
}

export { DEFAULT_EMBEDDER_ID };