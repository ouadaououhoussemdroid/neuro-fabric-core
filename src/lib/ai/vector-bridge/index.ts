/**
 * Bridges the AI embedding layer to the existing in-memory vector search
 * (`src/lib/vector-search`). The bridge embeds inputs through the AI facade
 * (with PCA fallback) and upserts the result into a VectorIndex, tagging
 * each entry with the model id so callers can keep neural and PCA stores
 * apart if they want.
 */
import { VectorIndex, type SearchHit } from "../../vector-search";
import { embed, type EmbedOptions } from "../embeddings";
import type { ModelInput } from "../types";

export interface NeuralIndexMeta {
  modelId: string;
  fellBack: boolean;
  fallbackReason?: string;
  [k: string]: unknown;
}

export class NeuralVectorIndex<M extends Record<string, unknown> = Record<string, unknown>> {
  private index = new VectorIndex<NeuralIndexMeta & M>();

  size(): number {
    return this.index.size();
  }

  async upsert(
    id: string,
    input: ModelInput,
    meta: M = {} as M,
    opts: EmbedOptions = {},
  ): Promise<NeuralIndexMeta & M> {
    const out = await embed(input, { normalize: true, ...opts });
    const fullMeta = {
      ...meta,
      modelId: out.modelId,
      fellBack: out.fellBack,
      fallbackReason: out.fallbackReason,
    } as NeuralIndexMeta & M;
    this.index.add({ id, vector: out.vector, meta: fullMeta });
    return fullMeta;
  }

  async search(
    input: ModelInput,
    k = 8,
    opts: EmbedOptions = {},
  ): Promise<SearchHit<NeuralIndexMeta & M>[]> {
    const out = await embed(input, { normalize: true, ...opts });
    return this.index.search(out.vector, k);
  }

  raw(): VectorIndex<NeuralIndexMeta & M> {
    return this.index;
  }
}