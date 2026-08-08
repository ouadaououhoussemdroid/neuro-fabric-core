/**
 * T-015 Fix 5 — Registry drift regression tests.
 *
 * Ensures the legacy `model-registry` (used by the /models UI page) cannot
 * drift from the AI layer's authoritative `src/lib/ai/models/registry.ts`:
 *
 *   - ACTIVE_EMBEDDER must equal DEFAULT_EMBEDDER_ID (the AI registry default).
 *   - The active embedder's descriptor embeddingDim must equal 32 (the
 *     canonical vector(32) contract).
 *   - The active embedder must be wasmCompatible (so it runs in-browser).
 *   - getModel(ACTIVE_EMBEDDER) must return a real entry (no stale IDs).
 *   - ACTIVE_DECODER must be a registered decoder (trained-logistic-v0).
 */
import { describe, it, expect } from "vitest";
import { ACTIVE_EMBEDDER, ACTIVE_DECODER, getModel, getModelsByType } from "../index";
import { DEFAULT_EMBEDDER_ID, getDescriptor, listModels } from "../../ai/models/registry";

describe("T-015 Fix 5: model-registry drift guard", () => {
  it("ACTIVE_EMBEDDER equals the AI registry DEFAULT_EMBEDDER_ID", () => {
    expect(ACTIVE_EMBEDDER).toBe(DEFAULT_EMBEDDER_ID);
  });

  it("ACTIVE_EMBEDDER is a real model in the AI registry (no stale IDs)", () => {
    expect(getModel(ACTIVE_EMBEDDER)).toBeDefined();
    expect(getDescriptor(ACTIVE_EMBEDDER)).toBeDefined();
  });

  it("active embedder descriptor declares embeddingDim=32 (canonical vector(32))", () => {
    const d = getDescriptor(ACTIVE_EMBEDDER);
    expect(d).toBeDefined();
    expect(d!.capabilities.embeddingDim).toBe(32);
  });

  it("active embedder is wasmCompatible (runs in-browser)", () => {
    const d = getDescriptor(ACTIVE_EMBEDDER);
    expect(d).toBeDefined();
    expect(d!.capabilities.wasmCompatible).not.toBe(false);
  });

  it("ACTIVE_DECODER is trained-logistic-v0 (the ONNX decoder decodeCognitiveState prefers)", () => {
    expect(ACTIVE_DECODER).toBe("trained-logistic-v0");
  });

  it("ACTIVE_DECODER appears in getModelsByType('decoder')", () => {
    const decoders = getModelsByType("decoder");
    expect(decoders.some((m) => m.id === ACTIVE_DECODER)).toBe(true);
  });

  it("getModelsByType('embedder') includes all AI registry embedding models", () => {
    const embedders = getModelsByType("embedder");
    const embedderIds = new Set(embedders.map((m) => m.id));
    // Every AI-registry embedding model should appear in the legacy list.
    const aiIds = new Set(listModels().map((m) => m.id));
    for (const id of aiIds) {
      expect(embedderIds.has(id), `AI registry ID "${id}" missing from legacy embedder list`).toBe(
        true,
      );
    }
    // Spot-check key IDs:
    expect(embedderIds.has("pca-legacy-v1")).toBe(true);
    expect(embedderIds.has("braindecode-eegconformer-prod")).toBe(true);
    expect(embedderIds.has("onnx-eegpt")).toBe(true);
  });

  it("no stale linear-ae or tfjs-autoencoder-v1 in the embedder list", () => {
    const embedders = getModelsByType("embedder");
    expect(embedders.some((m) => m.id === "linear-ae")).toBe(false);
    expect(embedders.some((m) => m.id === "tfjs-autoencoder-v1")).toBe(false);
    expect(embedders.some((m) => m.id === "raw-bandpower")).toBe(false);
  });
});
