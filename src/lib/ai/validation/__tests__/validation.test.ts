import { describe, it, expect } from "vitest";
import {
  validateEmbedding,
  l2Normalize,
  isUnitNorm,
  EmbeddingValidationError,
} from "../index";

describe("validateEmbedding", () => {
  it("accepts a well-formed vector", () => {
    expect(() => validateEmbedding([0.1, -0.2, 0.3])).not.toThrow();
  });
  it("rejects NaN / Infinity", () => {
    expect(() => validateEmbedding([0.1, NaN, 0.3])).toThrow(EmbeddingValidationError);
    expect(() => validateEmbedding([0.1, Infinity, 0.3])).toThrow(/non-finite/);
  });
  it("rejects dim mismatch", () => {
    expect(() => validateEmbedding([1, 2, 3], { expectedDim: 4 })).toThrow(
      /expected dim 4/,
    );
  });
  it("rejects all-zero by default", () => {
    expect(() => validateEmbedding([0, 0, 0])).toThrow(/zero/);
  });
  it("rejects out-of-range values", () => {
    expect(() => validateEmbedding([1e9, 1], { maxAbs: 100 })).toThrow(/exceeds/);
  });
});

describe("l2Normalize", () => {
  it("produces a unit vector", () => {
    const v = l2Normalize([3, 4]);
    expect(isUnitNorm(v)).toBe(true);
  });
  it("passes through the zero vector", () => {
    expect(l2Normalize([0, 0, 0])).toEqual([0, 0, 0]);
  });
});