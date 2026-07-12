import { describe, it, expect } from "vitest";
import { cosine, l2 } from "../cosine";

describe("cosine", () => {
  it("returns 1 for identical vectors", () => {
    expect(cosine([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 10);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 10);
  });

  it("returns -1 for exactly opposite vectors", () => {
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1, 10);
  });

  it("is scale-invariant (magnitude doesn't affect similarity)", () => {
    expect(cosine([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 10);
    expect(cosine([1, 2, 3], [100, 200, 300])).toBeCloseTo(1, 10);
  });

  it("computes a known intermediate value correctly", () => {
    // a=[1,0], b=[1,1] -> dot=1, |a|=1, |b|=sqrt(2) -> cos = 1/sqrt(2)
    expect(cosine([1, 0], [1, 1])).toBeCloseTo(1 / Math.sqrt(2), 10);
  });

  it("returns 0 (not NaN) when either vector is all-zero", () => {
    expect(cosine([0, 0], [1, 2])).toBe(0);
    expect(cosine([1, 2], [0, 0])).toBe(0);
    expect(cosine([0, 0], [0, 0])).toBe(0);
  });

  it("throws on dimension mismatch", () => {
    expect(() => cosine([1, 2], [1, 2, 3])).toThrow("dim mismatch");
  });
});

describe("l2", () => {
  it("computes the classic 3-4-5 right triangle distance", () => {
    expect(l2([0, 0], [3, 4])).toBe(5);
  });

  it("returns 0 for identical points", () => {
    expect(l2([1, 1, 1], [1, 1, 1])).toBe(0);
  });

  it("is symmetric", () => {
    const a = [1, 2, 3];
    const b = [4, -1, 7];
    expect(l2(a, b)).toBeCloseTo(l2(b, a), 10);
  });
});
