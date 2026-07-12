import { describe, it, expect } from "vitest";
import {
  aggregateSubjectSignature,
  medianVector,
  l2Norm,
  cosineDistance,
  type WindowEmbedding,
} from "../subject-aggregation";

function makeWindows(): WindowEmbedding[] {
  const windows: WindowEmbedding[] = [];
  // Session A: vectors clustered around [1, 0, 0]
  for (let i = 0; i < 10; i++) {
    windows.push({ vector: [1 + i * 0.01, 0.001 * i, 0], sessionId: "A" });
  }
  // Session B: vectors clustered around [1, 0, 0] (stable)
  for (let i = 0; i < 10; i++) {
    windows.push({ vector: [1 - i * 0.01, 0.001 * i, 0], sessionId: "B" });
  }
  return windows;
}

describe("medianVector", () => {
  it("computes the element-wise median", () => {
    const v = medianVector([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expect(v).toEqual([2, 5]);
  });
  it("averages for even-length inputs", () => {
    const v = medianVector([
      [1, 10],
      [3, 20],
    ]);
    expect(v).toEqual([2, 15]);
  });
  it("returns empty for no inputs", () => {
    expect(medianVector([])).toEqual([]);
  });
});

describe("l2Norm", () => {
  it("computes the Euclidean norm", () => {
    expect(l2Norm([3, 4])).toBeCloseTo(5, 5);
    expect(l2Norm([0, 0, 0])).toBe(0);
  });
});

describe("cosineDistance", () => {
  it("is 0 for identical vectors", () => {
    expect(cosineDistance([1, 0, 0], [1, 0, 0])).toBeCloseTo(0, 5);
  });
  it("is 1 for orthogonal vectors", () => {
    expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1, 5);
  });
  it("is 2 for opposite vectors", () => {
    expect(cosineDistance([1, 0], [-1, 0])).toBeCloseTo(2, 5);
  });
});

describe("aggregateSubjectSignature", () => {
  it("produces a per-session breakdown", () => {
    const sig = aggregateSubjectSignature(makeWindows());
    expect(sig.nSessions).toBe(2);
    expect(sig.sessions).toHaveLength(2);
    expect(sig.sessions[0].sessionId).toBe("A");
    expect(sig.sessions[0].nWindows).toBe(10);
    expect(sig.totalWindows).toBe(20);
  });

  it("signature is the median of session medians", () => {
    const sig = aggregateSubjectSignature(makeWindows());
    expect(sig.signature).toHaveLength(3);
    expect(sig.signature[0]).toBeCloseTo(1, 1);
  });

  it("stability score is high for stable sessions", () => {
    const sig = aggregateSubjectSignature(makeWindows());
    expect(sig.stability.stabilityScore).toBeGreaterThan(0.9);
    expect(sig.stability.cosineSpread).toBeLessThan(0.1);
  });

  it("stability score is low for divergent sessions", () => {
    const windows: WindowEmbedding[] = [
      ...Array.from({ length: 10 }, (_, i) => ({ vector: [1, 0, 0 + i * 0.01], sessionId: "A" })),
      ...Array.from({ length: 10 }, (_, i) => ({ vector: [0, 1, 0 + i * 0.01], sessionId: "B" })),
    ];
    const sig = aggregateSubjectSignature(windows);
    expect(sig.stability.cosineSpread).toBeGreaterThan(0.5);
    expect(sig.stability.stabilityScore).toBeLessThan(0.5);
  });

  it("handles empty input", () => {
    const sig = aggregateSubjectSignature([]);
    expect(sig.nSessions).toBe(0);
    expect(sig.signature).toEqual([]);
    expect(sig.stability.stabilityScore).toBe(1);
  });

  it("produces a dominant basis vector", () => {
    const sig = aggregateSubjectSignature(makeWindows());
    expect(sig.dominantBasis).toHaveLength(3);
    const norm = l2Norm(sig.dominantBasis);
    expect(norm).toBeCloseTo(1, 3); // basis should be unit-norm
  });
});
