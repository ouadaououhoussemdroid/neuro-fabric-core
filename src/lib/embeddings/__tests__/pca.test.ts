import { describe, it, expect } from "vitest";
import { fitPCA, transformPCA } from "../pca";

describe("fitPCA", () => {
  it("is deterministic — identical input produces byte-identical components (NFC-014 regression test)", () => {
    const X = Array.from({ length: 20 }, (_, i) =>
      Array.from({ length: 6 }, (_, j) => Math.sin(i * 0.3 + j) + j * 0.1),
    );
    const a = fitPCA(X, 3);
    const b = fitPCA(X, 3);
    expect(a.components).toEqual(b.components);
    expect(a.explainedVariance).toEqual(b.explainedVariance);
  });

  it("computes the mean as the actual per-column average of the input", () => {
    const X = [
      [1, 10],
      [3, 20],
      [5, 30],
    ];
    const model = fitPCA(X, 1);
    expect(model.mean[0]).toBeCloseTo(3, 10); // (1+3+5)/3
    expect(model.mean[1]).toBeCloseTo(20, 10); // (10+20+30)/3
  });

  it("finds the dominant direction for data varying along a single axis", () => {
    // Points lie exactly on the line y = 2x -> all variance is along [1, 2] (normalized)
    const X = Array.from({ length: 30 }, (_, i) => {
      const x = i - 15;
      return [x, 2 * x];
    });
    const model = fitPCA(X, 1);
    const pc1 = model.components[0];
    const expectedDir = [1 / Math.sqrt(5), 2 / Math.sqrt(5)];
    // Power iteration can converge to either sign of the eigenvector
    const dot = Math.abs(pc1[0] * expectedDir[0] + pc1[1] * expectedDir[1]);
    expect(dot).toBeCloseTo(1, 6);
  });

  it("orders components by descending explained variance", () => {
    // Strong variance along x, weak variance along y
    const X = Array.from({ length: 30 }, (_, i) => [
      (i - 15) * 10,
      (i % 3) - 1, // small, low-variance wobble
    ]);
    const model = fitPCA(X, 2);
    expect(model.explainedVariance[0]).toBeGreaterThan(model.explainedVariance[1]);
  });

  it("throws on empty input", () => {
    expect(() => fitPCA([], 2)).toThrow("empty input");
  });

  it("caps the number of components at min(k, d)", () => {
    const X = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const model = fitPCA(X, 10); // requesting more components than dimensions (d=2)
    expect(model.components).toHaveLength(2);
  });
});

describe("transformPCA", () => {
  it("projects the mean vector itself to (approximately) the origin", () => {
    const X = Array.from({ length: 20 }, (_, i) => [i, i * 2 + 1, Math.sin(i)]);
    const model = fitPCA(X, 2);
    const projected = transformPCA(model, model.mean);
    projected.forEach((v) => expect(v).toBeCloseTo(0, 8));
  });

  it("produces a vector with length equal to the number of components", () => {
    const X = Array.from({ length: 20 }, (_, i) => [i, i * 2, i * 3, i * 4]);
    const model = fitPCA(X, 2);
    expect(transformPCA(model, X[0])).toHaveLength(2);
  });

  it("round-trips reasonably for a point already in the training set (full-rank projection)", () => {
    // 2-D data, request 2 components (full rank) -> projection should be
    // exactly invertible via components^T (orthonormal basis)
    const X = Array.from({ length: 20 }, (_, i) => [Math.sin(i), Math.cos(i * 0.7)]);
    const model = fitPCA(X, 2);
    const x = X[5];
    const z = transformPCA(model, x);
    // Reconstruct: x_hat = mean + components^T @ z
    const d = model.mean.length;
    const xHat = new Array(d).fill(0);
    for (let i = 0; i < d; i++) {
      let s = model.mean[i];
      for (let k = 0; k < model.components.length; k++) s += model.components[k][i] * z[k];
      xHat[i] = s;
    }
    for (let i = 0; i < d; i++) expect(xHat[i]).toBeCloseTo(x[i], 3);
  });
});
