import { describe, it, expect } from "vitest";
import { VectorIndex } from "../index";

describe("VectorIndex", () => {
  it("starts empty", () => {
    const idx = new VectorIndex();
    expect(idx.size()).toBe(0);
  });

  it("add() and size() track item count", () => {
    const idx = new VectorIndex();
    idx.add({ id: "a", vector: [1, 0] });
    idx.add({ id: "b", vector: [0, 1] });
    expect(idx.size()).toBe(2);
  });

  it("addAll() adds every item", () => {
    const idx = new VectorIndex();
    idx.addAll([
      { id: "a", vector: [1, 0] },
      { id: "b", vector: [0, 1] },
      { id: "c", vector: [1, 1] },
    ]);
    expect(idx.size()).toBe(3);
  });

  it("search() ranks by descending cosine similarity to the query", () => {
    const idx = new VectorIndex();
    idx.addAll([
      { id: "orthogonal", vector: [0, 1] },
      { id: "identical", vector: [1, 0] },
      { id: "opposite", vector: [-1, 0] },
    ]);
    const hits = idx.search([1, 0], 3);
    expect(hits.map((h) => h.id)).toEqual(["identical", "orthogonal", "opposite"]);
    expect(hits[0].score).toBeCloseTo(1, 10);
    expect(hits[2].score).toBeCloseTo(-1, 10);
  });

  it("search() respects k, returning at most k hits", () => {
    const idx = new VectorIndex();
    idx.addAll(Array.from({ length: 10 }, (_, i) => ({ id: `v${i}`, vector: [i, 1] })));
    expect(idx.search([0, 1], 3)).toHaveLength(3);
  });

  it("search() returns all items when k exceeds the index size", () => {
    const idx = new VectorIndex();
    idx.addAll([
      { id: "a", vector: [1, 0] },
      { id: "b", vector: [0, 1] },
    ]);
    expect(idx.search([1, 0], 50)).toHaveLength(2);
  });

  it("search() defaults k to 8", () => {
    const idx = new VectorIndex();
    idx.addAll(Array.from({ length: 20 }, (_, i) => ({ id: `v${i}`, vector: [i, 1] })));
    expect(idx.search([0, 1])).toHaveLength(8);
  });

  it("search() carries meta through to hits", () => {
    const idx = new VectorIndex<{ label: string }>();
    idx.add({ id: "a", vector: [1, 0], meta: { label: "hello" } });
    const hits = idx.search([1, 0], 1);
    expect(hits[0].meta).toEqual({ label: "hello" });
  });

  it("nearest() returns null for an empty index", () => {
    expect(new VectorIndex().nearest([1, 0])).toBeNull();
  });

  it("nearest() returns the single best match", () => {
    const idx = new VectorIndex();
    idx.addAll([
      { id: "far", vector: [0, 1] },
      { id: "close", vector: [1, 0.01] },
    ]);
    expect(idx.nearest([1, 0])?.id).toBe("close");
  });
});
