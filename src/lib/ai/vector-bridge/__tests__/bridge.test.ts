import { describe, it, expect } from "vitest";
import { NeuralVectorIndex } from "../index";
import type { ModelInput } from "../../types";

function makeInput(freq: number): ModelInput {
  const T = 256;
  const sr = 128;
  const data: number[][] = [];
  for (let c = 0; c < 2; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * freq * t) / sr);
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: T }] };
}

describe("NeuralVectorIndex (PCA fallback path)", () => {
  it("upserts and searches end-to-end via the AI facade", async () => {
    const idx = new NeuralVectorIndex<{ label: string }>();
    await idx.upsert("a", makeInput(8), { label: "alpha" });
    await idx.upsert("b", makeInput(20), { label: "beta" });
    expect(idx.size()).toBe(2);
    const hits = await idx.search(makeInput(8), 2);
    expect(hits[0].id).toBe("a");
    expect(hits[0].score).toBeGreaterThan(hits[1].score);
    expect(hits[0].meta?.modelId).toBe("pca-legacy-v1");
    expect(hits[0].meta?.fellBack).toBe(false);
  });
});