import { describe, it, expect } from "vitest";
import { benchmarkAdapter } from "../index";
import type { ModelInput } from "../../types";

function makeInput(): ModelInput {
  const T = 256;
  const sr = 128;
  const data: number[][] = [];
  for (let c = 0; c < 2; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) ch[t] = Math.sin((2 * Math.PI * 10 * t) / sr);
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: T }] };
}

describe("benchmarkAdapter", () => {
  it("reports latency and dim for PCA", async () => {
    const r = await benchmarkAdapter("pca-legacy-v1", makeInput(), 3);
    expect(r.iterations).toBe(3);
    expect(r.embeddingDim).toBeGreaterThan(0);
    expect(r.latencyMsMean).toBeGreaterThanOrEqual(0);
    expect(r.fellBack).toBe(false);
    expect(r.error).toBeUndefined();
  });

  it("captures error when fallback is disabled and model id is unknown", async () => {
    const r = await benchmarkAdapter("nope", makeInput(), 2);
    expect(r.error).toMatch(/Unknown model id/);
  });
});