/**
 * Lightweight head-to-head benchmark runner for embedding adapters. Runs each
 * model N times against the same input, records latency, output dim, and (in
 * browsers exposing it) JS heap delta. Used by the inference dashboard and
 * the test suite to keep PCA and ONNX comparable.
 */
import { embed } from "../embeddings";
import type { ModelInput } from "../types";

export interface BenchmarkResult {
  modelId: string;
  iterations: number;
  latencyMsMean: number;
  latencyMsP50: number;
  latencyMsP95: number;
  embeddingDim: number;
  fellBack: boolean;
  fallbackReason?: string;
  heapDeltaBytes?: number;
  error?: string;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

function readHeap(): number | undefined {
  const perf = globalThis.performance as unknown as {
    memory?: { usedJSHeapSize: number };
  };
  return perf?.memory?.usedJSHeapSize;
}

export async function benchmarkAdapter(
  modelId: string,
  input: ModelInput,
  iterations = 5,
): Promise<BenchmarkResult> {
  const samples: number[] = [];
  let dim = 0;
  let fellBack = false;
  let fallbackReason: string | undefined;
  const heap0 = readHeap();
  try {
    for (let i = 0; i < iterations; i++) {
      const t0 = performance.now();
      const out = await embed(input, { modelId, fallbackToPCA: false });
      samples.push(performance.now() - t0);
      dim = out.dim;
      fellBack = out.fellBack;
      fallbackReason = out.fallbackReason;
    }
  } catch (err) {
    return {
      modelId,
      iterations: samples.length,
      latencyMsMean: 0,
      latencyMsP50: 0,
      latencyMsP95: 0,
      embeddingDim: dim,
      fellBack: false,
      error: (err as Error).message,
    };
  }
  const heap1 = readHeap();
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  return {
    modelId,
    iterations: samples.length,
    latencyMsMean: +mean.toFixed(3),
    latencyMsP50: +percentile(sorted, 50).toFixed(3),
    latencyMsP95: +percentile(sorted, 95).toFixed(3),
    embeddingDim: dim,
    fellBack,
    fallbackReason,
    heapDeltaBytes: heap0 != null && heap1 != null ? Math.max(0, heap1 - heap0) : undefined,
  };
}

export async function benchmarkAll(
  modelIds: string[],
  input: ModelInput,
  iterations = 5,
): Promise<BenchmarkResult[]> {
  const out: BenchmarkResult[] = [];
  for (const id of modelIds) out.push(await benchmarkAdapter(id, input, iterations));
  return out;
}
