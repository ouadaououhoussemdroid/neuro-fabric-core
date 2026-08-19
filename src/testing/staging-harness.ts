/**
 * Staging validation harness — extends the smoke-test harness with staging-specific
 * observability: high-resolution latency measurement, fallback-rate tracking,
 * and per-request diagnostics.
 *
 * Loaded by: staging-harness.html (served at /staging-harness.html by Vite dev server).
 * Used by: tests/browser/staging-latency.test.ts (Playwright real-browser tests).
 *
 * The harness exposes window.__stagingTest alongside window.__neuroTest.
 * All functions are the REAL production implementations — no stubs.
 */
import { embedEEG, type EmbedEEGOptions } from "@/lib/ai/inference/embed-eeg";
import { embed } from "@/lib/ai/embeddings";
import type { EmbedResult } from "@/lib/ai/embeddings";
import { setRolloutStage } from "@/lib/ai/rollout";
import { resetMetrics, metrics } from "@/lib/metrics";
import {
  __resetManifestCache,
  verifyRemoteArtifact,
  resolveVerification,
} from "@/lib/ai/artefacts/runtime-verifier";
import { hasModel, registerBraindecodeEEGConformer, unregisterModel } from "@/lib/ai/models/registry";
import { inferenceEngine } from "@/lib/ai/inference/engine";
import { renderPrometheusMetrics } from "@/lib/metrics";

/**
 * POST the browser-side metrics snapshot to the staging server's
 * /api/public/staging/metrics endpoint so it gets merged into the
 * server-side Prometheus registry, making browser WASM metrics visible
 * to the staging observation script.
 *
 * Accepts the structured snapshot from collectMetricsSnapshot() and
 * also sends the raw Prometheus text so the server can merge both.
 */
export async function reportMetricsToStagingServer(
  snapshotOrPrometheus: Record<string, any> | string,
  cronSecret?: string,
): Promise<{ status: string; timestamp: string } | null> {
  try {
    // Detect the staging server origin from the current page URL
    const origin = window.location.origin;

    // Normalize to { prometheus: text, metrics: structured }
    let prometheusText: string;
    let metricsObj: Record<string, any> | undefined;

    if (typeof snapshotOrPrometheus === "string") {
      // Already a Prometheus text string
      prometheusText = snapshotOrPrometheus;
    } else {
      // Structured snapshot from collectMetricsSnapshot()
      // Re-render Prometheus text from the in-memory registry
      prometheusText = renderPrometheusMetrics();
      metricsObj = snapshotOrPrometheus.metrics ?? snapshotOrPrometheus;
    }

    const response = await fetch(`${origin}/api/public/staging/metrics`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(cronSecret ? { Authorization: `Bearer ${cronSecret}` } : {}),
      },
      body: JSON.stringify({
        prometheus: prometheusText,
        ...(metricsObj ? { metrics: metricsObj } : {}),
      }),
    });
    if (!response.ok) {
      console.warn(`[staging-harness] metrics POST failed: ${response.status}`);
      return null;
    }
    return response.json();
  } catch (err) {
    console.warn(`[staging-harness] metrics POST error: ${(err as Error).message}`);
    return null;
  }
}

/**
 * High-resolution latency measurement around embedEEG().
 * Uses performance.mark() + performance.measure() for sub-millisecond precision.
 */
export async function measureEmbedLatency(
  input: unknown,
  opts?: EmbedEEGOptions,
  recordMetric: boolean = true,
): Promise<{
  result: EmbedResult;
  durationMs: number;
  markStart: string;
  markEnd: string;
}> {
  const markStart = `staging-embed-start-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const markEnd = `staging-embed-end-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  performance.mark(markStart);
  const result = await embedEEG(input as any, opts);
  performance.mark(markEnd);

  const measure = performance.measure("staging-embed-duration", {
    start: markStart,
    end: markEnd,
  });

  // Record latency in the Prometheus histogram so it flows through the
  // browser → POST → server → GET pipeline for staging validation.
  // Skip for warm-up iterations to avoid polluting P95 with cold-start costs.
  if (recordMetric) {
    metrics.uploadEmbedMs.observe({}, measure.duration);
  }

  return {
    result,
    durationMs: measure.duration,
    markStart,
    markEnd,
  };
}

export interface LatencySample {
  modelId: string;
  durationMs: number;
  fellBack: boolean;
  fallbackReason?: string;
  dim: number;
  timestamp: number;
}

const samples: LatencySample[] = [];

/** Number of warm-up iterations to discard (cold-start WASM compilation). */
const WARMUP_ITERATIONS = 3;

/**
 * Run N iterations of embedEEG() and collect latency + fallback statistics.
 * The first WARMUP_ITERATIONS are run but NOT included in the percentile
 * calculation — this excludes cold-start costs (WASM compilation, first
 * model load) from the latency gate, which mirrors production behaviour
 * where the model stays hot in the browser after initial page load.
 *
 * Results are stored in window.__stagingTest.samples for post-hoc analysis.
 */
export function runLatencyBenchmark(
  input: unknown,
  opts: EmbedEEGOptions,
  iterations: number,
): Promise<LatencySample[]> {
  return new Promise((resolve) => {
    samples.length = 0;
    (async () => {
      // Warm-up: run WARMUP_ITERATIONS but discard from results
      // Pass recordMetric=false to exclude cold-start WASM compilation from
      // the Prometheus latency histogram (mirrors production behavior where
      // the model stays hot after initial page load).
      for (let i = 0; i < WARMUP_ITERATIONS; i++) {
        await measureEmbedLatency(input, opts, false);
        await new Promise((r) => setTimeout(r, 10));
      }
      // Measured iterations
      for (let i = 0; i < iterations; i++) {
        const { result, durationMs } = await measureEmbedLatency(input, opts, true);
        samples.push({
          modelId: result.modelId,
          durationMs,
          fellBack: result.fellBack,
          fallbackReason: result.fallbackReason,
          dim: result.dim,
          timestamp: Date.now(),
        });
        await new Promise((r) => setTimeout(r, 10));
      }
      resolve([...samples]);
    })();
  });
}

/** Compute latency percentiles from collected samples. */
export function latencyPercentiles(samples: LatencySample[]): {
  n: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  fallbackCount: number;
  fallbackRate: number;
} {
  const sorted = [...samples].sort((a, b) => a.durationMs - b.durationMs);
  const durations = sorted.map((s) => s.durationMs);
  const n = durations.length;
  const percentile = (p: number) => {
    if (n === 0) return 0;
    const idx = Math.min(n - 1, Math.ceil((p / 100) * n) - 1);
    return sorted[idx].durationMs;
  };
  const fallbackCount = samples.filter((s) => s.fellBack).length;
  return {
    n,
    mean: durations.reduce((a, b) => a + b, 0) / n,
    p50: percentile(50),
    p95: percentile(95),
    p99: percentile(99),
    min: Math.min(...durations),
    max: Math.max(...durations),
    fallbackCount,
    fallbackRate: fallbackCount / n,
  };
}

/** Collect all current metrics as a JSON snapshot for staging monitoring. */
export function collectMetricsSnapshot(): Record<string, any> {
  const prometheus = renderPrometheusMetrics();
  const lines = prometheus.split("\n");
  const snapshot: Record<string, any> = {
    timestamp: new Date().toISOString(),
    metrics: {},
  };

  for (const line of lines) {
    if (line.startsWith("#") || !line.trim()) continue;

    // Match: metric_name{labels} value  OR  metric_name value
    const m = line.match(/^([\w:]+)_(\w+)(?:\{([^}]*)\})?\s+(\d+(?:\.\d+)?)$/);
    if (m) {
      // Reconstruct full metric name: prefix_counterName
      const name = `${m[1]}_${m[2]}`;
      const labels = m[3];
      const value = parseFloat(m[4]);
      if (!snapshot.metrics[name]) snapshot.metrics[name] = [];
      snapshot.metrics[name].push({ labels: labels ?? null, value });
    }
  }

  return snapshot;
}

/**
 * P2 instrumentation: set the ORT-Web WASM thread count.
 * This is test-code only — production embedEEG() keeps numThreads=1 (the ORT-Web
 * default), which P1/P2 measured as optimal for this 3.3 MB model (thread-pool
 * spin-up outweighs parallelism gains). Exposed here so the P2 ablation can
 * sweep numThreads=1 vs hardwareConcurrency and record the difference.
 */
export async function setOrtWasmThreads(n: number): Promise<void> {
  try {
    const mod = await import("onnxruntime-web");
    if (mod?.env?.wasm) {
      mod.env.wasm.numThreads = n;
    }
  } catch (err) {
    console.warn(`[staging-harness] setOrtWasmThreads: ${(err as Error).message}`);
  }
}

/**
 * P2 instrumentation: per-call embed facade (fresh session each call).
 * Mirrors the pre-P3 production path: createAdapter → load → embed → unload,
 * capturing the per-call session-create cost for A/B comparison against the
 * persistent InferenceEngine. Routes through the real `embed()` facade.
 */
export async function embedFacade(
  input: unknown,
  opts?: EmbedEEGOptions,
): Promise<EmbedResult> {
  return embed(input as any, {
    modelId: opts?.preferredModelId ?? "braindecode-eegconformer-prod-v2",
    fallbackChain: ["pca-legacy-v1"],
    fallbackToPCA: true,
    normalize: opts?.normalize !== false,
    expectedDim: opts?.expectedDim,
  });
}

function makeSyntheticInput(channels: number, samples: number, sampleRate: number): unknown {
  const data = Array.from({ length: channels }, (_, c) =>
    Array.from({ length: samples }, (_, t) => Math.sin((2 * Math.PI * (10 + c) * t) / sampleRate) * 0.5),
  );
  return {
    kind: "windows",
    windows: [{ data, sampleRate, start: 0, end: samples }],
  };
}

declare global {
  interface Window {
    __stagingTest: {
      measureEmbedLatency: typeof measureEmbedLatency;
      runLatencyBenchmark: typeof runLatencyBenchmark;
      latencyPercentiles: typeof latencyPercentiles;
      collectMetricsSnapshot: typeof collectMetricsSnapshot;
      reportMetricsToStagingServer: typeof reportMetricsToStagingServer;
      embedEEG: typeof embedEEG;
      setRolloutStage: typeof setRolloutStage;
      resetMetrics: typeof resetMetrics;
      __resetManifestCache: typeof __resetManifestCache;
      verifyRemoteArtifact: typeof verifyRemoteArtifact;
      resolveVerification: typeof resolveVerification;
      hasModel: typeof hasModel;
      registerEEGConformer: typeof registerBraindecodeEEGConformer;
      unregisterModel: typeof unregisterModel;
      metrics: typeof metrics;
      makeSyntheticInput: typeof makeSyntheticInput;
      samples: LatencySample[];
      WARMUP_ITERATIONS: number;
      /** P2/P3 instrumentation: ORT-Web thread count (test-only). */
      setOrtWasmThreads: typeof setOrtWasmThreads;
      /** P2 instrumentation: per-call embed facade for baseline measurement. */
      embedFacade: typeof embedFacade;
      /** P2/P3 instrumentation: cached InferenceEngine (production singleton). */
      inferenceEngine: typeof inferenceEngine;
    };
  }
}

if (typeof window !== "undefined") {
  (window as any).__stagingTest = {
    measureEmbedLatency,
    runLatencyBenchmark,
    latencyPercentiles,
    collectMetricsSnapshot,
    reportMetricsToStagingServer,
    embedEEG,
    setRolloutStage,
    resetMetrics,
    __resetManifestCache,
    verifyRemoteArtifact,
    resolveVerification,
    hasModel,
    registerEEGConformer: registerBraindecodeEEGConformer,
    unregisterModel,
    metrics,
    makeSyntheticInput,
    samples,
    WARMUP_ITERATIONS,
    setOrtWasmThreads,
    embedFacade,
    inferenceEngine,
  };
}
