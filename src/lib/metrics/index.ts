/**
 * T-008 — Lightweight Prometheus-format metrics collection.
 *
 * Provides Counter, Gauge, and Histogram primitives that can be read via the
 * `/api/public/metrics` endpoint. All state is in-process (single isolate),
 * suitable for dev/staging. In production edge runtimes, counters are
 * per-isolate and should be aggregated via a metrics scraper or pushed to a
 * central TSDB; this implementation prioritises visibility over perfect
 * accounting.
 *
 * Histogram buckets are exponential (base 2) to cover 1ms → 64s latency ranges.
 */

export interface Metric {
  name: string;
  type: "counter" | "gauge" | "histogram";
  help: string;
  values: { labels: Record<string, string>; value: number }[];
}

interface HistogramBucket {
  le: number;
  count: number;
}

interface HistogramValue {
  labels: Record<string, string>;
  sum: number;
  count: number;
  buckets: HistogramBucket[];
}

const HISTOGRAM_BUCKETS = [
  1, 2, 5, 10, 25, 50, 100, 250, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 750, 800, 850, 900, 950, 1000, 2500, 5000, 10000, 30000, 60000, 120000,
];

interface Registry {
  counters: Map<string, Map<string, number>>; // name → labelKey → value
  gauges: Map<string, Map<string, number>>;
  histograms: Map<string, Map<string, HistogramValue>>; // name → labelKey → {sum, count, buckets}
  help: Map<string, string>;
  types: Map<string, "counter" | "gauge" | "histogram">;
}

const registry: Registry = {
  counters: new Map(),
  gauges: new Map(),
  histograms: new Map(),
  help: new Map(),
  types: new Map(),
};

/** Serialize label set to a stable string key for lookup. */
function labelKey(labels: Record<string, string>): string {
  return Object.entries(labels)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}="${v}"`)
    .join(",");
}

/** Counter metric — monotonically increasing. */
export class Counter {
  private name: string;
  constructor(name: string, help: string) {
    this.name = name;
    if (!registry.types.has(name)) {
      registry.types.set(name, "counter");
      registry.help.set(name, help);
      registry.counters.set(name, new Map());
    }
  }
  inc(labels: Record<string, string> = {}, amount = 1): void {
    const store = registry.counters.get(this.name)!;
    const key = labelKey(labels);
    store.set(key, (store.get(key) ?? 0) + amount);
    // Also increment the unlabeled version (only if this call has labels).
    if (Object.keys(labels).length > 0) {
      const baseKey = labelKey({});
      store.set(baseKey, (store.get(baseKey) ?? 0) + amount);
    }
  }
  value(labels: Record<string, string> = {}): number {
    const store = registry.counters.get(this.name)!;
    return store.get(labelKey(labels)) ?? 0;
  }
}

/** Gauge metric — can go up and down. */
export class Gauge {
  private name: string;
  constructor(name: string, help: string) {
    this.name = name;
    if (!registry.types.has(name)) {
      registry.types.set(name, "gauge");
      registry.help.set(name, help);
      registry.gauges.set(name, new Map());
    }
  }
  set(labels: Record<string, string> = {}, value: number): void {
    const store = registry.gauges.get(this.name)!;
    store.set(labelKey(labels), value);
  }
  inc(labels: Record<string, string> = {}, amount = 1): void {
    const store = registry.gauges.get(this.name)!;
    const key = labelKey(labels);
    store.set(key, (store.get(key) ?? 0) + amount);
  }
  dec(labels: Record<string, string> = {}, amount = 1): void {
    const store = registry.gauges.get(this.name)!;
    const key = labelKey(labels);
    store.set(key, (store.get(key) ?? 0) - amount);
  }
  value(labels: Record<string, string> = {}): number {
    const store = registry.gauges.get(this.name)!;
    return store.get(labelKey(labels)) ?? 0;
  }
}

/** Histogram metric — tracks distributions (e.g., latencies). */
export class Histogram {
  private name: string;
  constructor(name: string, help: string) {
    this.name = name;
    if (!registry.types.has(name)) {
      registry.types.set(name, "histogram");
      registry.help.set(name, help);
      registry.histograms.set(name, new Map());
    }
  }
  observe(labels: Record<string, string> = {}, valueMs: number): void {
    const store = registry.histograms.get(this.name)!;
    const key = labelKey(labels);
    let entry = store.get(key);
    if (!entry) {
      entry = {
        labels,
        sum: 0,
        count: 0,
        buckets: HISTOGRAM_BUCKETS.map((le) => ({ le, count: 0 })),
      };
      store.set(key, entry);
    }
    entry.sum += valueMs;
    entry.count += 1;
    for (const b of entry.buckets) {
      if (valueMs <= b.le) b.count += 1;
    }
    // Also update the unlabeled (base) entry — but only when this call has
    // specific labels, to avoid double-counting the unlabeled aggregate.
    if (Object.keys(labels).length > 0) {
      const baseKey = labelKey({});
      let baseEntry = store.get(baseKey);
      if (!baseEntry) {
        baseEntry = {
          labels: {},
          sum: 0,
          count: 0,
          buckets: HISTOGRAM_BUCKETS.map((le) => ({ le, count: 0 })),
        };
        store.set(baseKey, baseEntry);
      }
      baseEntry.sum += valueMs;
      baseEntry.count += 1;
      for (const b of baseEntry.buckets) {
        if (valueMs <= b.le) b.count += 1;
      }
    }
  }

  /**
   * Merge pre-aggregated histogram data (e.g., from Prometheus text format
   * received via POST from a browser-side registry). Replaces the existing
   * values for the given labels rather than accumulating.
   *
   * `bucounts` maps bucket upper-bound (le) → cumulative count.
   */
  setAggregated(
    labels: Record<string, string> = {},
    data: { sum: number; count: number; bucketCounts: Map<number, number> },
  ): void {
    const store = registry.histograms.get(this.name)!;
    const key = labelKey(labels);
    let entry = store.get(key);
    if (!entry) {
      entry = {
        labels,
        sum: 0,
        count: 0,
        buckets: HISTOGRAM_BUCKETS.map((le) => ({ le, count: 0 })),
      };
      store.set(key, entry);
    }
    entry.sum = data.sum;
    entry.count = data.count;
    for (const b of entry.buckets) {
      const remote = data.bucketCounts.get(b.le);
      if (remote !== undefined) b.count = remote;
    }
  }
}

/** Format all collected metrics in Prometheus exposition format. */
export function renderPrometheusMetrics(): string {
  const lines: string[] = [];

  for (const [name, type] of registry.types) {
    const help = registry.help.get(name) ?? "";
    lines.push(`# HELP ${name} ${help}`);
    lines.push(`# TYPE ${name} ${type}`);

    if (type === "counter") {
      const store = registry.counters.get(name)!;
      for (const [labelKey, value] of store) {
        const labels = parseLabelKey(labelKey);
        const labelStr = formatLabels(labels);
        lines.push(`${name}${labelStr} ${value}`);
      }
    } else if (type === "gauge") {
      const store = registry.gauges.get(name)!;
      for (const [labelKey, value] of store) {
        const labels = parseLabelKey(labelKey);
        const labelStr = formatLabels(labels);
        lines.push(`${name}${labelStr} ${value}`);
      }
    } else if (type === "histogram") {
      const store = registry.histograms.get(name)!;
      for (const [labelKey, value] of store) {
        const labels = parseLabelKey(labelKey);
        const baseLabelStr = formatLabels(labels);
        for (const b of value.buckets) {
          const bucketLabels = { ...labels, le: String(b.le) };
          lines.push(`${name}_bucket${formatLabels(bucketLabels)} ${b.count}`);
        }
        lines.push(`${name}_sum${baseLabelStr} ${value.sum}`);
        lines.push(`${name}_count${baseLabelStr} ${value.count}`);
      }
    }
  }

  return lines.join("\n") + "\n";
}

function parseLabelKey(key: string): Record<string, string> {
  if (!key) return {};
  const labels: Record<string, string> = {};
  // Split on comma, but handle quoted values properly
  const parts = key.split(/,(?=(?:[^"]|$))/);
  for (const part of parts) {
    const m = part.match(/^([^=]+)="(.*)"$/);
    if (m) labels[m[1]] = m[2];
  }
  return labels;
}

function formatLabels(labels: Record<string, string>): string {
  const entries = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return "";
  return "{" + entries.map(([k, v]) => `${k}="${v}"`).join(",") + "}";
}

// ---------------------------------------------------------------------------
// Application-level metric definitions (singletons).
// ---------------------------------------------------------------------------

export const metrics = {
  // Upload pipeline
  uploadRequestsTotal: new Counter("neuro_fabric_upload_requests_total", "Total upload requests"),
  uploadBytesTotal: new Counter("neuro_fabric_upload_bytes_total", "Total bytes uploaded"),
  uploadErrorsTotal: new Counter("neuro_fabric_upload_errors_total", "Failed upload requests"),
  uploadParseMs: new Histogram(
    "neuro_fabric_upload_parse_ms",
    "Time spent parsing uploaded EEG files",
  ),
  uploadPreprocessMs: new Histogram(
    "neuro_fabric_upload_preprocess_ms",
    "Time spent preprocessing EEG signals",
  ),
  uploadEmbedMs: new Histogram("neuro_fabric_upload_embed_ms", "Time spent generating embeddings"),
  uploadDecodeMs: new Histogram(
    "neuro_fabric_upload_decode_ms",
    "Time spent running cognitive decoder",
  ),
  uploadTotalMs: new Histogram("neuro_fabric_upload_total_ms", "Total upload processing time"),

  // Rate limiting
  rateLimitedTotal: new Counter("neuro_fabric_rate_limited_total", "Total rate-limited requests"),

  // AI / Embedding
  embedFallbackTotal: new Counter(
    "neuro_fabric_embed_fallback_total",
    "Total embedding fallback events (ONNX→PCA)",
  ),
  embedLatencyMs: new Histogram("neuro_fabric_embed_latency_ms", "Embedding generation latency"),

  // T-016 — EEGConformer rollout observability (canary metrics)
  cohortChecksTotal: new Counter(
    "neuro_fabric_eegconformer_cohort_checks_total",
    "EEGConformer cohort eligibility checks",
  ),
  modelSelectedTotal: new Counter(
    "neuro_fabric_model_selected_total",
    "Total model selection events by model id",
  ),

  // T-016 — Runtime artifact SHA-256 verification
  artifactVerificationTotal: new Counter(
    "neuro_fabric_artifact_verification_total",
    "Runtime artifact SHA-256 verification outcomes",
  ),
  artifactVerifyMs: new Histogram(
    "neuro_fabric_artifact_verify_ms",
    "Time spent on artifact SHA-256 verification (fetch + hash)",
  ),

  // Vector search
  vectorSearchTotal: new Counter(
    "neuro_fabric_vector_search_total",
    "Total vector search operations",
  ),
  vectorSearchLatencyMs: new Histogram(
    "neuro_fabric_vector_search_latency_ms",
    "Vector search latency",
  ),
  vectorStoreErrorsTotal: new Counter(
    "neuro_fabric_vector_store_errors_total",
    "Total vector store (insert/search) errors",
  ),

  // Cross-subject evaluation (LOSO)
  evaluationRequestsTotal: new Counter(
    "neuro_fabric_evaluation_requests_total",
    "Total cross-subject evaluation requests",
  ),
  evaluationErrorsTotal: new Counter(
    "neuro_fabric_evaluation_errors_total",
    "Failed cross-subject evaluation requests",
  ),
  evaluationLatencyMs: new Histogram(
    "neuro_fabric_evaluation_latency_ms",
    "Cross-subject evaluation latency",
  ),

  // System
  inFlightUploads: new Gauge(
    "neuro_fabric_inflight_uploads",
    "Currently in-flight upload requests",
  ),
};

// T-008: export a function to reset all metrics (for testing).
//
// Clears the *value* stores (per-label counters, gauge readings, histogram
// observations) but preserves the schema registrations (types, help text, and
// the store Map objects themselves). This keeps existing Counter / Gauge /
// Histogram singleton instances — including the `metrics` object — fully
// functional after reset, since they look up their store by name at call time.
export function resetMetrics(): void {
  for (const store of registry.counters.values()) store.clear();
  for (const store of registry.gauges.values()) store.clear();
  for (const store of registry.histograms.values()) store.clear();
}
