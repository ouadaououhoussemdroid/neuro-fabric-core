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
  1, 2, 5, 10, 25, 50, 100, 250, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700,
  750, 800, 850, 900, 950, 1000, 2500, 5000, 10000, 30000, 60000, 120000,
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

  // T-036 — Tier-2 CBraMod foundation embed path (server-native, opt-in).
  foundationRequestsTotal: new Counter(
    "neuro_fabric_foundation_requests_total",
    "Total CBraMod foundation embed requests",
  ),
  foundationErrorsTotal: new Counter(
    "neuro_fabric_foundation_errors_total",
    "Failed CBraMod foundation requests (runtime/artifact/parse)",
  ),
  foundationBytesTotal: new Counter(
    "neuro_fabric_foundation_bytes_total",
    "Total EEG bytes submitted to the CBraMod foundation endpoint",
  ),
  foundationEmbedMs: new Histogram(
    "neuro_fabric_foundation_embed_ms",
    "CBraMod foundation embedding latency per window",
  ),

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

  // T-025 — GPU preprocessing observability
  gpuPreprocessingMs: new Histogram(
    "neuro_fabric_gpu_preprocessing_ms",
    "GPU preprocessing latency (bandpass, FFT, band-power)",
  ),
  gpuFallbackTotal: new Counter(
    "neuro_fabric_gpu_fallback_total",
    "Total fallback events from GPU to CPU preprocessing",
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

  // ─── M32 — Tier-1 downstream service layer metrics ──────────────────────

  // Subject Identity & Cohort Similarity
  subjectIdentityRequestsTotal: new Counter(
    "neuro_fabric_subject_identity_requests_total",
    "Total subject-identity similarity search requests",
  ),
  subjectIdentityErrorsTotal: new Counter(
    "neuro_fabric_subject_identity_errors_total",
    "Failed subject-identity similarity search requests",
  ),
  subjectIdentitySearchLatencyMs: new Histogram(
    "neuro_fabric_subject_identity_search_latency_ms",
    "Subject-identity search latency (ANN RPC + result formatting)",
  ),
  subjectIdentityResultsTotal: new Counter(
    "neuro_fabric_subject_identity_results_total",
    "Total similarity results returned across all subject-identity searches",
  ),
  subjectIdentityEmbeddingReusedTotal: new Counter(
    "neuro_fabric_subject_identity_embedding_reused_total",
    "Subject-identity searches that reused an existing Joint-2312 embedding (no re-embed)",
  ),
  subjectIdentityEmbeddingReembeddedTotal: new Counter(
    "neuro_fabric_subject_identity_embedding_reembedded_total",
    "Subject-identity searches that re-computed Joint-2312 (no existing embedding)",
  ),

  // Shared Tier-1 service layer
  tier1ServiceRequestsTotal: new Counter(
    "neuro_fabric_tier1_service_requests_total",
    "Total Tier-1 downstream service requests (all services)",
  ),
  tier1ServiceErrorsTotal: new Counter(
    "neuro_fabric_tier1_service_errors_total",
    "Total failed Tier-1 downstream service requests",
  ),
  tier1ServiceLatencyMs: new Histogram(
    "neuro_fabric_tier1_service_latency_ms",
    "Total Tier-1 service request processing latency",
  ),
  tier1AuditLogInsertsTotal: new Counter(
    "neuro_fabric_tier1_audit_log_inserts_total",
    "Total audit-log rows inserted by Tier-1 services",
  ),

  // ─── M33 — Cognitive State Intelligence service metrics ────────────────────

  cognitiveDecodeRequestsTotal: new Counter(
    "neuro_fabric_cognitive_decode_requests_total",
    "Total cognitive decode requests (workload/attention/arousal)",
  ),
  cognitiveDecodeErrorsTotal: new Counter(
    "neuro_fabric_cognitive_decode_errors_total",
    "Failed cognitive decode requests",
  ),
  cognitiveDecodeLatencyMs: new Histogram(
    "neuro_fabric_cognitive_decode_latency_ms",
    "Cognitive decode inference latency (ONNX probe forward pass)",
  ),
  cognitiveWorkloadPredictionsTotal: new Counter(
    "neuro_fabric_cognitive_workload_predictions_total",
    "Total workload predictions returned across all decode calls",
  ),
  cognitiveConfidenceDistribution: new Histogram(
    "neuro_fabric_cognitive_confidence_distribution",
    "Confidence score distribution for cognitive predictions",
  ),
  cognitiveEmbeddingReusedTotal: new Counter(
    "neuro_fabric_cognitive_embedding_reused_total",
    "Cognitive decode calls that reused an existing Joint-2312 embedding",
  ),
  cognitiveEmbeddingReembeddedTotal: new Counter(
    "neuro_fabric_cognitive_embedding_reembedded_total",
    "Cognitive decode calls that re-computed Joint-2312 (no existing embedding)",
  ),

  // ─── M34 — Anomaly Detection service metrics ──────────────────────────────

  anomalyDetectRequestsTotal: new Counter(
    "neuro_fabric_anomaly_detect_requests_total",
    "Total anomaly detection requests",
  ),
  anomalyDetectErrorsTotal: new Counter(
    "neuro_fabric_anomaly_detect_errors_total",
    "Failed anomaly detection requests",
  ),
  anomalyDetectLatencyMs: new Histogram(
    "neuro_fabric_anomaly_detect_latency_ms",
    "Anomaly detection inference latency (Mahalanobis distance computation)",
  ),
  anomalyScoresTotal: new Counter(
    "neuro_fabric_anomaly_scores_total",
    "Total anomaly scores returned across all detect calls",
  ),
  anomalyConfidenceDistribution: new Histogram(
    "neuro_fabric_anomaly_confidence_distribution",
    "Confidence score distribution for anomaly predictions",
  ),
  anomalyEmbeddingReusedTotal: new Counter(
    "neuro_fabric_anomaly_embedding_reused_total",
    "Anomaly detect calls that reused an existing Joint-2312 embedding",
  ),
  anomalyEmbeddingReembeddedTotal: new Counter(
    "neuro_fabric_anomaly_embedding_reembedded_total",
    "Anomaly detect calls that re-computed Joint-2312 (no existing embedding)",
  ),
  // ─── M39 — Sleep Staging service metrics ──────────────────────────────────────

  sleepDecodeRequestsTotal: new Counter(
    "neuro_fabric_sleep_decode_requests_total",
    "Total sleep staging decode requests",
  ),
  sleepDecodeErrorsTotal: new Counter(
    "neuro_fabric_sleep_decode_errors_total",
    "Failed sleep staging decode requests",
  ),
  sleepDecodeLatencyMs: new Histogram(
    "neuro_fabric_sleep_decode_latency_ms",
    "Sleep staging decode inference latency (ONNX probe forward pass)",
  ),
  sleepStagePredictionsTotal: new Counter(
    "neuro_fabric_sleep_stage_predictions_total",
    "Total sleep stage predictions returned across all decode calls",
  ),
  sleepConfidenceDistribution: new Histogram(
    "neuro_fabric_sleep_confidence_distribution",
    "Confidence score distribution for sleep predictions",
  ),
  sleepEmbeddingReusedTotal: new Counter(
    "neuro_fabric_sleep_embedding_reused_total",
    "Sleep decode calls that reused an existing Joint-2312 embedding",
  ),
  sleepEmbeddingReembeddedTotal: new Counter(
    "neuro_fabric_sleep_embedding_reembedded_total",
    "Sleep decode calls that re-computed Joint-2312 (no existing embedding)",
  ),

  // M48 — Predictive Neural Coding Engine metrics
  predictiveCodingRequestsTotal: new Counter(
    "neuro_fabric_predictive_coding_requests_total",
    "Total predictive coding inference requests (M48)",
  ),
  predictiveCodingErrorsTotal: new Counter(
    "neuro_fabric_predictive_coding_errors_total",
    "Failed predictive coding inference requests",
  ),
  predictiveCodingLatencyMs: new Histogram(
    "neuro_fabric_predictive_coding_latency_ms",
    "Predictive coding inference latency (prediction + surprise scoring)",
  ),
  predictiveCodingForecastHorizonTotal: new Counter(
    "neuro_fabric_predictive_coding_forecast_horizon_total",
    "Total prediction horizon steps used across all predictive coding calls",
    ["horizon"],
  ),
  predictiveCodingSurpriseScore: new Histogram(
    "neuro_fabric_predictive_coding_surprise_score",
    "Prediction error (surprise) score distribution per channel",
    ["band"],
  ),

  // M49 — Federated Brain Learning metrics
  federatedRoundRequestsTotal: new Counter(
    "neuro_fabric_federated_round_requests_total",
    "Total federated learning round requests (M49)",
  ),
  federatedRoundErrorsTotal: new Counter(
    "neuro_fabric_federated_round_errors_total",
    "Failed federated learning round coordination",
  ),
  federatedRoundLatencyMs: new Histogram(
    "neuro_fabric_federated_round_latency_ms",
    "Federated round coordination latency (server-side aggregation)",
  ),
  federatedClientsParticipatedTotal: new Counter(
    "neuro_fabric_federated_clients_participated_total",
    "Active client participation count per federated round",
    ["client_id"],
  ),
  federatedClientUpdatesTotal: new Counter(
    "neuro_fabric_federated_client_updates_total",
    "Received client model updates (delta weights)",
    ["client_id"],
  ),
  federatedAggregationConvergence: new Histogram(
    "neuro_fabric_federated_aggregation_convergence",
    "Global model convergence metric (weight delta L2 norm) per round",
  ),

  // M53 — Cross-Modal Neural Synchrony metrics
  multimodalRequestsTotal: new Counter(
    "neuro_fabric_multimodal_requests_total",
    "Total cross-modal fusion requests (M53)",
  ),
  multimodalModalityProcessedTotal: new Counter(
    "neuro_fabric_multimodal_modality_processed_total",
    "Per-modality embedding computations",
    ["modality"],
  ),
  multimodalFusionLatencyMs: new Histogram(
    "neuro_fabric_multimodal_fusion_latency_ms",
    "Cross-modal fusion latency (feature extraction + attention + synchrony)",
  ),
  multimodalSynchronyScore: new Histogram(
    "neuro_fabric_multimodal_synchrony_score",
    "Global synchrony score distribution (mean pairwise correlation)",
  ),

  // M54 — Adaptive Neurostimulation metrics
  neuroStimulationRequestsTotal: new Counter(
    "neuro_fabric_neurostimulation_requests_total",
    "Total neurostimulation session requests (M54)",
  ),
  neuroStimulationErrorsTotal: new Counter(
    "neuro_fabric_neurostimulation_errors_total",
    "Failed neurostimulation sessions (safety interlock, device error)",
  ),
  neuroStimulationLatencyMs: new Histogram(
    "neuro_fabric_neurostimulation_latency_ms",
    "Neurostimulation session setup + device command latency",
  ),
  neuroStimulationArtifactsTotal: new Counter(
    "neuro_fabric_neurostimulation_artifacts_total",
    "Total artifact detections during stimulation sessions",
    ["type"],
  ),
  neuroStimulationImpedanceGauge: new Gauge(
    "neuro_fabric_neurostimulation_impedance_kohm",
    "Most recent impedance reading (kΩ)",
  ),
  neuroStimulationSessionsActive: new Gauge(
    "neuro_fabric_neurostimulation_active",
    "Currently active neurostimulation sessions",
  ),

  // M55 — Quantum-Neuromorphic Computing metrics
  quantumCircuitExecutionsTotal: new Counter(
    "neuro_fabric_quantum_circuit_executions_total",
    "Total quantum circuit executions (M55)",
  ),
  quantumCircuitErrorsTotal: new Counter(
    "neuro_fabric_quantum_circuit_errors_total",
    "Failed quantum circuit executions",
  ),
  quantumCircuitLatencyMs: new Histogram(
    "neuro_fabric_quantum_circuit_latency_ms",
    "Quantum circuit execution latency (state prep + gates + sampling)",
  ),
  quantumVQEOptimizationLatencyMs: new Histogram(
    "neuro_fabric_quantum_vqe_optimization_latency_ms",
    "VQE parameter optimization latency (gradient-free search)",
  ),
  quantumStateDimensionGauge: new Gauge(
    "neuro_fabric_quantum_state_dimension",
    "Active quantum state vector dimension (2^numQubits)",
  ),
  quantumQubitsGauge: new Gauge(
    "neuro_fabric_quantum_qubits",
    "Number of qubits in the simulated quantum processor",
  ),
  quantumInferenceFidelityScore: new Histogram(
    "neuro_fabric_quantum_inference_fidelity",
    "Fidelity of quantum-neuromorphic hybrid inference output",
  ),

  // M37 — Alert threshold gauges (for production readiness monitoring)
  tier1AlertThresholds: {
    p95LatencyMs: new Gauge(
      "neuro_fabric_tier1_alert_p95_latency_ms_threshold",
      "Alert threshold: Tier-1 P95 latency in ms (default 2000)",
    ),
    p50LatencyMs: new Gauge(
      "neuro_fabric_tier1_alert_p50_latency_ms_threshold",
      "Alert threshold: Tier-1 P50 latency in ms (default 400)",
    ),
    errorRate: new Gauge(
      "neuro_fabric_tier1_alert_error_rate_threshold",
      "Alert threshold: Tier-1 error rate as fraction (default 0.05 = 5%)",
    ),
    fallbackRate: new Gauge(
      "neuro_fabric_tier1_alert_fallback_rate_threshold",
      "Alert threshold: Tier-1 fallback rate as fraction (default 0.005 = 0.5%)",
    ),
  },
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
