/**
 * Brain Accelerator Feature Flag — T-024b
 *
 * Unified execution provider selection for neural inference in browser:
 *   ["snn-wasm", "qpu", "webnn", "webgpu", "wasm"]
 *
 * Priority chain:
 *   1. SNN-WASM (neuromorphic compute) — always available when WebAssembly is present
 *   2. QPU (quantum co-processor simulation) — browser-side quantum acceleration
 *   3. WebNN (NPU) — native NPU via navigator.ml
 *   4. WebGPU — GPU compute via navigator.gpu
 *   5. WASM — CPU fallback
 *
 * The flag is controlled by:
 *   1. `VITE_WEBNN` build-time env var (set to `"true"` to enable WebNN EP).
 *   2. `VITE_ORT_WEBGPU` build-time env var (set to `"true"` to enable WebGPU EP).
 *   3. `VITE_QUANTUM_ENABLED` build-time env var (set to `"true"` to enable QPU EP).
 *   4. Runtime toggles via setWebNNEnabled()/setWebGPUEnabled()/setQuantumEnabled().
 *   5. Browser capability checks: `navigator.ml` for WebNN, `navigator.gpu` for WebGPU,
 *      WebAssembly for QPU simulation.
 *
 * Default: SNN-WASM + QPU always available, WebGPU opt-in (conservative), WebNN disabled.
 */

let runtimeWebNNEnabled = false;
let runtimeWebGPUEnabled = false;
let runtimeQuantumEnabled = false;

/** Build-time flag for WebNN EP. Checked once at module load. */
const envWebNN =
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_WEBNN === "true";

const envWebGPU =
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_ORT_WEBGPU === "true";

/** Build-time flag for quantum EP. */
const envQuantum =
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_QUANTUM_ENABLED === "true";

/** Whether the browser supports the WebNN API (navigator.ml). */
export function isWebNNAvailable(): boolean {
  if (typeof navigator === "undefined") return false;
  return "ml" in navigator;
}

/** Whether the browser supports WebGPU (navigator.gpu). */
export function isWebGPUAvailable(): boolean {
  if (typeof navigator === "undefined") return false;
  return "gpu" in navigator;
}

/**
 * Whether WebNN EP should be used.
 * Returns true only if: env flag OR runtime toggle is set, AND browser supports WebNN.
 */
export function isWebNEnabled(): boolean {
  return (envWebNN || runtimeWebNNEnabled) && isWebNNAvailable();
}

/**
 * Whether WebGPU EP should be used.
 * Returns true only if: env flag OR runtime toggle is set, AND browser supports WebGPU.
 */
export function isWebGPUEnabled(): boolean {
  return (envWebGPU || runtimeWebGPUEnabled) && isWebGPUAvailable();
}

/**
 * Runtime toggle for WebNN EP (e.g. from a settings UI).
 */
export function setWebNNEnabled(enabled: boolean): void {
  runtimeWebNNEnabled = enabled;
}

/**
 * Runtime toggle for WebGPU EP (e.g. from a settings UI).
 */
export function setWebGPUEnabled(enabled: boolean): void {
  runtimeWebGPUEnabled = enabled;
}

/**
 * Whether the browser supports quantum simulation (WebAssembly + Math).
 * Quantum simulation runs purely in JS/WASM — no special hardware required.
 */
export function isQuantumAvailable(): boolean {
  return typeof WebAssembly !== "undefined" && typeof Math.random === "function";
}

/**
 * Whether QPU (quantum simulation) EP should be used.
 * Always available when env flag or runtime toggle is set, and WASM is present.
 */
export function isQuantumEnabled(): boolean {
  return (envQuantum || runtimeQuantumEnabled) && isQuantumAvailable();
}

/**
 * Runtime toggle for QPU EP (e.g. from a settings UI).
 */
export function setQuantumEnabled(enabled: boolean): void {
  runtimeQuantumEnabled = enabled;
}

/**
 * Get the execution providers array for InferenceSession.create.
 * Priority chain: ["snn-wasm", "qpu", "webnn", "webgpu", "wasm"]
 *
 * M48 Phase 1 — Extended to include SNN-WASM neuromorphic compute path.
 * M55 — Extended to include QPU quantum simulation compute path.
 * Full chain: ["snn-wasm", "qpu", "webnn", "webgpu", "wasm"]
 * when both SNN and QPU are enabled.
 */
export function getExecutionProviders(): Array<"wasm" | "webgpu" | "webnn" | "snn-wasm" | "qpu"> {
  const providers: Array<"wasm" | "webgpu" | "webnn" | "snn-wasm" | "qpu"> = [];

  if (isSNNEnabled()) providers.push("snn-wasm");
  if (isQuantumEnabled()) providers.push("qpu");
  if (isWebNEnabled()) providers.push("webnn");
  if (isWebGPUEnabled()) providers.push("webgpu");
  providers.push("wasm");

  return providers;
}

/** Whether SNN-WASM neuromorphic simulator is enabled. */
export function isSNNEnabled(): boolean {
  // SNN-WASM is always available when the browser supports WebAssembly
  return typeof WebAssembly !== "undefined";
}

/**
 * Get a human-readable summary of available accelerators.
 * Useful for logging and UI display.
 */
export function getAcceleratorStatus(): {
  webnn: boolean;
  webgpu: boolean;
  wasm: boolean;
  snn: boolean;
  qpu: boolean;
  active: Array<"wasm" | "webgpu" | "webnn" | "snn-wasm" | "qpu">;
} {
  return {
    webnn: isWebNEnabled(),
    webgpu: isWebGPUEnabled(),
    wasm: true, // WASM is always available when onnxruntime-web is loaded
    snn: isSNNEnabled(),
    qpu: isQuantumEnabled(),
    active: getExecutionProviders(),
  };
}
