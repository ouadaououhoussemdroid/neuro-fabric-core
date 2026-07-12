/**
 * T-024 — WebGPU execution provider feature flag.
 *
 * Opt-in WebGPU EP for `onnxruntime-web`. When enabled, the ONNX adapter
 * will request `["webgpu", "wasm"]` execution providers — ORT automatically
 * falls back to WASM if WebGPU is unavailable in the browser.
 *
 * The flag is controlled by:
 *   1. `VITE_ORT_WEBGPU` build-time env var (set to `"true"` to enable).
 *   2. Runtime toggle via `setWebGPUEnabled(true)` (e.g. from a settings UI).
 *   3. Browser capability check: `navigator.gpu` must exist.
 *
 * Default: **disabled** (CPU/WASM fallback unchanged). This is intentionally
 * conservative — WebGPU EP is still maturing and enabling it by default
 * could cause silent failures on unsupported browsers.
 */

let runtimeEnabled = false;

/**
 * Whether WebGPU EP is enabled by the build-time env flag.
 * Checked once at module load.
 */
const envEnabled =
  typeof import.meta !== "undefined" &&
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_ORT_WEBGPU === "true";

/**
 * Whether the current browser supports WebGPU.
 */
export function isWebGPUAvailable(): boolean {
  if (typeof navigator === "undefined") return false;
  return "gpu" in navigator;
}

/**
 * Whether WebGPU EP should be used.
 * Returns true only if: env flag OR runtime toggle is set, AND the browser
 * supports WebGPU.
 */
export function isWebGPUEnabled(): boolean {
  return (envEnabled || runtimeEnabled) && isWebGPUAvailable();
}

/**
 * Runtime toggle for WebGPU EP (e.g. from a settings UI).
 * Overrides the build-time env flag.
 */
export function setWebGPUEnabled(enabled: boolean): void {
  runtimeEnabled = enabled;
}

/**
 * Get the execution providers array for `InferenceSession.create`.
 * Returns `["webgpu", "wasm"]` when WebGPU is enabled (with WASM fallback),
 * or `["wasm"]` otherwise.
 */
export function getExecutionProviders(): Array<"wasm" | "webgpu"> {
  return isWebGPUEnabled() ? ["webgpu", "wasm"] : ["wasm"];
}
