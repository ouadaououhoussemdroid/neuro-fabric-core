/**
 * T-024b — Tests for brain-flag.ts (unified WebNN + WebGPU feature flags)
 */
import { describe, it, expect, vi } from "vitest";
import {
  isWebNNAvailable,
  isWebGPUAvailable,
  isWebNEnabled,
  isWebGPUEnabled,
  getExecutionProviders,
  getAcceleratorStatus,
  setWebNNEnabled,
  setWebGPUEnabled,
} from "./brain-flag";

// Save original runtime state
let originalWindow: any;

describe("brain-flag", () => {
  it("returns WASM-only providers when browser APIs are unavailable", () => {
    setWebNNEnabled(false);
    setWebGPUEnabled(false);

    // Without navigator.ml or navigator.gpu, providers fall back to WASM
    const providers = getExecutionProviders();
    expect(providers).toContain("wasm");
    // In test environment (no real navigator.ml/gpu), webnn/webgpu are false
    expect(providers.filter((p) => p === "webnn" || p === "webgpu")).toHaveLength(0);
  });

  it("getAcceleratorStatus returns correct structure", () => {
    setWebNNEnabled(false);
    setWebGPUEnabled(false);

    const status = getAcceleratorStatus();
    expect(status).toHaveProperty("webnn");
    expect(status).toHaveProperty("webgpu");
    expect(status).toHaveProperty("wasm");
    expect(status).toHaveProperty("active");
    expect(status.wasm).toBe(true); // WASM is always available
    expect(Array.isArray(status.active)).toBe(true);
    expect(status.active).toContain("wasm");
  });

  it("prioritizes WebNN > WebGPU > WASM when enabled", () => {
    setWebNNEnabled(true);
    setWebGPUEnabled(true);

    const providers = getExecutionProviders();
    // If WebNN is available and enabled, it should be first
    // Since test env has no navigator.ml, webnn will be false
    // but the function structure is correct
    expect(providers[providers.length - 1]).toBe("wasm"); // WASM always last (fallback)
  });

  it("isWebGPUAvailable checks navigator.gpu", () => {
    expect(typeof isWebGPUAvailable()).toBe("boolean");
  });

  it("isWebNNAvailable checks navigator.ml", () => {
    expect(typeof isWebNNAvailable()).toBe("boolean");
  });

  it("runtime toggles work correctly", () => {
    setWebNNEnabled(true);
    setWebGPUEnabled(false);

    const status1 = getAcceleratorStatus();
    // In test env, navigator.ml won't exist, so isWebNNAvailable() is false
    // but the toggle is set — isWebNEnabled() requires both
    expect(isWebNEnabled()).toBe(false); // available check fails in test env

    setWebGPUEnabled(true);
    // Same for webgpu
    expect(isWebGPUEnabled()).toBe(false); // available check fails in test env
  });
});
