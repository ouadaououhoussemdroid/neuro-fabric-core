import { describe, it, expect, afterEach } from "vitest";
import {
  isWebGPUAvailable,
  isWebGPUEnabled,
  setWebGPUEnabled,
  getExecutionProviders,
} from "../webgpu-flag";

describe("T-024 WebGPU feature flag", () => {
  afterEach(() => {
    setWebGPUEnabled(false);
  });

  it("isWebGPUAvailable returns false in non-browser env", () => {
    // vitest runs in a jsdom-like env without navigator.gpu by default
    expect(isWebGPUAvailable()).toBe(false);
  });

  it("isWebGPUEnabled returns false by default", () => {
    expect(isWebGPUEnabled()).toBe(false);
  });

  it("getExecutionProviders returns wasm-only by default", () => {
    expect(getExecutionProviders()).toEqual(["wasm"]);
  });

  it("setWebGPUEnabled(true) does not enable without browser support", () => {
    setWebGPUEnabled(true);
    // Still false because navigator.gpu doesn't exist in test env
    expect(isWebGPUEnabled()).toBe(false);
    expect(getExecutionProviders()).toEqual(["wasm"]);
  });

  it("getExecutionProviders returns webgpu+wasm when both flag and GPU are available", () => {
    // Mock navigator.gpu
    const originalNav = globalThis.navigator;
    Object.defineProperty(globalThis, "navigator", {
      value: { ...originalNav, gpu: {} },
      writable: true,
      configurable: true,
    });
    setWebGPUEnabled(true);
    expect(isWebGPUAvailable()).toBe(true);
    expect(isWebGPUEnabled()).toBe(true);
    expect(getExecutionProviders()).toEqual(["webgpu", "wasm"]);
    // Restore
    Object.defineProperty(globalThis, "navigator", {
      value: originalNav,
      writable: true,
      configurable: true,
    });
  });
});
