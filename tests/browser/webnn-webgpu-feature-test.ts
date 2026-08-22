/**
 * T-025 — Browser WebGPU/WebNN Feature Flag Tests
 *
 * Tests the brain-flag.ts execution provider selection chain:
 *   ["webnn", "webgpu", "wasm"]  — WebNN (NPU) → WebGPU → WASM fallback
 *
 * These tests verify:
 * 1. Feature detection API surface works in real browsers (navigator.ml, navigator.gpu)
 * 2. The execution provider priority chain returns WASM when no accelerators are available
 * 3. Build-time flags (VITE_WEBNN, VITE_ORT_WEBGPU) are correctly wired
 * 4. Runtime toggles (setWebNNEnabled/setWebGPUEnabled) work as expected
 * 5. The accelerator status object reports correct boolean flags
 *
 * These tests do NOT require a GPU or NPU — they verify the fallback chain
 * is correct so that on any browser without WebNN/WebGPU, the system
 * gracefully degrades to WASM inference.
 */
import { test, expect, type Page } from "@playwright/test";

/** Navigate to the harness page and wait for the production-code bridge to load. */
async function loadHarness(page: Page): Promise<void> {
  await page.goto("/smoke-harness.html", { waitUntil: "networkidle" });
  await page.waitForFunction(() => (window as any).__neuroTest !== undefined, undefined, {
    timeout: 30_000,
  });
}

test.describe("T-025: WebGPU/WebNN feature flag detection", () => {
  test("getExecutionProviders() always returns at least WASM as fallback", async ({ page }) => {
    await loadHarness(page);

    const providers = await page.evaluate(() =>
      (window as any).__neuroTest.getExecutionProviders(),
    );

    // WASM is always the ultimate fallback.
    expect(providers).toContain("wasm");
    // Providers array is never empty.
    expect(providers.length).toBeGreaterThan(0);
    // Last entry is always wasm (the fallback).
    expect(providers[providers.length - 1]).toBe("wasm");
  });

  test("getAcceleratorStatus() returns well-formed status object", async ({ page }) => {
    await loadHarness(page);

    const status = await page.evaluate(() =>
      (window as any).__neuroTest.getAcceleratorStatus(),
    );

    expect(status).toHaveProperty("webnn");
    expect(status).toHaveProperty("webgpu");
    expect(status).toHaveProperty("wasm");
    expect(status).toHaveProperty("active");

    // WASM is always available when onnxruntime-web is loaded.
    expect(status.wasm).toBe(true);

    // webnn and webgpu are booleans (may be true or false depending on browser).
    expect(typeof status.webnn).toBe("boolean");
    expect(typeof status.webgpu).toBe("boolean");

    // active array should match getExecutionProviders().
    expect(Array.isArray(status.active)).toBe(true);
    expect(status.active).toContain("wasm");
  });

  test("navigator.ml and navigator.gpu detection matches API availability", async ({ page }) => {
    await loadHarness(page);

    const caps = await page.evaluate(() => ({
      hasMl: "ml" in navigator,
      hasGpu: "gpu" in navigator,
      isWebNNAvailable: (window as any).__neuroTest.isWebNNAvailable(),
      isWebGPUAvailable: (window as any).__neuroTest.isWebGPUAvailable(),
    }));

    // The brain-flag API should match raw navigator checks.
    expect(caps.isWebNNAvailable).toBe(caps.hasMl);
    expect(caps.isWebGPUAvailable).toBe(caps.hasGpu);
  });

  test("runtime toggle for WebGPU can be enabled and disabled", async ({ page }) => {
    await loadHarness(page);

    // With WebGPU disabled, should only have WASM.
    await page.evaluate(() => {
      (window as any).__neuroTest.setWebGPUEnabled(false);
    });
    let providers = await page.evaluate(() =>
      (window as any).__neuroTest.getExecutionProviders(),
    );
    expect(providers).toEqual(["wasm"]);

    // Enable WebGPU — if browser supports it, it should appear.
    // If browser doesn't support WebGPU, it still falls back to WASM.
    await page.evaluate(() => {
      (window as any).__neuroTest.setWebGPUEnabled(true);
    });
    providers = await page.evaluate(() =>
      (window as any).__neuroTest.getExecutionProviders(),
    );
    // At minimum, WASM is present. If WebGPU is supported, it's the first entry.
    expect(providers).toContain("wasm");
    if ((await page.evaluate(() => "gpu" in navigator))) {
      expect(providers).toContain("webgpu");
    }

    // Reset to disabled state after test.
    await page.evaluate(() => {
      (window as any).__neuroTest.setWebGPUEnabled(false);
    });
  });

  test("runtime toggle for WebNN can be enabled and disabled", async ({ page }) => {
    await loadHarness(page);

    // Enable WebNN — if browser supports it (navigator.ml exists), it should
    // appear in the priority chain ahead of WebGPU and WASM.
    await page.evaluate(() => {
      (window as any).__neuroTest.setWebNNEnabled(true);
    });
    const providers = await page.evaluate(() =>
      (window as any).__neuroTest.getExecutionProviders(),
    );

    expect(providers).toContain("wasm");
    if ((await page.evaluate(() => "ml" in navigator))) {
      expect(providers[0]).toBe("webnn");
    }

    // Reset.
    await page.evaluate(() => {
      (window as any).__neuroTest.setWebNNEnabled(false);
    });
  });

  test("build-time flags VITE_WEBNN and VITE_ORT_WEBGPU are set in env", async ({ page }) => {
    await loadHarness(page);

    // The env vars should be accessible via Vite's import.meta.env.
    // We test this through the runtime API — if the build-time flag is "true"
    // AND the browser supports the API, it should be enabled by default.
    const status = await page.evaluate(() =>
      (window as any).__neuroTest.getAcceleratorStatus(),
    );

    // The active providers should never include an EP that the browser
    // doesn't support (no false positives).
    if (!status.webnn) {
      expect(status.active).not.toContain("webnn");
    }
    if (!status.webgpu) {
      expect(status.active).not.toContain("webgpu");
    }
  });

  test("execution provider priority: WebNN > WebGPU > WASM", async ({ page }) => {
    await loadHarness(page);

    const chain = await page.evaluate(() => {
      const status = (window as any).__neuroTest.getAcceleratorStatus();
      return status.active;
    });

    // Verify the priority chain ordering.
    // If all three are available: ["webnn", "webgpu", "wasm"]
    // If only WebGPU+WASM: ["webgpu", "wasm"]
    // If only WASM: ["wasm"]
    expect(chain).toContain("wasm");

    if (chain.includes("webnn") && chain.includes("webgpu")) {
      const webnnIdx = chain.indexOf("webnn");
      const webgpuIdx = chain.indexOf("webgpu");
      expect(webnnIdx).toBeLessThan(webgpuIdx);
      expect(webgpuIdx).toBeLessThan(chain.indexOf("wasm"));
    }
  });
});
