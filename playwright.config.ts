/**
 * Playwright configuration for Browser WASM End-to-End Smoke Testing.
 *
 * T-016 Browser Verification: Exercises the real production inference path
 * (embedEEG → embed → createAdapter → ONNXAdapter → onnxruntime-web WASM
 * from /ort/ → verifyRemoteArtifact via crypto.subtle.digest) inside actual
 * Chromium and Firefox browsers, not just Node.js CPU EP.
 *
 * The dev server (Vite) serves the standalone smoke-harness.html which
 * imports real production code from src/testing/harness.ts.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "tests/browser",
  // Vite dev server serves smoke-harness.html at /
  webServer: {
    command: "vite dev --port 5173",
    url: "http://localhost:5173/smoke-harness.html",
    timeout: 60_000,
    reuseExistingServer: !process.env.CI,
  },
  // T-016: Default Playwright test timeout is 30s — too short for loading
  // 30MB ONNX models through WASM + WebAssembly.compile() + inference.
  // 360s accommodates FEMBA-tiny (32MB model + 27MB WASM binary) in Firefox
  // where WebAssembly compilation is up to 3x slower than Chromium.
  timeout: 360_000,
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
    // Give ONNX models time to download + initialize in WASM.
    actionTimeout: 300_000,
    navigationTimeout: 60_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
  ],
  // No retries in CI — if it fails, it needs investigation.
  retries: process.env.CI ? 0 : 1,
  reporter: process.env.CI ? "dot" : "list",
});
