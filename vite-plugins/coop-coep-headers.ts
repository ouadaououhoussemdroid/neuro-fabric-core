/**
 * Shared COOP/COEP/CORP header values for cross-origin isolation.
 *
 * WHY — Firefox V2 WASM latency blocker (T-035):
 *   Chromium V2 P95 442 ms (PASS) vs Firefox V2 P95 1576 ms (FAIL) for the same
 *   threaded `ort-wasm-simd-threaded.wasm` (13.5 MB) bundle. Without COOP + COEP
 *   the browser is NOT cross-origin isolated → `SharedArrayBuffer` is unavailable
 *   in Firefox → ORT's threaded WASM build degrades to a single thread (~3.5x
 *   slower). Chromium enables SAB heuristically, hiding the regression.
 *
 * WHAT THESE HEADERS DO:
 *   - Cross-Origin-Opener-Policy: same-origin  — isolates the browsing context
 *     group so no cross-origin opener can reach this page.
 *   - Cross-Origin-Embedder-Policy: require-corp — requires cross-origin
 *     resources to opt in (same-origin resources — like our /ort/*.wasm and
 *     /models/*.onnx — are exempt and load freely).
 *   - Cross-Origin-Resource-Policy: same-origin — hardens same-origin resources.
 *
 * Together they set `window.crossOriginIsolated === true` and
 * `typeof SharedArrayBuffer === "function"`, which is the prerequisite for
 * ORT-Web's threaded SIMD WASM build to actually spawn threads in Firefox.
 *
 * PLACEMENT (single source of truth, reused by all three layers):
 *   - Dev (vite.config.ts `server.headers`) — covers the real app document (`/`)
 *     and Vite-served static assets (404s, /models, /src, …).
 *   - Dev/test harness (vite-plugins/test-harness.ts) — the smoke/staging
 *     harness HTML documents and /ort/*.wasm are finalized with `res.end()` by a
 *     prepended middleware that bypasses Vite's `server.headers`, so they apply
 *     these headers at their own `res.end` sites.
 *   - Production (src/server.ts `withCrossOriginIsolationHeaders`) — applied to
 *     every Nitro SSR + static response.
 */
export const COOP_COEP_HEADERS: Record<string, string> = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};
