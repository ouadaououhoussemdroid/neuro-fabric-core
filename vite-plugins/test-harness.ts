/**
 * T-016 — Test harness + ORT WASM serving Vite plugin (dev only).
 *
 * Serves the standalone smoke-harness.html at /smoke-harness.html, bypassing
 * TanStack Start's SPA fallback middleware so the browser smoke tests can
 * import real production code via a plain <script type="module"> entry point.
 *
 * Additionally serves .wasm files from public/ort/ with the correct
 * `application/wasm` content-type. Under TanStack Start's SPA middleware,
 * .wasm requests are intercepted and return the SPA 404 page — this plugin
 * ensures they are served as static assets before the SPA fallback runs.
 *
 * In dev mode: configureServer injects middlewares that run before the SPA
 * fallback. In build mode: this plugin is a no-op (apply: "serve" only).
 *
 * This plugin is DEV + TEST only — it does not affect production builds
 * (the smoke-harness route is never linked from any production page).
 */
import type { Plugin } from "vite";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const HARNESS_PATHS = ["/smoke-harness.html", "/staging-harness.html"];
const FALLBACK_SCRIPT = "/src/testing/harness.ts";

/** Inline HTML template used when smoke-harness.html is absent. */
const FALLBACK_HTML = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Neuro-Fabric WASM Smoke Harness</title></head>
<body><script type="module" src="${FALLBACK_SCRIPT}"></script></body>
</html>`;

export function testHarnessPlugin(): Plugin {
  const cwd = process.cwd();

  return {
    name: "neurofabric-test-harness",
    apply: "serve", // dev server only — not built into production
    configureServer(server) {
      // --- Serve smoke-harness.html (bypass SPA fallback) ---
      // prepend: true ensures this runs BEFORE Vite's internal middlewares
      // and the TanStack Start SPA fallback, which would otherwise intercept
      // the route and return a 404 page.
      for (const harnessPath of HARNESS_PATHS) {
        server.middlewares.use(
          harnessPath,
          (_req, res, _next) => {
            const htmlFile = harnessPath.replace(/^\//, "");
            let html: string;
            try {
              html = readFileSync(join(cwd, htmlFile), "utf-8");
            } catch {
              html = FALLBACK_HTML;
            }
            res.statusCode = 200;
            res.setHeader("Content-Type", "text/html; charset=utf-8");
            res.end(html);
          },
          { prepend: true },
        );
      }

      // --- Serve .wasm files from public/ with correct content-type ---
      // TanStack Start's SPA middleware intercepts .wasm requests and returns
      // the SPA 404 page. prepend: true ensures this runs before all Vite
      // middlewares so the binary is served as application/wasm.
      server.middlewares.use(
        (req, res, next) => {
          const url = req.url ?? "";
          // Strip query string for file lookup.
          const cleanPath = url.split("?")[0];

          if (cleanPath.startsWith("/ort/") && cleanPath.endsWith(".wasm")) {
            const filePath = join(cwd, "public", cleanPath);
            if (existsSync(filePath)) {
              const bytes = readFileSync(filePath);
              res.statusCode = 200;
              res.setHeader("Content-Type", "application/wasm");
              res.setHeader("Content-Length", bytes.length);
              res.end(bytes);
              return;
            }
          }
          next();
        },
        { prepend: true },
      );

      // --- Serve .mjs?import requests for ORT WASM modules ---
      // onnxruntime-web uses `import("...ort-wasm-simd-threaded.jsep.mjs?import")`
      // internally. Vite's dev server transforms ?import ESM requests and may
      // break the module's internal import.meta.url / dynamic import structure.
      // prepend: true ensures we serve the raw .mjs file before Vite's
      // transform middleware runs.
      server.middlewares.use(
        (req, res, next) => {
          const url = req.url ?? "";
          const cleanPath = url.split("?")[0];
          if (url.includes("?import") && url.includes("/ort/") && cleanPath.endsWith(".mjs")) {
            const filePath = join(cwd, "public", cleanPath);
            if (existsSync(filePath)) {
              const bytes = readFileSync(filePath);
              res.statusCode = 200;
              res.setHeader("Content-Type", "text/javascript");
              res.setHeader("Content-Length", bytes.length);
              res.end(bytes);
              return;
            }
          }
          next();
        },
        { prepend: true },
      );
    },
  };
}
