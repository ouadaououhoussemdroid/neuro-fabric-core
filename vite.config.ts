// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, nitro (build-only using cloudflare as a default target),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import { ortWasmSelfHostPlugin } from "./vite-plugins/ort-wasm-self-host";
import { artefactManifestPlugin } from "./vite-plugins/artefact-manifest";
import { testHarnessPlugin } from "./vite-plugins/test-harness";

// T-003: enable Nitro's cross-platform WebSocket support (Node.js runtime).
// T-008: self-host ORT WASM with SHA-384 integrity (removes jsdelivr dep).
// T-009: generate content-hashed ONNX artefact manifest.
// T-016: serve the browser smoke-test harness at /smoke-harness.html (dev only).
export default defineConfig({
  tanstackStart: {
    // Redirect TanStack Start's bundled server entry to src/server.ts (our SSR error wrapper).
    // nitro/vite builds from this
    server: { entry: "server" },
  },
  nitro: {
    features: { websocket: true },
  } as never,
  vite: {
    plugins: [ortWasmSelfHostPlugin(), artefactManifestPlugin(), testHarnessPlugin()],
    // T-016: Exclude onnxruntime-web from Vite's dependency pre-bundling so that
    // its internal dynamic import() of the WASM module is not transformed
    // (which causes "?import" resolution failures). The package is served as-is
    // from node_modules/, which is how it works in production builds.
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
  },
} as never);
