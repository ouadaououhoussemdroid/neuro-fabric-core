// @lovable.dev/vite-tanstack-config already includes the following — do NOT add them manually
// or the app will break with duplicate plugins:
//   - tanstackStart, viteReact, tailwindcss, tsConfigPaths, nitro (build-only using cloudflare as a default target),
//     componentTagger (dev-only), VITE_* env injection, @ path alias, React/TanStack dedupe,
//     error logger plugins, and sandbox detection (port/host/strictPort).
// You can pass additional config via defineConfig({ vite: { ... }, etc... }) if needed.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";
import { ortWasmSelfHostPlugin } from "./vite-plugins/ort-wasm-self-host";
import { artefactManifestPlugin } from "./vite-plugins/artefact-manifest";

// T-003: enable Nitro's cross-platform WebSocket support (Node.js runtime).
// T-008: self-host ORT WASM with SHA-384 integrity (removes jsdelivr dep).
// T-009: generate content-hashed ONNX artefact manifest.
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
    plugins: [ortWasmSelfHostPlugin(), artefactManifestPlugin()],
  },
} as never);
