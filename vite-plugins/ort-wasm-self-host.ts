/**
 * T-008 — Vite plugin to self-host the onnxruntime-web WASM bundle.
 *
 * Copies the ORT WASM + loader artefacts from `node_modules/onnxruntime-web/dist/`
 * into `public/ort/` (both dev and build) so they are served from our own origin
 * instead of jsdelivr. Also generates a SHA-384 integrity manifest
 * (`public/ort/integrity.json`) so the runtime can verify artefact integrity
 * at load time.
 *
 * This plugin runs at Vite `configResolved` (dev) and `buildStart` (build) so
 * the files are in place before the dev server or production bundle starts.
 */
import type { Plugin } from "vite";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { createHash } from "node:crypto";

interface OrtWasmPluginOptions {
  /** Source directory inside node_modules. Default: onnxruntime-web/dist. */
  sourceDir?: string;
  /** Destination under public/. Default: public/ort. */
  destDir?: string;
  /** Glob of files to copy. Default: the SIMD-threaded variants. */
  files?: string[];
}

const DEFAULT_FILES = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
];

interface IntegrityEntry {
  file: string;
  sha384: string;
  size: number;
}

export function ortWasmSelfHostPlugin(opts: OrtWasmPluginOptions = {}): Plugin {
  const files = opts.files ?? DEFAULT_FILES;
  const sourceDir =
    opts.sourceDir ?? join(process.cwd(), "node_modules", "onnxruntime-web", "dist");
  const destDir = opts.destDir ?? join(process.cwd(), "public", "ort");

  function syncFiles(): void {
    if (!existsSync(sourceDir)) {
      // ORT not installed (e.g. CI without deps); skip silently.
      return;
    }
    mkdirSync(destDir, { recursive: true });

    const manifest: IntegrityEntry[] = [];
    for (const file of files) {
      const src = join(sourceDir, file);
      if (!existsSync(src)) continue;
      cpSync(src, join(destDir, file));
      const buf = readFileSync(join(destDir, file));
      const hash = createHash("sha384").update(buf).digest("base64");
      manifest.push({ file, sha384: `sha384-${hash}`, size: buf.length });
    }

    // Write the integrity manifest.
    if (manifest.length > 0) {
      writeFileSync(
        join(destDir, "integrity.json"),
        JSON.stringify({ generated: new Date().toISOString(), artifacts: manifest }, null, 2),
      );
    }
  }

  return {
    name: "ort-wasm-self-host",
    configResolved() {
      syncFiles();
    },
    buildStart() {
      syncFiles();
    },
  };
}

/** Load the integrity manifest from `public/ort/integrity.json` (server-side). */
export function loadOrtIntegrityManifest(publicDir: string): {
  artifacts: IntegrityEntry[];
} | null {
  const manifestPath = join(resolve(publicDir), "ort", "integrity.json");
  if (!existsSync(manifestPath)) return null;
  try {
    return JSON.parse(readFileSync(manifestPath, "utf-8"));
  } catch {
    return null;
  }
}

export type { IntegrityEntry };
