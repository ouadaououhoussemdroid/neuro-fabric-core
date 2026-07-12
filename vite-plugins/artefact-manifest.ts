/**
 * T-009 — Vite plugin to generate the ONNX artefact manifest at build time.
 *
 * Scans `public/models/*.onnx`, computes SHA-256 hashes, and writes
 * `public/models/manifest.json` so the runtime can verify artefact
 * integrity at load (see `src/lib/ai/artefacts/hashed-artefact.ts`).
 */
import type { Plugin } from "vite";
import {
  generateArtefactManifest,
  writeArtefactManifest,
} from "../src/lib/ai/artefacts/hashed-artefact";

interface ArtefactManifestPluginOptions {
  modelsDir?: string;
}

export function artefactManifestPlugin(opts: ArtefactManifestPluginOptions = {}): Plugin {
  const modelsDir = opts.modelsDir ?? "public/models";
  return {
    name: "artefact-manifest",
    configResolved() {
      try {
        writeArtefactManifest(modelsDir);
      } catch {
        // No models directory or no .onnx files — skip.
      }
    },
    buildStart() {
      try {
        writeArtefactManifest(modelsDir);
      } catch {
        // ignore
      }
    },
  };
}
