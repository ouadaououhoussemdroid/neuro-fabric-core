/**
 * T-009 — Vite plugin to generate the ONNX artefact manifest at build time.
 *
 * Scans `public/models/*.onnx`, computes SHA-256 hashes, and writes
 * `public/models/manifest.json` so the runtime can verify artefact
 * integrity at load (see `src/lib/ai/artefacts/hashed-artefact.ts`).
 *
 * T-016 — Merges in Tier 4 sidecar metadata (registry IDs, WASM
 * compatibility flags) from `manifest-metadata.ts`.
 */
import type { Plugin } from "vite";
import {
  generateArtefactManifest,
  writeArtefactManifest,
} from "../src/lib/ai/artefacts/hashed-artefact";
import { TIER4_MANIFEST_METADATA } from "../src/lib/ai/artefacts/manifest-metadata";

interface ArtefactManifestPluginOptions {
  modelsDir?: string;
}

export function artefactManifestPlugin(opts: ArtefactManifestPluginOptions = {}): Plugin {
  const modelsDir = opts.modelsDir ?? "public/models";
  return {
    name: "artefact-manifest",
    configResolved() {
      try {
        writeArtefactManifest(modelsDir, undefined, TIER4_MANIFEST_METADATA);
      } catch {
        // No models directory or no .onnx files — skip.
      }
    },
    buildStart() {
      try {
        writeArtefactManifest(modelsDir, undefined, TIER4_MANIFEST_METADATA);
      } catch {
        // ignore
      }
    },
  };
}
