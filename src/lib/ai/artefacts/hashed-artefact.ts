/**
 * T-009 — Content-hashed ONNX artefact manager.
 *
 * Replaces the static `/models/eegconformer.onnx` reference with a
 * content-addressed scheme: each model artefact is referenced by a
 * `sha256-…` URL, and the loader verifies the fetched bytes match the
 * expected hash before instantiating the ONNX session.
 *
 * The manifest (`public/models/manifest.json`) maps model ids to:
 *   - url: the fetchable URL (same-origin under /models/ or a storage bucket)
 *   - sha256: the hex digest of the expected bytes
 *   - size: byte length, for size-cap checks before download
 *
 * At build time, `scripts/generate_model_manifest.py` (or the Vite plugin
 * below) scans `public/models/*.onnx` and writes the manifest. At load time,
 * `fetchArtifact()` fetches, verifies, and returns an ArrayBuffer.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, resolve, basename } from "node:path";

export interface ArtefactManifestEntry {
  /** Artefact id (filename without extension). */
  id: string;
  /** Registry id this artifact backs (e.g. "braindecode-eegconformer-prod"). */
  registryId?: string;
  /** Fetchable URL or same-origin path. */
  url: string;
  /** SHA-256 hex digest of the expected bytes. */
  sha256: string;
  /** Byte length. */
  size: number;
  /** Whether the ONNX ops are supported by ORT-WASM (browser). */
  wasmCompatible?: boolean;
  /** ONNX op names that block WASM execution (when wasmCompatible is false). */
  wasmBlockers?: string[];
  /** External data file URL (when ONNX uses external storage). */
  externalData?: string;
  /** SHA-256 of the external data file. */
  sha256ExternalData?: string;
  /** External data file size in bytes. */
  sizeExternalData?: number;
}

export interface ArtefactManifest {
  /** ISO timestamp of generation. */
  generated: string;
  /** Map of model id → entry. */
  models: Record<string, ArtefactManifestEntry>;
}

/** SRI-style hash string: `sha256-<base64>`. */
export function toSRI(hex: string): string {
  // Convert hex to base64 for Subresource Integrity compatibility.
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return "sha256-" + Buffer.from(bytes).toString("base64");
}

/** Compute the SHA-256 hex digest of a byte buffer. */
export function sha256Hex(data: Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * Verify a fetched artefact against an expected SHA-256.
 * Throws if the hash does not match.
 */
export function verifyArtefact(data: Uint8Array, expectedSha256: string): void {
  const actual = sha256Hex(data);
  if (actual !== expectedSha256) {
    throw new Error(
      `Artefact integrity check failed: expected sha256=${expectedSha256.slice(0, 16)}…, got ${actual.slice(0, 16)}…`,
    );
  }
}

/**
 * Fetch an ONNX artefact by model id, verifying its SHA-256 hash.
 *
 * Resolution order for the URL:
 *   1. If the manifest entry has a full URL, fetch it.
 *   2. If it's a path (starts with `/`), fetch from same origin.
 *
 * @param manifest The loaded artefact manifest.
 * @param modelId  The model id to fetch.
 * @returns Verified artefact bytes as an ArrayBuffer.
 */
export async function fetchArtifact(
  manifest: ArtefactManifest,
  modelId: string,
): Promise<ArrayBuffer> {
  const entry = manifest.models[modelId];
  if (!entry) {
    throw new Error(`Artefact manifest has no entry for model: ${modelId}`);
  }

  const response = await fetch(entry.url);
  if (!response.ok) {
    throw new Error(`Artefact fetch failed (${response.status}) for ${entry.url}`);
  }
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  if (bytes.byteLength !== entry.size) {
    throw new Error(
      `Artefact size mismatch: expected ${entry.size} bytes, got ${bytes.byteLength}`,
    );
  }
  verifyArtefact(bytes, entry.sha256);
  return buffer;
}

/**
 * Load the artefact manifest from a same-origin path.
 * Default: `/models/manifest.json`.
 */
export async function loadArtefactManifest(
  url = "/models/manifest.json",
): Promise<ArtefactManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load artefact manifest from ${url} (${response.status})`);
  }
  return response.json();
}

/**
 * Optional sidecar metadata that enriches auto-generated manifest entries.
 * Maps artifact filename (without `.onnx`) → extra metadata.
 */
export type ManifestMetadata = Record<
  string,
  {
    registryId?: string;
    wasmCompatible?: boolean;
    wasmBlockers?: string[];
  }
>;

/**
 * Build-time: scan a directory for `.onnx` files and write a manifest.
 *
 * Each file becomes a manifest entry with:
 *   - id: the filename without extension (e.g. "eegconformer" → "eegconformer")
 *   - url: `/<dir>/<filename>` (same-origin)
 *   - sha256: hex digest of the file bytes
 *   - size: byte length
 *
 * Pass `metadata` to enrich entries with registry IDs, WASM compatibility
 * flags, and WASM blocker op lists.
 */
export function generateArtefactManifest(
  modelsDir: string,
  metadata?: ManifestMetadata,
): ArtefactManifest {
  const abs = resolve(modelsDir);
  const entries: Record<string, ArtefactManifestEntry> = {};

  if (existsSync(abs)) {
    for (const file of readdirSync(abs)) {
      if (!file.endsWith(".onnx")) continue;
      const fullPath = join(abs, file);
      const bytes = readFileSync(fullPath);
      const id = basename(file, ".onnx");
      const relativePath = `/${basename(abs)}/${file}`;
      const meta = metadata?.[id];
      const entry: ArtefactManifestEntry = {
        id,
        registryId: meta?.registryId,
        url: relativePath,
        sha256: sha256Hex(new Uint8Array(bytes)),
        size: bytes.byteLength,
        wasmCompatible: meta?.wasmCompatible ?? true,
        wasmBlockers: meta?.wasmBlockers,
      };
      // Detect + hash external data file (<model>.onnx.data)
      const dataFile = `${file}.data`;
      const dataPath = join(abs, dataFile);
      if (existsSync(dataPath)) {
        const dataBytes = readFileSync(dataPath);
        entry.externalData = `/${basename(abs)}/${dataFile}`;
        entry.sha256ExternalData = sha256Hex(new Uint8Array(dataBytes));
        entry.sizeExternalData = dataBytes.byteLength;
      }
      entries[id] = entry;
    }
  }

  return {
    generated: new Date().toISOString(),
    models: entries,
  };
}

/** Write the manifest to `<modelsDir>/manifest.json`. */
export function writeArtefactManifest(
  modelsDir: string,
  manifest?: ArtefactManifest,
  metadata?: ManifestMetadata,
): void {
  const m = manifest ?? generateArtefactManifest(modelsDir, metadata);
  const outPath = join(resolve(modelsDir), "manifest.json");
  writeFileSync(outPath, JSON.stringify(m, null, 2));
}
