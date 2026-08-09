/**
 * T-016 — Runtime artifact verification.
 *
 * Lazily loads the build-time manifest (`public/models/manifest.json`) and
 * provides a single entry point — `verifyRemoteArtifact` — that fetches a
 * same-origin artifact, checks its byte length, and verifies its SHA-256
 * against the manifest before the bytes are handed to ONNX Runtime.
 *
 * If the manifest cannot be loaded (e.g. build artefacts absent in some
 * test environments), verification is silently skipped so the caller can
 * degrade to the existing un-verified URL path. This keeps the mechanism
 * fail-open during development while enforcing integrity in production
 * where the manifest is always present.
 *
 * Browser compatibility: this module uses the Web Crypto API
 * (`crypto.subtle.digest`) for SHA-256 and the Fetch API for network
 * requests. It does NOT import `node:crypto` / `node:fs`, so it is safe
 * to load in the browser bundle. The `ArtefactManifest` type is imported
 * as a type-only reference from `hashed-artefact.ts` (stripped at compile
 * time) to avoid pulling Node.js built-ins into the client.
 */
import type { ArtefactManifest } from "./hashed-artefact";
import { log } from "../../logging";
import { metrics } from "@/lib/metrics";

let cachedManifest: ArtefactManifest | null = null;
let manifestLoadPromise: Promise<ArtefactManifest | null> | null = null;

/**
 * Load and cache the artefact manifest. Returns `null` if the manifest
 * cannot be fetched (non-fatal — callers fall back to un-verified loading).
 *
 * Uses the Fetch API (browser-compatible, also available in Node 18+).
 */
export async function getArtefactManifest(): Promise<ArtefactManifest | null> {
  if (cachedManifest) return cachedManifest;
  if (manifestLoadPromise) return manifestLoadPromise;

  manifestLoadPromise = loadManifest()
    .then((m) => {
      cachedManifest = m;
      return m;
    })
    .catch((err: unknown) => {
      log("warn", "artefact.manifest.load_failed", {
        error: (err as Error).message,
      });
      manifestLoadPromise = null;
      return null;
    });
  return manifestLoadPromise;
}

/** Fetch the manifest JSON from a same-origin path (default: /models/manifest.json). */
async function loadManifest(url = "/models/manifest.json"): Promise<ArtefactManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load artefact manifest from ${url} (${response.status})`);
  }
  return response.json();
}

export interface VerificationInfo {
  sha256: string;
  size: number;
  sha256External?: string;
  sizeExternal?: number;
  externalDataUrl?: string;
}

/**
 * Look up verification metadata for a same-origin artifact URL by scanning
 * the manifest entries. Returns `null` when no matching entry is found.
 */
export async function resolveVerification(url: string): Promise<VerificationInfo | null> {
  const manifest = await getArtefactManifest();
  if (!manifest) return null;
  for (const entry of Object.values(manifest.models)) {
    if (entry.url === url) {
      return {
        sha256: entry.sha256,
        size: entry.size,
        sha256External: entry.sha256ExternalData,
        sizeExternal: entry.sizeExternalData,
        externalDataUrl: entry.externalData,
      };
    }
  }
  return null;
}

/**
 * Compute the SHA-256 hex digest of a byte buffer using the Web Crypto API.
 * Available in all modern browsers and Node.js 18+.
 */
async function sha256Hex(data: Uint8Array): Promise<string> {
  // Uint8Array is a valid BufferSource at runtime, but TypeScript 5.9's
  // generic Uint8Array<TArrayBuffer> type isn't directly assignable to
  // BufferSource in the DOM lib. Cast to satisfy the type checker.
  const hashBuffer = await crypto.subtle.digest("SHA-256", data as BufferSource);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Fetch a remote artifact, verify its size and SHA-256 against the manifest,
 * and (when the manifest declares an external-data file) verify that too.
 *
 * Throws on any integrity violation so the caller's catch block can fall
 * back to the PCA adapter. Returns `void` on success.
 *
 * Metrics emitted:
 *   - `neuro_fabric_artifact_verification_total{result="pass"}`
 *   - `neuro_fabric_artifact_verification_total{result="fail",reason="..."}`
 *   - `neuro_fabric_artifact_verify_ms` histogram
 */
export async function verifyRemoteArtifact(url: string): Promise<void> {
  const info = await resolveVerification(url);
  if (!info) {
    // No manifest entry for this URL — skip verification (backward compatible).
    log("debug", "artefact.verify.skip", { url, reason: "no manifest entry" });
    return;
  }

  const t0 = performance.now();
  metrics.artifactVerificationTotal.inc({ result: "attempt" });

  try {
    // --- Main artifact ---
    const resp = await fetch(url);
    if (!resp.ok) {
      recordFail("fetch_error", url, t0);
      throw new Error(`Artefact fetch failed (${resp.status}) for ${url}`);
    }
    const buf = await resp.arrayBuffer();
    const bytes = new Uint8Array(buf);

    if (bytes.byteLength !== info.size) {
      recordFail("size_mismatch", url, t0);
      throw new Error(
        `Artefact size mismatch: expected ${info.size} bytes, got ${bytes.byteLength} (${url})`,
      );
    }

    const actualHash = await sha256Hex(bytes);
    if (actualHash !== info.sha256) {
      recordFail("hash_mismatch", url, t0);
      throw new Error(
        `Artefact integrity check failed: expected sha256=${info.sha256.slice(0, 16)}…, got ${actualHash.slice(0, 16)}… (${url})`,
      );
    }

    // --- External data file (if declared) ---
    if (info.externalDataUrl && info.sha256External && info.sizeExternal) {
      const extResp = await fetch(info.externalDataUrl);
      if (!extResp.ok) {
        recordFail("external_fetch_error", url, t0);
        throw new Error(
          `External data fetch failed (${extResp.status}) for ${info.externalDataUrl}`,
        );
      }
      const extBuf = await extResp.arrayBuffer();
      const extBytes = new Uint8Array(extBuf);
      if (extBytes.byteLength !== info.sizeExternal) {
        recordFail("external_size_mismatch", url, t0);
        throw new Error(
          `External data size mismatch for ${info.externalDataUrl}: expected ${info.sizeExternal}, got ${extBytes.byteLength}`,
        );
      }
      const extHash = await sha256Hex(extBytes);
      if (extHash !== info.sha256External) {
        recordFail("external_hash_mismatch", url, t0);
        throw new Error(`External data integrity check failed for ${info.externalDataUrl}`);
      }
    }

    recordPass(url, t0);
  } catch (err) {
    log("error", "artefact.verify.fail", {
      url,
      error: (err as Error).message,
    });
    throw err;
  }
}

function recordPass(url: string, t0: number): void {
  const elapsed = performance.now() - t0;
  metrics.artifactVerifyMs.observe({}, elapsed);
  metrics.artifactVerificationTotal.inc({ result: "pass" });
  log("info", "artefact.verify.pass", {
    url,
    durationMs: +elapsed.toFixed(2),
  });
}

function recordFail(reason: string, url: string, t0: number): void {
  const elapsed = performance.now() - t0;
  metrics.artifactVerifyMs.observe({}, elapsed);
  // Increment the result="fail" counter (matching recordPass's pattern so
  // tests can query { result: "fail" } without knowing the specific reason).
  metrics.artifactVerificationTotal.inc({ result: "fail" });
  // Also increment with the reason label for detailed observability.
  metrics.artifactVerificationTotal.inc({ result: "fail", reason });
  log("error", "artefact.verify.fail", {
    url,
    reason,
    durationMs: +elapsed.toFixed(2),
  });
}

/** Reset the cached manifest (test helper). */
export function __resetManifestCache(): void {
  cachedManifest = null;
  manifestLoadPromise = null;
}
