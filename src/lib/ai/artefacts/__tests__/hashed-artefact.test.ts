import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  sha256Hex,
  verifyArtefact,
  toSRI,
  generateArtefactManifest,
  writeArtefactManifest,
} from "../hashed-artefact";

const TMP = join(process.cwd(), "tmp-artefact-test");

describe("sha256Hex", () => {
  it("computes a deterministic SHA-256 hex digest", () => {
    const data = new Uint8Array([1, 2, 3]);
    const hash = sha256Hex(data);
    expect(hash).toHaveLength(64); // 32 bytes → 64 hex chars
    expect(hash).toMatch(/^[0-9a-f]+$/);
  });

  it("returns the same hash for the same input", () => {
    const a = sha256Hex(new Uint8Array([42]));
    const b = sha256Hex(new Uint8Array([42]));
    expect(a).toBe(b);
  });
});

describe("toSRI", () => {
  it("converts a hex hash to an SRI base64 string", () => {
    const hex = "ab".repeat(32); // 32 bytes
    const sri = toSRI(hex);
    expect(sri).toMatch(/^sha256-/);
    // SRI base64 of 32 bytes is 44 chars (with padding).
    expect(sri.length).toBe("sha256-".length + 44);
  });
});

describe("verifyArtefact", () => {
  it("passes when the hash matches", () => {
    const data = new Uint8Array([1, 2, 3, 4]);
    const hash = sha256Hex(data);
    expect(() => verifyArtefact(data, hash)).not.toThrow();
  });

  it("throws when the hash does not match", () => {
    const data = new Uint8Array([1, 2, 3, 4]);
    const wrongHash = "0".repeat(64);
    expect(() => verifyArtefact(data, wrongHash)).toThrow(/integrity check failed/);
  });
});

describe("generateArtefactManifest", () => {
  beforeEach(() => {
    if (existsSync(TMP)) rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
  });
  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  it("generates a manifest with SHA-256 hashes for .onnx files", () => {
    const bytes = new Uint8Array([0x08, 0x05, 0x12, 0x00]);
    writeFileSync(join(TMP, "eegconformer.onnx"), bytes);
    writeFileSync(join(TMP, "other.txt"), "ignored");

    const manifest = generateArtefactManifest(TMP);
    expect(Object.keys(manifest.models)).toHaveLength(1);
    expect(manifest.models.eegconformer).toBeDefined();
    expect(manifest.models.eegconformer.sha256).toBe(sha256Hex(bytes));
    expect(manifest.models.eegconformer.size).toBe(4);
    expect(manifest.models.eegconformer.url).toContain("eegconformer.onnx");
  });

  it("returns an empty manifest when no .onnx files exist", () => {
    writeFileSync(join(TMP, "readme.md"), "no models here");
    const manifest = generateArtefactManifest(TMP);
    expect(Object.keys(manifest.models)).toHaveLength(0);
  });
});

describe("writeArtefactManifest", () => {
  beforeEach(() => {
    if (existsSync(TMP)) rmSync(TMP, { recursive: true, force: true });
    mkdirSync(TMP, { recursive: true });
  });
  afterEach(() => {
    rmSync(TMP, { recursive: true, force: true });
  });

  it("writes manifest.json to the models directory", () => {
    writeFileSync(join(TMP, "model.onnx"), new Uint8Array([1, 2]));
    writeArtefactManifest(TMP);
    const manifestPath = join(TMP, "manifest.json");
    expect(existsSync(manifestPath)).toBe(true);
    const written = JSON.parse(readFileSync(manifestPath, "utf-8"));
    expect(written.models.model).toBeDefined();
    expect(written.models.model.sha256).toHaveLength(64);
  });
});
