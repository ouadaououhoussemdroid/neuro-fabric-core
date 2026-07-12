import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, writeFileSync, rmSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import {
  ortWasmSelfHostPlugin,
  loadOrtIntegrityManifest,
} from "../../vite-plugins/ort-wasm-self-host";

const TMP_DIR = join(process.cwd(), "tmp-ort-test");

describe("ortWasmSelfHostPlugin", () => {
  beforeEach(() => {
    if (existsSync(TMP_DIR)) rmSync(TMP_DIR, { recursive: true, force: true });
    mkdirSync(TMP_DIR, { recursive: true });
  });
  afterEach(() => {
    rmSync(TMP_DIR, { recursive: true, force: true });
  });

  it("returns a named Vite plugin with configResolved and buildStart hooks", () => {
    const plugin = ortWasmSelfHostPlugin();
    expect(plugin.name).toBe("ort-wasm-self-host");
    expect(typeof plugin.configResolved).toBe("function");
    expect(typeof plugin.buildStart).toBe("function");
  });

  it("copies files and generates integrity.json when source exists", () => {
    const sourceDir = join(TMP_DIR, "src");
    const destDir = join(TMP_DIR, "dest");
    mkdirSync(sourceDir, { recursive: true });
    writeFileSync(join(sourceDir, "ort-wasm-simd-threaded.wasm"), "fake-wasm-bytes");
    writeFileSync(join(sourceDir, "ort-wasm-simd-threaded.mjs"), "export default {};");

    const plugin = ortWasmSelfHostPlugin({ sourceDir, destDir });
    // Trigger the sync by calling configResolved.
    (plugin.configResolved as (cfg: unknown) => void)({});

    expect(existsSync(join(destDir, "ort-wasm-simd-threaded.wasm"))).toBe(true);
    expect(existsSync(join(destDir, "ort-wasm-simd-threaded.mjs"))).toBe(true);
    expect(existsSync(join(destDir, "integrity.json"))).toBe(true);

    const manifest = JSON.parse(readFileSync(join(destDir, "integrity.json"), "utf-8"));
    expect(manifest.artifacts).toHaveLength(2);
    expect(manifest.artifacts[0].file).toBe("ort-wasm-simd-threaded.wasm");
    expect(manifest.artifacts[0].sha384).toMatch(/^sha384-/);
    expect(manifest.artifacts[0].size).toBe(15);
  });

  it("skips silently when source dir does not exist", () => {
    const plugin = ortWasmSelfHostPlugin({
      sourceDir: join(TMP_DIR, "nonexistent"),
      destDir: join(TMP_DIR, "dest2"),
    });
    expect(() => (plugin.configResolved as (cfg: unknown) => void)({})).not.toThrow();
    expect(existsSync(join(TMP_DIR, "dest2"))).toBe(false);
  });

  it("generates correct SHA-384 hashes", () => {
    const sourceDir = join(TMP_DIR, "src");
    const destDir = join(TMP_DIR, "dest");
    mkdirSync(sourceDir, { recursive: true });
    const content = "test-content";
    writeFileSync(join(sourceDir, "ort-wasm-simd-threaded.wasm"), content);
    const plugin = ortWasmSelfHostPlugin({ sourceDir, destDir });
    (plugin.configResolved as (cfg: unknown) => void)({});

    const manifest = JSON.parse(readFileSync(join(destDir, "integrity.json"), "utf-8"));
    const entry = manifest.artifacts.find(
      (a: { file: string }) => a.file === "ort-wasm-simd-threaded.wasm",
    );
    // Verify the hash matches a manual computation.
    const expected = "sha384-" + createHash("sha384").update(content).digest("base64");
    expect(entry.sha384).toBe(expected);
  });
});

describe("loadOrtIntegrityManifest", () => {
  const TMP_PUBLIC = join(process.cwd(), "tmp-ort-public");

  beforeEach(() => {
    if (existsSync(TMP_PUBLIC)) rmSync(TMP_PUBLIC, { recursive: true, force: true });
    mkdirSync(join(TMP_PUBLIC, "ort"), { recursive: true });
  });
  afterEach(() => {
    rmSync(TMP_PUBLIC, { recursive: true, force: true });
  });

  it("returns null when manifest is absent", () => {
    expect(loadOrtIntegrityManifest(TMP_PUBLIC)).toBeNull();
  });

  it("returns parsed manifest when present", () => {
    writeFileSync(
      join(TMP_PUBLIC, "ort", "integrity.json"),
      JSON.stringify({ artifacts: [{ file: "test.wasm", sha384: "sha384-abc", size: 100 }] }),
    );
    const manifest = loadOrtIntegrityManifest(TMP_PUBLIC);
    expect(manifest).not.toBeNull();
    expect(manifest!.artifacts).toHaveLength(1);
    expect(manifest!.artifacts[0].file).toBe("test.wasm");
  });
});
