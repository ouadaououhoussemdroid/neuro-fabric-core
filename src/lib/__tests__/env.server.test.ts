import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { requireServerEnv } from "../env.server";

const ORIGINAL_ENV = { ...process.env };

describe("requireServerEnv", () => {
  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it("returns the requested vars when all are present and valid", () => {
    process.env.SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_PUBLISHABLE_KEY = "pub-key";
    const result = requireServerEnv(["SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"]);
    expect(result).toEqual({
      SUPABASE_URL: "https://example.supabase.co",
      SUPABASE_PUBLISHABLE_KEY: "pub-key",
    });
  });

  it("throws listing every missing var, not just the first", () => {
    delete process.env.SUPABASE_URL;
    delete process.env.SUPABASE_PUBLISHABLE_KEY;
    expect(() => requireServerEnv(["SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"])).toThrow(
      /missing: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY/,
    );
  });

  it("treats an empty string as missing", () => {
    process.env.SUPABASE_URL = "";
    expect(() => requireServerEnv(["SUPABASE_URL"])).toThrow(/missing: SUPABASE_URL/);
  });

  it("throws for a present-but-invalid value (e.g. non-URL SUPABASE_URL)", () => {
    process.env.SUPABASE_URL = "not-a-url";
    expect(() => requireServerEnv(["SUPABASE_URL"])).toThrow(/invalid: SUPABASE_URL/);
  });

  it("reports missing and invalid vars together in one error", () => {
    process.env.SUPABASE_URL = "not-a-url";
    delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    expect(() => requireServerEnv(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"])).toThrow(
      "Server misconfigured (missing: SUPABASE_SERVICE_ROLE_KEY; invalid: SUPABASE_URL). Connect Supabase in Lovable Cloud.",
    );
  });
});
