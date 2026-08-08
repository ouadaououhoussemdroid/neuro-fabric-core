import { describe, it, expect } from "vitest";
import { sanitizeFilename } from "../upload";

describe("T-028 filename sanitization", () => {
  it("strips path traversal directory components (POSIX)", () => {
    // basename of "../../../etc/passwd.csv" is "passwd.csv"
    expect(sanitizeFilename("../../../etc/passwd.csv")).toBe("passwd.csv");
    // Collapses ".." sequences within filename (double-dots → single dot)
    expect(sanitizeFilename("..hidden..file.csv")).toBe(".hidden.file.csv");
  });

  it("replaces dangerous characters with underscores", () => {
    const result = sanitizeFilename('my..file<>:"|?*.csv');
    // Dots collapse to single dot; unsafe chars become underscores
    expect(result).toBe("my.file_______.csv");
    expect(result.includes("..")).toBe(false);
  });

  it("handles Windows-style backslash paths", () => {
    // basename of "Windows\\System32\\file.csv" is "file.csv"
    expect(sanitizeFilename("Windows\\System32\\file.csv")).toBe("file.csv");
  });

  it("truncates extremely long filenames to 255 chars", () => {
    const longName = "a".repeat(300) + ".csv";
    const result = sanitizeFilename(longName);
    expect(result.length).toBeLessThanOrEqual(255);
  });

  it("falls back to 'upload' for empty or fully-sanitized-away names", () => {
    expect(sanitizeFilename("")).toBe("upload");
    expect(sanitizeFilename("///")).toBe("upload");
    expect(sanitizeFilename("\\\\")).toBe("upload");
  });

  it("preserves valid filenames", () => {
    expect(sanitizeFilename("my_signal_recording.edf")).toBe("my_signal_recording.edf");
    expect(sanitizeFilename("subject_01_baseline.csv")).toBe("subject_01_baseline.csv");
    expect(sanitizeFilename("session-data.npy")).toBe("session-data.npy");
  });

  it("prevents path traversal in stored names", () => {
    const malicious = sanitizeFilename("../../../../etc/shadow.csv");
    expect(malicious).not.toContain("/");
    expect(malicious).not.toContain("\\");
    expect(malicious).not.toContain("..");
  });
});
