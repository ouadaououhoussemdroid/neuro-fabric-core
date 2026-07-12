import { describe, it, expect } from "vitest";
import { KNOWN_DATASETS, mapDatasetRow, type DatasetClient } from "../manifest";

function mockClient(
  insertData: Record<string, unknown>[] = [],
  selectData: Record<string, unknown>[] = [],
): DatasetClient {
  return {
    from: () => ({
      insert: () => ({
        select: async () => ({ data: insertData, error: null }),
      }),
      select: () => ({
        order: async () => ({ data: selectData, error: null }),
      }),
      delete: () => ({ eq: async () => ({ error: null }) }),
    }),
  };
}

describe("KNOWN_DATASETS", () => {
  it("includes BCI-IV-2a, BCI-IV-2b, and PhysioNetMI", () => {
    const names = KNOWN_DATASETS.map((d) => d.name);
    expect(names).toContain("BCI-IV-2a");
    expect(names).toContain("BCI-IV-2b");
    expect(names).toContain("PhysioNetMI");
  });

  it("every dataset has a license", () => {
    for (const d of KNOWN_DATASETS) {
      expect(d.license).toBeTruthy();
    }
  });
});

describe("mapDatasetRow", () => {
  it("maps snake_case DB columns to camelCase TS fields", () => {
    const row = {
      id: "uuid-1",
      user_id: "user-uuid",
      name: "BCI-IV-2a",
      license: "BSD-3-Clause",
      raw_sha256: "abc123",
      source_url: "http://example.com",
      n_subjects: 9,
      n_channels: 22,
      sample_rate: 250,
      n_classes: 4,
      preprocessing_sha256: "def456",
      metadata: { paradigm: "motor_imagery" },
      created_at: "2026-07-11T00:00:00Z",
      updated_at: "2026-07-11T00:00:00Z",
    };
    const entry = mapDatasetRow(row);
    expect(entry.name).toBe("BCI-IV-2a");
    expect(entry.license).toBe("BSD-3-Clause");
    expect(entry.rawSha256).toBe("abc123");
    expect(entry.nSubjects).toBe(9);
    expect(entry.nChannels).toBe(22);
    expect(entry.sampleRate).toBe(250);
    expect(entry.nClasses).toBe(4);
    expect(entry.preprocessingSha256).toBe("def456");
  });

  it("handles null fields", () => {
    const entry = mapDatasetRow({ id: "x", name: "test", license: "MIT" });
    expect(entry.userId).toBeNull();
    expect(entry.rawSha256).toBeNull();
    expect(entry.nSubjects).toBeNull();
    expect(entry.metadata).toEqual({});
  });
});
