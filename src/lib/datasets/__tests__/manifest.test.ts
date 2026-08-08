import { describe, it, expect } from "vitest";
import {
  KNOWN_DATASETS,
  mapDatasetRow,
  listDatasets,
  insertDataset,
  type DatasetClient,
} from "../manifest";

function mockClient(
  insertData: Record<string, unknown>[] = [],
  selectData: Record<string, unknown>[] | null = [],
  error: boolean = false,
): DatasetClient {
  const err = error ? { message: "DB error" } : null;
  return {
    from: () => ({
      insert: () => ({
        select: async () => ({ data: error ? null : insertData, error: err }),
      }),
      select: () => ({
        order: async () => ({ data: error ? null : selectData, error: err }),
      }),
      delete: () => ({ eq: async () => ({ error: err }) }),
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

describe("listDatasets", () => {
  it("returns mapped datasets on success", async () => {
    const client = mockClient(
      [],
      [
        {
          id: "uuid-1",
          user_id: null,
          name: "BCI-IV-2a",
          license: "BSD-3-Clause",
          raw_sha256: "abc",
          source_url: "http://example.com",
          n_subjects: 9,
          n_channels: 22,
          sample_rate: 250,
          n_classes: 4,
          preprocessing_sha256: null,
          metadata: { paradigm: "mi" },
          created_at: "2026-07-11T00:00:00Z",
          updated_at: "2026-07-11T00:00:00Z",
        },
      ],
    );
    const datasets = await listDatasets(client);
    expect(datasets).toHaveLength(1);
    expect(datasets[0].name).toBe("BCI-IV-2a");
  });

  it("returns empty array on DB error", async () => {
    const client = mockClient([], [], true);
    const datasets = await listDatasets(client);
    expect(datasets).toEqual([]);
  });

  it("returns empty array when no data", async () => {
    const client = mockClient([], null);
    const datasets = await listDatasets(client);
    expect(datasets).toEqual([]);
  });
});

describe("insertDataset", () => {
  it("inserts and returns the created entry", async () => {
    const client = mockClient([
      {
        id: "new-uuid",
        user_id: "u1",
        name: "MyDataset",
        license: "MIT",
        raw_sha256: "sha123",
        source_url: "http://example.com/data",
        n_subjects: 5,
        n_channels: 10,
        sample_rate: 128,
        n_classes: 2,
        preprocessing_sha256: "psha",
        metadata: { foo: "bar" },
        created_at: "2026-07-11T00:00:00Z",
        updated_at: "2026-07-11T00:00:00Z",
      },
    ]);
    const result = await insertDataset(client, {
      name: "MyDataset",
      license: "MIT",
      rawSha256: "sha123",
      sourceUrl: "http://example.com/data",
      nSubjects: 5,
      nChannels: 10,
      sampleRate: 128,
      nClasses: 2,
      preprocessingSha256: "psha",
      metadata: { foo: "bar" },
      userId: "u1",
    });
    expect(result).not.toBeNull();
    expect(result!.name).toBe("MyDataset");
    expect(result!.userId).toBe("u1");
  });

  it("returns null on DB error", async () => {
    const client = mockClient([], [], true);
    const result = await insertDataset(client, {
      name: "Test",
      license: "MIT",
      metadata: {},
    });
    expect(result).toBeNull();
  });

  it("returns null when no data returned", async () => {
    const client = mockClient([], null);
    const result = await insertDataset(client, {
      name: "Test",
      license: "MIT",
      metadata: {},
    });
    expect(result).toBeNull();
  });
});
