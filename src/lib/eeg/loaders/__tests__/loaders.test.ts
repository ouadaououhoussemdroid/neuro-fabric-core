import { describe, it, expect, vi } from "vitest";
import { physionet } from "../physionet";
import { bciCompetitionIV2a } from "../bci-competition";
import { tuhEeg } from "../tuh";

const BCI_MIRROR = "https://example.com/bci";

// Mock the EDF parser so load() tests don't need real binary data.
vi.mock("../../parsers/edf", () => ({
  parseEDF: vi.fn().mockResolvedValue({
    channels: ["Fz", "Cz", "Pz"],
    data: [[1, 2, 3]],
    sampleRate: 256,
  }),
}));

describe("Dataset loaders", () => {
  describe("physionet (eegmmidb)", () => {
    it("has the correct name", () => {
      expect(physionet.name).toBe("physionet-eegmmidb");
    });

    it("lists 109 subjects × 14 runs = 1526 records", async () => {
      const records = await physionet.list();
      expect(records).toHaveLength(1526);
    });

    it("marks runs 1-2 as baseline, runs 3-14 as motor-imagery", async () => {
      const records = await physionet.list();
      const s001 = records.filter((r) => r.subject === "S001");
      expect(s001.length).toBe(14);
      expect(s001.slice(0, 2).every((r) => r.task === "baseline")).toBe(true);
      expect(s001.slice(2).every((r) => r.task === "motor-imagery")).toBe(true);
    });

    it("generates correct URLs with leading zeros", async () => {
      const records = await physionet.list();
      const first = records[0];
      expect(first.url).toBe("https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf");
    });

    it("load() calls fetcher and parses EDF", async () => {
      const mockResponse = {
        ok: true,
        arrayBuffer: async () => new ArrayBuffer(16),
      };
      const fetcher = vi.fn().mockResolvedValue(mockResponse);
      await physionet.load(
        {
          id: "S001R01",
          subject: "S001",
          session: "run-01",
          url: "https://example.com/test.edf",
          format: "edf",
        },
        fetcher,
      );
      expect(fetcher).toHaveBeenCalledWith("https://example.com/test.edf");
    });

    it("load() throws on HTTP error", async () => {
      const mockResponse = {
        ok: false,
        status: 404,
        statusText: "Not Found",
      };
      const fetcher = vi.fn().mockResolvedValue(mockResponse);
      await expect(
        physionet.load(
          {
            id: "test",
            subject: "S001",
            url: "https://example.com/missing.edf",
            format: "edf",
          },
          fetcher,
        ),
      ).rejects.toThrow("404");
    });
  });

  describe("bciCompetitionIV2a", () => {
    it("has the correct name", () => {
      const loader = bciCompetitionIV2a(BCI_MIRROR);
      expect(loader.name).toBe("bci-competition-iv-2a");
    });

    it("lists records for all 9 subjects × 2 phases = 18 records", async () => {
      const loader = bciCompetitionIV2a(BCI_MIRROR);
      const records = await loader.list();
      expect(records).toHaveLength(18);
      const subjects = new Set(records.map((r) => r.subject));
      expect(subjects.size).toBe(9);
    });

    it("generates correct URLs", async () => {
      const loader = bciCompetitionIV2a(BCI_MIRROR);
      const records = await loader.list();
      const first = records[0];
      expect(first.url).toBe(`${BCI_MIRROR}/A01T.edf`);
    });

    it("identifies all records as EDF format", async () => {
      const loader = bciCompetitionIV2a(BCI_MIRROR);
      const records = await loader.list();
      expect(records.every((r) => r.format === "edf")).toBe(true);
    });

    it("throws when mirrorBase is empty", () => {
      expect(() => bciCompetitionIV2a("")).toThrow("mirrorBase required");
    });

    it("strips trailing slash from mirrorBase", async () => {
      const loader = bciCompetitionIV2a(`${BCI_MIRROR}/`);
      const records = await loader.list();
      expect(records[0].url).toBe(`${BCI_MIRROR}/A01T.edf`);
    });
  });

  describe("tuhEeg", () => {
    it("returns empty list when no mirror configured", async () => {
      const loader = tuhEeg();
      const records = await loader.list();
      expect(records).toEqual([]);
    });

    it("returns empty list when index is empty", async () => {
      const loader = tuhEeg("https://mirror.example.com", []);
      const records = await loader.list();
      expect(records).toEqual([]);
    });

    it("maps index entries to records", async () => {
      const index = [{ subject: "001", session: "01", file: "001/01/001_01_01.edf" }];
      const loader = tuhEeg("https://mirror.example.com", index);
      const records = await loader.list();
      expect(records).toHaveLength(1);
      expect(records[0].id).toBe("001/01/001/01/001_01_01.edf");
      expect(records[0].url).toBe("https://mirror.example.com/001/01/001_01_01.edf");
    });

    it("throws on load when mirrorBase not configured", async () => {
      const loader = tuhEeg();
      await expect(
        loader.load(
          { id: "test", subject: "s", url: "https://example.com/test.edf", format: "edf" },
          vi.fn(),
        ),
      ).rejects.toThrow("mirrorBase not configured");
    });
  });
});
