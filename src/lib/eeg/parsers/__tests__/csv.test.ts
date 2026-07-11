import { describe, it, expect } from "vitest";
import { parseCSV } from "../csv";

describe("parseCSV", () => {
  it("throws if sampleRate is missing or invalid", () => {
    expect(() => parseCSV("1,2\n3,4", 0)).toThrow("sampleRate is required");
    expect(() => parseCSV("1,2\n3,4", NaN)).toThrow("sampleRate is required");
  });

  it("throws on an empty file", () => {
    expect(() => parseCSV("", 128)).toThrow("file is empty");
  });

  it("auto-generates channel names when the first row is numeric (no header)", () => {
    const signal = parseCSV("1,2,3\n4,5,6", 128);
    expect(signal.channels).toEqual(["ch0", "ch1", "ch2"]);
    // no header -> all rows are data
    expect(signal.data[0]).toEqual([1, 4]);
    expect(signal.data[1]).toEqual([2, 5]);
    expect(signal.data[2]).toEqual([3, 6]);
  });

  it("uses the first row as channel labels when it's non-numeric", () => {
    const signal = parseCSV("Fz,Cz,Pz\n1,2,3\n4,5,6", 128);
    expect(signal.channels).toEqual(["Fz", "Cz", "Pz"]);
    expect(signal.data[0]).toEqual([1, 4]);
  });

  it("throws if there are no data rows after the header", () => {
    expect(() => parseCSV("Fz,Cz", 128)).toThrow("no data rows after header");
  });

  it("supports comma, semicolon, and tab delimiters", () => {
    expect(parseCSV("1,2\n3,4", 128).data[0]).toEqual([1, 3]);
    expect(parseCSV("1;2\n3;4", 128).data[0]).toEqual([1, 3]);
    expect(parseCSV("1\t2\n3\t4", 128).data[0]).toEqual([1, 3]);
  });

  it("forward-fills a non-finite value from the previous sample in that channel", () => {
    const signal = parseCSV("1,10\n2,20\nNaN,30\n4,40", 128);
    // channel 0: [1, 2, NaN->forward-filled to 2, 4]
    expect(signal.data[0]).toEqual([1, 2, 2, 4]);
    expect(signal.meta?.nan_count).toBe(1);
    expect(signal.meta?.warnings).toHaveLength(1);
    expect((signal.meta?.warnings as string[])[0]).toContain("row 3, ch ch0");
  });

  it("forward-fills to 0 when the non-finite value has no valid prior sample", () => {
    // A leading "NaN" cell in the very first row would be misread as a text
    // header (the header heuristic treats isNaN(Number(cell)) as "text"),
    // so use an explicit header row to keep "NaN" as the first *data* row.
    // Padded with clean rows to stay under the whole-file 20% threshold.
    const signal = parseCSV("ch0,ch1\nNaN,10\n2,20\n3,30\n4,40\n5,50", 128);
    expect(signal.data[0]).toEqual([0, 2, 3, 4, 5]);
  });

  it("throws when more than 20% of all values are non-finite (whole-file threshold)", () => {
    // 3 channels x 10 rows = 30 values; make 7 of them (>20%) non-finite
    const rows: string[] = [];
    for (let r = 0; r < 10; r++) {
      const cells = [String(r), String(r + 1), r < 7 ? "NaN" : String(r + 2)];
      rows.push(cells.join(","));
    }
    expect(() => parseCSV(rows.join("\n"), 128)).toThrow(/non-finite/);
  });

  it("throws when a single channel exceeds the 50% per-channel threshold even if the whole file is under 20%", () => {
    // 8 channels x 10 rows = 80 values. Channel 0 fully NaN (10/80 = 12.5%,
    // under the whole-file 20% cutoff) but 100% for that one channel.
    const rows: string[] = [];
    for (let r = 0; r < 10; r++) {
      const cells = ["NaN"];
      for (let c = 1; c < 8; c++) cells.push(String(r + c));
      rows.push(cells.join(","));
    }
    expect(() => parseCSV(rows.join("\n"), 128)).toThrow(/channel\(s\) exceed 50% non-finite/);
  });

  it("does not throw when non-finite values are spread thinly across channels", () => {
    // 8 channels x 10 rows, 5 NaNs spread one-per-channel across 5 different
    // channels (well under 20% file-wide and under 50% for any one channel)
    const rows: string[] = [];
    for (let r = 0; r < 10; r++) {
      const cells: string[] = [];
      for (let c = 0; c < 8; c++) cells.push(r === c ? "NaN" : String(r + c));
      rows.push(cells.join(","));
    }
    expect(() => parseCSV(rows.join("\n"), 128)).not.toThrow();
  });

  it("reports zero warnings and nan_count for a clean file", () => {
    const signal = parseCSV("1,2\n3,4\n5,6", 128);
    expect(signal.meta?.nan_count).toBe(0);
    expect(signal.meta?.nan_percent).toBe(0);
    expect(signal.meta?.warnings).toEqual([]);
  });

  it("sets sampleRate to the value passed in", () => {
    const signal = parseCSV("1,2\n3,4", 250);
    expect(signal.sampleRate).toBe(250);
  });
});
