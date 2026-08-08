import { describe, it, expect } from "vitest";
import { sleepEDF } from "../sleep-edf";

describe("sleepEDF loader", () => {
  it("should have the correct name", () => {
    expect(sleepEDF.name).toBe("sleep-edf-sc");
  });

  it("should list 101 records (100 subjects, subject 000 has 2 nights)", async () => {
    const records = await sleepEDF.list();
    expect(records.length).toBe(101);
  });

  it("should give subject S000 two nights (sessions)", async () => {
    const records = await sleepEDF.list();
    const subject0Records = records.filter((r) => r.subject === "S0000");
    expect(subject0Records.length).toBe(2);
    expect(subject0Records[0].session).toBe("night-1");
    expect(subject0Records[1].session).toBe("night-2");
  });

  it("should give other subjects one night each", async () => {
    const records = await sleepEDF.list();
    const otherSubjects = records.filter((r) => r.subject !== "S0000");
    // 99 subjects × 1 night = 99 records
    expect(otherSubjects.length).toBe(99);
    expect(otherSubjects.every((r) => r.session === "night-1")).toBe(true);
  });

  it("should generate correct PhysioNet URLs", async () => {
    const records = await sleepEDF.list();
    const subject1Records = records.filter((r) => r.subject === "S0001");
    expect(subject1Records.length).toBe(1);
    expect(subject1Records[0].url).toBe(
      "https://physionet.org/files/sleep-edf/1.0.0/SC/S0001/S0001-n1.edf",
    );
  });

  it("should identify all records as EDF format", async () => {
    const records = await sleepEDF.list();
    expect(records.every((r) => r.format === "edf")).toBe(true);
  });

  it("should set sampleRate to 100 Hz for all records", async () => {
    const records = await sleepEDF.list();
    expect(records.every((r) => r.sampleRate === 100)).toBe(true);
  });

  it("should tag all records with sleep-staging task", async () => {
    const records = await sleepEDF.list();
    expect(records.every((r) => r.task === "sleep-staging")).toBe(true);
  });

  it("should produce unique ids for all records", async () => {
    const records = await sleepEDF.list();
    const ids = records.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
