import { describe, it, expect } from "vitest";
import { EEGPTAdapter } from "../eegpt-adapter";
import { NotImplementedError, type ModelInput } from "../../types";

const adapter = new EEGPTAdapter();
const dummyInput: ModelInput = {
  kind: "windows",
  windows: [{ data: [[1, 2, 3]], sampleRate: 250, start: 0, end: 3 }],
};

describe("T-016 EEGPT honest stub", () => {
  it("declares implemented: false in its descriptor", () => {
    expect(adapter.descriptor.capabilities.implemented).toBe(false);
  });

  it("marks itself as experimental", () => {
    expect(adapter.descriptor.isExperimental).toBe(true);
  });

  it("descriptor name includes 'Scheduled' (not 'Planned' or 'Available')", () => {
    expect(adapter.descriptor.name).toContain("Scheduled");
  });

  it("descriptor description references T-016 and the unblock conditions", () => {
    expect(adapter.descriptor.description).toContain("T-016");
    expect(adapter.descriptor.description).toContain("license");
  });

  it("isLoaded() returns false", () => {
    expect(adapter.isLoaded()).toBe(false);
  });

  it("load() throws NotImplementedError", async () => {
    await expect(adapter.load()).rejects.toThrow(NotImplementedError);
  });

  it("embed() throws NotImplementedError", async () => {
    await expect(adapter.embed(dummyInput)).rejects.toThrow(NotImplementedError);
  });

  it("predict() throws NotImplementedError", async () => {
    await expect(adapter.predict(dummyInput)).rejects.toThrow(NotImplementedError);
  });

  it("unload() is a no-op (does not throw)", async () => {
    await expect(adapter.unload()).resolves.toBeUndefined();
  });
});
