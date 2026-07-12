import { describe, it, expect } from "vitest";
import { hasModel, getDescriptor, listModels } from "../registry";

describe("T-015 Braindecode model zoo registration", () => {
  it("registers EEGConformer in the zoo", () => {
    expect(hasModel("braindecode-eegconformer-prod")).toBe(true);
    const d = getDescriptor("braindecode-eegconformer-prod");
    expect(d).toBeDefined();
    expect(d!.kind).toBe("braindecode");
  });

  it("registers EEGNetv4 in the zoo", () => {
    expect(hasModel("braindecode-eegnetv4-default")).toBe(true);
    const d = getDescriptor("braindecode-eegnetv4-default");
    expect(d).toBeDefined();
    expect(d!.capabilities.channels).toBe(22);
    expect(d!.capabilities.sampleRate).toBe(128);
    expect(d!.capabilities.windowSamples).toBe(256);
  });

  it("registers ShallowFBCSPNet in the zoo", () => {
    expect(hasModel("braindecode-shallowfbcspnet-default")).toBe(true);
    const d = getDescriptor("braindecode-shallowfbcspnet-default");
    expect(d).toBeDefined();
    expect(d!.capabilities.channels).toBe(22);
    expect(d!.capabilities.sampleRate).toBe(250);
    expect(d!.capabilities.windowSamples).toBe(1125);
  });

  it("registers Deep4Net in the zoo", () => {
    expect(hasModel("braindecode-deep4net-default")).toBe(true);
    const d = getDescriptor("braindecode-deep4net-default");
    expect(d).toBeDefined();
    expect(d!.capabilities.channels).toBe(22);
    expect(d!.capabilities.sampleRate).toBe(250);
    expect(d!.capabilities.windowSamples).toBe(1125);
  });

  it("all four Braindecode architectures appear in listModels()", () => {
    const ids = listModels().map((m) => m.id);
    expect(ids).toContain("braindecode-eegconformer-prod");
    expect(ids).toContain("braindecode-eegnetv4-default");
    expect(ids).toContain("braindecode-shallowfbcspnet-default");
    expect(ids).toContain("braindecode-deep4net-default");
  });

  it("PCA baseline is still registered alongside the zoo", () => {
    expect(hasModel("pca-legacy-v1")).toBe(true);
  });
});
