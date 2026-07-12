import { describe, it, expect } from "vitest";
import { parseSaliencySidecar, normalizeSaliency } from "../saliency";

const validSidecar = {
  artefact_hash: "abc123",
  saliency_path: "training/artefacts/eegconformer-bciiv2a/saliency.npz",
  n_samples: 5,
  channels: 22,
  samples: 1000,
  channel_saliency: [
    0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.15, 0.55, 0.95, 0.35, 0.75, 0.25, 0.85, 0.45,
    0.65, 0.12, 0.52, 0.92, 0.32,
  ],
};

describe("parseSaliencySidecar", () => {
  it("parses a valid sidecar", () => {
    const s = parseSaliencySidecar(validSidecar);
    expect(s).not.toBeNull();
    expect(s!.artefactHash).toBe("abc123");
    expect(s!.channels).toBe(22);
    expect(s!.channelSaliency).toHaveLength(22);
  });

  it("returns null for invalid input", () => {
    expect(parseSaliencySidecar(null)).toBeNull();
    expect(parseSaliencySidecar("not an object")).toBeNull();
    expect(parseSaliencySidecar({})).toBeNull();
  });

  it("returns null when channel_saliency is missing", () => {
    expect(
      parseSaliencySidecar({
        artefact_hash: "a",
        saliency_path: "b",
        n_samples: 1,
        channels: 1,
        samples: 1,
      }),
    ).toBeNull();
  });
});

describe("normalizeSaliency", () => {
  it("normalizes values to [0, 1]", () => {
    const normalized = normalizeSaliency([1, 5, 10, 3]);
    expect(normalized[0]).toBeCloseTo(0, 5);
    expect(normalized[2]).toBeCloseTo(1, 5);
    for (const v of normalized) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("returns 0.5 for constant input", () => {
    const normalized = normalizeSaliency([3, 3, 3, 3]);
    expect(normalized).toEqual([0.5, 0.5, 0.5, 0.5]);
  });

  it("returns empty for empty input", () => {
    expect(normalizeSaliency([])).toEqual([]);
  });
});
