/**
 * T-025 — Tests for WebTransport datagram EEG protocol.
 *
 * Verifies binary frame encoding/decoding, bandwidth calculations,
 * chunking/reassembly, and frame validation.
 */
import { describe, it, expect } from "vitest";
import {
  encodeWTFrame,
  decodeWTFrame,
  encodeWTBatch,
  decodeWTBatch,
  calculateBandwidth,
  chunkFrame,
  reassembleFrames,
  validateFrame,
  DATAGRAM_MAGIC,
  DATAGRAM_VERSION,
  MAX_DATAGRAM_SIZE,
} from "../eeg-webtransport";
import type { WTFrame } from "../eeg-webtransport";

function makeTestFrame(seq: number, channels: number, samples: number): WTFrame {
  return {
    seq,
    channelCount: channels,
    samples,
    sampleRate: 250,
    isFirstFrame: seq === 0,
    channels: seq === 0 ? Array.from({ length: channels }, (_, i) => `ch${i}`) : [],
    data: Float32Array.from({ length: channels * samples }, (_, i) => Math.sin(i * 0.1) * 50),
  };
}

describe("eeg-webtransport", () => {
  describe("encode / decode round-trip", () => {
    it("encodes and decodes a single-channel frame correctly", () => {
      const frame = makeTestFrame(0, 1, 100);
      const encoded = encodeWTFrame(frame);
      expect(encoded).toBeInstanceOf(Uint8Array);
      expect(encoded.length).toBeGreaterThan(16); // header only minimum

      const decoded = decodeWTFrame(encoded);
      expect(decoded.seq).toBe(0);
      expect(decoded.channelCount).toBe(1);
      expect(decoded.samples).toBe(100);
      expect(decoded.sampleRate).toBe(250);
      expect(decoded.isFirstFrame).toBe(true);
      expect(decoded.channels).toEqual(["ch0"]);
      expect(decoded.data.length).toBe(100);
    });

    it("encodes and decodes multi-channel frame with channel labels", () => {
      const frame = makeTestFrame(0, 22, 1000);
      const encoded = encodeWTFrame(frame);
      expect(encoded.length).toBeGreaterThan(0);

      const decoded = decodeWTFrame(encoded);
      expect(decoded.channelCount).toBe(22);
      expect(decoded.samples).toBe(1000);
      expect(decoded.channels).toHaveLength(22);
      expect(decoded.channels[0]).toBe("ch0");
      expect(decoded.channels[21]).toBe("ch21");
      expect(decoded.data.length).toBe(22 * 1000);
    });

    it("preserves data values through round-trip", () => {
      const frame = makeTestFrame(5, 4, 500);
      frame.isFirstFrame = false; // subsequent frame, no channel labels
      const encoded = encodeWTFrame(frame);
      const decoded = decodeWTFrame(encoded);

      // Verify first few values match
      for (let i = 0; i < 10; i++) {
        expect(decoded.data[i]).toBeCloseTo(frame.data[i], 5);
      }
    });

    it("throws on invalid magic bytes", () => {
      const badBuf = new Uint8Array(20);
      badBuf[0] = 0; // wrong magic
      badBuf[1] = 0;
      expect(() => decodeWTFrame(badBuf)).toThrow(/Invalid datagram magic/);
    });

    it("throws on unsupported version", () => {
      const buf = new Uint8Array(20);
      const view = new DataView(buf.buffer);
      view.setUint16(0, DATAGRAM_MAGIC, true);
      view.setUint8(2, 99); // wrong version
      expect(() => decodeWTFrame(buf)).toThrow(/Unsupported datagram version/);
    });
  });

  describe("batch encoding/decoding", () => {
    it("encodes and decodes a batch of frames", () => {
      const frames = [
        makeTestFrame(0, 22, 500),
        makeTestFrame(1, 22, 500),
        makeTestFrame(2, 22, 500),
      ];
      frames[1].isFirstFrame = false;
      frames[2].isFirstFrame = false;

      const datagrams = encodeWTBatch(frames);
      expect(datagrams).toHaveLength(3);

      const decoded = decodeWTBatch(datagrams);
      expect(decoded).toHaveLength(3);
      expect(decoded[0].seq).toBe(0);
      expect(decoded[1].seq).toBe(1);
      expect(decoded[2].seq).toBe(2);
    });
  });

  describe("bandwidth calculation", () => {
    it("calculates correct bandwidth for 22-channel EEG", () => {
      // 22 channels × 250 Hz × 4 bytes × 8 bits = 176,000 bps
      const bw = calculateBandwidth(22, 250);
      expect(bw).toBe(22 * 250 * 4 * 8);
    });

    it("scales linearly with channel count", () => {
      const bw1 = calculateBandwidth(1, 250);
      const bw22 = calculateBandwidth(22, 250);
      expect(bw22 / bw1).toBe(22);
    });
  });

  describe("chunking and reassembly", () => {
    it("chunks large frames into valid datagrams", () => {
      const frame = makeTestFrame(0, 22, 2000); // 22 × 2000 × 4 = 176,000 bytes — exceeds max
      const chunks = chunkFrame(frame);

      expect(chunks.length).toBeGreaterThan(1);

      // Each chunk should be valid
      for (const chunk of chunks) {
        const validation = validateFrame(chunk);
        expect(validation.valid).toBe(true);
      }
    });

    it("reassembles chunked frames correctly", () => {
      const frame = makeTestFrame(0, 22, 500);
      const chunks = chunkFrame(frame);
      const reassembled = reassembleFrames(chunks);

      expect(reassembled).not.toBeNull();
      expect(reassembled!.data.length).toBe(frame.data.length);
    });
  });

  describe("frame validation", () => {
    it("validates correct frames", () => {
      const frame = makeTestFrame(0, 4, 100);
      const result = validateFrame(frame);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it("rejects invalid channel count", () => {
      const frame = makeTestFrame(0, 0, 100);
      frame.channelCount = 0;
      const result = validateFrame(frame);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain("channelCount must be >= 1");
    });

    it("rejects oversized datagrams", () => {
      const frame = makeTestFrame(0, 22, 10000);
      const result = validateFrame(frame);
      expect(result.valid).toBe(false);
    });
  });

  describe("wire format constants", () => {
    it("uses correct magic bytes", () => {
      expect(DATAGRAM_MAGIC).toBe(0x4e45); // 'NE'
    });

    it("uses version 1", () => {
      expect(DATAGRAM_VERSION).toBe(1);
    });

    it("maximum datagram size is 8192", () => {
      expect(MAX_DATAGRAM_SIZE).toBe(8192);
    });
  });
});
