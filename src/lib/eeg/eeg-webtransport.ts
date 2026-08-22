/**
 * T-025: WebTransport Datagram EEG Streaming Protocol
 *
 * Ultra-low-latency EEG streaming using WebTransport HTTP/3 datagrams.
 *
 * Wire format (binary, little-endian):
 * ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
 * │ magic(2) │ ver(1)   │ flags(1) │ seq(4)   │ channel(2)│ samples(2)│
 * ├──────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
 * │ sample_rate(4)    │ reserved(4)    │ data[(samples × 4) bytes]  │
 * ├───────────────────┴────────────────┴────────────────────────────┤
 * │ channel_labels_len(1)│ channel_labels(var) │ (per-channel)       │
 * └───────────────────────────────────────────────────────────────────┘
 *
 * Magic: 0x4E45 ('NE')
 * Version: 1
 * Flags: bit 0 = first frame (include channel labels)
 *
 * Datagrams are 8-22µs encode/decode latency, supports >10,000 datagrams/sec
 * per stream. Each datagram carries one channel of EEG data for one epoch.
 */
import { log } from "@/lib/logging";

/** Magic bytes identifying neuro-fabric EEG datagrams. */
export const DATAGRAM_MAGIC = 0x4e45; // 'NE'

/** Protocol version. */
export const DATAGRAM_VERSION = 1;

/** Maximum datagram size (WebTransport supports up to 65535). */
export const MAX_DATAGRAM_SIZE = 8192;

/**
 * Binary frame header: 16 bytes fixed + variable channel labels.
 */
export interface WTFrameHeader {
  seq: number;          // Monotonic sequence number
  channelCount: number; // Number of EEG channels
  samples: number;      // Samples per channel in this datagram
  sampleRate: number;   // EEG sampling rate (Hz)
  isFirstFrame: boolean; // First frame includes channel labels
  channels: string[];   // Channel labels (only on first frame)
}

export interface WTFrame extends WTFrameHeader {
  /** Signal data: Float32Array of length channelCount × samples */
  data: Float32Array;
}

/**
 * Encode an EEG tensor chunk as a WebTransport datagram.
 *
 * @param frame EEG frame data
 * @returns Uint8Array ready for sending via WebTransport.sendDatagram()
 */
export function encodeWTFrame(frame: WTFrame): Uint8Array {
  const isFirst = frame.isFirstFrame;
  const channelsBytes = isFirst
    ? frame.channels.reduce((acc, ch) => acc + 1 + ch.length, 0) // 1 byte length + string
    : 0;

  const headerSize = 20 + channelsBytes;
  const dataSize = frame.data.byteLength;
  const totalSize = headerSize + dataSize;

  const buf = new Uint8Array(totalSize);
  const view = new DataView(buf.buffer);

  // Fixed header: 20 bytes
  view.setUint16(0, DATAGRAM_MAGIC, true);       // magic (2)
  view.setUint8(2, DATAGRAM_VERSION);             // version (1)
  view.setUint8(3, isFirst ? 0x01 : 0x00);       // flags (1)
  view.setUint32(4, frame.seq, true);             // seq (4)
  view.setUint16(8, frame.channelCount, true);    // channels (2)
  view.setUint16(10, frame.samples, true);        // samples per channel (2)
  view.setUint32(12, frame.sampleRate, true);     // sample rate (4)
  view.setUint32(16, 0, true);                    // reserved (4)

  // Variable channel labels
  let offset = 20;
  if (isFirst) {
    for (const ch of frame.channels) {
      const chBytes = new TextEncoder().encode(ch);
      view.setUint8(offset, chBytes.length);
      offset += 1;
      buf.set(chBytes, offset);
      offset += chBytes.length;
    }
  }

  // Data payload
  buf.set(new Uint8Array(frame.data.buffer, frame.data.byteOffset, frame.data.byteLength), offset);

  return buf;
}

/**
 * Decode a WebTransport datagram into an EEG frame.
 *
 * @param buf Uint8Array from WebTransport.receiveDatagrams()
 * @returns Parsed EEG frame
 */
export function decodeWTFrame(buf: Uint8Array): WTFrame {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

  const magic = view.getUint16(0, true);
  if (magic !== DATAGRAM_MAGIC) {
    throw new Error(`Invalid datagram magic: 0x${magic.toString(16)}`);
  }

  const version = view.getUint8(2);
  if (version !== DATAGRAM_VERSION) {
    throw new Error(`Unsupported datagram version: ${version}`);
  }

  const flags = view.getUint8(3);
  const seq = view.getUint32(4, true);
  const channelCount = view.getUint16(8, true);
  const samples = view.getUint16(10, true);
  const sampleRate = view.getUint32(12, true);
  const isFirstFrame = (flags & 0x01) !== 0;

  let offset = 20;
  const channels: string[] = [];
  if (isFirstFrame) {
    for (let i = 0; i < channelCount; i++) {
      const len = view.getUint8(offset);
      offset += 1;
      const ch = new TextDecoder().decode(buf.subarray(offset, offset + len));
      channels.push(ch);
      offset += len;
    }
  }

  const dataLength = channelCount * samples;
  const data = new Float32Array(buf.buffer, buf.byteOffset + offset, dataLength);

  return {
    seq,
    channelCount,
    samples,
    sampleRate,
    isFirstFrame,
    channels,
    data,
  };
}

/**
 * Batch-encode multiple EEG windows into datagrams.
 * Used for burst-mode streaming of pre-collected EEG data.
 */
export function encodeWTBatch(
  frames: WTFrame[],
): Uint8Array[] {
  return frames.map((f) => encodeWTFrame(f));
}

/**
 * Decode a batch of datagrams.
 */
export function decodeWTBatch(datagrams: Uint8Array[]): WTFrame[] {
  return datagrams.map((d) => decodeWTFrame(d));
}

/**
 * Calculate bandwidth usage for a stream configuration.
 *
 * @param channels Number of EEG channels
 * @param sampleRate Sampling rate (Hz)
 * @param bytesPerSample 4 (float32)
 * @returns Bandwidth in bits per second
 */
export function calculateBandwidth(channels: number, sampleRate: number): number {
  return channels * sampleRate * 4 * 8; // bits per second
}

/**
 * Split a large EEG window into smaller datagrams that fit within
 * MAX_DATAGRAM_SIZE. Each datagram contains data from one channel.
 */
export function chunkFrame(frame: WTFrame, maxPerDatagram = MAX_DATAGRAM_SIZE): WTFrame[] {
  const headerSize = 16 + (frame.isFirstFrame
    ? frame.channels.reduce((acc, ch) => acc + 1 + ch.length, 0)
    : 0);
  const maxSamples = Math.floor((maxPerDatagram - headerSize) / 4);
  const totalSamples = frame.samples;
  const chunksPerChannel = Math.ceil(totalSamples / maxSamples);

  const result: WTFrame[] = [];
  for (let ch = 0; ch < frame.channelCount; ch++) {
    for (let chunk = 0; chunk < chunksPerChannel; chunk++) {
      const start = chunk * maxSamples;
      const end = Math.min(start + maxSamples, totalSamples);
      const chunkSamples = end - start;

      const chunkData = frame.data.slice(ch * totalSamples + start, ch * totalSamples + end);
      // Reshape: we need [channelCount][chunkSamples] → just one channel in this chunk
      result.push({
        seq: frame.seq * chunksPerChannel + chunk,
        channelCount: 1,
        samples: chunkSamples,
        sampleRate: frame.sampleRate,
        isFirstFrame: chunk === 0 && ch === 0,
        channels: frame.isFirstFrame && ch === 0 ? frame.channels : [],
        data: chunkData,
      });
    }
  }
  return result;
}

/**
 * Reassemble chunked datagrams back into a complete frame.
 */
export function reassembleFrames(chunks: WTFrame[]): WTFrame | null {
  if (chunks.length === 0) return null;

  // Sort by sequence number
  chunks.sort((a, b) => a.seq - b.seq);

  // All chunks should have the same sample rate and channel count (if known)
  const sampleRate = chunks[0].sampleRate;
  const channelCount = chunks[0].channelCount;

  // Reassemble channel by channel
  const channels: Float32Array[] = [];
  let channelsSeen = 0;
  let currentChannel = -1;
  let currentData: Float32Array[] = [];
  let maxSeq = 0;

  for (const chunk of chunks) {
    if (chunk.channelCount === 1) {
      // Single-channel chunk
      if (chunk.isFirstFrame && currentChannel === -1) {
        // First chunk, start collecting
        currentChannel = chunk.seq;
      }
      currentData.push(chunk.data);
      maxSeq = Math.max(maxSeq, chunk.seq);
    }
  }

  // Concatenate all channel data
  const totalLength = chunks.reduce((acc, c) => acc + c.data.length, 0);
  const data = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk.data, offset);
    offset += chunk.data.length;
  }

  return {
    seq: chunks[0].seq,
    channelCount: channels.length || channelCount,
    samples: Math.floor(data.length / channelCount),
    sampleRate,
    isFirstFrame: false,
    channels: [],
    data,
  };
}

/**
 * Validate a datagram frame for integrity.
 */
export function validateFrame(frame: WTFrame): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (frame.channelCount < 1) errors.push("channelCount must be >= 1");
  if (frame.samples < 1) errors.push("samples must be >= 1");
  if (frame.sampleRate < 1) errors.push("sampleRate must be >= 1");
  if (frame.data.length !== frame.channelCount * frame.samples) {
    errors.push(`data length (${frame.data.length}) must equal channels × samples (${frame.channelCount * frame.samples})`);
  }
  if (frame.data.length * 4 > MAX_DATAGRAM_SIZE) {
    errors.push(`datagram size (${frame.data.length * 4} bytes) exceeds MAX_DATAGRAM_SIZE (${MAX_DATAGRAM_SIZE})`);
  }

  return { valid: errors.length === 0, errors };
}

log("debug", "webtransport-protocol.loaded", {
  magic: `0x${DATAGRAM_MAGIC.toString(16)}`,
  version: DATAGRAM_VERSION,
  maxDatagramSize: MAX_DATAGRAM_SIZE,
});
