/**
 * T-004 — LSL source adapter (TypeScript side).
 *
 * The actual LSL→WS relay lives in `scripts/lsl_bridge.py` (a Python sidecar
 * using pylsl). This module provides the `AcquisitionSource` factory that the
 * gateway (T-003) registers when an LSL stream is available — it consumes
 * frames from the WS gateway itself, so the TS side just needs to decode the
 * wire format back into {@link EEGSignal} chunks.
 *
 * In the typical deployment, the bridge connects to the gateway as a *client*
 * and registers itself as a source. Downstream consumers (dashboards, the
 * embed pipeline) connect to the same gateway URL and receive frames. This
 * adapter is therefore mostly a thin decoder used in tests and by the
 * embedding pipeline when it needs to read LSL data directly.
 */
import type { EEGSignal } from "./types";
import type { AcquisitionSource } from "./acquisition";
import type { EEGStreamFrame } from "./stream-gateway";

/**
 * Decode a wire-format {@link EEGStreamFrame} (produced by the LSL bridge or
 * any gateway client) back into an {@link EEGSignal}.
 *
 * The first frame in a stream carries the channel labels; subsequent frames
 * omit them to save bandwidth, so the caller threads the labels forward.
 */
export function decodeFrame(frame: EEGStreamFrame, knownChannels?: string[]): EEGSignal {
  const channels = frame.channels.length > 0 ? frame.channels : (knownChannels ?? []);
  return {
    channels,
    data: frame.data,
    sampleRate: frame.sampleRate,
    meta: {
      source: frame.source,
      modelId: frame.model_id,
      seq: frame.seq,
      ts: frame.ts,
    },
  };
}

/**
 * Build an {@link AcquisitionSource} from an async iterable of raw JSON frame
 * strings (e.g. from a WebSocket). This is what a client of the gateway uses
 * to turn the stream back into `EEGSignal` chunks for the embedding pipeline.
 */
export function createLSLSource(
  frames: AsyncIterable<string>,
  sampleRate: number,
  channels: string[],
): AcquisitionSource {
  let knownChannels = channels;
  return {
    type: "lsl",
    metadata: { sampleRate, channels },
    async *stream() {
      for await (const raw of frames) {
        const frame = JSON.parse(raw) as EEGStreamFrame;
        if (frame.channels.length > 0) knownChannels = frame.channels;
        yield decodeFrame(frame, knownChannels);
      }
    },
  };
}
