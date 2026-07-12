/**
 * T-003 — WebSocket EEG gateway logic.
 *
 * Decoupled from the Nitro `defineWebSocketHandler` route so the fan-out,
 * sequencing, and model-tagging logic is testable in isolation.
 *
 * A {@link StreamGateway} owns:
 *   - a registry of named {@link AcquisitionSource} adapters (e.g. "file:abc",
 *     "brainflow:cyton-1", "lsl:open");
 *   - per-peer sequence counters so each client receives monotonically
 *     increasing `seq` numbers per stream;
 *   - a `model_id` header stamped on every frame so downstream consumers
 *     know which embedding model the stream should be paired with.
 */
import type { AcquisitionSource } from "./acquisition";
import type { EEGSignal } from "./types";
import { log } from "@/lib/logging";

export interface EEGStreamFrame {
  /** Monotonic per-peer sequence number, starting at 0. */
  seq: number;
  /** Embedding model id this stream is associated with. */
  model_id: string;
  /** Source identifier this frame originated from. */
  source: string;
  /** Channel labels for this stream (echoed on first frame / on change). */
  channels: string[];
  /** Sample rate of this stream. */
  sampleRate: number;
  /** Signal payload, shape [C][N]. */
  data: number[][];
  /** Server-side timestamp (ISO 8601) when the frame was emitted. */
  ts: string;
}

export interface StreamGatewayOptions {
  /** Default embedding model id stamped onto frames. */
  defaultModelId?: string;
}

export class StreamGateway {
  private sources = new Map<string, AcquisitionSource>();
  private peerSeq = new Map<string, number>();
  private readonly defaultModelId: string;

  constructor(opts: StreamGatewayOptions = {}) {
    this.defaultModelId = opts.defaultModelId ?? "eegconformer-v1";
  }

  /** Register a named acquisition source so clients can subscribe to it. */
  register(sourceId: string, source: AcquisitionSource): void {
    this.sources.set(sourceId, source);
    log("info", "ws.gateway.source_registered", { sourceId, type: source.type });
  }

  /** Remove a previously registered source. */
  unregister(sourceId: string): void {
    const s = this.sources.get(sourceId);
    if (s) {
      void s.close?.();
      this.sources.delete(sourceId);
      log("info", "ws.gateway.source_unregistered", { sourceId });
    }
  }

  has(sourceId: string): boolean {
    return this.sources.has(sourceId);
  }

  /**
   * Stream all chunks from `sourceId` to a callback. Each frame is stamped
   * with a per-peer sequence number and the model id. Returns when the
   * source's async iterable completes or the `shouldStop` predicate returns
   * true (e.g. when the peer disconnects).
   */
  async pump(
    sourceId: string,
    peerId: string,
    send: (frame: EEGStreamFrame) => void,
    shouldStop: () => boolean = () => false,
  ): Promise<void> {
    const source = this.sources.get(sourceId);
    if (!source) {
      throw new GatewayError(`unknown source: ${sourceId}`, 404);
    }

    let seq = this.peerSeq.get(peerId) ?? 0;
    let firstFrame = true;

    for await (const chunk of source.stream()) {
      if (shouldStop()) break;

      const frame: EEGStreamFrame = {
        seq,
        model_id: this.defaultModelId,
        source: sourceId,
        channels: firstFrame ? chunk.channels : [],
        sampleRate: chunk.sampleRate,
        data: chunk.data,
        ts: new Date().toISOString(),
      };
      send(frame);
      seq++;
      this.peerSeq.set(peerId, seq);
      firstFrame = false;
    }
  }

  /** Reset the sequence counter for a disconnecting peer. */
  resetPeer(peerId: string): void {
    this.peerSeq.delete(peerId);
  }

  /** Next sequence number for a peer without pumping (for testing). */
  peekSeq(peerId: string): number {
    return this.peerSeq.get(peerId) ?? 0;
  }
}

export class GatewayError extends Error {
  status: number;
  constructor(message: string, status = 400) {
    super(message);
    this.status = status;
  }
}

/**
 * Parse a source id from the route path segment. Source ids are opaque
 * strings registered by the operator (e.g. "file:rec-123", "brainflow:cyton").
 * We only validate non-emptiness here; resolution happens via {@link StreamGateway}.
 */
export function parseSourceId(raw: string | undefined): string {
  if (!raw || raw.length === 0) {
    throw new GatewayError("missing source parameter", 400);
  }
  return raw;
}

/** Serialise a frame to the wire format (JSON string). */
export function encodeFrame(frame: EEGStreamFrame): string {
  return JSON.stringify(frame);
}

/** Type guard for a parsed incoming client message. */
export interface ClientMessage {
  type: "subscribe" | "unsubscribe";
  source?: string;
}

export function parseClientMessage(raw: string): ClientMessage | null {
  try {
    const obj = JSON.parse(raw) as unknown;
    if (typeof obj !== "object" || obj === null) return null;
    const t = (obj as Record<string, unknown>).type;
    if (t === "subscribe" || t === "unsubscribe") {
      const src = (obj as Record<string, unknown>).source;
      return { type: t, source: typeof src === "string" ? src : undefined };
    }
    return null;
  } catch {
    return null;
  }
}

/** Convenience: build a gateway pre-populated with file sources from a map. */
export function createGateway(
  entries: Array<[string, AcquisitionSource]>,
  opts?: StreamGatewayOptions,
): StreamGateway {
  const gw = new StreamGateway(opts);
  for (const [id, src] of entries) gw.register(id, src);
  return gw;
}

// Re-export EEGSignal type for route consumers.
export type { EEGSignal };
