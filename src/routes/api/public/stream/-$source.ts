/**
 * T-003 — WebSocket EEG gateway route.
 *
 * Route: /api/public/stream/:source
 *
 * Clients connect with `new WebSocket("ws://host/api/public/stream/file:rec-1")`.
 * The gateway fans out {@link AcquisitionSource} chunks as JSON frames with
 * per-peer sequence numbers and a `model_id` header (see stream-gateway.ts).
 *
 * The gateway is a singleton shared across connections; sources are
 * registered by the operator (file uploads, BrainFlow boards, LSL streams)
 * and clients subscribe to a named source via the path segment.
 */
import { defineWebSocketHandler } from "nitro";
import { StreamGateway, parseSourceId, type EEGStreamFrame } from "@/lib/eeg/stream-gateway";
import { log } from "@/lib/logging";

// Singleton gateway instance. Sources are registered elsewhere (e.g. when a
// file is uploaded or a board connection is opened). Lives for the lifetime
// of the server process.
const gateway = new StreamGateway({ defaultModelId: "eegconformer-v1" });

// Expose the gateway so other modules (upload route, future BrainFlow/LSL
// adapters) can register sources.
export { gateway };

export default defineWebSocketHandler({
  open(peer) {
    const sourceId = parseSourceId(extractSource(peer));
    log("info", "ws.gateway.peer_open", { peerId: peer.id, sourceId });

    if (!gateway.has(sourceId)) {
      peer.send(JSON.stringify({ error: `unknown source: ${sourceId}`, status: 404 }));
      peer.close(1008, "unknown source");
      return;
    }

    // Track this peer's active pump so we can stop it on close.
    let stopped = false;
    peer.context.stop = () => {
      stopped = true;
    };

    void gateway
      .pump(
        sourceId,
        peer.id,
        (frame: EEGStreamFrame) => {
          peer.send(JSON.stringify(frame));
        },
        () => stopped,
      )
      .then(() => {
        log("info", "ws.gateway.stream_end", { peerId: peer.id, sourceId });
        peer.close(1000, "stream complete");
      })
      .catch((err: unknown) => {
        log("error", "ws.gateway.pump_failed", {
          peerId: peer.id,
          sourceId,
          error: (err as Error).message,
        });
        peer.close(1011, "stream error");
      });
  },

  message(peer, message) {
    // Clients can send a simple "ping" keepalive; respond with a pong.
    const text = message.text();
    if (text === "ping") {
      peer.send("pong");
    }
  },

  close(peer, details) {
    const stop = peer.context.stop as (() => void) | undefined;
    stop?.();
    gateway.resetPeer(peer.id);
    log("info", "ws.gateway.peer_close", {
      peerId: peer.id,
      code: details?.code,
      reason: details?.reason,
    });
  },

  error(peer, error) {
    log("error", "ws.gateway.peer_error", {
      peerId: peer.id,
      error: (error as Error).message,
    });
  },
});

/**
 * Extract the `source` path segment from the upgrade request URL.
 * Route is `/api/public/stream/:source`, so the last path component is the id.
 */
function extractSource(peer: { request?: Request }): string | undefined {
  const url = peer.request?.url;
  if (!url) return undefined;
  const parts = new URL(url).pathname.split("/").filter(Boolean);
  return parts[parts.length - 1];
}
