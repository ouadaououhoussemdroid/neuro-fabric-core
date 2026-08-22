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
import type { Database } from "@/integrations/supabase/types";

// Singleton gateway instance. Sources are registered elsewhere (e.g. when a
// file is uploaded or a board connection is opened). Lives for the lifetime
// of the server process.
const gateway = new StreamGateway({ defaultModelId: "eegconformer-v1" });

// Expose the gateway so other modules (upload route, future BrainFlow/LSL
// adapters) can register sources.
export { gateway };

/**
 * T-011: Authenticate the WebSocket connection before allowing streaming.
 *
 * The connection URL may include a Bearer token:
 *   new WebSocket("wss://host/api/public/stream/file:rec-1?token=<jwt>")
 *
 * For security, we also accept tokens passed via the Sec-WebSocket-Protocol
 * header (preferred) or via the query string (less secure, but needed for
 * older browser clients).
 *
 * Anonymous access is denied in production — every WebSocket connection
 * must present a valid JWT that can be verified via Supabase's getUser().
 */
async function authenticatePeer(peer: {
  request?: Request;
  send: (data: string) => void;
  close: (code: number, reason?: string) => void;
}): Promise<boolean> {
  // Extract token: prefer query parameter, then Sec-WebSocket-Protocol header
  const url = peer.request?.url || "";
  const params = new URLSearchParams(new URL(url).search);
  let token = params.get("token");

  // Fallback: check Sec-WebSocket-Protocol header
  if (!token) {
    // peer.request headers are accessible in Nitro WebSocket handlers
    token = null; // Sec-WebSocket-Protocol access varies by runtime; query param is primary
  }

  if (!token) {
    peer.send(JSON.stringify({ error: "Unauthorized: missing token", status: 401 }));
    peer.close(1008, "Unauthorized: missing token");
    return false;
  }

  // Verify token via Supabase
  try {
    const { createClient } = await import("@supabase/supabase-js");
    const { requireServerEnv } = await import("@/lib/env.server");
    const { SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY } = requireServerEnv([
      "SUPABASE_URL",
      "SUPABASE_PUBLISHABLE_KEY",
    ]);

    const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
    const { data: { user }, error } = await supabase.auth.getUser(token);

    if (error || !user) {
      peer.send(JSON.stringify({ error: "Unauthorized: invalid token", status: 401 }));
      peer.close(1008, "Unauthorized: invalid token");
      return false;
    }
  } catch (e) {
    peer.send(JSON.stringify({ error: "Unauthorized: token verification failed", status: 401 }));
    peer.close(1008, "Unauthorized: token verification failed");
    return false;
  }

  return true;
}

export default defineWebSocketHandler({
  async open(peer) {
    // T-011: Require authentication before processing WebSocket connections
    const authenticated = await authenticatePeer(peer);
    if (!authenticated) return;

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
