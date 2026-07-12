"""T-004 — Lab Streaming Layer (LSL) bridge.

Tiny Python sidecar that relays EEG samples from one or more LSL outlets
to the Neuro-Fabric WebSocket EEG gateway (T-003). Ships as an optional
dependency for academic pilots using LSL-compatible hardware.

Usage:
    python scripts/lsl_bridge.py --gateway ws://localhost:3000/api/public/stream/lsl:open
    python scripts/lsl_bridge.py --gateway <url> --stream-type EEG --timeout 30

Architecture:
    LSL outlet → pylsl StreamInlet → chunk buffer → JSON frame → WS gateway

Frames follow the gateway's wire contract:
    {"seq": int, "model_id": str, "source": str,
     "channels": [str], "sampleRate": int, "data": [[float]], "ts": iso8601}

Dependencies (not in training/requirements.txt — install separately):
    pip install pylsl websocket-client
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

try:
    import numpy as np
except ImportError:
    print("[lsl-bridge] numpy is required: pip install numpy", file=sys.stderr)
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bridge LSL → Neuro-Fabric WS gateway")
    ap.add_argument(
        "--gateway",
        required=True,
        help="WebSocket URL of the gateway route "
        "(e.g. ws://localhost:3000/api/public/stream/lsl:open)",
    )
    ap.add_argument(
        "--stream-type",
        default="EEG",
        help="LSL stream type to resolve (default: EEG).",
    )
    ap.add_argument(
        "--stream-name",
        default=None,
        help="Optional LSL stream name to resolve (resolves any if omitted).",
    )
    ap.add_argument(
        "--max-streams",
        type=int,
        default=1,
        help="Maximum number of LSL inlets to relay concurrently (default: 1).",
    )
    ap.add_argument(
        "--chunk-samples",
        type=float,
        default=0.5,
        help="Target chunk duration in seconds relayed per WS frame (default: 0.5).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for an LSL stream to appear (default: 5.0).",
    )
    ap.add_argument(
        "--model-id",
        default="eegconformer-v1",
        help="Embedding model id stamped onto each WS frame (default: eegconformer-v1).",
    )
    ap.add_argument(
        "--source-id",
        default=None,
        help="Source id for the gateway registration (default: lsl:<stream-name>).",
    )
    return ap.parse_args()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_inlets(
    stream_type: str,
    stream_name: str | None,
    max_streams: int,
    timeout: float,
) -> list[Any]:
    """Resolve LSL streams matching the type/name and return StreamInlet objects."""
    import pylsl

    print(f"[lsl-bridge] resolving LSL streams type={stream_type!r} "
          f"name={stream_name!r} timeout={timeout}s")
    streams = pylsl.resolve_byprop(
        "type", stream_type, timeout=timeout, minimum=1
    ) if stream_name is None else pylsl.resolve_byprop(
        "name", stream_name, timeout=timeout, minimum=1
    )
    if not streams:
        return []
    inlets: list[Any] = []
    for info in streams[:max_streams]:
        inlet = pylsl.StreamInlet(info)
        inlets.append(inlet)
        print(f"[lsl-bridge] connected to stream {info.name()!r} "
              f"({info.channel_count()}ch @ {info.nominal_srate()} Hz)")
    return inlets


def _relay_inlet(
    inlet: Any,
    gateway_url: str,
    model_id: str,
    source_id: str,
    chunk_seconds: float,
    stop_flag: dict[str, bool],
) -> None:
    """Relay a single LSL inlet to the WS gateway until stopped or EOF."""
    try:
        from websocket import create_connection
    except ImportError:
        print("[lsl-bridge] websocket-client is required: pip install websocket-client",
              file=sys.stderr)
        sys.exit(1)

    info = inlet.info()
    n_chans = info.channel_count()
    srate = info.nominal_srate()
    if srate <= 0:
        srate = 250  # irregular rate fallback
    chunk_samples = max(1, int(round(srate * chunk_seconds)))

    # Build channel labels from the LSL channel description.
    channels: list[str] = []
    ch_xml = info.desc().child("channels").child("channel")
    for _ in range(n_chans):
        label = ch_xml.child_value("label") or ""
        channels.append(label)
        ch_xml = ch_xml.next_sibling()

    print(f"[lsl-bridge] opening WS {gateway_url} (source={source_id})")
    ws = create_connection(gateway_url, timeout=10)

    seq = 0
    try:
        while not stop_flag.get("stop", False):
            chunk, timestamps = inlet.pull_chunk(
                max_samples=chunk_samples, timeout=chunk_seconds
            )
            if not chunk:
                continue
            data = np.asarray(chunk, dtype=np.float32).T.tolist()  # [C][N]
            frame = {
                "seq": seq,
                "model_id": model_id,
                "source": source_id,
                "channels": channels if seq == 0 else [],
                "sampleRate": int(srate),
                "data": data,
                "ts": _iso_now(),
            }
            ws.send(json.dumps(frame))
            seq += 1
    except Exception as e:
        print(f"[lsl-bridge] relay error on {source_id}: {e}", file=sys.stderr)
    finally:
        try:
            ws.close()
        except Exception:
            pass
        print(f"[lsl-bridge] relay stopped for {source_id} (sent {seq} frames)")


def main() -> None:
    args = _parse_args()
    inlets = _resolve_inlets(args.stream_type, args.stream_name, args.max_streams, args.timeout)
    if not inlets:
        print(f"[lsl-bridge] no LSL streams found within {args.timeout}s", file=sys.stderr)
        sys.exit(1)

    stop_flag: dict[str, bool] = {"stop": False}
    threads: list[Any] = []

    import threading

    for inlet in inlets:
        info = inlet.info()
        sid = args.source_id or f"lsl:{info.name()}"
        t = threading.Thread(
            target=_relay_inlet,
            args=(inlet, args.gateway, args.model_id, sid, args.chunk_samples, stop_flag),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"[lsl-bridge] relaying {len(threads)} stream(s). Press Ctrl+C to stop.")
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[lsl-bridge] shutting down...")
        stop_flag["stop"] = True
        for t in threads:
            t.join(timeout=5)
        print("[lsl-bridge] stopped.")


if __name__ == "__main__":
    main()
