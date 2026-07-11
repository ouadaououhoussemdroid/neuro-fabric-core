import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";

export interface TelemetryState {
  latencyMs: number;
  throughputTps: number;
  activeSessions: number;
  apiRequests: number;
  isLive: boolean;
}

const REFRESH_MS = 8_000;

function p50(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * 0.5)] ?? 0;
}

export function useTelemetry(): TelemetryState {
  const [state, setState] = useState<TelemetryState>({
    latencyMs: 0,
    throughputTps: 0,
    activeSessions: 0,
    apiRequests: 0,
    isLive: false,
  });

  const presenceChannel = useRef<ReturnType<typeof supabase.channel> | null>(null);
  const sessionCount = useRef(0);

  useEffect(() => {
    const channel = supabase.channel("telemetry-presence", {
      config: { presence: { key: crypto.randomUUID() } },
    });

    channel
      .on("presence", { event: "sync" }, () => {
        const state = channel.presenceState();
        sessionCount.current = Object.keys(state).length;
        setState((prev) => ({ ...prev, activeSessions: sessionCount.current }));
      })
      .subscribe(async (status) => {
        if (status === "SUBSCRIBED") {
          await channel.track({ joined_at: Date.now() });
        }
      });

    presenceChannel.current = channel;
    return () => {
      channel.unsubscribe();
    };
  }, []);

  useEffect(() => {
    async function fetchMetrics() {
      const now = new Date();
      const startOfDay = new Date(now);
      startOfDay.setHours(0, 0, 0, 0);
      const sixtySecondsAgo = new Date(now.getTime() - 60_000);

      const { data: latencyRows } = await supabase
        .from("eeg_analyses")
        .select("processing_time_ms")
        .order("created_at", { ascending: false })
        .limit(50);

      const latencyValues = (latencyRows ?? [])
        .map((r) => r.processing_time_ms)
        .filter((v): v is number => typeof v === "number" && v > 0);

      const latencyMs = p50(latencyValues);

      const { count: tpsCount } = await supabase
        .from("eeg_analyses")
        .select("id", { count: "exact", head: true })
        .gte("created_at", sixtySecondsAgo.toISOString());

      const throughputTps = tpsCount ?? 0;

      const { count: todayCount } = await supabase
        .from("eeg_analyses")
        .select("id", { count: "exact", head: true })
        .gte("created_at", startOfDay.toISOString());

      const apiRequests = todayCount ?? 0;

      setState((prev) => ({
        ...prev,
        latencyMs: latencyMs > 0 ? +latencyMs.toFixed(1) : prev.latencyMs,
        throughputTps,
        apiRequests,
        isLive: true,
      }));
    }

    fetchMetrics();
    const id = setInterval(fetchMetrics, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  return state;
}
