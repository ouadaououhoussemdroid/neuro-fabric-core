import { useEffect, useState } from "react";
import { Activity, Cpu, GitBranch, Radio, Sparkles, Waves } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, ResponsiveContainer } from "recharts";
import { useTelemetry } from "@/hooks/use-telemetry";

function useHistory(value: number, len = 28) {
  const [data, setData] = useState<{ x: number; v: number }[]>(() =>
    Array.from({ length: len }, (_, i) => ({ x: i, v: 0 })),
  );
  useEffect(() => {
    if (value === 0) return;
    setData((prev) => {
      const next = prev.slice(1);
      next.push({ x: (prev[prev.length - 1]?.x ?? 0) + 1, v: value });
      return next;
    });
  }, [value]);
  return data;
}

export function LiveDot({ className = "" }: { className?: string }) {
  return (
    <span className={`relative inline-flex h-2 w-2 ${className}`}>
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-neuro opacity-60" />
      <span className="relative inline-flex h-2 w-2 rounded-full bg-neuro" />
    </span>
  );
}

function Widget({
  icon: Icon,
  label,
  value,
  sub,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="glass relative overflow-hidden rounded-xl p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md border border-border bg-muted/40">
            <Icon className="h-3.5 w-3.5 text-neuro" />
          </div>
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {label}
          </span>
        </div>
        <LiveDot />
      </div>
      <div className="mt-3 flex items-end justify-between gap-2">
        <div>
          <div className="font-mono text-xl font-semibold tabular-nums tracking-tight">{value}</div>
          {sub && <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sub}</div>}
        </div>
        {children && <div className="h-10 w-24">{children}</div>}
      </div>
    </div>
  );
}

export function LiveOpsBand() {
  const t = useTelemetry();

  const latSeries = useHistory(t.latencyMs);
  const gpuSeries = useHistory(t.gpuUtil / 100);
  const tpsSeries = useHistory(t.throughputTps);

  return (
    <div className="mx-auto max-w-7xl px-4">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <LiveDot />
          {t.isLive ? "Live · neuro-core-7 · us-east" : "Connecting…"}
        </div>
        <div className="hidden items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground md:flex">
          <GitBranch className="h-3 w-3 text-neuro" /> nwf-7b-embed · v3.4.1 · canary 12%
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Widget
          icon={Activity}
          label="p50 latency"
          value={t.latencyMs > 0 ? `${t.latencyMs} ms` : "—"}
          sub="rolling 60s"
        >
          <ResponsiveContainer>
            <AreaChart data={latSeries}>
              <defs>
                <linearGradient id="lat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0.7} />
                  <stop offset="100%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="v"
                stroke="oklch(0.85 0.18 195)"
                strokeWidth={1}
                fill="url(#lat)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Widget>

        <Widget
          icon={Cpu}
          label="GPU H100 util"
          value={t.gpuUtil > 0 ? `${t.gpuUtil}%` : "—"}
          sub="8× cluster · neuro-core-7"
        >
          <ResponsiveContainer>
            <BarChart data={gpuSeries}>
              <Bar
                dataKey="v"
                fill="oklch(0.7 0.22 295)"
                isAnimationActive={false}
                radius={[1, 1, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Widget>

        <Widget
          icon={Radio}
          label="Throughput"
          value={t.throughputTps > 0 ? `${t.throughputTps} tps` : "—"}
          sub="inference / sec"
        >
          <ResponsiveContainer>
            <AreaChart data={tpsSeries}>
              <defs>
                <linearGradient id="tps" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.82 0.18 170)" stopOpacity={0.7} />
                  <stop offset="100%" stopColor="oklch(0.82 0.18 170)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="v"
                stroke="oklch(0.82 0.18 170)"
                strokeWidth={1}
                fill="url(#tps)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Widget>

        <Widget
          icon={Waves}
          label="Active sessions"
          value={t.activeSessions > 0 ? t.activeSessions.toLocaleString() : "—"}
          sub="streaming embed channels"
        />

        <Widget
          icon={Sparkles}
          label="Synthetic samples"
          value={t.syntheticSamples > 0 ? t.syntheticSamples.toLocaleString() : "—"}
          sub="generated · all time"
        />

        <Widget
          icon={Activity}
          label="API requests"
          value={t.apiRequests > 0 ? t.apiRequests.toLocaleString() : "—"}
          sub="total · today"
        />
      </div>
    </div>
  );
}

export function StreamingLatent({
  cols = 64,
  rows = 4,
  speed = 110,
}: {
  cols?: number;
  rows?: number;
  speed?: number;
}) {
  const [grid, setGrid] = useState<number[][]>(() =>
    Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.random() * 2 - 1)),
  );
  useEffect(() => {
    const id = setInterval(() => {
      setGrid((g) =>
        g.map((row) => {
          const shifted = row.slice(1);
          shifted.push(Math.random() * 2 - 1);
          return shifted;
        }),
      );
    }, speed);
    return () => clearInterval(id);
  }, [cols, rows, speed]);
  return (
    <div className="space-y-[2px]">
      {grid.map((row, r) => (
        <div
          key={r}
          className="grid gap-[2px]"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}
        >
          {row.map((v, i) => (
            <div
              key={i}
              className="h-3 rounded-[2px] transition-colors"
              style={{
                background:
                  v >= 0
                    ? `oklch(0.78 0.16 200 / ${0.15 + Math.abs(v) * 0.8})`
                    : `oklch(0.7 0.22 295 / ${0.15 + Math.abs(v) * 0.8})`,
              }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function StreamingJson({ text, speed = 6 }: { text: string; speed?: number }) {
  const [shown, setShown] = useState("");
  useEffect(() => {
    setShown("");
    if (!text) return;
    let i = 0;
    const id = setInterval(() => {
      i += speed;
      setShown(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, 16);
    return () => clearInterval(id);
  }, [text, speed]);
  return (
    <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap text-[12px] leading-relaxed text-muted-foreground">
      {shown}
      {shown.length < text.length && (
        <span className="ml-0.5 inline-block h-3 w-1.5 -mb-0.5 animate-pulse bg-neuro" />
      )}
    </pre>
  );
}

function useTick(ms = 1500) {
  const [n, setN] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setN((v) => v + 1), ms);
    return () => clearInterval(id);
  }, [ms]);
  return n;
}

export function InferenceStages({ active }: { active: boolean }) {
  const stages = ["ingest", "preprocess", "tokenize", "encode", "project", "respond"];
  const tick = useTick(380);
  const cur = active ? tick % (stages.length + 2) : -1;
  return (
    <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider">
      {stages.map((s, i) => {
        const done = active && i < cur;
        const now = active && i === cur;
        return (
          <span
            key={s}
            className={`rounded border px-1.5 py-0.5 transition-colors ${
              now
                ? "border-neuro/60 bg-neuro/15 text-foreground"
                : done
                  ? "border-border bg-muted/30 text-muted-foreground"
                  : "border-border/60 text-muted-foreground/60"
            }`}
          >
            {s}
          </span>
        );
      })}
    </div>
  );
}
