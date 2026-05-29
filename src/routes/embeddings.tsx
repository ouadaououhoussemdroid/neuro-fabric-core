import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section, StatPill } from "@/components/ui-bits";
import { Search, Sparkles, Layers, Activity, Zap, Brain, Orbit } from "lucide-react";
import { Area, AreaChart, ResponsiveContainer, RadialBar, RadialBarChart, PolarAngleAxis } from "recharts";

export const Route = createFileRoute("/embeddings")({
  head: () => ({ meta: [
    { title: "Embeddings Explorer — NeuroWeave" },
    { name: "description", content: "Interactive 3D latent space, brain-state clusters, semantic search, and real-time embedding generation." },
    { property: "og:title", content: "Embeddings Explorer — NeuroWeave" },
    { property: "og:description", content: "Zoomable 3D latent map, vector similarity metrics, and brain-state transitions." },
  ]}),
  component: EmbeddingsPage,
});

const CLUSTERS = [
  { id: "attention.high", hue: 200, label: "Attention · High" },
  { id: "rest.eyes-closed", hue: 170, label: "Rest · Eyes Closed" },
  { id: "visual.faces", hue: 295, label: "Visual · Faces" },
  { id: "visual.scenes", hue: 320, label: "Visual · Scenes" },
  { id: "motor.left", hue: 220, label: "Motor · Left" },
  { id: "motor.right", hue: 260, label: "Motor · Right" },
  { id: "language.semantic", hue: 50, label: "Language · Semantic" },
  { id: "affect.stress", hue: 12, label: "Affect · Stress" },
];

const SUBJECTS = ["subj_0421/run_03", "subj_0118/run_07", "subj_0712/run_01", "subj_0099/run_12", "subj_0322/run_02", "subj_0501/run_05", "subj_0884/run_09", "subj_0277/run_04"];

type Pt3 = { x: number; y: number; z: number; c: number; label: string; sim: number };

function generatePoints(n: number): Pt3[] {
  // Cluster centers in a unit cube
  const centers = CLUSTERS.map((_, i) => {
    const a = (i / CLUSTERS.length) * Math.PI * 2;
    return {
      x: Math.cos(a) * 0.55,
      y: Math.sin(a * 1.3) * 0.45,
      z: Math.sin(a) * 0.55,
    };
  });
  return Array.from({ length: n }, (_, i) => {
    const c = i % CLUSTERS.length;
    const center = centers[c];
    const r = Math.pow(Math.random(), 0.6) * 0.22;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    return {
      x: center.x + r * Math.sin(phi) * Math.cos(theta),
      y: center.y + r * Math.sin(phi) * Math.sin(theta),
      z: center.z + r * Math.cos(phi),
      c,
      label: SUBJECTS[i % SUBJECTS.length],
      sim: 0.6 + Math.random() * 0.4,
    };
  });
}

function LatentSpace3D({
  points,
  activeCluster,
  zoom,
  onHover,
}: {
  points: Pt3[];
  activeCluster: number | null;
  zoom: number;
  onHover: (i: number | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ on: boolean; lx: number; ly: number; rx: number; ry: number }>({ on: false, lx: 0, ly: 0, rx: 0.5, ry: 0.6 });
  const autoRef = useRef(true);
  const stateRef = useRef({ rx: 0.5, ry: 0.6 });
  const hoverRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (autoRef.current) {
        stateRef.current.ry += 0.0025;
      } else {
        stateRef.current.rx += (dragRef.current.rx - stateRef.current.rx) * 0.15;
        stateRef.current.ry += (dragRef.current.ry - stateRef.current.ry) * 0.15;
      }
      const { rx, ry } = stateRef.current;
      const cosX = Math.cos(rx), sinX = Math.sin(rx);
      const cosY = Math.cos(ry), sinY = Math.sin(ry);

      const cx = W / 2;
      const cy = H / 2;
      const scale = Math.min(W, H) * 0.42 * zoom;

      // grid floor
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      for (let i = -5; i <= 5; i++) {
        const t = i / 5;
        const project = (x: number, z: number) => {
          const y = 0.6;
          const x1 = x * cosY - z * sinY;
          const z1 = x * sinY + z * cosY;
          const y1 = y * cosX - z1 * sinX;
          const z2 = y * sinX + z1 * cosX;
          const persp = 1 / (1 + z2 * 0.6);
          return { px: cx + x1 * scale * persp, py: cy + y1 * scale * persp };
        };
        const a = project(t, -1), b = project(t, 1);
        const a2 = project(-1, t), b2 = project(1, t);
        ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(a2.px, a2.py); ctx.lineTo(b2.px, b2.py); ctx.stroke();
      }

      // project points
      const projected = points.map((p, i) => {
        const x1 = p.x * cosY - p.z * sinY;
        const z1 = p.x * sinY + p.z * cosY;
        const y1 = p.y * cosX - z1 * sinX;
        const z2 = p.y * sinX + z1 * cosX;
        const persp = 1 / (1 + z2 * 0.6);
        return {
          i,
          px: cx + x1 * scale * persp,
          py: cy + y1 * scale * persp,
          depth: z2,
          size: (2.2 + (1 - z2) * 2.2) * dpr,
          c: p.c,
        };
      });
      projected.sort((a, b) => b.depth - a.depth);

      const hueFor = (c: number) => CLUSTERS[c].hue;
      for (const p of projected) {
        const dim = activeCluster !== null && activeCluster !== p.c;
        const alpha = dim ? 0.12 : 0.55 + (1 - p.depth) * 0.4;
        const hue = hueFor(p.c);
        // glow
        const grd = ctx.createRadialGradient(p.px, p.py, 0, p.px, p.py, p.size * 5);
        grd.addColorStop(0, `oklch(0.85 0.2 ${hue} / ${alpha * 0.5})`);
        grd.addColorStop(1, `oklch(0.85 0.2 ${hue} / 0)`);
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(p.px, p.py, p.size * 5, 0, Math.PI * 2); ctx.fill();
        // core
        ctx.fillStyle = `oklch(0.88 0.18 ${hue} / ${alpha})`;
        ctx.beginPath(); ctx.arc(p.px, p.py, p.size, 0, Math.PI * 2); ctx.fill();
      }

      // animated flow particles between two cluster centers
      const t = performance.now() / 1000;
      for (let k = 0; k < 18; k++) {
        const seed = k * 0.37;
        const u = ((t * 0.25 + seed) % 1);
        const aIdx = k % CLUSTERS.length;
        const bIdx = (k + 3) % CLUSTERS.length;
        const a = points.find((p) => p.c === aIdx)!;
        const b = points.find((p) => p.c === bIdx)!;
        const x = a.x + (b.x - a.x) * u;
        const y = a.y + (b.y - a.y) * u;
        const z = a.z + (b.z - a.z) * u;
        const x1 = x * cosY - z * sinY;
        const z1 = x * sinY + z * cosY;
        const y1 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        const persp = 1 / (1 + z2 * 0.6);
        const px = cx + x1 * scale * persp;
        const py = cy + y1 * scale * persp;
        ctx.fillStyle = `oklch(0.95 0.16 200 / ${0.6 * (1 - Math.abs(u - 0.5) * 2)})`;
        ctx.beginPath(); ctx.arc(px, py, 1.6 * dpr, 0, Math.PI * 2); ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    const onDown = (e: PointerEvent) => {
      autoRef.current = false;
      dragRef.current.on = true;
      dragRef.current.lx = e.clientX;
      dragRef.current.ly = e.clientY;
      (e.target as Element).setPointerCapture?.(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * dpr;
      const my = (e.clientY - rect.top) * dpr;
      // hover detect
      let nearest: number | null = null;
      let best = 18 * dpr;
      // we need projected positions; recompute lightweight
      const { rx, ry } = stateRef.current;
      const cosX = Math.cos(rx), sinX = Math.sin(rx);
      const cosY = Math.cos(ry), sinY = Math.sin(ry);
      const cx = canvas.width / 2, cy = canvas.height / 2;
      const scale = Math.min(canvas.width, canvas.height) * 0.42 * zoom;
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const x1 = p.x * cosY - p.z * sinY;
        const z1 = p.x * sinY + p.z * cosY;
        const y1 = p.y * cosX - z1 * sinX;
        const z2 = p.y * sinX + z1 * cosX;
        const persp = 1 / (1 + z2 * 0.6);
        const px = cx + x1 * scale * persp;
        const py = cy + y1 * scale * persp;
        const d = Math.hypot(px - mx, py - my);
        if (d < best) { best = d; nearest = i; }
      }
      if (nearest !== hoverRef.current) {
        hoverRef.current = nearest;
        onHover(nearest);
      }

      if (dragRef.current.on) {
        const dx = e.clientX - dragRef.current.lx;
        const dy = e.clientY - dragRef.current.ly;
        dragRef.current.ry += dx * 0.01;
        dragRef.current.rx += dy * 0.01;
        dragRef.current.lx = e.clientX;
        dragRef.current.ly = e.clientY;
      }
    };
    const onUp = () => { dragRef.current.on = false; };
    const onLeave = () => { if (hoverRef.current !== null) { hoverRef.current = null; onHover(null); } };

    canvas.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [points, activeCluster, zoom, onHover]);

  return <canvas ref={canvasRef} className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing" />;
}

function EmbeddingsPage() {
  const points = useMemo(() => generatePoints(420), []);
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [query, setQuery] = useState("attention.high");
  const [genTick, setGenTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setGenTick((t) => t + 1), 1400);
    return () => clearInterval(id);
  }, []);

  const ranked = useMemo(
    () => [...points]
      .map((p) => ({ ...p, sim: p.sim * (activeCluster === null || p.c === activeCluster ? 1 : 0.45) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 8),
    [points, activeCluster],
  );

  const dimSpectrum = useMemo(
    () => Array.from({ length: 48 }, (_, i) => ({
      d: i,
      v: 0.3 + 0.7 * Math.abs(Math.sin(i * 0.4 + genTick * 0.3)) * (1 - i / 64),
    })),
    [genTick],
  );

  const cogState = [
    { name: "attention", value: 72, fill: "oklch(0.78 0.16 200)" },
    { name: "arousal", value: 48, fill: "oklch(0.7 0.22 295)" },
    { name: "workload", value: 61, fill: "oklch(0.82 0.18 170)" },
    { name: "valence", value: 38, fill: "oklch(0.75 0.2 50)" },
  ];

  const flowChart = useMemo(
    () => Array.from({ length: 40 }, (_, i) => ({
      t: i,
      v: 0.5 + 0.4 * Math.sin(i * 0.4 + genTick * 0.6) + Math.random() * 0.08,
    })),
    [genTick],
  );

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Neuro Embeddings Explorer · v2"
          title="A foundation-model view of the brain's latent space."
          sub="Project 1024-d brain-signal embeddings into an interactive 3D manifold. Search by semantic similarity, overlay cognitive states, and watch new embeddings stream in real time."
        />

        <div className="mb-6 grid gap-3 md:grid-cols-4">
          <StatPill label="Total vectors" value="62.3M" />
          <StatPill label="Latent dims" value="1,024" />
          <StatPill label="ANN p99" value="9.4 ms" />
          <StatPill label="Brain-states" value="184" />
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
          {/* Latent space */}
          <GlassCard className="p-0 overflow-hidden">
            <div className="flex flex-wrap items-center gap-3 border-b border-border/60 px-5 py-3">
              <div className="flex flex-1 items-center gap-2 rounded-md border border-border bg-background/40 px-3 py-1.5">
                <Search className="h-3.5 w-3.5 text-muted-foreground" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="similarity_search('attention.high', k=8)"
                  className="w-full bg-transparent text-xs outline-none placeholder:text-muted-foreground/70 font-mono"
                />
                <span className="font-mono text-[10px] text-neuro">cosine</span>
              </div>
              <div className="flex items-center gap-1.5 rounded-md border border-border bg-background/40 px-2 py-1.5">
                <button onClick={() => setZoom((z) => Math.max(0.6, z - 0.15))} className="px-1.5 font-mono text-xs text-muted-foreground hover:text-foreground">−</button>
                <span className="font-mono text-[10px] text-muted-foreground">{zoom.toFixed(2)}×</span>
                <button onClick={() => setZoom((z) => Math.min(2.4, z + 0.15))} className="px-1.5 font-mono text-xs text-muted-foreground hover:text-foreground">+</button>
              </div>
              <span className="font-mono text-[10px] uppercase text-muted-foreground">umap · n=420 · 3D</span>
            </div>

            <div className="relative aspect-[16/11] grid-bg">
              <LatentSpace3D points={points} activeCluster={activeCluster} zoom={zoom} onHover={setHover} />

              {/* HUD top-left */}
              <div className="pointer-events-none absolute left-4 top-4 space-y-1.5">
                <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-background/60 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                  <span className="h-1.5 w-1.5 rounded-full bg-neuro animate-pulse-glow" /> streaming embeddings · 412/s
                </div>
                <div className="font-mono text-[10px] text-muted-foreground">drag to rotate · scroll buttons to zoom</div>
              </div>

              {/* HUD hover label */}
              {hover !== null && (
                <div className="pointer-events-none absolute right-4 top-4 rounded-md border border-border bg-background/80 px-3 py-2 backdrop-blur">
                  <div className="font-mono text-[10px] uppercase text-muted-foreground">vector</div>
                  <div className="font-mono text-xs">{points[hover].label}</div>
                  <div className="mt-1 flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: `oklch(0.85 0.2 ${CLUSTERS[points[hover].c].hue})` }} />
                    <span className="font-mono text-[10px] text-muted-foreground">{CLUSTERS[points[hover].c].id}</span>
                  </div>
                </div>
              )}

              {/* Cluster legend */}
              <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
                {CLUSTERS.map((c, i) => {
                  const active = activeCluster === i;
                  return (
                    <button
                      key={c.id}
                      onClick={() => setActiveCluster(active ? null : i)}
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px] transition ${
                        active ? "border-neuro/60 bg-neuro/10 text-foreground" : "border-border bg-background/60 text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: `oklch(0.85 0.2 ${c.hue})` }} />
                      {c.id}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Real-time embedding flow */}
            <div className="border-t border-border/60 px-5 py-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  <Activity className="h-3 w-3 text-neuro" /> Real-time embedding throughput
                </div>
                <span className="font-mono text-[10px] text-neuro">live</span>
              </div>
              <div className="h-20">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={flowChart} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="flowGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0.6} />
                        <stop offset="100%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="v" stroke="oklch(0.85 0.18 195)" strokeWidth={1.5} fill="url(#flowGrad)" isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </GlassCard>

          {/* Right column */}
          <div className="space-y-4">
            <GlassCard>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  <Sparkles className="h-3 w-3 text-neuro" /> Top matches · cosine
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">k=8</span>
              </div>
              <ul className="mt-3 divide-y divide-border/60">
                {ranked.map((r, i) => (
                  <li key={i} className="flex items-center justify-between gap-3 py-2 text-sm">
                    <span className="flex items-center gap-2 min-w-0">
                      <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: `oklch(0.85 0.2 ${CLUSTERS[r.c].hue})` }} />
                      <span className="font-mono text-xs text-muted-foreground truncate">{r.label}</span>
                    </span>
                    <span className="flex items-center gap-2">
                      <div className="h-1 w-16 overflow-hidden rounded-full bg-muted">
                        <div className="h-full bg-neuro-gradient" style={{ width: `${r.sim * 100}%` }} />
                      </div>
                      <span className="w-10 text-right font-mono text-xs text-neuro">{r.sim.toFixed(3)}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </GlassCard>

            <GlassCard>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  <Layers className="h-3 w-3 text-neuro" /> Dimensionality panel · top 48 / 1024
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">σ live</span>
              </div>
              <div className="mt-3 grid grid-cols-12 gap-1">
                {dimSpectrum.map((d) => (
                  <div key={d.d} className="h-10 overflow-hidden rounded-sm bg-muted/40">
                    <div
                      className="h-full w-full"
                      style={{
                        background: `linear-gradient(to top, oklch(0.78 0.16 200) ${d.v * 100}%, transparent ${d.v * 100}%)`,
                      }}
                    />
                  </div>
                ))}
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>d₀</span><span>d₂₄</span><span>d₄₇</span>
              </div>
            </GlassCard>

            <GlassCard>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  <Brain className="h-3 w-3 text-neuro" /> Cognitive state overlay
                </div>
                <span className="font-mono text-[10px] text-muted-foreground">decoded</span>
              </div>
              <div className="mt-2 grid grid-cols-[1fr_auto] items-center gap-3">
                <div className="space-y-2">
                  {cogState.map((c) => (
                    <div key={c.name}>
                      <div className="flex justify-between text-xs">
                        <span className="capitalize">{c.name}</span>
                        <span className="font-mono text-muted-foreground">{c.value}%</span>
                      </div>
                      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full" style={{ width: `${c.value}%`, background: c.fill }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="h-24 w-24">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart innerRadius="55%" outerRadius="100%" data={cogState} startAngle={90} endAngle={-270}>
                      <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                      <RadialBar background dataKey="value" cornerRadius={4} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>

        {/* Bottom: brain-state transitions */}
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <GlassCard className="lg:col-span-2">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                <Orbit className="h-3 w-3 text-neuro" /> Brain-state transition graph · last 5 min
              </div>
              <span className="font-mono text-[10px] text-muted-foreground">markov · stationary</span>
            </div>
            <div className="relative h-56">
              <svg viewBox="0 0 600 220" className="absolute inset-0 h-full w-full">
                <defs>
                  <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                    <path d="M0,0 L6,3 L0,6 z" fill="oklch(0.78 0.16 200 / 0.6)" />
                  </marker>
                </defs>
                {CLUSTERS.map((c, i) => {
                  const angle = (i / CLUSTERS.length) * Math.PI * 2 - Math.PI / 2;
                  const cx = 300 + Math.cos(angle) * 170;
                  const cy = 110 + Math.sin(angle) * 80;
                  return { c, i, cx, cy };
                }).map((n, _, all) => (
                  <g key={n.i}>
                    {all.filter((m) => m.i !== n.i && (n.i + m.i) % 3 === 0).map((m) => {
                      const w = 0.15 + ((n.i * 7 + m.i * 3) % 7) / 20;
                      return (
                        <line
                          key={`${n.i}-${m.i}`}
                          x1={n.cx} y1={n.cy} x2={m.cx} y2={m.cy}
                          stroke={`oklch(0.78 0.16 200 / ${w})`}
                          strokeWidth={w * 4}
                          markerEnd="url(#arr)"
                        />
                      );
                    })}
                  </g>
                ))}
                {CLUSTERS.map((c, i) => {
                  const angle = (i / CLUSTERS.length) * Math.PI * 2 - Math.PI / 2;
                  const cx = 300 + Math.cos(angle) * 170;
                  const cy = 110 + Math.sin(angle) * 80;
                  return (
                    <g key={c.id}>
                      <circle cx={cx} cy={cy} r={22} fill={`oklch(0.85 0.2 ${c.hue} / 0.18)`} stroke={`oklch(0.85 0.2 ${c.hue} / 0.7)`} />
                      <text x={cx} y={cy + 36} textAnchor="middle" fontSize="9" fontFamily="JetBrains Mono" fill="oklch(0.78 0.018 250)">{c.id}</text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </GlassCard>

          <GlassCard>
            <div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <Zap className="h-3 w-3 text-neuro" /> Foundation model · neuroweave-1024
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between"><span className="text-muted-foreground">Backbone</span><span className="font-mono text-xs">NeuroFormer-L</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Parameters</span><span className="font-mono text-xs">1.4 B</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Pretrain hours</span><span className="font-mono text-xs">84,200 h EEG</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Latent dim</span><span className="font-mono text-xs">1,024</span></div>
              <div className="flex justify-between"><span className="text-muted-foreground">Probe accuracy</span><span className="font-mono text-xs text-neuro">94.2%</span></div>
            </div>
            <div className="mt-4 rounded-md border border-border bg-background/40 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
              <span className="text-neuro">{"→"}</span> embed(signal) <span className="text-muted-foreground/60">// returns 1024-d</span><br />
              <span className="text-neuro">{"→"}</span> ann.search(v, k=8)<br />
              <span className="text-neuro">{"→"}</span> decode_state(v) <span className="text-muted-foreground/60">// → cognitive</span>
            </div>
          </GlassCard>
        </div>
      </Section>
    </SiteShell>
  );
}