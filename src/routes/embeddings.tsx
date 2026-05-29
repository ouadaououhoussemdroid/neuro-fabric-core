import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section, StatPill } from "@/components/ui-bits";
import { Search } from "lucide-react";

export const Route = createFileRoute("/embeddings")({
  head: () => ({ meta: [
    { title: "Embeddings Explorer — NeuroWeave" },
    { name: "description", content: "Search, cluster, and visualize neural latent representations." },
    { property: "og:title", content: "Embeddings Explorer — NeuroWeave" },
    { property: "og:description", content: "Explore the NeuroWeave latent space." },
  ]}),
  component: EmbeddingsPage,
});

type Pt = { x: number; y: number; cluster: number; label: string; sim: number };
const CLUSTERS = ["attention.high", "rest.eyes-closed", "visual.faces", "visual.scenes", "motor.left", "motor.right", "language", "stress"];
const LABELS = ["subj_0421/run_03", "subj_0118/run_07", "subj_0712/run_01", "subj_0099/run_12", "subj_0322/run_02", "subj_0501/run_05"];

function gen(n: number): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    const c = i % CLUSTERS.length;
    const cx = ((c % 4) + 0.5) * 0.25;
    const cy = (Math.floor(c / 4) + 0.5) * 0.5;
    return {
      x: cx + (Math.random() - 0.5) * 0.12,
      y: cy + (Math.random() - 0.5) * 0.18,
      cluster: c,
      label: LABELS[i % LABELS.length],
      sim: 0.6 + Math.random() * 0.4,
    };
  });
}

function EmbeddingsPage() {
  const points = useMemo(() => gen(220), []);
  const [active, setActive] = useState<number | null>(null);
  const [q, setQ] = useState("");

  const ranked = useMemo(() => [...points].sort((a, b) => b.sim - a.sim).slice(0, 8), [points]);

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Neuro Embeddings Explorer"
          title="Search and cluster the latent space."
          sub="Project 768-d brain-signal embeddings to 2-D with UMAP, search by similarity, and inspect neural representations."
        />

        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <GlassCard className="p-0">
            <div className="flex flex-wrap items-center gap-3 border-b border-border/60 px-5 py-3">
              <div className="flex flex-1 items-center gap-2 rounded-md border border-border bg-background/40 px-3 py-1.5">
                <Search className="h-3.5 w-3.5 text-muted-foreground" />
                <input
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="similarity_search('attention.high', k=8)"
                  className="w-full bg-transparent text-xs outline-none placeholder:text-muted-foreground/70"
                />
              </div>
              <span className="font-mono text-[10px] uppercase text-muted-foreground">umap · n=220</span>
            </div>
            <div className="relative aspect-[16/10] overflow-hidden">
              <svg viewBox="0 0 100 60" className="absolute inset-0 h-full w-full">
                <defs>
                  <radialGradient id="halo">
                    <stop offset="0%" stopColor="oklch(0.78 0.16 200)" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="oklch(0.78 0.16 200)" stopOpacity="0" />
                  </radialGradient>
                </defs>
                {points.map((p, i) => {
                  const cx = p.x * 100;
                  const cy = p.y * 60;
                  const hues = [200, 220, 260, 295, 320, 170, 50, 12];
                  const color = `oklch(0.78 0.18 ${hues[p.cluster]})`;
                  return (
                    <g key={i} onMouseEnter={() => setActive(i)} onMouseLeave={() => setActive(null)} style={{ cursor: "pointer" }}>
                      {active === i && <circle cx={cx} cy={cy} r={4} fill="url(#halo)" />}
                      <circle cx={cx} cy={cy} r={active === i ? 0.9 : 0.55} fill={color} opacity={0.9} />
                    </g>
                  );
                })}
              </svg>
              <div className="absolute bottom-3 left-3 flex flex-wrap gap-2">
                {CLUSTERS.map((c, i) => {
                  const hues = [200, 220, 260, 295, 320, 170, 50, 12];
                  return (
                    <span key={c} className="inline-flex items-center gap-1.5 rounded-full border border-border bg-background/60 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: `oklch(0.78 0.18 ${hues[i]})` }} />
                      {c}
                    </span>
                  );
                })}
              </div>
            </div>
          </GlassCard>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <StatPill label="Total vectors" value="62.3M" />
              <StatPill label="ANN p99" value="9.4 ms" />
              <StatPill label="Clusters" value="184" />
              <StatPill label="Workspaces" value="27" />
            </div>
            <GlassCard>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Top matches · cosine</div>
              <ul className="mt-3 divide-y divide-border/60">
                {ranked.map((r, i) => (
                  <li key={i} className="flex items-center justify-between py-2.5 text-sm">
                    <span className="font-mono text-xs text-muted-foreground">{r.label}</span>
                    <span className="font-mono text-xs text-neuro">{r.sim.toFixed(3)}</span>
                  </li>
                ))}
              </ul>
            </GlassCard>
            <GlassCard>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Brain-state vector</div>
              <div className="mt-3 space-y-2">
                {[
                  ["attention", 0.72],
                  ["arousal", 0.41],
                  ["workload", 0.58],
                  ["valence", 0.34],
                  ["stress", 0.22],
                ].map(([k, v]) => (
                  <div key={k as string}>
                    <div className="flex justify-between text-xs"><span>{k}</span><span className="font-mono text-muted-foreground">{(v as number).toFixed(2)}</span></div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
                      <div className="h-full bg-neuro-gradient" style={{ width: `${(v as number) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </GlassCard>
          </div>
        </div>
      </Section>
    </SiteShell>
  );
}