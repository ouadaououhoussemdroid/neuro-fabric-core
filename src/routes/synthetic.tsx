import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { Area, AreaChart, Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { Download } from "lucide-react";

export const Route = createFileRoute("/synthetic")({
  head: () => ({ meta: [
    { title: "Synthetic Neurodata Lab — NeuroWeave" },
    { name: "description", content: "Generate condition-controlled synthetic EEG datasets." },
    { property: "og:title", content: "Synthetic Neurodata Lab — NeuroWeave" },
    { property: "og:description", content: "Generate consented, label-rich synthetic EEG." },
  ]}),
  component: SyntheticPage,
});

function SyntheticPage() {
  const [attention, setAttention] = useState(0.7);
  const [stress, setStress] = useState(0.2);
  const [workload, setWorkload] = useState(0.4);
  const [category, setCategory] = useState("faces");

  const signal = useMemo(() => {
    const fAlpha = 1.5 - attention;
    const noise = 0.15 + stress * 0.4;
    const amp = 0.5 + workload * 0.4;
    return Array.from({ length: 160 }, (_, i) => ({
      t: i,
      v: Math.sin(i / fAlpha) * amp + Math.sin(i / 2.1) * 0.25 + (Math.random() - 0.5) * noise,
    }));
  }, [attention, stress, workload]);

  const bands = useMemo(() => [
    { band: "δ", power: 0.2 + Math.random() * 0.1 },
    { band: "θ", power: 0.3 + stress * 0.3 },
    { band: "α", power: 0.4 + attention * 0.4 },
    { band: "β", power: 0.3 + workload * 0.4 },
    { band: "γ", power: 0.15 + workload * 0.2 },
  ], [attention, stress, workload]);

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Synthetic Neurodata Lab"
          title="Generate label-rich EEG on demand."
          sub="Condition synthetic signals on attention, stress, cognitive load, and visual category — then export as Parquet or BIDS."
        />

        <div className="grid gap-4 lg:grid-cols-[1fr_2fr]">
          <GlassCard>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Conditions</div>
            <div className="mt-4 space-y-5">
              <Slider label="attention" value={attention} onChange={setAttention} />
              <Slider label="stress" value={stress} onChange={setStress} />
              <Slider label="cognitive load" value={workload} onChange={setWorkload} />
              <div>
                <div className="text-xs text-muted-foreground">visual category</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {["faces", "scenes", "objects", "animals", "text"].map((c) => (
                    <button
                      key={c}
                      onClick={() => setCategory(c)}
                      className={`rounded-md border px-2.5 py-1 text-[11px] ${
                        category === c ? "border-neuro/60 bg-neuro/10" : "border-border text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {c}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <button className="mt-6 inline-flex w-full items-center justify-center gap-2 rounded-md bg-neuro-gradient px-4 py-2 text-xs font-medium text-background glow">
              <Download className="h-3.5 w-3.5" /> Export dataset (Parquet)
            </button>
            <button className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-md border border-border bg-card/40 px-4 py-2 text-xs hover:bg-card">
              Export BIDS bundle (.zip)
            </button>
          </GlassCard>

          <div className="space-y-4">
            <GlassCard>
              <div className="flex items-center justify-between">
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Synthesized signal · Cz</div>
                <div className="font-mono text-[10px] text-muted-foreground">cat={category} · n=1024</div>
              </div>
              <div className="mt-3 h-44">
                <ResponsiveContainer>
                  <AreaChart data={signal}>
                    <defs>
                      <linearGradient id="syn" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area type="monotone" dataKey="v" stroke="oklch(0.85 0.18 195)" strokeWidth={1.2} fill="url(#syn)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>

            <GlassCard>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Band power</div>
              <div className="mt-3 h-40">
                <ResponsiveContainer>
                  <BarChart data={bands}>
                    <XAxis dataKey="band" stroke="oklch(0.6 0.02 260)" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis hide />
                    <Bar dataKey="power" fill="oklch(0.78 0.16 200)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </div>
        </div>
      </Section>
    </SiteShell>
  );
}

function Slider({ label, value, onChange }: { label: string; value: number; onChange: (n: number) => void }) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs"><span className="text-muted-foreground">{label}</span><span className="font-mono text-neuro">{value.toFixed(2)}</span></div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-2 w-full accent-[oklch(0.78_0.16_200)]"
      />
    </div>
  );
}