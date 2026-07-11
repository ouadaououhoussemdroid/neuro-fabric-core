import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { ReconstructionShowcase } from "@/components/recon-showcase";
import { Line, LineChart, ResponsiveContainer } from "recharts";
import { Sparkles } from "lucide-react";

export const Route = createFileRoute("/eeg2image")({
  head: () => ({
    meta: [
      { title: "EEG2Image — NeuroWeave" },
      {
        name: "description",
        content: "A concept demo of visual-cortex EEG-to-image reconstruction.",
      },
      { property: "og:title", content: "EEG2Image — NeuroWeave" },
      {
        property: "og:description",
        content: "NeuroWeave EEG-to-image reconstruction concept demo.",
      },
    ],
  }),
  component: EEG2ImagePage,
});

const SAMPLES = [
  {
    label: "subj_0421 · visual.faces",
    caption: "a young woman with red hair, frontal portrait",
    conf: 0.78,
    align: 0.84,
    hue: 12,
  },
  {
    label: "subj_0118 · visual.scenes",
    caption: "a snow-covered mountain range at dusk",
    conf: 0.71,
    align: 0.79,
    hue: 220,
  },
  {
    label: "subj_0712 · visual.animals",
    caption: "a golden retriever sitting on green grass",
    conf: 0.74,
    align: 0.81,
    hue: 70,
  },
];

function EEG2ImagePage() {
  const [idx, setIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [out, setOut] = useState<(typeof SAMPLES)[number] | null>(null);

  const trace = Array.from({ length: 120 }, (_, i) => ({
    t: i,
    v: Math.sin(i / 4) * 0.6 + Math.sin(i / 1.8) * 0.3 + (Math.random() - 0.5) * 0.2,
  }));

  const run = () => {
    setRunning(true);
    setOut(null);
    setTimeout(() => {
      setOut(SAMPLES[idx]);
      setRunning(false);
    }, 1100);
  };

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="EEG2Image Reconstruction · concept demo"
          title="Decode what the visual cortex sees."
          sub="A concept preview of visual-cortex EEG-to-image reconstruction. This page cycles through fixed sample outputs to illustrate the intended UX — it is not connected to a live image-generation model."
        />

        <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
          <GlassCard>
            <div className="flex flex-wrap items-center gap-2">
              {SAMPLES.map((s, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setIdx(i);
                    setOut(null);
                  }}
                  className={`rounded-md border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider ${
                    idx === i
                      ? "border-neuro/60 bg-neuro/10"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <div className="mt-4 h-40">
              <ResponsiveContainer>
                <LineChart data={trace}>
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="oklch(0.85 0.18 195)"
                    strokeWidth={1.1}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <button
              onClick={run}
              disabled={running}
              className="mt-4 inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-4 py-2 text-xs font-medium text-background glow disabled:opacity-60"
            >
              <Sparkles className="h-3.5 w-3.5" /> {running ? "Reconstructing…" : "Reconstruct"}
            </button>
          </GlassCard>

          <GlassCard>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Reconstructed output
            </div>
            <div className="mt-3 relative aspect-square overflow-hidden rounded-lg border border-border">
              <div
                className="absolute inset-0 transition-all"
                style={{
                  background: out
                    ? `radial-gradient(60% 60% at 50% 40%, oklch(0.78 0.2 ${out.hue} / 0.7), transparent 65%), radial-gradient(40% 40% at 70% 70%, oklch(0.7 0.22 295 / 0.5), transparent 70%), linear-gradient(135deg, oklch(0.2 0.04 260), oklch(0.16 0.02 260))`
                    : "linear-gradient(135deg, oklch(0.2 0.02 260), oklch(0.16 0.01 260))",
                  filter: running ? "blur(18px)" : "blur(0)",
                  opacity: out ? 1 : 0.5,
                }}
              />
              {!out && !running && (
                <div className="relative grid h-full place-items-center text-xs text-muted-foreground">
                  no output yet
                </div>
              )}
              {running && (
                <div className="absolute inset-0 grid place-items-center">
                  <div className="font-mono text-xs text-neuro animate-pulse">sampling latent…</div>
                </div>
              )}
            </div>
            {out && (
              <>
                <p className="mt-4 text-sm italic text-muted-foreground">"{out.caption}"</p>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <Metric label="Confidence" value={out.conf} />
                  <Metric label="Latent alignment" value={out.align} />
                </div>
              </>
            )}
          </GlassCard>
        </div>

        <div className="mt-20">
          <ReconstructionShowcase />
        </div>
      </Section>
    </SiteShell>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-neuro">{value.toFixed(2)}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-neuro-gradient" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}
