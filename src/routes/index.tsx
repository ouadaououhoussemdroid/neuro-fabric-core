import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Brain, Database, Eye, Layers, Sparkles, Waves, Zap } from "lucide-react";
import { SiteShell } from "@/components/site-shell";
import { NeuralBackground } from "@/components/neural-bg";
import { Eyebrow, GlassCard, Section, StatPill } from "@/components/ui-bits";
import { Area, AreaChart, ResponsiveContainer } from "recharts";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "NeuroWeave — Foundation Models for Brain Signals" },
      { name: "description", content: "NeuroAI APIs for embeddings, cognitive decoding, synthetic neurodata, and EEG-to-image reconstruction." },
      { property: "og:title", content: "NeuroWeave — Foundation Models for Brain Signals" },
      { property: "og:description", content: "NeuroAI APIs for embeddings, cognitive decoding, synthetic neurodata, and EEG-to-image reconstruction." },
    ],
  }),
  component: Index,
});

const PRODUCTS = [
  { icon: Waves, title: "EEG Embeddings API", desc: "768-d brain-signal vectors for search, clustering, and downstream ML. Sub-50 ms latency." , tag: "embed.v3" },
  { icon: Sparkles, title: "Synthetic Neurodata Engine", desc: "Generate consented, label-rich EEG datasets conditioned on cognitive state.", tag: "synth.v2" },
  { icon: Brain, title: "Cognitive State Intelligence", desc: "Decode attention, stress, workload, and intent with calibrated confidence.", tag: "cog.v4" },
  { icon: Eye, title: "EEG2Image Reconstruction", desc: "Latent-aligned diffusion that turns visual-cortex signals into images.", tag: "vision.v1" },
  { icon: Layers, title: "Neuro Foundation Models", desc: "Pretrained transformers across 18k+ subjects, fine-tunable in a single API call.", tag: "nwf.7B" },
  { icon: Database, title: "Neurodata Lakehouse", desc: "Versioned, governance-ready storage for raw signal, labels, and derived embeddings.", tag: "lake" },
];

function Index() {
  return (
    <SiteShell>
      <Hero />
      <Logos />
      <ProductsGrid />
      <PipelinePreview />
      <MetricsBand />
      <ClosingCta />
    </SiteShell>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <NeuralBackground density={80} />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/40 to-background" />
      </div>
      <div className="relative mx-auto max-w-7xl px-4 pt-24 pb-28 md:pt-32 md:pb-36">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <Eyebrow>NeuroAI Infrastructure · Series A</Eyebrow>
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="mt-5 max-w-4xl text-5xl font-semibold leading-[1.05] tracking-tight md:text-7xl"
        >
          Foundation Models for{" "}
          <span className="text-gradient">Brain Signals</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="mt-6 max-w-2xl text-lg text-muted-foreground"
        >
          NeuroAI APIs for embeddings, cognitive decoding, synthetic neurodata, and EEG-to-image reconstruction — built for researchers, startups, and sovereign labs.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.22 }}
          className="mt-9 flex flex-wrap items-center gap-3"
        >
          <Link to="/pricing" className="group inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-5 py-3 text-sm font-medium text-background glow">
            Request access <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <Link to="/developers" className="inline-flex items-center gap-2 rounded-md border border-border bg-card/40 px-5 py-3 text-sm font-medium hover:bg-card">
            Explore APIs
          </Link>
          <div className="ml-2 hidden items-center gap-2 font-mono text-xs text-muted-foreground md:flex">
            <Zap className="h-3.5 w-3.5 text-neuro" /> 42 ms p50 inference · SOC 2 · HIPAA-ready
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mt-16 grid grid-cols-2 gap-3 md:grid-cols-4"
        >
          <StatPill label="Subjects pretrained" value="18,421" />
          <StatPill label="Hours of EEG" value="62.3k" />
          <StatPill label="Embedding dim" value="768" />
          <StatPill label="API p50 latency" value="42 ms" />
        </motion.div>
      </div>
    </section>
  );
}

function Logos() {
  const labs = ["MIT CSAIL", "Max Planck", "ETH Zürich", "Stanford NPC", "Inria", "RIKEN", "DeepMind", "Allen Institute"];
  return (
    <div className="border-y border-border/60 bg-background/40">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-6 px-4 py-8">
        <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">Trusted in research at</span>
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3 opacity-70">
          {labs.map((l) => (
            <span key={l} className="text-sm tracking-tight text-muted-foreground">{l}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function ProductsGrid() {
  return (
    <Section>
      <div className="flex items-end justify-between">
        <div>
          <Eyebrow>Platform</Eyebrow>
          <h2 className="mt-4 max-w-2xl text-3xl font-semibold tracking-tight md:text-4xl">A complete stack for neural intelligence.</h2>
        </div>
        <Link to="/architecture" className="hidden text-sm text-muted-foreground hover:text-foreground md:inline">View architecture →</Link>
      </div>
      <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {PRODUCTS.map((p) => (
          <motion.div key={p.title} whileHover={{ y: -3 }} transition={{ type: "spring", stiffness: 200, damping: 20 }}>
            <GlassCard className="group h-full transition-colors hover:border-neuro/40">
              <div className="flex items-center justify-between">
                <div className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-muted/40">
                  <p.icon className="h-5 w-5 text-neuro" />
                </div>
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{p.tag}</span>
              </div>
              <h3 className="mt-5 text-lg font-semibold tracking-tight">{p.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{p.desc}</p>
              <div className="mt-6 flex items-center gap-2 text-xs text-neuro opacity-80 group-hover:opacity-100">
                Read API spec <ArrowRight className="h-3.5 w-3.5" />
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>
    </Section>
  );
}

function PipelinePreview() {
  const data = Array.from({ length: 80 }, (_, i) => ({
    x: i,
    a: Math.sin(i / 5) * 0.6 + Math.sin(i / 2.3) * 0.3 + (Math.random() - 0.5) * 0.15,
    b: Math.cos(i / 4) * 0.5 + Math.sin(i / 7) * 0.4 + (Math.random() - 0.5) * 0.1,
  }));
  return (
    <Section>
      <div className="grid items-center gap-10 lg:grid-cols-2">
        <div>
          <Eyebrow>Pipeline</Eyebrow>
          <h2 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">From raw signal to grounded intelligence.</h2>
          <p className="mt-3 max-w-lg text-muted-foreground">
            Stream 64-channel EEG into the NeuroWeave runtime. We preprocess, embed in a shared latent space, and route to embeddings, cognitive state, or vision reconstruction APIs — all with the same authenticated client.
          </p>
          <Link to="/architecture" className="mt-6 inline-flex items-center gap-2 text-sm text-neuro">View full architecture <ArrowRight className="h-4 w-4" /></Link>
        </div>
        <GlassCard className="p-0">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
            <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
              <span className="h-2 w-2 rounded-full bg-neuro animate-pulse-glow" />
              live · stream://subject_0421
            </div>
            <span className="font-mono text-[10px] uppercase text-muted-foreground">Fp1 · Cz · O2</span>
          </div>
          <div className="h-44 w-full">
            <ResponsiveContainer>
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0.6} />
                    <stop offset="100%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.7 0.22 295)" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="oklch(0.7 0.22 295)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="a" stroke="oklch(0.85 0.18 195)" strokeWidth={1.2} fill="url(#g1)" />
                <Area type="monotone" dataKey="b" stroke="oklch(0.78 0.2 295)" strokeWidth={1.2} fill="url(#g2)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 divide-x divide-border/60 border-t border-border/60 text-center">
            {[["SNR", "27.4 dB"], ["α/β ratio", "1.84"], ["Attention", "0.71"]].map(([k, v]) => (
              <div key={k} className="px-3 py-3">
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
                <div className="mt-1 text-sm font-semibold">{v}</div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </Section>
  );
}

function MetricsBand() {
  return (
    <Section>
      <GlassCard className="relative overflow-hidden p-10">
        <div className="pointer-events-none absolute -right-20 -top-20 h-80 w-80 rounded-full bg-neuro/20 blur-3xl" />
        <div className="pointer-events-none absolute -left-10 -bottom-20 h-72 w-72 rounded-full bg-accent/20 blur-3xl" />
        <div className="relative grid items-center gap-10 md:grid-cols-2">
          <div>
            <h3 className="text-3xl font-semibold tracking-tight">Built like real infrastructure.</h3>
            <p className="mt-3 text-muted-foreground">Multi-region inference, customer-managed encryption keys, audit logs, and a deterministic replay engine for clinical-grade reproducibility.</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <StatPill label="Inference regions" value="9" />
            <StatPill label="Throughput" value="2.1M req/min" />
            <StatPill label="Uptime SLA" value="99.99%" />
            <StatPill label="Compliance" value="SOC 2 · HIPAA" />
          </div>
        </div>
      </GlassCard>
    </Section>
  );
}

function ClosingCta() {
  return (
    <Section className="text-center">
      <Eyebrow>Get started</Eyebrow>
      <h2 className="mx-auto mt-4 max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">Bring brain signals into your software stack.</h2>
      <p className="mx-auto mt-4 max-w-xl text-muted-foreground">Join research labs and BCI startups building on the NeuroWeave foundation.</p>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Link to="/pricing" className="rounded-md bg-neuro-gradient px-5 py-3 text-sm font-medium text-background glow">Request access</Link>
        <Link to="/playground" className="rounded-md border border-border bg-card/40 px-5 py-3 text-sm font-medium hover:bg-card">Try the playground</Link>
      </div>
    </Section>
  );
}
