import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Brain, Database, Eye, Layers, Lock, Network, Sparkles, Waves, Zap } from "lucide-react";
import { SiteShell } from "@/components/site-shell";
import { NeuralBackground } from "@/components/neural-bg";
import { Eyebrow, GlassCard, Section } from "@/components/ui-bits";

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
      <ProductsGrid />
      <Capabilities />
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
          <Link to="/signup" className="group inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-5 py-3 text-sm font-medium text-background glow">
            Sign up <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <Link to="/signin" className="inline-flex items-center gap-2 rounded-md border border-border bg-card/40 px-5 py-3 text-sm font-medium hover:bg-card">
            Sign in
          </Link>
          <Link to="/developers" className="inline-flex items-center gap-2 rounded-md border border-border bg-card/40 px-5 py-3 text-sm font-medium hover:bg-card">
            Explore APIs
          </Link>
        </motion.div>
      </div>
    </section>
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

function Capabilities() {
  const items = [
    { icon: Network, title: "Open dataset loaders", desc: "Load PhysioNet, BCI Competition, and TUH EEG corpora directly from the playground." },
    { icon: Zap, title: "Real preprocessing", desc: "Bandpass filtering, normalization, and windowing implemented in TypeScript — runs server-side on every upload." },
    { icon: Brain, title: "Cognitive decoding", desc: "Lightweight estimators for attention, workload, and arousal computed from spectral features." },
    { icon: Lock, title: "Per-user data scope", desc: "All uploads and embeddings are scoped to the authenticated user via row-level security." },
  ];
  return (
    <Section>
      <Eyebrow>Capabilities</Eyebrow>
      <h2 className="mt-4 max-w-2xl text-3xl font-semibold tracking-tight md:text-4xl">What you can do today.</h2>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        NeuroWeave is in active development. Below is what is wired up and runnable in the current build — no placeholder metrics.
      </p>
      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {items.map((it) => (
          <GlassCard key={it.title} className="h-full">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-muted/40">
                <it.icon className="h-5 w-5 text-neuro" />
              </div>
              <h3 className="text-base font-semibold tracking-tight">{it.title}</h3>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">{it.desc}</p>
          </GlassCard>
        ))}
      </div>
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
        <Link to="/signup" className="rounded-md bg-neuro-gradient px-5 py-3 text-sm font-medium text-background glow">Sign up</Link>
        <Link to="/playground" className="rounded-md border border-border bg-card/40 px-5 py-3 text-sm font-medium hover:bg-card">Try the playground</Link>
      </div>
    </Section>
  );
}
