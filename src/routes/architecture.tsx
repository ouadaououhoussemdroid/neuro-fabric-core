import { createFileRoute } from "@tanstack/react-router";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { ArrowRight, Cpu, Database, Eye, Layers, Shield, Waves } from "lucide-react";

export const Route = createFileRoute("/architecture")({
  head: () => ({ meta: [
    { title: "Architecture — NeuroWeave" },
    { name: "description", content: "From EEG ingestion to embeddings, latent space, and reconstruction APIs." },
    { property: "og:title", content: "Architecture — NeuroWeave" },
    { property: "og:description", content: "NeuroWeave platform architecture." },
  ]}),
  component: ArchitecturePage,
});

const STAGES = [
  { icon: Waves, name: "EEG Ingest", desc: "BIDS, EDF, LSL, OpenBCI, Neuralink-Compat" },
  { icon: Cpu, name: "Preprocess", desc: "ICA · band-pass · artifact removal" },
  { icon: Layers, name: "Embeddings", desc: "NWF-7B encoder · 768-d latent" },
  { icon: Database, name: "Latent Space", desc: "pgvector · ANN · governance" },
  { icon: Eye, name: "Reconstruction", desc: "Diffusion · captioning · decoders" },
  { icon: Shield, name: "APIs", desc: "REST · gRPC · WebSocket · SDKs" },
];

function ArchitecturePage() {
  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Product Architecture"
          title="EEG to intelligence, end to end."
          sub="A unified runtime for neural data: ingestion, preprocessing, embeddings, latent storage, and decoder APIs — all with a single auth layer and audit trail."
        />

        <div className="relative">
          <div className="grid gap-3 md:grid-cols-6">
            {STAGES.map((s, i) => (
              <GlassCard key={s.name} className="relative">
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Stage {i + 1}</div>
                <s.icon className="mt-3 h-5 w-5 text-neuro" />
                <div className="mt-3 text-sm font-semibold">{s.name}</div>
                <div className="mt-1 text-xs text-muted-foreground">{s.desc}</div>
                {i < STAGES.length - 1 && (
                  <ArrowRight className="absolute -right-3 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-neuro md:block" />
                )}
              </GlassCard>
            ))}
          </div>
        </div>

        <div className="mt-16 grid gap-4 lg:grid-cols-3">
          {[
            { title: "Runtime", body: "Multi-region inference fleet on bare-metal GPUs with token-bucket fairness, custom CUDA kernels for 1D temporal convolutions, and streaming-first model serving." },
            { title: "Storage", body: "Append-only neurodata lakehouse with Iceberg tables. Embeddings live in a sharded pgvector cluster with copy-on-write isolation per workspace." },
            { title: "Governance", body: "Per-subject consent ledger, region-pinned encryption, audit-logged inference, and a deterministic replay engine to reproduce any past output." },
          ].map((c) => (
            <GlassCard key={c.title}>
              <div className="text-sm font-semibold">{c.title}</div>
              <div className="mt-2 text-sm text-muted-foreground">{c.body}</div>
            </GlassCard>
          ))}
        </div>
      </Section>
    </SiteShell>
  );
}