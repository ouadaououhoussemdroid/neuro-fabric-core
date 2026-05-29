import { createFileRoute } from "@tanstack/react-router";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { Brain, Lock, Microscope, Sparkles } from "lucide-react";

export const Route = createFileRoute("/about")({
  head: () => ({ meta: [
    { title: "About — NeuroWeave" },
    { name: "description", content: "Building the foundation layer for human–AI interfaces." },
    { property: "og:title", content: "About — NeuroWeave" },
    { property: "og:description", content: "NeuroWeave is the NeuroAI infrastructure company." },
  ]}),
  component: AboutPage,
});

function AboutPage() {
  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="About NeuroWeave"
          title="Infrastructure for the human–AI interface."
          sub="We build the foundation layer that turns brain signals into software primitives: embeddings, decoders, and synthetic data — shared across labs, startups, and sovereign programs."
        />

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[
            { icon: Brain, title: "Brain foundation models", body: "Pretrained on 18k+ subjects across consented public and partner datasets." },
            { icon: Microscope, title: "NeuroAI infrastructure", body: "Multi-region runtime, governance, and reproducibility for clinical-grade work." },
            { icon: Sparkles, title: "Synthetic neurodata", body: "Augment scarce datasets with consented, label-rich generated EEG." },
            { icon: Lock, title: "Ethical AI", body: "Subject consent ledger, per-region encryption, and a public model card for every release." },
          ].map((c) => (
            <GlassCard key={c.title}>
              <div className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-muted/40"><c.icon className="h-5 w-5 text-neuro" /></div>
              <div className="mt-4 text-sm font-semibold">{c.title}</div>
              <p className="mt-2 text-sm text-muted-foreground">{c.body}</p>
            </GlassCard>
          ))}
        </div>

        <div className="mt-12 grid gap-6 lg:grid-cols-2">
          <GlassCard>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Mission</div>
            <h3 className="mt-3 text-2xl font-semibold tracking-tight">Make brain data as composable as language.</h3>
            <p className="mt-3 text-muted-foreground">
              Today, every BCI team rebuilds the same pipeline — acquisition, preprocessing, models, evaluation. NeuroWeave collapses that work into a single API surface so teams can spend their effort on the part that's actually new: the interface between minds and machines.
            </p>
          </GlassCard>
          <GlassCard>
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Principles</div>
            <ul className="mt-3 space-y-3 text-sm text-muted-foreground">
              <li><span className="font-semibold text-foreground">Consent first.</span> Every dataset is tracked to a consenting subject. Withdrawal propagates to derived embeddings.</li>
              <li><span className="font-semibold text-foreground">Reproducible by default.</span> Deterministic replay of any inference, with versioned weights and configs.</li>
              <li><span className="font-semibold text-foreground">Open where it matters.</span> Public model cards, evaluations, and a free Research tier for accredited labs.</li>
              <li><span className="font-semibold text-foreground">Sovereign-ready.</span> On-prem and air-gapped deployments for national programs.</li>
            </ul>
          </GlassCard>
        </div>
      </Section>
    </SiteShell>
  );
}