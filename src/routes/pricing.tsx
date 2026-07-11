import { createFileRoute, Link } from "@tanstack/react-router";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { EEGLive } from "@/components/eeg-live";
import { Check } from "lucide-react";

export const Route = createFileRoute("/pricing")({
  head: () => ({
    meta: [
      { title: "Pricing — NeuroWeave" },
      { name: "description", content: "Research, Startup, Enterprise, and Sovereign AI plans." },
      { property: "og:title", content: "Pricing — NeuroWeave" },
      { property: "og:description", content: "NeuroWeave plans for research and production." },
    ],
  }),
  component: PricingPage,
});

const TIERS = [
  {
    name: "Research",
    price: "Free",
    per: "for accredited labs",
    desc: "For academic teams running non-commercial experiments.",
    features: [
      "10k embeddings / mo",
      "1k reconstructions / mo",
      "Public synthetic datasets",
      "Community Slack",
    ],
    cta: "Apply for credits",
    highlight: false,
  },
  {
    name: "Startup",
    price: "$499",
    per: "/ month",
    desc: "Build BCI products with production-grade SLAs.",
    features: [
      "1M embeddings / mo",
      "50k reconstructions / mo",
      "Custom fine-tunes",
      "SOC 2 reports",
      "Email support",
    ],
    cta: "Start trial",
    highlight: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    per: "annual contract",
    desc: "For BCI hardware vendors and clinical platforms.",
    features: [
      "Unlimited inference",
      "Dedicated regions",
      "VPC peering · CMEK",
      "HIPAA BAA",
      "Solutions engineering",
    ],
    cta: "Talk to sales",
    highlight: false,
  },
  {
    name: "Sovereign AI",
    price: "Contact",
    per: "on-prem · air-gapped",
    desc: "For governments and defense research orgs.",
    features: [
      "Self-hosted runtime",
      "Weights licensing",
      "FedRAMP High roadmap",
      "24/7 on-call",
      "Dedicated SE team",
    ],
    cta: "Request brief",
    highlight: false,
  },
];

function PricingPage() {
  return (
    <SiteShell>
      <Section>
        <div className="relative">
          <div className="pointer-events-none absolute inset-0 -z-10 opacity-[0.12] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_80%)]">
            <EEGLive height={260} />
          </div>
          <PageHeader
            eyebrow="Pricing"
            title="Plans for every stage of neural intelligence."
            sub="From a lone PhD student to a sovereign research program — choose the right runtime."
          />
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {TIERS.map((t) => (
            <GlassCard
              key={t.name}
              className={`flex h-full flex-col ${t.highlight ? "border-neuro/40 ring-glow" : ""}`}
            >
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold">{t.name}</div>
                {t.highlight && (
                  <span className="rounded-full bg-neuro/15 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-neuro">
                    popular
                  </span>
                )}
              </div>
              <div className="mt-4 flex items-baseline gap-1">
                <div className="text-3xl font-semibold tracking-tight">{t.price}</div>
                <div className="text-xs text-muted-foreground">{t.per}</div>
              </div>
              <p className="mt-2 text-sm text-muted-foreground">{t.desc}</p>
              <ul className="mt-5 space-y-2 text-sm">
                {t.features.map((f) => (
                  <li key={f} className="flex items-start gap-2">
                    <Check className="mt-0.5 h-4 w-4 text-neuro" />
                    <span className="text-muted-foreground">{f}</span>
                  </li>
                ))}
              </ul>
              <Link
                to="/about"
                className={`mt-6 inline-flex items-center justify-center rounded-md px-3 py-2 text-xs font-medium ${
                  t.highlight
                    ? "bg-neuro-gradient text-background glow"
                    : "border border-border bg-card/40 hover:bg-card"
                }`}
              >
                {t.cta}
              </Link>
            </GlassCard>
          ))}
        </div>
      </Section>
    </SiteShell>
  );
}
