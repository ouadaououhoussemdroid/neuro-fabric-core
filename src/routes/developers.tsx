import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section, StatPill } from "@/components/ui-bits";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { Copy, KeyRound, Plus } from "lucide-react";

export const Route = createFileRoute("/developers")({
  head: () => ({
    meta: [
      { title: "Developer Platform (concept) — NeuroWeave" },
      {
        name: "description",
        content:
          "Concept SDK design and API docs for NeuroWeave — the SDKs shown are not yet published.",
      },
      { property: "og:title", content: "Developer Platform (concept) — NeuroWeave" },
      { property: "og:description", content: "Concept SDK design for the NeuroWeave APIs." },
    ],
  }),
  component: DevPage,
});

const PY = `from neuroweave import NeuroWeave

nw = NeuroWeave(api_key="nw-live-…")

# Embed a 64-channel EEG window
emb = nw.embeddings.create(
    signal=signal,        # (n_channels, n_samples)
    sample_rate=250,
    model="nwf-7b-embed",
)

matches = nw.embeddings.search(emb.vector, k=8)
print(matches[0].subject, matches[0].similarity)`;

const JS = `import { NeuroWeave } from "@neuroweave/sdk";

const nw = new NeuroWeave({ apiKey: process.env.NW_API_KEY! });

const recon = await nw.vision.reconstruct({
  signal,            // Float32Array[64][1000]
  sampleRate: 250,
  model: "nw-vision-v1",
});

console.log(recon.caption, recon.confidence);`;

function DevPage() {
  const [tab, setTab] = useState<"py" | "js">("py");
  const usage = Array.from({ length: 30 }, (_, i) => ({
    d: i,
    r: 2000 + Math.sin(i / 2) * 600 + Math.random() * 400,
  }));

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Developer Platform · concept"
          title="Production APIs, batteries included."
          sub="A concept design for Python and TypeScript SDKs. Neither package is published yet — the working API today is the single REST endpoint documented below in the pipeline docs."
        />

        <div className="mb-6 rounded-lg border border-neuro/30 bg-neuro/5 px-4 py-3 text-sm text-muted-foreground">
          The <code className="rounded bg-muted px-1 py-0.5 text-xs">neuroweave</code> /{" "}
          <code className="rounded bg-muted px-1 py-0.5 text-xs">@neuroweave/sdk</code> packages
          below are a concept design, not published packages. The real, callable endpoint today is{" "}
          <code className="rounded bg-muted px-1 py-0.5 text-xs">POST /api/eeg/upload</code> — see{" "}
          <Link to="/playground" className="text-neuro hover:underline">
            /playground
          </Link>{" "}
          to try it. The request stats below are illustrative sample data.
        </div>

        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <GlassCard className="p-0">
            <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
              <div className="flex gap-2">
                <TabBtn active={tab === "py"} onClick={() => setTab("py")}>
                  Python
                </TabBtn>
                <TabBtn active={tab === "js"} onClick={() => setTab("js")}>
                  TypeScript
                </TabBtn>
              </div>
              <button className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
                <Copy className="h-3.5 w-3.5" /> Copy
              </button>
            </div>
            <pre className="overflow-x-auto p-5 font-mono text-[12.5px] leading-relaxed text-muted-foreground">
              {tab === "py" ? PY : JS}
            </pre>
          </GlassCard>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <StatPill label="Requests · 30d (demo)" value="1.24M" />
              <StatPill label="Error rate (demo)" value="0.04%" />
              <StatPill label="Avg latency (demo)" value="48 ms" />
              <StatPill label="Quota used (demo)" value="62%" />
            </div>
            <GlassCard>
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Requests · last 30d
              </div>
              <div className="mt-3 h-32">
                <ResponsiveContainer>
                  <AreaChart data={usage}>
                    <defs>
                      <linearGradient id="u" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="oklch(0.78 0.16 200)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="r"
                      stroke="oklch(0.85 0.18 195)"
                      strokeWidth={1.2}
                      fill="url(#u)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </GlassCard>
          </div>
        </div>

        <div className="mt-8 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <GlassCard>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <KeyRound className="h-4 w-4 text-neuro" />
                <span className="text-sm font-semibold">API keys</span>
              </div>
              <button className="inline-flex items-center gap-1.5 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background">
                <Plus className="h-3.5 w-3.5" /> New key
              </button>
            </div>
            <ul className="mt-4 divide-y divide-border/60">
              {[
                { name: "production", key: "nw-live-9f2b…d41a", used: "1,204,219" },
                { name: "staging", key: "nw-test-aa31…7c0e", used: "38,201" },
                { name: "ci-bot", key: "nw-live-c421…ff90", used: "12,884" },
              ].map((k) => (
                <li key={k.name} className="flex items-center justify-between py-3 text-sm">
                  <div>
                    <div className="font-medium">{k.name}</div>
                    <div className="font-mono text-xs text-muted-foreground">{k.key}</div>
                  </div>
                  <div className="font-mono text-xs text-muted-foreground">{k.used} req</div>
                </li>
              ))}
            </ul>
          </GlassCard>

          <GlassCard>
            <div className="text-sm font-semibold">Docs · quickstart</div>
            <ul className="mt-4 space-y-2 text-sm">
              {[
                "Authentication",
                "Embeddings API",
                "Reconstruction API",
                "Synthetic Generation",
                "Streaming WebSocket",
                "Webhooks",
              ].map((s) => (
                <li
                  key={s}
                  className="flex items-center justify-between rounded-md border border-border/60 bg-background/30 px-3 py-2 text-muted-foreground hover:text-foreground"
                >
                  <span>{s}</span>
                  <span className="font-mono text-[10px]">→</span>
                </li>
              ))}
            </ul>
          </GlassCard>
        </div>
      </Section>
    </SiteShell>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-xs font-mono uppercase tracking-wider ${active ? "bg-neuro/10 text-foreground" : "text-muted-foreground hover:text-foreground"}`}
    >
      {children}
    </button>
  );
}
