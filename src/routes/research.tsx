import { createFileRoute } from "@tanstack/react-router";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section, StatPill } from "@/components/ui-bits";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export const Route = createFileRoute("/research")({
  head: () => ({
    meta: [
      { title: "Research Dashboard — NeuroWeave" },
      {
        name: "description",
        content: "Experiment tracking, datasets, training metrics, and benchmarks.",
      },
      { property: "og:title", content: "Research Dashboard — NeuroWeave" },
      { property: "og:description", content: "Track experiments and benchmarks." },
    ],
  }),
  component: ResearchPage,
});

const TRAIN = Array.from({ length: 40 }, (_, i) => ({
  step: i * 250,
  loss: 3.4 * Math.exp(-i / 12) + 0.8 + Math.random() * 0.05,
  val: 3.6 * Math.exp(-i / 11) + 0.9 + Math.random() * 0.05,
}));

const EXPERIMENTS = [
  { id: "nwf-7b/exp_2014", model: "nwf-7b", acc: 0.842, loss: 0.91, status: "running" },
  { id: "nwf-7b/exp_2013", model: "nwf-7b", acc: 0.831, loss: 0.94, status: "done" },
  { id: "nwf-3b/exp_1880", model: "nwf-3b", acc: 0.812, loss: 1.02, status: "done" },
  { id: "nw-vision/exp_0421", model: "nw-vision-v1", acc: 0.71, loss: 1.34, status: "done" },
  { id: "nw-synth/exp_0188", model: "nw-synth-v2", acc: 0.66, loss: 1.41, status: "done" },
];

function ResearchPage() {
  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Research Dashboard"
          title="Experiments, metrics, and benchmarks."
          sub="Track every training run, dataset version, and benchmark — designed for neuroscience teams that ship."
        />

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatPill label="Active runs" value="12" />
          <StatPill label="GPU hours · 7d" value="8,412" />
          <StatPill label="Datasets" value="73" />
          <StatPill label="Best benchmark" value="84.2% acc" />
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <GlassCard>
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">nwf-7b · exp_2014 · training</div>
              <span className="font-mono text-[10px] text-muted-foreground">
                step 10,000 / 50,000
              </span>
            </div>
            <div className="mt-4 h-64">
              <ResponsiveContainer>
                <LineChart data={TRAIN}>
                  <XAxis
                    dataKey="step"
                    stroke="oklch(0.6 0.02 260)"
                    tick={{ fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="oklch(0.6 0.02 260)"
                    tick={{ fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "oklch(0.18 0.014 260)",
                      border: "1px solid oklch(1 0 0 / 0.1)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="oklch(0.85 0.18 195)"
                    strokeWidth={1.5}
                    dot={false}
                    name="train loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="val"
                    stroke="oklch(0.7 0.22 295)"
                    strokeWidth={1.5}
                    dot={false}
                    name="val loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>

          <GlassCard>
            <div className="text-sm font-semibold">Benchmarks</div>
            <ul className="mt-4 space-y-3">
              {[
                ["ThingsEEG · top-1", 0.842],
                ["MindEye · CLIP-r", 0.71],
                ["DEAP · arousal", 0.78],
                ["BCI-IV 2a · kappa", 0.69],
              ].map(([n, v]) => (
                <li key={n as string}>
                  <div className="flex justify-between text-xs">
                    <span>{n}</span>
                    <span className="font-mono text-neuro">{(v as number).toFixed(3)}</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-neuro-gradient"
                      style={{ width: `${(v as number) * 100}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </GlassCard>
        </div>

        <GlassCard className="mt-6 p-0">
          <div className="border-b border-border/60 px-5 py-3 text-sm font-semibold">
            Experiments
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                <th className="px-5 py-2">Run</th>
                <th className="px-5 py-2">Model</th>
                <th className="px-5 py-2">Accuracy</th>
                <th className="px-5 py-2">Loss</th>
                <th className="px-5 py-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {EXPERIMENTS.map((e) => (
                <tr key={e.id}>
                  <td className="px-5 py-3 font-mono text-xs">{e.id}</td>
                  <td className="px-5 py-3 text-muted-foreground">{e.model}</td>
                  <td className="px-5 py-3 font-mono text-neuro">{e.acc.toFixed(3)}</td>
                  <td className="px-5 py-3 font-mono text-muted-foreground">{e.loss.toFixed(2)}</td>
                  <td className="px-5 py-3">
                    <span
                      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] ${e.status === "running" ? "bg-neuro/10 text-neuro" : "bg-muted text-muted-foreground"}`}
                    >
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${e.status === "running" ? "bg-neuro animate-pulse-glow" : "bg-muted-foreground"}`}
                      />
                      {e.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>
      </Section>
    </SiteShell>
  );
}
