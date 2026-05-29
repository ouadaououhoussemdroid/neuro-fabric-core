import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { Play, Upload } from "lucide-react";

export const Route = createFileRoute("/playground")({
  head: () => ({ meta: [
    { title: "API Playground — NeuroWeave" },
    { name: "description", content: "Run embeddings, reconstruction, and synthetic generation on sample EEG." },
    { property: "og:title", content: "API Playground — NeuroWeave" },
    { property: "og:description", content: "Interactive NeuroWeave API playground." },
  ]}),
  component: PlaygroundPage,
});

type Endpoint = "embed" | "reconstruct" | "synthesize";

const SAMPLE_SIGNAL = Array.from({ length: 256 }, (_, i) => ({
  t: i,
  v: Math.sin(i / 6) * 0.7 + Math.sin(i / 2.1) * 0.3 + (Math.random() - 0.5) * 0.25,
}));

function PlaygroundPage() {
  const [endpoint, setEndpoint] = useState<Endpoint>("embed");
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState<string>("");

  const latent = useMemo(
    () => Array.from({ length: 64 }, () => Math.random() * 2 - 1),
    [output]
  );

  const run = () => {
    setRunning(true);
    setOutput("");
    setTimeout(() => {
      const payloads: Record<Endpoint, object> = {
        embed: {
          model: "nwf-7b-embed",
          subject: "anon_0421",
          dim: 768,
          latency_ms: 41.2,
          embedding: [
            ...Array.from({ length: 8 }, () => +(Math.random() * 2 - 1).toFixed(4)),
            "…+760 more",
          ],
        },
        reconstruct: {
          model: "nw-vision-v1",
          alignment_score: 0.842,
          confidence: 0.71,
          caption: "a golden retriever sitting on green grass",
          image_url: "ipfs://Qm…/reconstruction_421.png",
        },
        synthesize: {
          model: "nw-synth-v2",
          condition: { attention: 0.7, stress: 0.2, workload: 0.4 },
          n_samples: 1024,
          dataset_uri: "s3://neuroweave-synth/datasets/run_0421.parquet",
        },
      };
      setOutput(JSON.stringify(payloads[endpoint], null, 2));
      setRunning(false);
    }, 850);
  };

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="API Playground"
          title="Run NeuroWeave on a sample signal."
          sub="Upload an EEG sample, pick an endpoint, and inspect the JSON output and latent vector visualization."
        />

        <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
          <GlassCard>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
                <span className="h-2 w-2 rounded-full bg-neuro animate-pulse-glow" /> sample.edf · 64ch · 250 Hz
              </div>
              <button className="inline-flex items-center gap-2 rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs hover:bg-card">
                <Upload className="h-3.5 w-3.5" /> Upload EEG
              </button>
            </div>
            <div className="mt-4 h-44">
              <ResponsiveContainer>
                <LineChart data={SAMPLE_SIGNAL}>
                  <XAxis dataKey="t" hide />
                  <YAxis hide domain={[-1.5, 1.5]} />
                  <Line type="monotone" dataKey="v" stroke="oklch(0.85 0.18 195)" strokeWidth={1.2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-2">
              {(["embed", "reconstruct", "synthesize"] as Endpoint[]).map((e) => (
                <button
                  key={e}
                  onClick={() => setEndpoint(e)}
                  className={`rounded-md border px-3 py-1.5 text-xs font-mono uppercase tracking-wider ${
                    endpoint === e ? "border-neuro/60 bg-neuro/10 text-foreground" : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                >
                  /v1/{e}
                </button>
              ))}
              <button
                onClick={run}
                disabled={running}
                className="ml-auto inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-4 py-2 text-xs font-medium text-background glow disabled:opacity-60"
              >
                <Play className="h-3.5 w-3.5" /> {running ? "Running…" : "Run inference"}
              </button>
            </div>

            <div className="mt-6">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Latent vector · 64-d preview</div>
              <div className="mt-2 grid grid-cols-32 gap-[2px]" style={{ gridTemplateColumns: "repeat(32, minmax(0, 1fr))" }}>
                {latent.map((v, i) => (
                  <div
                    key={i}
                    className="h-6 rounded-[2px]"
                    style={{
                      background: v >= 0
                        ? `oklch(0.78 0.16 200 / ${0.2 + Math.abs(v) * 0.8})`
                        : `oklch(0.7 0.22 295 / ${0.2 + Math.abs(v) * 0.8})`,
                    }}
                  />
                ))}
              </div>
            </div>
          </GlassCard>

          <GlassCard className="font-mono text-xs">
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <span className="text-muted-foreground">POST · api.neuroweave.ai/v1/{endpoint}</span>
              <span className="text-neuro">200 OK</span>
            </div>
            <pre className="mt-4 max-h-[420px] overflow-auto whitespace-pre-wrap text-[12px] leading-relaxed text-muted-foreground">
{output || "// Run inference to see response payload…"}
            </pre>
          </GlassCard>
        </div>
      </Section>
    </SiteShell>
  );
}