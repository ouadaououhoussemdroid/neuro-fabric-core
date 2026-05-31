import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section } from "@/components/ui-bits";
import { InferenceStages, LiveDot, StreamingJson, StreamingLatent } from "@/components/live-ops";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { Image as ImageIcon, Play, Upload } from "lucide-react";

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
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [seed, setSeed] = useState(0);

 const [uploadedFile, setUploadedFile] = useState<File | null>(null);

const run = async () => {
  if (!uploadedFile) return;
  setRunning(true);
  setOutput("");
  setLatencyMs(null);

  const started = performance.now();
  const form = new FormData();
  form.append("file", uploadedFile);

  try {
    const res = await fetch("/api/eeg/upload", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setOutput(JSON.stringify(data, null, 2));
    setLatencyMs(+(performance.now() - started).toFixed(0));
  } catch (err: any) {
    setOutput(`Error: ${err.message}`);
  } finally {
    setRunning(false);
  }
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
                <LiveDot /> sample.edf · 64ch · 250 Hz
              </div>
              <label className="inline-flex items-center gap-2 rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs hover:bg-card cursor-pointer">
  <Upload className="h-3.5 w-3.5" />
  {uploadedFile ? uploadedFile.name : "Upload EEG"}
  <input
    type="file"
    accept=".edf,.csv,.npy"
    className="hidden"
    onChange={(e) => setUploadedFile(e.target.files?.[0] ?? null)}
  />
</label>
          
            </div>
            <div className="mt-4 h-44">
              <ResponsiveContainer>
                <LineChart data={SAMPLE_SIGNAL}>
                  <XAxis dataKey="t" hide />
                
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
                className="ml-auto inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-4 py-2 text-xs font-medium text-background glow transition-transform hover:scale-[1.02] disabled:opacity-60"
              >
                <Play className="h-3.5 w-3.5" /> {running ? "Running…" : "Run inference"}
              </button>
            </div>

            <div className="mt-5">
              <InferenceStages active={running} />
            </div>

            <div className="mt-6">
              <div className="flex items-center justify-between">
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Latent vector · streaming · 64×4
                </div>
                {latencyMs != null && (
                  <div className="font-mono text-[10px] text-neuro">{latencyMs} ms · 200 OK</div>
                )}
              </div>
              <div className="mt-3">
                <StreamingLatent cols={64} rows={4} speed={running ? 60 : 140} />
              </div>
              {endpoint === "reconstruct" && output && (
                <ReconstructionPreview seed={seed} />
              )}
            </div>
          </GlassCard>

          <GlassCard className="font-mono text-xs">
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <span className="text-muted-foreground">POST · api.neuroweave.ai/v1/{endpoint}</span>
              <span className={running ? "text-muted-foreground" : "text-neuro"}>
                {running ? "streaming…" : output ? "200 OK" : "idle"}
              </span>
            </div>
            <div className="mt-4">
              {output ? (
                <StreamingJson text={output} />
              ) : (
                <pre className="text-[12px] text-muted-foreground/70">// Run inference to see response payload…</pre>
              )}
            </div>
          </GlassCard>
        </div>
      </Section>
    </SiteShell>
  );
}

function ReconstructionPreview({ seed }: { seed: number }) {
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    setProgress(0);
    const id = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          clearInterval(id);
          return 100;
        }
        return p + 6;
      });
    }, 60);
    return () => clearInterval(id);
  }, [seed]);
  return (
    <div className="mt-6 grid gap-3 sm:grid-cols-[1fr_1.2fr]">
      <div className="glass relative aspect-square overflow-hidden rounded-lg">
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(60% 60% at 35% 35%, oklch(0.85 0.18 195 / 0.55), transparent 60%), radial-gradient(50% 50% at 70% 70%, oklch(0.7 0.22 295 / 0.5), transparent 60%), linear-gradient(135deg, oklch(0.2 0.02 260), oklch(0.16 0.02 260))",
            filter: `blur(${Math.max(0, 14 - progress / 8)}px) saturate(${1 + progress / 100})`,
          }}
        />
        <div className="absolute inset-0 grid-bg opacity-30" />
        <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded border border-border bg-background/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          <ImageIcon className="h-3 w-3 text-neuro" /> reconstruction · diff step {Math.min(40, Math.round(progress * 0.4))}/40
        </div>
      </div>
      <div className="glass rounded-lg p-3">
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Decode · vision.v1</div>
        <div className="mt-2 text-sm">"a golden retriever sitting on green grass"</div>
        <div className="mt-4 space-y-2">
          {[
            ["Alignment", 0.84],
            ["Confidence", 0.71],
            ["CLIP cos sim", 0.78],
          ].map(([k, v]) => (
            <div key={k as string}>
              <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>{k}</span>
                <span className="text-foreground">{(v as number).toFixed(2)}</span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted/40">
                <div
                  className="h-full bg-neuro-gradient"
                  style={{ width: `${(v as number) * 100 * (progress / 100)}%`, transition: "width 80ms linear" }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
