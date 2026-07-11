import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { parseEDF, parseCSV, parseNPY } from "@/lib/eeg/parsers";
import { checkSignalQuality, qualityColor, qualityLabel } from "@/lib/signal-quality";
import type { SignalQualityReport } from "@/lib/signal-quality";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { supabase } from "@/integrations/supabase/client";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle, Upload, XCircle } from "lucide-react";

export const Route = createFileRoute("/upload")({
  component: UploadPage,
});

async function loadProfile() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user;
  if (!user) return null;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();
  return profile;
}

function UploadPage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const [file, setFile] = useState<File | null>(null);
  const [sampleRate, setSampleRate] = useState("256");
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [quality, setQuality] = useState<SignalQualityReport | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";

  const handleFileChange = async (f: File | null) => {
    setFile(f);
    setQuality(null);
    setResult(null);
    setError(null);
    if (!f) return;
    setChecking(true);
    try {
      const lower = f.name.toLowerCase();
      let signal;
      if (lower.endsWith(".edf") || lower.endsWith(".bdf")) {
        signal = parseEDF(await f.arrayBuffer());
      } else if (lower.endsWith(".csv") || lower.endsWith(".tsv")) {
        const fs = Number(sampleRate);
        if (!Number.isFinite(fs) || fs <= 0) return;
        signal = parseCSV(await f.text(), fs);
      } else if (lower.endsWith(".npy")) {
        const fs = Number(sampleRate);
        if (!Number.isFinite(fs) || fs <= 0) return;
        signal = parseNPY(await f.arrayBuffer(), fs);
      } else return;
      setQuality(checkSignalQuality(signal));
    } catch (e) {
      setError(`Could not pre-check file: ${(e as Error).message}`);
    } finally {
      setChecking(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    if (sampleRate) form.append("sampleRate", sampleRate);
    try {
      const res = await fetch("/api/eeg/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>Analysis</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">Upload EEG File</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Supported: EDF, BDF, CSV, TSV, NPY — max 10 MB
      </p>

      <div className="mt-8 flex flex-col gap-4 max-w-2xl">
        <GlassCard>
          <label className="flex flex-col items-center gap-3 cursor-pointer py-6">
            <Upload className="h-8 w-8 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {file ? file.name : "Tap to choose a file"}
            </span>
            <input
              type="file"
              accept=".edf,.bdf,.csv,.tsv,.npy"
              className="hidden"
              onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
            />
          </label>
        </GlassCard>

        {file &&
          (file.name.toLowerCase().endsWith(".csv") ||
            file.name.toLowerCase().endsWith(".npy") ||
            file.name.toLowerCase().endsWith(".tsv")) && (
            <GlassCard>
              <label className="text-xs text-muted-foreground">Sample Rate (Hz)</label>
              <input
                type="number"
                value={sampleRate}
                onChange={(e) => setSampleRate(e.target.value)}
                className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm"
                placeholder="e.g. 256"
              />
            </GlassCard>
          )}

        {checking && <p className="text-xs text-muted-foreground">Checking signal quality…</p>}
        {quality && <QualityCard report={quality} />}

        {error && (
          <GlassCard className="border-destructive/40 bg-destructive/5">
            <div className="flex items-center gap-2 text-destructive">
              <XCircle className="h-4 w-4 shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          </GlassCard>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading || quality?.overall === "bad"}
          className="rounded-lg bg-neuro px-6 py-3 text-sm font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
        >
          {loading
            ? "Processing…"
            : quality?.overall === "bad"
              ? "Fix signal quality issues first"
              : "Upload & Analyse"}
        </button>

        {result && <ResultCard result={result} />}
      </div>
    </DashboardShell>
  );
}

function QualityCard({ report }: { report: SignalQualityReport }) {
  const Icon =
    report.overall === "good"
      ? CheckCircle
      : report.overall === "warning"
        ? AlertTriangle
        : XCircle;
  const color = qualityColor(report.overall);
  return (
    <GlassCard>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5" style={{ color }} />
          <span className="text-sm font-semibold" style={{ color }}>
            Signal Quality: {qualityLabel(report.overall)}
          </span>
        </div>
        <span className="font-mono text-sm font-bold" style={{ color }}>
          {report.score}/100
        </span>
      </div>
      <div className="mt-3 h-2 rounded-full bg-muted/60">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${report.score}%`, backgroundColor: color }}
        />
      </div>
      {report.warnings.map((w, i) => (
        <p key={i} className="mt-2 text-xs text-amber-400 flex items-center gap-1">
          <AlertTriangle className="h-3 w-3 shrink-0" /> {w}
        </p>
      ))}
      {report.errors.map((e, i) => (
        <p key={i} className="mt-1 text-xs text-destructive flex items-center gap-1">
          <XCircle className="h-3 w-3 shrink-0" /> {e}
        </p>
      ))}
      {report.channels.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold text-muted-foreground">Channel breakdown</p>
          <div className="flex flex-col gap-1.5">
            {report.channels.map((ch) => (
              <div key={ch.channel} className="flex items-start justify-between text-xs">
                <span className="font-mono font-medium w-12 shrink-0">{ch.channel}</span>
                <div className="flex-1 flex flex-col gap-0.5">
                  {ch.issues.length === 0 ? (
                    <span className="text-green-500">No issues</span>
                  ) : (
                    ch.issues.map((issue, i) => (
                      <span key={i} style={{ color: qualityColor(ch.level) }}>
                        {issue}
                      </span>
                    ))
                  )}
                </div>
                <span className="ml-2 font-mono text-muted-foreground">{ch.rms.toFixed(1)} µV</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </GlassCard>
  );
}

function ResultCard({ result }: { result: Record<string, unknown> }) {
  const decoder = result.decoder as Record<string, number> | undefined;
  const signal = result.signal as Record<string, unknown> | undefined;
  const timings = result.timings as Record<string, number> | undefined;
  return (
    <GlassCard className="border-neuro/30">
      <p className="text-sm font-semibold text-neuro mb-3">✅ Analysis complete</p>
      {decoder && (
        <div className="grid grid-cols-3 gap-2 mb-4">
          <MetricBar label="Attention" value={decoder.attention} color="#818cf8" />
          <MetricBar label="Workload" value={decoder.workload} color="#a78bfa" />
          <MetricBar label="Arousal" value={decoder.arousal} color="#f59e0b" />
        </div>
      )}
      {signal && (
        <p className="text-xs text-muted-foreground">
          {String(signal.channels)} · {String(signal.sampleRate)} Hz
        </p>
      )}
      {timings && <p className="text-xs text-muted-foreground mt-1">⏱ {timings.total_ms} ms</p>}
    </GlassCard>
  );
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold" style={{ color }}>
          {pct}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-muted/60">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}
