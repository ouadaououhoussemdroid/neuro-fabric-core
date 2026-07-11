import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { useMOABB, MOABB_DATASETS } from "@/hooks/use-moabb";
import {
  AlertTriangle,
  BookOpen,
  Brain,
  CheckCircle,
  Database,
  Loader2,
  Play,
  Users,
} from "lucide-react";

export const Route = createFileRoute("/_authenticated/datasets")({
  component: DatasetsPage,
});

async function loadProfile() {
  const { data: userData } = await supabase.auth.getUser();
  if (!userData.user) return null;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", userData.user.id)
    .maybeSingle();
  return profile;
}

function DatasetsPage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const { progress, epochs, loadDataset } = useMOABB();
  const [selected, setSelected] = useState<string>("BNCI2014_001");
  const [subjects, setSubjects] = useState("1");
  const [labelStats, setLabelStats] = useState<Record<string, number>>({});

  const handleLoad = async () => {
    const subjList = subjects
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => n > 0);
    try {
      const result = await loadDataset(selected, subjList);
      const stats: Record<string, number> = {};
      for (const ep of result.epochs) stats[ep.label] = (stats[ep.label] ?? 0) + 1;
      setLabelStats(stats);
    } catch (err) {
      console.error("datasets.loadDataset failed", err);
    }
  };

  const isRunning = ["fetching", "loading-moabb", "processing"].includes(progress.phase);
  const isDone = progress.phase === "done";
  const isError = progress.phase === "error";
  const pyReady = !!window.pyodideInstance;

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>Scientific Datasets</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">MOABB Datasets</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Mother of All BCI Benchmarks — standardized access to 10+ peer-reviewed EEG datasets
        including BCI Competition IV. Requires MNE-Python to be loaded first.
      </p>

      {!pyReady && (
        <GlassCard className="mt-6 border-amber-400/30">
          <div className="flex items-center gap-2 text-amber-400">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <p className="text-sm">
              MNE-Python must be loaded first. Go to{" "}
              <a href="/mne" className="underline">
                MNE Analysis
              </a>{" "}
              and click "Load MNE-Python".
            </p>
          </div>
        </GlassCard>
      )}

      <div className="mt-6 grid gap-3">
        {MOABB_DATASETS.map((ds) => (
          <GlassCard
            key={ds.id}
            className={`cursor-pointer transition-colors ${selected === ds.id ? "border-neuro/60 bg-neuro/5" : "hover:border-neuro/20"}`}
            onClick={() => setSelected(ds.id)}
          >
            <div className="flex items-start gap-2">
              <div
                className={`mt-1 h-2 w-2 shrink-0 rounded-full ${selected === ds.id ? "bg-neuro" : "bg-muted"}`}
              />
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-semibold">{ds.name}</span>
                  <span className="font-mono text-[11px] text-muted-foreground">{ds.id}</span>
                </div>
                <p className="text-xs text-muted-foreground">{ds.description}</p>
                <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Users className="h-3 w-3" /> {ds.n_subjects} subjects
                  </span>
                  <span className="flex items-center gap-1">
                    <Brain className="h-3 w-3" /> {ds.n_channels} channels
                  </span>
                  <span className="flex items-center gap-1">
                    <Database className="h-3 w-3" /> {ds.sfreq} Hz
                  </span>
                  <span className="flex items-center gap-1">
                    <BookOpen className="h-3 w-3" /> {ds.scientific_ref}
                  </span>
                </div>
              </div>
            </div>
          </GlassCard>
        ))}
      </div>

      <GlassCard className="mt-4">
        <p className="text-xs font-semibold text-muted-foreground mb-3">Load Configuration</p>
        <label className="text-xs text-muted-foreground">Subjects (comma-separated)</label>
        <input
          value={subjects}
          onChange={(e) => setSubjects(e.target.value)}
          disabled={isRunning}
          className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
          placeholder="1,2,3"
        />
        <p className="mt-1 text-[11px] text-muted-foreground">Start with 1 subject (~50MB).</p>
        <button
          onClick={handleLoad}
          disabled={isRunning || !pyReady}
          className="mt-4 flex items-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40 hover:opacity-90"
        >
          {isRunning ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" /> Load Dataset
            </>
          )}
        </button>
      </GlassCard>

      {isRunning && (
        <GlassCard className="mt-4">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 className="h-4 w-4 animate-spin text-neuro" />
            <span className="text-sm text-neuro">{progress.message}</span>
          </div>
          <div className="h-2 rounded-full bg-muted/60">
            <div
              className="h-full rounded-full bg-neuro transition-all"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
        </GlassCard>
      )}

      {isError && (
        <GlassCard className="mt-4 border-destructive/40">
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-4 w-4" />
            <p className="text-sm">{progress.error}</p>
          </div>
        </GlassCard>
      )}

      {isDone && epochs.length > 0 && (
        <div className="mt-4 flex flex-col gap-3">
          <GlassCard className="border-neuro/30">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-sm font-semibold text-green-400">{progress.message}</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Channels: </span>
                <span className="font-mono">{epochs[0]?.data.length}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Samples/epoch: </span>
                <span className="font-mono">{epochs[0]?.data[0]?.length}</span>
              </div>
            </div>
          </GlassCard>

          {Object.keys(labelStats).length > 0 && (
            <GlassCard>
              <p className="text-xs font-semibold text-muted-foreground mb-3">
                🏷️ Label Distribution
              </p>
              {Object.entries(labelStats).map(([label, count]) => {
                const total = Object.values(labelStats).reduce((s, v) => s + v, 0);
                return (
                  <div key={label} className="mb-2">
                    <div className="flex justify-between text-[11px] mb-1">
                      <span className="font-mono text-muted-foreground">{label}</span>
                      <span className="font-mono text-neuro">
                        {count} ({Math.round((count / total) * 100)}%)
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-muted/60">
                      <div
                        className="h-full rounded-full bg-neuro"
                        style={{ width: `${Math.round((count / total) * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </GlassCard>
          )}

          <GlassCard>
            <p className="text-xs font-semibold text-muted-foreground mb-3">Sample Epochs</p>
            <div className="flex flex-col gap-1.5">
              {epochs.slice(0, 5).map((ep, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded border border-border bg-muted/20 px-3 py-2 text-[11px]"
                >
                  <span className="font-mono text-muted-foreground">
                    S{ep.subject} · {ep.session}
                  </span>
                  <span className="font-mono font-semibold text-neuro">{ep.label}</span>
                  <span className="text-muted-foreground">
                    {ep.data.length}ch × {ep.data[0]?.length}pts
                  </span>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}
    </DashboardShell>
  );
}
