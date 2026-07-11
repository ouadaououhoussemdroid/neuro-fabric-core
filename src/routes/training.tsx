import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { runTrainingPipeline } from "@/lib/training/pipeline";
import type { TrainingProgress, TrainedWeights } from "@/lib/training/pipeline";
import { Activity, AlertTriangle, CheckCircle, Download, Play, Square } from "lucide-react";

export const Route = createFileRoute("/training")({
  component: TrainingPage,
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

function TrainingPage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<TrainingProgress[]>([]);
  const [weights, setWeights] = useState<TrainedWeights | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [subjects, setSubjects] = useState(10);
  const [epochs, setEpochs] = useState(150);
  const abortRef = useRef(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const addLog = (p: TrainingProgress) => {
    setLogs((prev) => [...prev.slice(-100), p]);
    setTimeout(() => logsEndRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  const handleStart = async () => {
    setRunning(true);
    setLogs([]);
    setWeights(null);
    setError(null);
    abortRef.current = false;
    try {
      const result = await runTrainingPipeline({
        maxSubjects: subjects,
        runsPerSubject: 4,
        epochs,
        onProgress: (p) => {
          if (abortRef.current) throw new Error("Cancelled");
          addLog(p);
        },
      });
      setWeights(result);
    } catch (err) {
      setError((err as Error).message);
      addLog({ phase: "error", error: (err as Error).message });
    } finally {
      setRunning(false);
    }
  };

  const handleDownload = () => {
    if (!weights) return;
    const blob = new Blob([JSON.stringify(weights, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tfjs-eeg-v1-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const lastLog = logs[logs.length - 1];
  const prog =
    lastLog?.epoch && lastLog?.totalEpochs
      ? Math.round((lastLog.epoch / lastLog.totalEpochs) * 100)
      : 0;

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>ML Training</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">Train on PhysioNet</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Fetches real EEG data from PhysioNet, extracts band-power features, and trains a 3-layer MLP
        decoder in the browser.
      </p>

      <GlassCard className="mt-6">
        <p className="text-xs font-semibold text-muted-foreground mb-3">Configuration</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-muted-foreground">Subjects (1–109)</label>
            <input
              type="number"
              min={1}
              max={109}
              value={subjects}
              disabled={running}
              onChange={(e) => setSubjects(Math.min(109, Math.max(1, Number(e.target.value))))}
              className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground">Epochs</label>
            <input
              type="number"
              min={10}
              max={500}
              value={epochs}
              disabled={running}
              onChange={(e) => setEpochs(Math.min(500, Math.max(10, Number(e.target.value))))}
              className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
            />
          </div>
        </div>
        <p className="mt-3 text-[11px] text-amber-400">
          ⚠️ Keep this tab open during training. Est. time: ~
          {Math.round(subjects * 0.5 + epochs * 0.05)} min
        </p>
      </GlassCard>

      <div className="mt-4 flex gap-3">
        {!running ? (
          <button
            onClick={handleStart}
            className="flex items-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white hover:opacity-90"
          >
            <Play className="h-4 w-4" /> Start Training
          </button>
        ) : (
          <button
            onClick={() => (abortRef.current = true)}
            className="flex items-center gap-2 rounded-lg bg-destructive px-5 py-2.5 text-sm font-semibold text-white hover:opacity-90"
          >
            <Square className="h-4 w-4" /> Stop
          </button>
        )}
        {weights && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 rounded-lg border border-border px-5 py-2.5 text-sm font-semibold hover:bg-muted/40"
          >
            <Download className="h-4 w-4" /> Download Weights
          </button>
        )}
      </div>

      {running && lastLog?.phase === "training" && (
        <GlassCard className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-neuro flex items-center gap-1">
              <Activity className="h-3.5 w-3.5 animate-pulse" /> Training…
            </span>
            <span className="font-mono text-xs text-muted-foreground">
              Epoch {lastLog.epoch}/{lastLog.totalEpochs}
            </span>
          </div>
          <div className="h-2 rounded-full bg-muted/60">
            <div
              className="h-full rounded-full bg-neuro transition-all"
              style={{ width: `${prog}%` }}
            />
          </div>
          <div className="mt-2 flex gap-4 text-[11px] font-mono text-muted-foreground">
            {lastLog.loss !== undefined && <span>train: {lastLog.loss.toFixed(4)}</span>}
            {lastLog.valLoss !== undefined && <span>val: {lastLog.valLoss.toFixed(4)}</span>}
          </div>
        </GlassCard>
      )}

      {weights && (
        <GlassCard className="mt-4 border-neuro/30">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="h-5 w-5 text-green-400" />
            <span className="text-sm font-semibold text-green-400">Training Complete!</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-muted-foreground">Subjects: </span>
              <span className="font-mono">{weights.meta.n_subjects}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Samples: </span>
              <span className="font-mono">{weights.meta.n_samples}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Epochs: </span>
              <span className="font-mono">{weights.meta.epochs_trained}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Val MSE: </span>
              <span className="font-mono text-neuro">{weights.meta.val_mse.toFixed(4)}</span>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Download the JSON and replace W1/B1/W2/B2/W3/B3 in{" "}
            <span className="font-mono">src/lib/decoder/tfjs-decoder.ts</span>
          </p>
        </GlassCard>
      )}

      {error && (
        <GlassCard className="mt-4 border-destructive/40">
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <p className="text-sm">{error}</p>
          </div>
        </GlassCard>
      )}

      {logs.length > 0 && (
        <GlassCard className="mt-4">
          <p className="text-xs font-semibold text-muted-foreground mb-2">Log</p>
          <div className="h-48 overflow-y-auto font-mono text-[11px] space-y-0.5">
            {logs.map((log, i) => (
              <div
                key={i}
                className={
                  log.phase === "error"
                    ? "text-destructive"
                    : log.phase === "done"
                      ? "text-green-400"
                      : log.phase === "training"
                        ? "text-neuro"
                        : "text-muted-foreground"
                }
              >
                [{log.phase}]{" "}
                {log.message ??
                  (log.epoch
                    ? `epoch ${log.epoch}/${log.totalEpochs} — loss: ${log.loss?.toFixed(4)} val: ${log.valLoss?.toFixed(4)}`
                    : (log.error ?? ""))}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </GlassCard>
      )}
    </DashboardShell>
  );
}
