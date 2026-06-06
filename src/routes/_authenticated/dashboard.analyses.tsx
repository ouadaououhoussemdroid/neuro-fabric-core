import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, Clock, Database, FileAudio, Trash2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";

export const Route = createFileRoute("/_authenticated/dashboard/analyses")({
  head: () => ({ meta: [{ title: "My Analyses · NeuroWeave" }] }),
  component: AnalysesPage,
});

type Analysis = {
  id: string;
  file_name: string;
  file_size_bytes: number;
  sample_rate: number;
  num_channels: number;
  num_samples: number;
  embedding_model: string;
  attention: number;
  workload: number;
  arousal: number;
  processing_time_ms: number;
  created_at: string;
};

async function loadAnalyses() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  
  const { data: profile } = await supabase
    .from("profiles")
    .select("full_name, role")
    .eq("id", user.id)
    .maybeSingle();

  const { data: analyses } = await supabase
    .from("eeg_analyses")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(50);

  return {
    analyses: (analyses ?? []) as Analysis[],
    profile: profile ?? { full_name: null, role: "individual" },
  };
}

function AnalysesPage() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["analyses"],
    queryFn: loadAnalyses,
  });

  const handleDelete = async (id: string) => {
    await supabase.from("eeg_analyses").delete().eq("id", id);
    refetch();
  };

  if (isLoading || !data) {
    return (
      <DashboardShell fullName="..." role="individual">
        <div className="text-sm text-muted-foreground">Loading analyses...</div>
      </DashboardShell>
    );
  }

  const name = data.profile.full_name ?? "User";
  const role = data.profile.role as "individual" | "researcher" | "enterprise";

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>History</Eyebrow>
      <div className="mt-4 flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight">My Analyses</h1>
        <span className="rounded border border-border bg-muted/40 px-2.5 py-1 font-mono text-xs text-muted-foreground">
          {data.analyses.length} total
        </span>
      </div>

      {data.analyses.length === 0 ? (
        <GlassCard className="mt-8 text-center py-16">
          <Database className="mx-auto h-10 w-10 text-muted-foreground/40" />
          <p className="mt-4 text-sm text-muted-foreground">No analyses yet.</p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            Upload an EEG file from the Playground to get started.
          </p>
        </GlassCard>
      ) : (
        <div className="mt-6 flex flex-col gap-3">
          {data.analyses.map((a) => (
            <AnalysisCard key={a.id} analysis={a} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </DashboardShell>
  );
}

function AnalysisCard({ analysis: a, onDelete }: { analysis: Analysis; onDelete: (id: string) => void }) {
  const date = new Date(a.created_at).toLocaleString();
  const sizeMB = (a.file_size_bytes / (1024 * 1024)).toFixed(2);
  const durationSec = (a.num_samples / a.sample_rate).toFixed(1);

  return (
    <GlassCard className="group">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-border bg-muted/40">
            <FileAudio className="h-5 w-5 text-neuro" />
          </div>
          <div>
            <p className="text-sm font-semibold truncate max-w-[220px]">{a.file_name}</p>
            <p className="text-xs text-muted-foreground">
              {sizeMB} MB · {a.num_channels} ch · {durationSec}s · {a.sample_rate} Hz
            </p>
          </div>
        </div>
        <button
          onClick={() => onDelete(a.id)}
          className="shrink-0 rounded p-1.5 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        <MetricBar label="Attention" value={a.attention} color="#60a5fa" />
        <MetricBar label="Workload" value={a.workload} color="#a78bfa" />
        <MetricBar label="Arousal" value={a.arousal} color="#fbbf24" />
      </div>

      <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" /> {date}
        </span>
        <span className="flex items-center gap-1">
          <Activity className="h-3 w-3" /> {a.processing_time_ms}ms · {a.embedding_model}
        </span>
      </div>
    </GlassCard>
  );
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-semibold" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-muted/60">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
