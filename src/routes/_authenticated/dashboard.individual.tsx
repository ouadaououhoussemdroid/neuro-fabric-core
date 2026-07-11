import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Clock, FileAudio, History, Sparkles, Zap } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { EEGLive } from "@/components/eeg-live";
import { Eyebrow, GlassCard, StatPill } from "@/components/ui-bits";

export const Route = createFileRoute("/_authenticated/dashboard/individual")({
  head: () => ({ meta: [{ title: "Individual Dashboard · NeuroWeave" }] }),
  component: IndividualDashboard,
});

async function load() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();
  const { count: totalAnalyses } = await supabase
    .from("eeg_analyses")
    .select("id", { count: "exact", head: true });
  const { data: analyses } = await supabase
    .from("eeg_analyses")
    .select("id, file_name, attention, workload, arousal, created_at, processing_time_ms")
    .order("created_at", { ascending: false })
    .limit(3);
  const { data: latencyData } = await supabase
    .from("eeg_analyses")
    .select("processing_time_ms")
    .order("created_at", { ascending: false })
    .limit(20);
  const avgLatency =
    latencyData && latencyData.length > 0
      ? Math.round(latencyData.reduce((s, r) => s + r.processing_time_ms, 0) / latencyData.length)
      : null;
  const memberSince = profile?.created_at
    ? new Date(profile.created_at).toLocaleDateString("en-US", { month: "short", year: "numeric" })
    : "—";
  return {
    user,
    profile,
    analyses: analyses ?? [],
    totalAnalyses: totalAnalyses ?? 0,
    avgLatency,
    memberSince,
  };
}

function IndividualDashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dash-individual"], queryFn: load });
  if (isLoading || !data)
    return (
      <DashboardShell fullName="…" role="individual">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </DashboardShell>
    );
  const name = data.profile?.full_name ?? data.user.email?.split("@")[0] ?? "there";
  return (
    <DashboardShell fullName={name} role="individual">
      <Eyebrow>Individual workspace</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
        Welcome back, {name}.
      </h1>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Live neural feed simulating a 4-band EEG decoder.
      </p>
      <div className="mt-8">
        <GlassCard>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Activity className="h-4 w-4 text-neuro" /> Live EEG · 4-band decoder
            </div>
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              channel · Cz
            </span>
          </div>
          <EEGLive />
        </GlassCard>
      </div>
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <StatPill label="Analyses total" value={String(data.totalAnalyses)} />
        <StatPill
          label="Avg latency"
          value={data.avgLatency !== null ? `${data.avgLatency} ms` : "—"}
        />
        <StatPill label="Member since" value={data.memberSince} />
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <QuickAction
          icon={Sparkles}
          title="Run embedding"
          desc="Generate a 768-d brain vector from a signal upload."
          to="/embeddings"
        />
        <QuickAction
          icon={Zap}
          title="Test API"
          desc="Send a request from the developer playground."
          to="/playground"
        />
        <QuickAction
          icon={History}
          title="My analyses"
          desc={`View your ${data.totalAnalyses} saved analyses.`}
          to="/dashboard/analyses"
        />
        <QuickAction
          icon={FileAudio}
          title="Upload EEG"
          desc="Upload a new EDF, CSV, or NPY file for analysis."
          to="/upload"
        />
      </div>
      {data.analyses.length > 0 && (
        <div className="mt-8">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent Analyses</h2>
            <Link to="/dashboard/analyses" className="text-xs text-neuro hover:underline">
              View all →
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            {data.analyses.map((a) => (
              <GlassCard key={a.id} className="flex items-center justify-between py-3">
                <div className="flex items-center gap-2">
                  <FileAudio className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium truncate max-w-[140px]">{a.file_name}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>A:{Math.round(a.attention * 100)}%</span>
                  <span>W:{Math.round(a.workload * 100)}%</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {new Date(a.created_at).toLocaleDateString()}
                  </span>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}
    </DashboardShell>
  );
}

function QuickAction({
  icon: Icon,
  title,
  desc,
  to,
}: {
  icon: typeof Activity;
  title: string;
  desc: string;
  to: string;
}) {
  return (
    <Link to={to}>
      <GlassCard className="group h-full transition-colors hover:border-neuro/40">
        <div className="flex items-center justify-between">
          <div className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-muted/40">
            <Icon className="h-5 w-5 text-neuro" />
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-neuro" />
        </div>
        <h3 className="mt-4 text-base font-semibold tracking-tight">{title}</h3>
        <p className="mt-1 text-sm text-muted-foreground">{desc}</p>
      </GlassCard>
    </Link>
  );
}
