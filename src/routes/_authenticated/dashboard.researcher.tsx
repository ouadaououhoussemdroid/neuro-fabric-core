import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Clock, Database, FileAudio, FlaskConical, FolderPlus, History } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { EEGLive } from "@/components/eeg-live";
import { Eyebrow, GlassCard, StatPill } from "@/components/ui-bits";

export const Route = createFileRoute("/_authenticated/dashboard/researcher")({
  head: () => ({ meta: [{ title: "Researcher Dashboard · NeuroWeave" }] }),
  component: ResearcherDashboard,
});

async function load() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  const { data: profile } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
  const { data: researcher } = await supabase.from("researcher_profiles").select("*").eq("user_id", user.id).maybeSingle();
  const { count: totalAnalyses } = await supabase.from("eeg_analyses").select("id", { count: "exact", head: true });
  const { data: recentAnalyses } = await supabase.from("eeg_analyses").select("id, file_name, attention, workload, arousal, created_at, num_channels, sample_rate").order("created_at", { ascending: false }).limit(3);
  const { data: allFiles } = await supabase.from("eeg_analyses").select("file_name");
  const uniqueFormats = new Set((allFiles ?? []).map((f) => f.file_name.split(".").pop()?.toLowerCase() ?? "")).size;
  const memberSince = profile?.created_at ? new Date(profile.created_at).toLocaleDateString("en-US", { month: "short", year: "numeric" }) : "—";
  return { user, profile, researcher, totalAnalyses: totalAnalyses ?? 0, recentAnalyses: recentAnalyses ?? [], uniqueFormats, memberSince };
}

function ResearcherDashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dash-researcher"], queryFn: load });
  if (isLoading || !data) return <DashboardShell fullName="…" role="researcher"><div className="text-sm text-muted-foreground">Loading…</div></DashboardShell>;
  const name = data.profile?.full_name ?? data.user.email?.split("@")[0] ?? "researcher";
  const inst = data.researcher?.institution_name;
  const field = data.researcher?.research_field;
  return (
    <DashboardShell fullName={name} role="researcher">
      <Eyebrow>Researcher workspace</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">Welcome, {name}{inst && <span className="ml-2 text-muted-foreground text-xl">· {inst}</span>}</h1>
      {field && <p className="mt-1 text-xs text-muted-foreground">Field: {field}</p>}
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">Multi-channel decoder live feed across delta, theta, alpha, and beta bands.</p>
      <div className="mt-8"><GlassCard><div className="mb-3 flex items-center justify-between"><div className="flex items-center gap-2 text-sm font-semibold"><Activity className="h-4 w-4 text-neuro" /> Live EEG · spectral bands</div><span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">montage · 10-20</span></div><EEGLive height={300} /></GlassCard></div>
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <StatPill label="Total analyses" value={String(data.totalAnalyses)} />
        <StatPill label="File formats used" value={String(data.uniqueFormats)} />
        <StatPill label="Member since" value={data.memberSince} />
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <QuickAction icon={FolderPlus} title="New project" desc="Create a tracked experiment with versioned runs." to="/research" />
        <QuickAction icon={Database} title="Generate dataset" desc="Synthesize a labeled EEG corpus with the synth engine." to="/synthetic" />
        <QuickAction icon={History} title="My analyses" desc={`Browse your ${data.totalAnalyses} saved analyses.`} to="/dashboard/analyses" />
        <QuickAction icon={FileAudio} title="Upload EEG" desc="Upload a new EDF, CSV, or NPY file for analysis." to="/upload" />
      </div>
      {data.recentAnalyses.length > 0 && (
        <div className="mt-8">
          <div className="mb-3 flex items-center justify-between"><h2 className="text-sm font-semibold">Recent Analyses</h2><Link to="/dashboard/analyses" className="text-xs text-neuro hover:underline">View all →</Link></div>
          <div className="flex flex-col gap-2">
            {data.recentAnalyses.map((a) => (
              <GlassCard key={a.id} className="flex items-center justify-between py-3">
                <div className="flex items-center gap-2"><FileAudio className="h-4 w-4 text-muted-foreground" /><div><p className="text-sm font-medium truncate max-w-[140px]">{a.file_name}</p><p className="text-xs text-muted-foreground">{a.num_channels} ch · {a.sample_rate} Hz</p></div></div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground"><span>A:{Math.round(a.attention * 100)}%</span><span className="flex items-center gap-1"><Clock className="h-3 w-3" />{new Date(a.created_at).toLocaleDateString()}</span></div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}
    </DashboardShell>
  );
}

function QuickAction({ icon: Icon, title, desc, to }: { icon: typeof FlaskConical; title: string; desc: string; to: string }) {
  return (
    <Link to={to}><GlassCard className="group h-full transition-colors hover:border-neuro/40">
      <div className="flex items-center justify-between"><div className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-muted/40"><Icon className="h-5 w-5 text-neuro" /></div><ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-neuro" /></div>
      <h3 className="mt-4 text-base font-semibold tracking-tight">{title}</h3><p className="mt-1 text-sm text-muted-foreground">{desc}</p>
    </GlassCard></Link>
  );
}
