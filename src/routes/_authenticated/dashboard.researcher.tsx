import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Database, FlaskConical, FolderPlus } from "lucide-react";
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
  return { user, profile, researcher };
}

function ResearcherDashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dash-researcher"], queryFn: load });
  if (isLoading || !data) {
    return <DashboardShell fullName="…" role="researcher"><div className="text-sm text-muted-foreground">Loading…</div></DashboardShell>;
  }
  const name = data.profile?.full_name ?? data.user.email?.split("@")[0] ?? "researcher";
  const inst = data.researcher?.institution_name;
  return (
    <DashboardShell fullName={name} role="researcher">
      <Eyebrow>Researcher workspace</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
        Welcome, {name}
        {inst && <span className="ml-2 text-muted-foreground">· {inst}</span>}
      </h1>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Multi-channel decoder live feed across delta, theta, alpha, and beta bands.
      </p>

      <div className="mt-8">
        <GlassCard>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold"><Activity className="h-4 w-4 text-neuro" /> Live EEG · spectral bands</div>
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">montage · 10-20</span>
          </div>
          <EEGLive height={300} />
        </GlassCard>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <StatPill label="Active projects" value="4" />
        <StatPill label="Datasets" value="17" />
        <StatPill label="API keys" value="3" />
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <QuickAction icon={FolderPlus} title="New project" desc="Create a tracked experiment with versioned runs." to="/research" />
        <QuickAction icon={Database} title="Generate dataset" desc="Synthesize a labeled EEG corpus with the synth engine." to="/synthetic" />
      </div>
    </DashboardShell>
  );
}

function QuickAction({ icon: Icon, title, desc, to }: { icon: typeof FlaskConical; title: string; desc: string; to: string }) {
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