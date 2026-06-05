import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Building2, CreditCard, UserPlus } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { EEGLive } from "@/components/eeg-live";
import { Eyebrow, GlassCard, StatPill } from "@/components/ui-bits";

export const Route = createFileRoute("/_authenticated/dashboard/enterprise")({
  head: () => ({ meta: [{ title: "Enterprise Dashboard · NeuroWeave" }] }),
  component: EnterpriseDashboard,
});

async function load() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  const { data: profile } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
  const { data: enterprise } = await supabase.from("enterprise_profiles").select("*").eq("user_id", user.id).maybeSingle();
  return { user, profile, enterprise };
}

function EnterpriseDashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dash-enterprise"], queryFn: load });
  if (isLoading || !data) {
    return <DashboardShell fullName="…" role="enterprise"><div className="text-sm text-muted-foreground">Loading…</div></DashboardShell>;
  }
  const name = data.profile?.full_name ?? data.user.email?.split("@")[0] ?? "team";
  const company = data.enterprise?.company_name;
  return (
    <DashboardShell fullName={name} role="enterprise">
      <Eyebrow>Enterprise workspace</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
        {company ? `Welcome, ${company}` : `Welcome, ${name}`}
      </h1>
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Aggregated live neural feed across your team's active sessions.
      </p>

      <div className="mt-8">
        <GlassCard>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold"><Activity className="h-4 w-4 text-neuro" /> Live EEG · team aggregate</div>
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">streams · 12 active</span>
          </div>
          <EEGLive height={300} />
        </GlassCard>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <StatPill label="Team members" value="14" />
        <StatPill label="Total API usage" value="3.42M" />
        <StatPill label="Billing cycle" value="Jun 1 – Jun 30" />
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <QuickAction icon={UserPlus} title="Invite member" desc="Add a teammate and assign workspace permissions." to="/about" />
        <QuickAction icon={CreditCard} title="View billing" desc="Inspect usage, invoices, and current plan limits." to="/pricing" />
      </div>
    </DashboardShell>
  );
}

function QuickAction({ icon: Icon, title, desc, to }: { icon: typeof Building2; title: string; desc: string; to: string }) {
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