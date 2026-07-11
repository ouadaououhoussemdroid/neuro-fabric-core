import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Building2,
  Clock,
  CreditCard,
  FileAudio,
  History,
  UserPlus,
} from "lucide-react";
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
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();
  const { data: enterprise } = await supabase
    .from("enterprise_profiles")
    .select("*")
    .eq("user_id", user.id)
    .maybeSingle();
  const { count: totalAnalyses } = await supabase
    .from("eeg_analyses")
    .select("id", { count: "exact", head: true });
  const { data: recentAnalyses } = await supabase
    .from("eeg_analyses")
    .select("id, file_name, attention, workload, arousal, created_at, file_size_bytes")
    .order("created_at", { ascending: false })
    .limit(3);
  const { data: sizeData } = await supabase.from("eeg_analyses").select("file_size_bytes");
  const totalMB = sizeData
    ? (sizeData.reduce((s, r) => s + r.file_size_bytes, 0) / 1024 / 1024).toFixed(1)
    : "0";
  const now = new Date();
  const billingStart = new Date(now.getFullYear(), now.getMonth(), 1).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
  const billingEnd = new Date(now.getFullYear(), now.getMonth() + 1, 0).toLocaleDateString(
    "en-US",
    { month: "short", day: "numeric" },
  );
  return {
    user,
    profile,
    enterprise,
    totalAnalyses: totalAnalyses ?? 0,
    recentAnalyses: recentAnalyses ?? [],
    totalMB,
    billingCycle: `${billingStart} – ${billingEnd}`,
  };
}

function EnterpriseDashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dash-enterprise"], queryFn: load });
  if (isLoading || !data)
    return (
      <DashboardShell fullName="…" role="enterprise">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </DashboardShell>
    );
  const name = data.profile?.full_name ?? data.user.email?.split("@")[0] ?? "team";
  const company = data.enterprise?.company_name;
  const industry = data.enterprise?.industry;
  return (
    <DashboardShell fullName={name} role="enterprise">
      <Eyebrow>Enterprise workspace</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
        {company ? `Welcome, ${company}` : `Welcome, ${name}`}
      </h1>
      {industry && <p className="mt-1 text-xs text-muted-foreground">Industry: {industry}</p>}
      <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
        Aggregated live neural feed across your team's active sessions.
      </p>
      <div className="mt-8">
        <GlassCard>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <Activity className="h-4 w-4 text-neuro" /> Live EEG · team aggregate
            </div>
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              streams · active
            </span>
          </div>
          <EEGLive height={300} />
        </GlassCard>
      </div>
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <StatPill label="Total analyses" value={String(data.totalAnalyses)} />
        <StatPill label="Data processed" value={`${data.totalMB} MB`} />
        <StatPill label="Billing cycle" value={data.billingCycle} />
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <QuickAction
          icon={UserPlus}
          title="Invite member"
          desc="Add a teammate and assign workspace permissions."
          to="/about"
        />
        <QuickAction
          icon={CreditCard}
          title="View billing"
          desc="Inspect usage, invoices, and current plan limits."
          to="/pricing"
        />
        <QuickAction
          icon={History}
          title="All analyses"
          desc={`Browse ${data.totalAnalyses} analyses across the workspace.`}
          to="/dashboard/analyses"
        />
        <QuickAction
          icon={FileAudio}
          title="Upload EEG"
          desc="Upload a new EDF, CSV, or NPY file for analysis."
          to="/upload"
        />
      </div>
      {data.recentAnalyses.length > 0 && (
        <div className="mt-8">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent Activity</h2>
            <Link to="/dashboard/analyses" className="text-xs text-neuro hover:underline">
              View all →
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            {data.recentAnalyses.map((a) => (
              <GlassCard key={a.id} className="flex items-center justify-between py-3">
                <div className="flex items-center gap-2">
                  <FileAudio className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium truncate max-w-[140px]">{a.file_name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(a.file_size_bytes / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span>A:{Math.round(a.attention * 100)}%</span>
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
  icon: typeof Building2;
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
