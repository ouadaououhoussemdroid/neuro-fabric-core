import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Building2, FlaskConical, KeyRound, Mail, User as UserIcon, Users } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, Eyebrow, StatPill } from "@/components/ui-bits";

export const Route = createFileRoute("/_authenticated/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard · NeuroWeave" }] }),
  component: Dashboard,
});

async function loadDashboard() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  const { data: profile } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
  let researcher = null, enterprise = null;
  if (profile?.role === "researcher") {
    const { data } = await supabase.from("researcher_profiles").select("*").eq("user_id", user.id).maybeSingle();
    researcher = data;
  } else if (profile?.role === "enterprise") {
    const { data } = await supabase.from("enterprise_profiles").select("*").eq("user_id", user.id).maybeSingle();
    enterprise = data;
  }
  return { user, profile, researcher, enterprise };
}

function Dashboard() {
  const { data, isLoading } = useQuery({ queryKey: ["dashboard"], queryFn: loadDashboard });

  if (isLoading || !data) {
    return (
      <SiteShell>
        <div className="mx-auto max-w-7xl px-4 py-20 text-sm text-muted-foreground">Loading…</div>
      </SiteShell>
    );
  }

  const { user, profile, researcher, enterprise } = data;
  const role = profile?.role ?? "individual";
  const name = profile?.full_name ?? user.email?.split("@")[0] ?? "there";
  const memberSince = profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : "—";

  return (
    <SiteShell>
      <div className="mx-auto max-w-7xl px-4 py-12">
        <Eyebrow>{role} dashboard</Eyebrow>
        <h1 className="mt-4 text-3xl font-semibold tracking-tight md:text-4xl">
          {role === "enterprise" && enterprise ? `Welcome, ${enterprise.company_name}` : `Welcome, ${name}`}
          {role === "researcher" && researcher && <span className="ml-2 text-muted-foreground">· {researcher.institution_name}</span>}
        </h1>

        <div className="mt-8 grid gap-6 lg:grid-cols-3">
          <GlassCard className="lg:col-span-1">
            <div className="flex items-center gap-3">
              <div className="grid h-12 w-12 place-items-center rounded-lg bg-neuro-gradient text-background">
                {role === "enterprise" ? <Building2 className="h-5 w-5" /> : role === "researcher" ? <FlaskConical className="h-5 w-5" /> : <UserIcon className="h-5 w-5" />}
              </div>
              <div>
                <div className="text-sm font-semibold">{name}</div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground"><Mail className="h-3 w-3" /> {user.email}</div>
              </div>
            </div>
            <div className="mt-5 space-y-2 border-t border-border/60 pt-4 text-xs text-muted-foreground">
              <div className="flex justify-between"><span>Member since</span><span className="text-foreground">{memberSince}</span></div>
              <div className="flex justify-between"><span>Account type</span><span className="font-mono uppercase text-foreground">{role}</span></div>
              {researcher && (
                <>
                  <div className="flex justify-between"><span>Research field</span><span className="text-foreground">{researcher.research_field ?? "—"}</span></div>
                  {researcher.publication_url && (
                    <div className="flex justify-between"><span>Publications</span><a href={researcher.publication_url} target="_blank" rel="noreferrer" className="text-neuro hover:underline">Link</a></div>
                  )}
                </>
              )}
              {enterprise && (
                <>
                  <div className="flex justify-between"><span>Industry</span><span className="text-foreground">{enterprise.industry ?? "—"}</span></div>
                  <div className="flex justify-between"><span>Size</span><span className="text-foreground">{enterprise.company_size ?? "—"}</span></div>
                  {enterprise.website && (
                    <div className="flex justify-between"><span>Website</span><a href={enterprise.website} target="_blank" rel="noreferrer" className="text-neuro hover:underline">Visit</a></div>
                  )}
                </>
              )}
            </div>
          </GlassCard>

          <div className="lg:col-span-2 space-y-6">
            <div className="grid gap-3 sm:grid-cols-3">
              {role === "individual" && (
                <>
                  <StatPill label="Total sessions" value="0" />
                  <StatPill label="Last analysis" value="Never" />
                  <StatPill label="Account type" value="Individual" />
                </>
              )}
              {role === "researcher" && (
                <>
                  <StatPill label="Total sessions" value="0" />
                  <StatPill label="Publications linked" value={researcher?.publication_url ? "1" : "0"} />
                  <StatPill label="Institution" value={researcher?.institution_name ?? "—"} />
                </>
              )}
              {role === "enterprise" && (
                <>
                  <StatPill label="Team members" value="0" />
                  <StatPill label="API keys" value="0" />
                  <StatPill label="Total sessions" value="0" />
                </>
              )}
            </div>

            {role === "enterprise" ? (
              <>
                <GlassCard>
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-semibold"><Users className="h-4 w-4 text-neuro" /> Team members</div>
                    <button className="rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs hover:bg-card">Invite team member</button>
                  </div>
                  <div className="rounded-lg border border-dashed border-border/60 px-4 py-10 text-center text-sm text-muted-foreground">No team members yet.</div>
                </GlassCard>
                <GlassCard>
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm font-semibold"><KeyRound className="h-4 w-4 text-neuro" /> API keys</div>
                    <button className="rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs hover:bg-card">Generate API key</button>
                  </div>
                  <div className="rounded-lg border border-dashed border-border/60 px-4 py-10 text-center text-sm text-muted-foreground">No API keys generated yet.</div>
                </GlassCard>
              </>
            ) : (
              <GlassCard>
                <div className="mb-3 flex items-center gap-2 text-sm font-semibold"><Activity className="h-4 w-4 text-neuro" /> Session history</div>
                <div className="rounded-lg border border-dashed border-border/60 px-4 py-10 text-center text-sm text-muted-foreground">
                  Run your first EEG analysis in the Playground.
                </div>
              </GlassCard>
            )}

            <Link to="/playground" className="inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-5 py-3 text-sm font-medium text-background glow">
              Go to Playground <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </SiteShell>
  );
}