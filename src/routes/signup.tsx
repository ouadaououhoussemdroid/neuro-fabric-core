import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Brain, Building2, FlaskConical, User } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { lovable } from "@/integrations/lovable";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/signup")({
  head: () => ({ meta: [{ title: "Sign up · NeuroWeave" }] }),
  component: SignUp,
});

type Role = "individual" | "researcher" | "enterprise";

function SignUp() {
  const nav = useNavigate();
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [role, setRole] = useState<Role>("individual");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  // researcher
  const [institution, setInstitution] = useState("");
  const [publicationUrl, setPublicationUrl] = useState("");
  const [researchField, setResearchField] = useState("");
  // enterprise
  const [companyName, setCompanyName] = useState("");
  const [companySize, setCompanySize] = useState("");
  const [website, setWebsite] = useState("");
  const [industry, setIndustry] = useState("");

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onGoogle() {
    setError(null);
    const res = await lovable.auth.signInWithOAuth("google", { redirect_uri: window.location.origin + "/dashboard" });
    if (res.error) setError(res.error.message);
  }

  async function onFinish() {
    setError(null);
    if (role === "researcher" && !institution.trim()) return setError("Institution is required");
    if (role === "enterprise" && !companyName.trim()) return setError("Company name is required");
    setLoading(true);
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { emailRedirectTo: `${window.location.origin}/dashboard`, data: { full_name: fullName } },
      });
      if (error) throw error;
      const userId = data.user?.id;
      if (!userId) throw new Error("Account created. Please confirm your email, then sign in.");

      // Insert profile (needs active session — will exist if email confirmation is off)
      const { error: pErr } = await supabase.from("profiles").insert({ id: userId, role, full_name: fullName });
      if (pErr) throw pErr;

      if (role === "researcher") {
        const { error: rErr } = await supabase.from("researcher_profiles").insert({
          user_id: userId,
          institution_name: institution,
          publication_url: publicationUrl || null,
          research_field: researchField || null,
        });
        if (rErr) throw rErr;
      } else if (role === "enterprise") {
        const { error: eErr } = await supabase.from("enterprise_profiles").insert({
          user_id: userId,
          company_name: companyName,
          company_size: companySize || null,
          website: website || null,
          industry: industry || null,
        });
        if (eErr) throw eErr;
      }

      nav({ to: "/dashboard" });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sign up failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SiteShell>
      <div className="mx-auto max-w-xl px-4 py-16">
        <div className="mb-6 flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-md bg-neuro-gradient glow">
            <Brain className="h-4 w-4 text-background" />
          </div>
          <Eyebrow>Sign up · step {step} of 3</Eyebrow>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">Create your NeuroWeave account</h1>

        <GlassCard className="mt-8">
          {step === 1 && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">Choose the account type that best fits you.</p>
              {[
                { id: "individual", icon: User, title: "Individual", desc: "Personal access to the playground, embeddings, and reconstruction tools." },
                { id: "researcher", icon: FlaskConical, title: "Researcher", desc: "Academic features, dataset linking, publication tracking, citations." },
                { id: "enterprise", icon: Building2, title: "Enterprise", desc: "Team seats, API keys, governance, and SLA-backed inference." },
              ].map((opt) => {
                const Icon = opt.icon;
                const active = role === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setRole(opt.id as Role)}
                    className={`flex w-full items-start gap-3 rounded-lg border p-4 text-left transition-colors ${active ? "border-neuro/60 bg-neuro/10" : "border-border bg-card/30 hover:bg-card/60"}`}
                  >
                    <div className="grid h-10 w-10 place-items-center rounded-md border border-border bg-muted/40">
                      <Icon className="h-4 w-4 text-neuro" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{opt.title}</div>
                      <div className="mt-1 text-xs text-muted-foreground">{opt.desc}</div>
                    </div>
                  </button>
                );
              })}
              <button onClick={() => setStep(2)} className="w-full rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow">Continue</button>
            </div>
          )}

          {step === 2 && (
            <div className="space-y-3">
              <button type="button" onClick={onGoogle} className="w-full rounded-md border border-border bg-card/40 px-4 py-2.5 text-sm font-medium hover:bg-card">
                Continue with Google
              </button>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <div className="h-px flex-1 bg-border" /> or <div className="h-px flex-1 bg-border" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="fn">Full name</Label>
                <Input id="fn" required value={fullName} onChange={(e) => setFullName(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="em">Email</Label>
                <Input id="em" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pw">Password</Label>
                <Input id="pw" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} />
              </div>
              <div className="flex gap-2">
                <button onClick={() => setStep(1)} className="rounded-md border border-border bg-card/40 px-4 py-2.5 text-sm hover:bg-card">Back</button>
                <button
                  onClick={() => {
                    if (!fullName || !email || password.length < 8) return setError("Fill all fields (password ≥ 8 chars)");
                    setError(null);
                    if (role === "individual") onFinish();
                    else setStep(3);
                  }}
                  className="flex-1 rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow"
                >
                  {role === "individual" ? (loading ? "Creating…" : "Create account") : "Continue"}
                </button>
              </div>
            </div>
          )}

          {step === 3 && role === "researcher" && (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label>Institution name</Label>
                <Input required value={institution} onChange={(e) => setInstitution(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Publication URL</Label>
                <Input type="url" placeholder="https://" value={publicationUrl} onChange={(e) => setPublicationUrl(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <Label>Research field</Label>
                <Input placeholder="e.g. computational neuroscience" value={researchField} onChange={(e) => setResearchField(e.target.value)} />
              </div>
              <div className="flex gap-2">
                <button onClick={() => setStep(2)} className="rounded-md border border-border bg-card/40 px-4 py-2.5 text-sm hover:bg-card">Back</button>
                <button onClick={onFinish} disabled={loading} className="flex-1 rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow disabled:opacity-60">
                  {loading ? "Creating…" : "Create account"}
                </button>
              </div>
            </div>
          )}

          {step === 3 && role === "enterprise" && (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label>Company name</Label>
                <Input required value={companyName} onChange={(e) => setCompanyName(e.target.value)} />
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-1.5">
                  <Label>Company size</Label>
                  <Input placeholder="e.g. 11–50" value={companySize} onChange={(e) => setCompanySize(e.target.value)} />
                </div>
                <div className="space-y-1.5">
                  <Label>Industry</Label>
                  <Input placeholder="e.g. healthtech" value={industry} onChange={(e) => setIndustry(e.target.value)} />
                </div>
              </div>
              <div className="space-y-1.5">
                <Label>Website</Label>
                <Input type="url" placeholder="https://" value={website} onChange={(e) => setWebsite(e.target.value)} />
              </div>
              <div className="flex gap-2">
                <button onClick={() => setStep(2)} className="rounded-md border border-border bg-card/40 px-4 py-2.5 text-sm hover:bg-card">Back</button>
                <button onClick={onFinish} disabled={loading} className="flex-1 rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow disabled:opacity-60">
                  {loading ? "Creating…" : "Create account"}
                </button>
              </div>
            </div>
          )}

          {error && <div className="mt-3 rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div>}
        </GlassCard>

        <p className="mt-6 text-center text-sm text-muted-foreground">
          Already have an account? <Link to="/signin" className="text-foreground hover:underline">Sign in</Link>
        </p>
      </div>
    </SiteShell>
  );
}