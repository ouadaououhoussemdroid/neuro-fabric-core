import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Brain } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { lovable } from "@/integrations/lovable";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/signin")({
  head: () => ({ meta: [{ title: "Sign in · NeuroWeave" }] }),
  component: SignIn,
});

function SignIn() {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setLoading(false);
    if (error) return setError(error.message);
    nav({ to: "/dashboard" });
  }

  async function onGoogle() {
    setError(null);
    const res = await lovable.auth.signInWithOAuth("google", {
      redirect_uri: window.location.origin + "/dashboard",
    });
    if (res.error) setError(res.error.message);
    else if (!("redirected" in res && res.redirected)) nav({ to: "/dashboard" });
  }

  return (
    <SiteShell>
      <div className="mx-auto max-w-md px-4 py-20">
        <div className="mb-6 flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-md bg-neuro-gradient glow">
            <Brain className="h-4 w-4 text-background" />
          </div>
          <Eyebrow>Sign in</Eyebrow>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">Welcome back</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Sign in to access your NeuroWeave dashboard.
        </p>

        <GlassCard className="mt-8 space-y-4">
          <button
            type="button"
            onClick={onGoogle}
            className="w-full rounded-md border border-border bg-card/40 px-4 py-2.5 text-sm font-medium hover:bg-card"
          >
            Continue with Google
          </button>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <div className="h-px flex-1 bg-border" /> or <div className="h-px flex-1 bg-border" />
          </div>
          <form onSubmit={onSubmit} className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            {error && (
              <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {error}
              </div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow disabled:opacity-60"
            >
              {loading ? "Signing in…" : "Sign in"}
            </button>
          </form>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <Link to="/reset-password" className="hover:text-foreground">
              Forgot password?
            </Link>
            <Link to="/signup" className="hover:text-foreground">
              Create account
            </Link>
          </div>
        </GlassCard>
      </div>
    </SiteShell>
  );
}
