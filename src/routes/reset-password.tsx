import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export const Route = createFileRoute("/reset-password")({
  head: () => ({ meta: [{ title: "Reset password · NeuroWeave" }] }),
  component: ResetPassword,
});

function ResetPassword() {
  const [isRecovery, setIsRecovery] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.location.hash.includes("type=recovery")) setIsRecovery(true);
    const { data: sub } = supabase.auth.onAuthStateChange((evt) => {
      if (evt === "PASSWORD_RECOVERY") setIsRecovery(true);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  async function sendLink(e: React.FormEvent) {
    e.preventDefault();
    setError(null); setMsg(null); setLoading(true);
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${window.location.origin}/reset-password`,
    });
    setLoading(false);
    if (error) setError(error.message);
    else setMsg("Check your inbox for a reset link.");
  }

  async function updatePassword(e: React.FormEvent) {
    e.preventDefault();
    setError(null); setMsg(null); setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });
    setLoading(false);
    if (error) setError(error.message);
    else setMsg("Password updated. You can now sign in.");
  }

  return (
    <SiteShell>
      <div className="mx-auto max-w-md px-4 py-20">
        <Eyebrow>{isRecovery ? "Set new password" : "Reset password"}</Eyebrow>
        <h1 className="mt-4 text-3xl font-semibold tracking-tight">{isRecovery ? "Choose a new password" : "Forgot your password?"}</h1>

        <GlassCard className="mt-8">
          {!isRecovery ? (
            <form onSubmit={sendLink} className="space-y-3">
              <div className="space-y-1.5">
                <Label>Email</Label>
                <Input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
              </div>
              <button disabled={loading} className="w-full rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow disabled:opacity-60">
                {loading ? "Sending…" : "Send reset link"}
              </button>
            </form>
          ) : (
            <form onSubmit={updatePassword} className="space-y-3">
              <div className="space-y-1.5">
                <Label>New password</Label>
                <Input type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} />
              </div>
              <button disabled={loading} className="w-full rounded-md bg-neuro-gradient px-4 py-2.5 text-sm font-medium text-background glow disabled:opacity-60">
                {loading ? "Updating…" : "Update password"}
              </button>
            </form>
          )}
          {msg && <div className="mt-3 rounded border border-neuro/40 bg-neuro/10 px-3 py-2 text-xs">{msg}</div>}
          {error && <div className="mt-3 rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{error}</div>}
        </GlassCard>

        <p className="mt-6 text-center text-sm text-muted-foreground">
          <Link to="/signin" className="hover:text-foreground">Back to sign in</Link>
        </p>
      </div>
    </SiteShell>
  );
}