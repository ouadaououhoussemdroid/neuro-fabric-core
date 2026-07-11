import { Link, useNavigate } from "@tanstack/react-router";
import { Brain, LogOut } from "lucide-react";
import { ReactNode } from "react";
import { supabase } from "@/integrations/supabase/client";
import { NeuralBackground } from "@/components/neural-bg";
import { EmbedFallbackBadge } from "@/components/embed-fallback-badge";

type Role = "individual" | "researcher" | "enterprise";

const ROLE_STYLES: Record<Role, { label: string; cls: string }> = {
  individual: { label: "Individual", cls: "border-neuro/40 bg-neuro/10 text-neuro" },
  researcher: { label: "Researcher", cls: "border-violet-400/40 bg-violet-400/10 text-violet-300" },
  enterprise: { label: "Enterprise", cls: "border-amber-400/40 bg-amber-400/10 text-amber-300" },
};

export function DashboardShell({
  fullName,
  role,
  children,
}: {
  fullName: string;
  role: Role;
  children: ReactNode;
}) {
  const navigate = useNavigate();
  const r = ROLE_STYLES[role];

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-aurora opacity-60" />
      <div className="pointer-events-none fixed inset-0 -z-10 grid-bg" />
      <div className="pointer-events-none fixed inset-0 -z-10 opacity-30">
        <NeuralBackground density={50} />
      </div>

      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/60 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="grid h-7 w-7 place-items-center rounded-md bg-neuro-gradient glow">
              <Brain className="h-4 w-4 text-background" />
            </div>
            <span className="text-sm font-semibold tracking-tight">NeuroWeave</span>
            <span className="ml-1 rounded border border-border/80 bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              v0.9 beta
            </span>
          </Link>
          <div className="ml-auto flex items-center gap-2">
            <div className="hidden items-center gap-2 rounded-md border border-border bg-card/40 px-2.5 py-1.5 text-xs sm:flex">
              <span className="font-medium">{fullName}</span>
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${r.cls}`}
              >
                {r.label}
              </span>
            </div>
            <button
              onClick={async () => {
                await supabase.auth.signOut();
                navigate({ to: "/" });
              }}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs font-medium hover:bg-card"
            >
              <LogOut className="h-3.5 w-3.5" /> Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto max-w-7xl px-4 py-10">{children}</main>
      <EmbedFallbackBadge />
    </div>
  );
}
