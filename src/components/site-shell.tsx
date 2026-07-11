import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { Brain, Github, LayoutDashboard, LogOut, Twitter, User as UserIcon } from "lucide-react";
import { ReactNode } from "react";
import { useSession } from "@/lib/auth-hooks";
import { supabase } from "@/integrations/supabase/client";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const NAV = [
  { to: "/", label: "Home" },
  { to: "/architecture", label: "Architecture" },
  { to: "/studio", label: "Studio" },
  { to: "/playground", label: "Playground" },
  { to: "/embeddings", label: "Embeddings" },
  { to: "/eeg2image", label: "EEG2Image" },
  { to: "/synthetic", label: "Synthetic" },
  { to: "/developers", label: "Developers" },
  { to: "/research", label: "Research" },
  { to: "/pricing", label: "Pricing" },
  { to: "/about", label: "About" },
] as const;

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-aurora opacity-70" />
      <div className="pointer-events-none fixed inset-0 -z-10 grid-bg" />
      <Nav />
      <main className="relative">{children}</main>
      <Footer />
    </div>
  );
}

function Nav() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/60 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4">
        <Link to="/" className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md bg-neuro-gradient glow">
            <Brain className="h-4 w-4 text-background" />
          </div>
          <span className="text-sm font-semibold tracking-tight">NeuroWeave</span>
          <span className="ml-1 rounded border border-border/80 bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            v0.9 beta
          </span>
        </Link>
        <nav className="hidden flex-1 items-center gap-1 md:flex">
          {NAV.map((n) => {
            const active = path === n.to || (n.to !== "/" && path.startsWith(n.to));
            return (
              <Link
                key={n.to}
                to={n.to}
                className={`rounded-md px-2.5 py-1.5 text-xs transition-colors ${
                  active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <Link
            to="/developers"
            className="hidden text-xs text-muted-foreground hover:text-foreground md:inline"
          >
            Docs
          </Link>
          <AuthControls />
        </div>
      </div>
    </header>
  );
}

function AuthControls() {
  const { session, loading } = useSession();
  const navigate = useNavigate();
  if (loading) return <div className="h-7 w-20" />;
  if (!session) {
    return (
      <>
        <Link
          to="/signin"
          className="rounded-md border border-border bg-card/40 px-3 py-1.5 text-xs font-medium hover:bg-card"
        >
          Sign in
        </Link>
        <Link
          to="/signup"
          className="rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow transition-transform hover:scale-[1.02]"
        >
          Sign up
        </Link>
      </>
    );
  }
  const email = session.user.email ?? "Account";
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 rounded-md border border-border bg-card/40 px-2.5 py-1.5 text-xs font-medium hover:bg-card">
          <UserIcon className="h-3.5 w-3.5 text-neuro" />
          <span className="max-w-[120px] truncate">{email}</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem asChild>
          <Link to="/dashboard">
            <LayoutDashboard className="mr-2 h-4 w-4" /> Dashboard
          </Link>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={async () => {
            await supabase.auth.signOut();
            navigate({ to: "/" });
          }}
        >
          <LogOut className="mr-2 h-4 w-4" /> Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function Footer() {
  return (
    <footer className="mt-32 border-t border-border/60">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-12 md:grid-cols-4">
        <div>
          <div className="flex items-center gap-2">
            <div className="grid h-7 w-7 place-items-center rounded-md bg-neuro-gradient">
              <Brain className="h-4 w-4 text-background" />
            </div>
            <span className="text-sm font-semibold">NeuroWeave</span>
          </div>
          <p className="mt-3 max-w-xs text-xs text-muted-foreground">
            Foundation models, embeddings, and synthetic neurodata infrastructure for the next
            decade of brain-computer software.
          </p>
          <div className="mt-4 flex gap-2 text-muted-foreground">
            <a className="rounded p-1.5 hover:bg-muted" href="#">
              <Github className="h-4 w-4" />
            </a>
            <a className="rounded p-1.5 hover:bg-muted" href="#">
              <Twitter className="h-4 w-4" />
            </a>
          </div>
        </div>
        <FooterCol
          title="Platform"
          items={[
            { to: "/playground", label: "API Playground" },
            { to: "/embeddings", label: "Embeddings Explorer" },
            { to: "/eeg2image", label: "EEG2Image" },
            { to: "/synthetic", label: "Synthetic Lab" },
          ]}
        />
        <FooterCol
          title="Developers"
          items={[
            { to: "/developers", label: "Documentation" },
            { to: "/developers", label: "SDKs" },
            { to: "/developers", label: "API Keys" },
            { to: "/research", label: "Research" },
          ]}
        />
        <FooterCol
          title="Company"
          items={[
            { to: "/about", label: "About" },
            { to: "/pricing", label: "Pricing" },
            { to: "/about", label: "Ethics" },
            { to: "/about", label: "Contact" },
          ]}
        />
      </div>
      <div className="border-t border-border/60">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 text-xs text-muted-foreground">
          <span>© {new Date().getFullYear()} NeuroWeave Labs, Inc.</span>
        </div>
      </div>
    </footer>
  );
}

function FooterCol({ title, items }: { title: string; items: { to: string; label: string }[] }) {
  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      <ul className="mt-3 space-y-2 text-sm">
        {items.map((it, i) => (
          <li key={i}>
            <Link to={it.to} className="text-muted-foreground hover:text-foreground">
              {it.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
