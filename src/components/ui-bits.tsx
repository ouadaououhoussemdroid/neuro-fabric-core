import { ReactNode } from "react";

export function Section({
  children,
  className = "",
  id,
}: {
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={`mx-auto max-w-7xl px-4 py-20 ${className}`}>
      {children}
    </section>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/40 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
      <span className="h-1.5 w-1.5 rounded-full bg-neuro animate-pulse-glow" />
      {children}
    </span>
  );
}

export function GlassCard({
  children,
  className = "",
  onClick,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
}) {
  return (
    <div className={`glass rounded-xl p-6 ${className}`} onClick={onClick}>
      {children}
    </div>
  );
}

export function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-lg px-4 py-3">
      <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-xl font-semibold tracking-tight">{value}</div>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  sub,
}: {
  eyebrow: string;
  title: string;
  sub?: string;
}) {
  return (
    <div className="mb-10">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight md:text-5xl">{title}</h1>
      {sub && <p className="mt-3 max-w-2xl text-base text-muted-foreground">{sub}</p>}
    </div>
  );
}
