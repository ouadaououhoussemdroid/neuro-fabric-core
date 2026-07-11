import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { getModelsByType, ACTIVE_DECODER, ACTIVE_EMBEDDER } from "@/lib/model-registry";
import { Brain, CheckCircle, FlaskConical, Sparkles } from "lucide-react";

export const Route = createFileRoute("/models")({
  component: ModelsPage,
});

async function loadProfile() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user;
  if (!user) return null;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();
  return profile;
}

function ModelsPage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const decoders = getModelsByType("decoder");
  const embedders = getModelsByType("embedder");

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>System</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">Model Registry</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        All registered decoder and embedder models.
      </p>

      <GlassCard className="mt-6 border-neuro/30 bg-neuro/5">
        <p className="text-xs font-semibold text-neuro mb-2">⚡ Currently Active</p>
        <div className="flex gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Decoder: </span>
            <span className="font-mono font-semibold">{ACTIVE_DECODER}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Embedder: </span>
            <span className="font-mono font-semibold">{ACTIVE_EMBEDDER}</span>
          </div>
        </div>
      </GlassCard>

      <div className="mt-8">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="h-4 w-4 text-neuro" />
          <h2 className="text-sm font-semibold">Decoders</h2>
        </div>
        <div className="flex flex-col gap-3">
          {decoders.map((m) => (
            <ModelCard key={m.id} model={m} isActive={m.id === ACTIVE_DECODER} />
          ))}
        </div>
      </div>

      <div className="mt-8">
        <div className="flex items-center gap-2 mb-3">
          <Sparkles className="h-4 w-4 text-neuro" />
          <h2 className="text-sm font-semibold">Embedders</h2>
        </div>
        <div className="flex flex-col gap-3">
          {embedders.map((m) => (
            <ModelCard key={m.id} model={m} isActive={m.id === ACTIVE_EMBEDDER} />
          ))}
        </div>
      </div>
    </DashboardShell>
  );
}

function ModelCard({
  model: m,
  isActive,
}: {
  model: ReturnType<typeof getModelsByType>[0];
  isActive: boolean;
}) {
  return (
    <GlassCard className={isActive ? "border-neuro/40" : ""}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-semibold">{m.id}</span>
            <span className="text-xs text-muted-foreground">v{m.version}</span>
            {isActive && (
              <span className="flex items-center gap-1 text-[11px] text-neuro font-semibold">
                <CheckCircle className="h-3 w-3" /> Active
              </span>
            )}
            {m.isExperimental && (
              <span className="flex items-center gap-1 text-[11px] text-amber-400 font-semibold">
                <FlaskConical className="h-3 w-3" /> Experimental
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{m.description}</p>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
        <div>
          <span className="text-muted-foreground">Input: </span>
          <span className="font-mono">{m.inputShape}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Output: </span>
          <span className="font-mono">{m.outputShape}</span>
        </div>
        {m.metrics &&
          Object.entries(m.metrics).map(([k, v]) => (
            <div key={k}>
              <span className="text-muted-foreground">{k}: </span>
              <span className="font-mono text-neuro">
                {typeof v === "number" ? v.toFixed(1) : String(v)}
              </span>
            </div>
          ))}
      </div>
    </GlassCard>
  );
}
