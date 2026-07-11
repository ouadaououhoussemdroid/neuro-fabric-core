import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import {
  CheckCircle,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  Loader2,
  Plus,
  Trash2,
  XCircle,
} from "lucide-react";

export const Route = createFileRoute("/_authenticated/experiments")({
  component: ExperimentsPage,
});

type Experiment = {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  tags: string[];
  status: string;
  created_at: string;
  updated_at: string;
};
type Run = {
  id: string;
  experiment_id: string;
  name: string | null;
  status: string;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  notes: string | null;
  duration_ms: number | null;
  created_at: string;
};

async function loadAll() {
  const { data: userData } = await supabase.auth.getUser();
  const user = userData.user!;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", user.id)
    .maybeSingle();
  const { data: experiments } = await supabase
    .from("experiments")
    .select("*")
    .order("created_at", { ascending: false });
  const { data: runs } = await supabase
    .from("experiment_runs")
    .select("*")
    .order("created_at", { ascending: false });
  return { profile, experiments: (experiments ?? []) as Experiment[], runs: (runs ?? []) as Run[] };
}

function ExperimentsPage() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["experiments"], queryFn: loadAll });
  const [expanded, setExpanded] = useState<string | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  const createExp = useMutation({
    mutationFn: async () => {
      const { data: userData } = await supabase.auth.getUser();
      await supabase.from("experiments").insert({
        user_id: userData.user!.id,
        name: newName.trim(),
        description: newDesc.trim() || null,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      setShowNew(false);
      setNewName("");
      setNewDesc("");
    },
  });

  const deleteExp = useMutation({
    mutationFn: async (id: string) => {
      await supabase.from("experiments").delete().eq("id", id);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }),
  });

  const updateStatus = useMutation({
    mutationFn: async ({ id, status }: { id: string; status: string }) => {
      await supabase
        .from("experiments")
        .update({ status, updated_at: new Date().toISOString() })
        .eq("id", id);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }),
  });

  if (isLoading || !data)
    return (
      <DashboardShell fullName="…" role="researcher">
        <div className="text-sm text-muted-foreground">Loading…</div>
      </DashboardShell>
    );

  const name = data.profile?.full_name ?? "User";
  const role = (data.profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const runsMap = data.runs.reduce<Record<string, Run[]>>((acc, r) => {
    (acc[r.experiment_id] ??= []).push(r);
    return acc;
  }, {});

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>Research</Eyebrow>
      <div className="mt-4 flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight">Experiments</h1>
        <button
          onClick={() => setShowNew(true)}
          className="flex items-center gap-1.5 rounded-lg bg-neuro px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
        >
          <Plus className="h-4 w-4" /> New
        </button>
      </div>

      {showNew && (
        <GlassCard className="mt-4 border-neuro/30">
          <p className="text-sm font-semibold mb-3">New Experiment</p>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Experiment name"
            className="w-full rounded border border-border bg-background px-3 py-2 text-sm mb-2"
          />
          <textarea
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            placeholder="Description (optional)"
            className="w-full rounded border border-border bg-background px-3 py-2 text-sm mb-3 h-16 resize-none"
          />
          <div className="flex gap-2">
            <button
              onClick={() => createExp.mutate()}
              disabled={!newName.trim() || createExp.isPending}
              className="rounded bg-neuro px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-40"
            >
              {createExp.isPending ? "Creating…" : "Create"}
            </button>
            <button
              onClick={() => setShowNew(false)}
              className="rounded border border-border px-4 py-1.5 text-sm"
            >
              Cancel
            </button>
          </div>
        </GlassCard>
      )}

      {data.experiments.length === 0 ? (
        <GlassCard className="mt-8 text-center py-16">
          <FlaskConical className="mx-auto h-10 w-10 text-muted-foreground/40" />
          <p className="mt-4 text-sm text-muted-foreground">No experiments yet.</p>
        </GlassCard>
      ) : (
        <div className="mt-6 flex flex-col gap-3">
          {data.experiments.map((exp) => {
            const runs = runsMap[exp.id] ?? [];
            const isOpen = expanded === exp.id;
            return (
              <GlassCard key={exp.id} className="group">
                <div className="flex items-center justify-between gap-2">
                  <button
                    onClick={() => setExpanded(isOpen ? null : exp.id)}
                    className="flex items-center gap-2 flex-1 text-left"
                  >
                    {isOpen ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0" />
                    )}
                    <div>
                      <p className="text-sm font-semibold">{exp.name}</p>
                      {exp.description && (
                        <p className="text-xs text-muted-foreground">{exp.description}</p>
                      )}
                    </div>
                  </button>
                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className={`text-[11px] font-semibold capitalize ${exp.status === "active" ? "text-green-400" : exp.status === "completed" ? "text-neuro" : "text-muted-foreground"}`}
                    >
                      {exp.status}
                    </span>
                    <select
                      value={exp.status}
                      onChange={(e) => updateStatus.mutate({ id: exp.id, status: e.target.value })}
                      className="text-xs bg-muted/40 border border-border rounded px-1 py-0.5"
                    >
                      <option value="active">Active</option>
                      <option value="completed">Completed</option>
                      <option value="archived">Archived</option>
                    </select>
                    <button
                      onClick={() => deleteExp.mutate(exp.id)}
                      className="opacity-0 group-hover:opacity-100 rounded p-1 text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {isOpen && (
                  <div className="mt-4 border-t border-border pt-4">
                    <p className="text-xs font-semibold text-muted-foreground mb-2">
                      {runs.length} run{runs.length !== 1 ? "s" : ""}
                    </p>
                    {runs.length === 0 ? (
                      <p className="text-xs text-muted-foreground">No runs yet.</p>
                    ) : (
                      <div className="flex flex-col gap-2">
                        {runs.map((run) => (
                          <div
                            key={run.id}
                            className="rounded border border-border bg-muted/20 px-3 py-2"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                {run.status === "completed" ? (
                                  <CheckCircle className="h-3.5 w-3.5 text-green-400" />
                                ) : run.status === "failed" ? (
                                  <XCircle className="h-3.5 w-3.5 text-destructive" />
                                ) : (
                                  <Loader2 className="h-3.5 w-3.5 text-neuro animate-spin" />
                                )}
                                <span className="text-xs font-medium">
                                  {run.name ?? `Run ${run.id.slice(0, 6)}`}
                                </span>
                              </div>
                              <span className="text-[11px] text-muted-foreground">
                                {run.duration_ms ? `${run.duration_ms}ms` : "—"}
                              </span>
                            </div>
                            {Object.keys(run.metrics).length > 0 && (
                              <div className="mt-1.5 flex flex-wrap gap-2">
                                {Object.entries(run.metrics).map(([k, v]) => (
                                  <span key={k} className="font-mono text-[11px] text-neuro">
                                    {k}:{" "}
                                    {typeof v === "number" ? (v as number).toFixed(3) : String(v)}
                                  </span>
                                ))}
                              </div>
                            )}
                            {run.notes && (
                              <p className="mt-1 text-[11px] text-muted-foreground">{run.notes}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="mt-2 flex items-center gap-3 text-[11px] text-muted-foreground">
                  <span>{new Date(exp.created_at).toLocaleDateString()}</span>
                  <span>{runs.length} runs</span>
                </div>
              </GlassCard>
            );
          })}
        </div>
      )}
    </DashboardShell>
  );
}
