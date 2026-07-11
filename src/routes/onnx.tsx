import { createFileRoute } from "@tanstack/react-router";
import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { useOnnxTrainer } from "@/hooks/use-onnx-trainer";
import { usePyodide } from "@/hooks/use-pyodide";
import { AlertTriangle, Brain, CheckCircle, Download, Loader2, Play, Zap } from "lucide-react";

export const Route = createFileRoute("/onnx")({
  component: OnnxPage,
});

async function loadProfile() {
  const { data: userData } = await supabase.auth.getUser();
  if (!userData.user) return null;
  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", userData.user.id)
    .maybeSingle();
  return profile;
}

const DATASETS = [
  { id: "BNCI2014_001", name: "BCI Competition IV 2a" },
  { id: "BNCI2014_004", name: "BCI Competition IV 2b" },
  { id: "PhysionetMI", name: "PhysioNet Motor Imagery" },
];

const CLASSIFIERS = [
  { id: "lda", name: "LDA", desc: "Fast, works well for EEG" },
  { id: "svm", name: "SVM", desc: "Higher accuracy, slower" },
  { id: "rf", name: "Random Forest", desc: "Robust to noise" },
];

function OnnxPage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const { state: pyState, initialize } = usePyodide();
  const { state, trainAndExport, infer, downloadOnnx, labels } = useOnnxTrainer();
  const [dataset, setDataset] = useState("BNCI2014_001");
  const [subjects, setSubjects] = useState("1");
  const [classifier, setClassifier] = useState<"lda" | "svm" | "rf">("lda");
  const [inferResult, setInferResult] = useState<{
    label: string;
    probabilities: Record<string, number>;
    latencyMs: number;
  } | null>(null);
  const [testFeatures, setTestFeatures] = useState("");

  const pyReady = pyState.status === "ready";
  const isTraining = ["installing", "extracting", "training", "exporting", "loading"].includes(
    state.phase,
  );
  const isReady = state.phase === "ready";

  const handleTrain = async () => {
    const subjList = subjects
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => n > 0);
    try {
      await trainAndExport(dataset, subjList, classifier);
    } catch (err) {
      console.error("onnx.trainAndExport failed", err);
    }
  };

  const handleInfer = useCallback(async () => {
    try {
      const features = testFeatures.split(",").map(Number).filter(Number.isFinite);
      if (features.length === 0) return;
      const result = await infer(features);
      setInferResult(result);
    } catch (err) {
      console.error(err);
    }
  }, [testFeatures, infer]);

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>ONNX Inference</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">Train & Export ONNX</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Trains Scikit-learn on MOABB data inside Pyodide, exports to ONNX, runs inference via
        onnxruntime-web — entirely in browser.
      </p>

      {!pyReady && (
        <GlassCard className="mt-6">
          <p className="text-xs font-semibold text-muted-foreground mb-3">
            Step 1 — Load Python Runtime
          </p>
          {pyState.status === "idle" ? (
            <button
              onClick={initialize}
              className="flex items-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white hover:opacity-90"
            >
              <Play className="h-4 w-4" />
              Load MNE-Python
            </button>
          ) : (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Loader2 className="h-4 w-4 animate-spin text-neuro" />
                <span className="text-sm text-neuro">{pyState.message}</span>
              </div>
              <div className="h-2 rounded-full bg-muted/60">
                <div
                  className="h-full rounded-full bg-neuro transition-all"
                  style={{ width: `${pyState.progress}%` }}
                />
              </div>
            </div>
          )}
        </GlassCard>
      )}

      {pyReady && (
        <GlassCard className="mt-6">
          <p className="text-xs font-semibold text-muted-foreground mb-3">
            Step 2 — Configure Training
          </p>
          <div className="mb-3">
            <label className="text-xs text-muted-foreground">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              disabled={isTraining}
              className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
            >
              {DATASETS.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </div>
          <div className="mb-3">
            <label className="text-xs text-muted-foreground">Subjects</label>
            <input
              value={subjects}
              onChange={(e) => setSubjects(e.target.value)}
              disabled={isTraining}
              className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm disabled:opacity-50"
              placeholder="1"
            />
          </div>
          <div className="mb-4">
            <label className="text-xs text-muted-foreground mb-2 block">Classifier</label>
            <div className="grid grid-cols-3 gap-2">
              {CLASSIFIERS.map((c) => (
                <button
                  key={c.id}
                  disabled={isTraining}
                  onClick={() => setClassifier(c.id as "lda" | "svm" | "rf")}
                  className={`rounded border px-3 py-2 text-left transition-colors disabled:opacity-50 ${classifier === c.id ? "border-neuro bg-neuro/10" : "border-border hover:border-neuro/40"}`}
                >
                  <p className="text-xs font-semibold">{c.name}</p>
                  <p className="text-[11px] text-muted-foreground">{c.desc}</p>
                </button>
              ))}
            </div>
          </div>
          <button
            onClick={handleTrain}
            disabled={isTraining}
            className="flex items-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40 hover:opacity-90"
          >
            {isTraining ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Training…
              </>
            ) : (
              <>
                <Brain className="h-4 w-4" />
                Train + Export ONNX
              </>
            )}
          </button>
        </GlassCard>
      )}

      {isTraining && (
        <GlassCard className="mt-4">
          <div className="flex items-center gap-2 mb-2">
            <Loader2 className="h-4 w-4 animate-spin text-neuro" />
            <span className="text-sm text-neuro">{state.message}</span>
          </div>
          <div className="h-2 rounded-full bg-muted/60">
            <div
              className="h-full rounded-full bg-neuro transition-all"
              style={{ width: `${state.progress}%` }}
            />
          </div>
        </GlassCard>
      )}

      {state.phase === "error" && (
        <GlassCard className="mt-4 border-destructive/40">
          <div className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <p className="text-sm">{state.error}</p>
          </div>
        </GlassCard>
      )}

      {isReady && (
        <div className="mt-4 flex flex-col gap-3">
          <GlassCard className="border-neuro/30">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-400" />
                <span className="text-sm font-semibold text-green-400">Model Ready</span>
              </div>
              <button
                onClick={() => downloadOnnx(`eeg-${classifier}-${dataset}.onnx`)}
                className="flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs hover:bg-muted/40"
              >
                <Download className="h-3.5 w-3.5" />
                Download .onnx
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-muted-foreground">Classifier: </span>
                <span className="font-mono">{classifier.toUpperCase()}</span>
              </div>
              <div>
                <span className="text-muted-foreground">CV Accuracy: </span>
                <span className="font-mono text-neuro">
                  {state.accuracy !== undefined ? `${(state.accuracy * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Model size: </span>
                <span className="font-mono">{state.modelSizeKB} KB</span>
              </div>
              <div>
                <span className="text-muted-foreground">Labels: </span>
                <span className="font-mono">{labels.join(", ")}</span>
              </div>
            </div>
          </GlassCard>

          <GlassCard>
            <p className="text-xs font-semibold text-muted-foreground mb-3">⚡ Test Inference</p>
            <label className="text-xs text-muted-foreground">
              Band-power features (comma-separated)
            </label>
            <textarea
              value={testFeatures}
              onChange={(e) => setTestFeatures(e.target.value)}
              className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-xs font-mono h-16 resize-none"
              placeholder="0.12, 0.08, 0.35, 0.28, 0.17, ..."
            />
            <button
              onClick={handleInfer}
              disabled={!testFeatures.trim() || state.phase === "inferring"}
              className="mt-3 flex items-center gap-2 rounded-lg bg-neuro px-4 py-2 text-sm font-semibold text-white disabled:opacity-40 hover:opacity-90"
            >
              {state.phase === "inferring" ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running…
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" />
                  Run Inference
                </>
              )}
            </button>
            {inferResult && (
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold">
                    Predicted: <span className="text-neuro font-mono">{inferResult.label}</span>
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    {inferResult.latencyMs}ms
                  </span>
                </div>
                {Object.entries(inferResult.probabilities).map(([label, prob]) => (
                  <div key={label} className="mb-1.5">
                    <div className="flex justify-between text-[11px] mb-0.5">
                      <span className="font-mono text-muted-foreground">{label}</span>
                      <span className="font-mono text-neuro">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-muted/60">
                      <div
                        className="h-full rounded-full bg-neuro"
                        style={{ width: `${(prob * 100).toFixed(1)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </GlassCard>
        </div>
      )}
    </DashboardShell>
  );
}
