import { createFileRoute } from "@tanstack/react-router";
import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { DashboardShell } from "@/components/dashboard-shell";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { usePyodide } from "@/hooks/use-pyodide";
import {
  AlertTriangle,
  Brain,
  CheckCircle,
  FlaskConical,
  Loader2,
  Play,
  Save,
  Upload,
} from "lucide-react";

export const Route = createFileRoute("/_authenticated/mne")({
  component: MnePage,
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

async function loadExperiments() {
  const { data } = await supabase
    .from("experiments")
    .select("id, name")
    .eq("status", "active")
    .order("created_at", { ascending: false });
  return data ?? [];
}

interface MNEResult {
  n_channels: number;
  n_times: number;
  sfreq: number;
  ch_names: string[];
  band_powers: Record<string, number[]>;
  epochs_count: number;
  artifacts_rejected: number;
  attention: number;
  workload: number;
  arousal: number;
  mne_version: string;
  duration_ms: number;
}

const PYTHON_CODE = `
import numpy as np, mne, json
from io import BytesIO, StringIO
import csv as csv_mod

filename = eeg_filename
sfreq = float(eeg_sfreq)
buf = BytesIO(bytes(eeg_bytes))

try:
    if filename.lower().endswith(('.edf', '.bdf')):
        raw = mne.io.read_raw_edf(buf, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
    elif filename.lower().endswith('.csv'):
        text = bytes(eeg_bytes).decode('utf-8')
        rows = list(csv_mod.reader(StringIO(text)))
        header = rows[0] if not rows[0][0].replace('.','',1).lstrip('-').isdigit() else None
        data_rows = rows[1:] if header else rows
        data = np.array([[float(v) for v in r if v.strip()] for r in data_rows if r]).T
        ch_names = header[:data.shape[0]] if header else [f'ch{i}' for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=False)
    else:
        raise ValueError(f"Unsupported: {filename}")
except Exception as e:
    raise RuntimeError(f"Parse error: {e}")

raw.set_eeg_reference('average', projection=False, verbose=False)
raw.filter(1.0, 40.0, method='iir', verbose=False)
raw.notch_filter(freqs=60, verbose=False)

epoch_len = 2.0; overlap = 0.5; step = epoch_len*(1-overlap)
n_samples = raw.n_times; ep_samples = int(epoch_len*sfreq); st_samples = int(step*sfreq)
epochs_data = [raw.get_data(start=s, stop=s+ep_samples) for s in range(0, n_samples-ep_samples, st_samples)]
epochs_array = np.array(epochs_data)
ch_std = np.std(epochs_array, axis=(0,2), keepdims=True)
ch_mean = np.mean(epochs_array, axis=(0,2), keepdims=True)
mask = np.all(np.abs(epochs_array-ch_mean) < 5*ch_std, axis=(1,2))
clean = epochs_array[mask]; rejected = len(epochs_array)-len(clean)
if len(clean)==0: raise RuntimeError("All epochs rejected")

from scipy import signal as sp
bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}
band_powers = {b:[] for b in bands}
for epoch in clean:
    for ch in epoch:
        freqs,psd = sp.welch(ch, fs=sfreq, nperseg=min(256,len(ch)))
        for band,(lo,hi) in bands.items():
            idx = np.logical_and(freqs>=lo, freqs<=hi)
            band_powers[band].append(float(np.mean(psd[idx])))

mean_bp = {b:float(np.mean(v)) for b,v in band_powers.items()}
total = sum(mean_bp.values())+1e-9
norm = {b:v/total for b,v in mean_bp.items()}
def sig(x): return float(1/(1+np.exp(-np.log(max(1e-9,x)))))
attention = sig(norm['beta']/max(1e-9, norm['alpha']+norm['theta']))
workload  = sig(norm['theta']/max(1e-9, norm['alpha']))
arousal   = float(min(1, norm['beta']+norm['gamma']))

json.dumps({'n_channels':raw.info['nchan'],'n_times':raw.n_times,'sfreq':float(sfreq),
  'ch_names':raw.ch_names[:16],'band_powers':{b:[round(v,6) for v in vals[:8]] for b,vals in band_powers.items()},
  'epochs_count':len(clean),'artifacts_rejected':int(rejected),
  'attention':round(attention,4),'workload':round(workload,4),'arousal':round(arousal,4),
  'mne_version':mne.__version__})
`;

function MnePage() {
  const { data: profile } = useQuery({ queryKey: ["profile"], queryFn: loadProfile });
  const { data: experiments } = useQuery({
    queryKey: ["experiments-list"],
    queryFn: loadExperiments,
  });
  const name = profile?.full_name ?? "User";
  const role = (profile?.role ?? "individual") as "individual" | "researcher" | "enterprise";
  const { state, initialize, runPython, setGlobal } = usePyodide();
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<MNEResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sampleRate, setSampleRate] = useState("256");
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [runName, setRunName] = useState("");

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setSavedId(null);
    const t0 = performance.now();
    try {
      const buffer = await file.arrayBuffer();
      setGlobal("eeg_bytes", Array.from(new Uint8Array(buffer)));
      setGlobal("eeg_filename", file.name);
      setGlobal("eeg_sfreq", Number(sampleRate));
      const res = (await runPython(PYTHON_CODE)) as string;
      const duration_ms = Math.round(performance.now() - t0);
      setResult({ ...JSON.parse(res), duration_ms });
    } catch (err) {
      setError((err as Error).message);
    }
  }, [file, runPython, setGlobal, sampleRate]);

  const handleSave = useCallback(async () => {
    if (!result || !file) return;
    setSaving(true);
    try {
      const { data: userData } = await supabase.auth.getUser();
      const userId = userData.user!.id;
      const meanBp = Object.fromEntries(
        Object.entries(result.band_powers).map(([b, v]) => [
          b,
          v.reduce((s, x) => s + x, 0) / Math.max(1, v.length),
        ]),
      );
      const embeddingVector = Object.values(meanBp);
      const { data: analysis, error: aErr } = await supabase
        .from("eeg_analyses")
        .insert({
          user_id: userId,
          file_name: file.name,
          file_size_bytes: file.size,
          sample_rate: Math.round(result.sfreq),
          num_channels: result.n_channels,
          num_samples: result.n_times,
          embedding: embeddingVector,
          embedding_dimensions: embeddingVector.length,
          embedding_model: `mne-${result.mne_version}`,
          attention: result.attention,
          workload: result.workload,
          arousal: result.arousal,
          processing_time_ms: result.duration_ms,
        })
        .select("id")
        .single();
      if (aErr) throw new Error(aErr.message);
      setSavedId(analysis.id);
      if (selectedExp) {
        await supabase.from("experiment_runs").insert({
          experiment_id: selectedExp,
          user_id: userId,
          analysis_id: analysis.id,
          name: runName || `MNE run — ${file.name}`,
          status: "completed",
          params: {
            mne_version: result.mne_version,
            sfreq: result.sfreq,
            n_channels: result.n_channels,
          },
          metrics: {
            attention: result.attention,
            workload: result.workload,
            arousal: result.arousal,
            epochs_clean: result.epochs_count,
          },
          duration_ms: result.duration_ms,
          completed_at: new Date().toISOString(),
        });
      }
    } catch (err) {
      setError(`Save failed: ${(err as Error).message}`);
    } finally {
      setSaving(false);
    }
  }, [result, file, selectedExp, runName]);

  const isReady = state.status === "ready";
  const isRunning = state.status === "running";
  const isLoading = state.status === "loading-pyodide" || state.status === "loading-packages";

  return (
    <DashboardShell fullName={name} role={role}>
      <Eyebrow>Scientific Analysis</Eyebrow>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight">MNE-Python Analysis</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Real scientific EEG processing. Results saved to your analyses and experiments.
      </p>

      {state.status === "idle" && (
        <GlassCard className="mt-6">
          <div className="flex items-center gap-3 mb-3">
            <FlaskConical className="h-5 w-5 text-neuro" />
            <p className="text-sm font-semibold">Load MNE-Python Runtime</p>
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            Downloads ~50MB Python runtime + MNE, NumPy, SciPy. One-time, cached in browser.
          </p>
          <button
            onClick={initialize}
            className="flex items-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white hover:opacity-90"
          >
            <Play className="h-4 w-4" /> Load MNE-Python
          </button>
        </GlassCard>
      )}

      {isLoading && (
        <GlassCard className="mt-6">
          <div className="flex items-center gap-2 mb-3">
            <Loader2 className="h-4 w-4 animate-spin text-neuro" />
            <span className="text-sm font-semibold text-neuro">{state.message}</span>
          </div>
          <div className="h-2 rounded-full bg-muted/60">
            <div
              className="h-full rounded-full bg-neuro transition-all duration-500"
              style={{ width: `${state.progress}%` }}
            />
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground">{state.progress}%</p>
        </GlassCard>
      )}

      {(isReady || isRunning) && (
        <>
          <GlassCard className="mt-6 border-green-500/20">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-400" />
              <span className="text-xs font-semibold text-green-400">MNE-Python ready ✓</span>
            </div>
          </GlassCard>
          <GlassCard className="mt-4">
            <p className="text-xs font-semibold text-muted-foreground mb-3">Upload EEG File</p>
            <label className="flex flex-col items-center gap-2 cursor-pointer py-4 rounded border border-dashed border-border hover:border-neuro/40 transition-colors">
              <Upload className="h-6 w-6 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">
                {file ? file.name : "Choose EDF, BDF, or CSV"}
              </span>
              <input
                type="file"
                accept=".edf,.bdf,.csv"
                className="hidden"
                onChange={(e) => {
                  setFile(e.target.files?.[0] ?? null);
                  setResult(null);
                  setError(null);
                  setSavedId(null);
                }}
              />
            </label>
            {file?.name.toLowerCase().endsWith(".csv") && (
              <div className="mt-3">
                <label className="text-xs text-muted-foreground">Sample Rate (Hz)</label>
                <input
                  type="number"
                  value={sampleRate}
                  onChange={(e) => setSampleRate(e.target.value)}
                  className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm"
                />
              </div>
            )}
            <button
              onClick={handleAnalyze}
              disabled={!file || isRunning}
              className="mt-4 w-full flex items-center justify-center gap-2 rounded-lg bg-neuro px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40 hover:opacity-90"
            >
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Processing…
                </>
              ) : (
                <>
                  <Brain className="h-4 w-4" /> Analyse with MNE-Python
                </>
              )}
            </button>
          </GlassCard>

          {error && (
            <GlassCard className="mt-4 border-destructive/40">
              <div className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <p className="text-sm">{error}</p>
              </div>
            </GlassCard>
          )}

          {result && (
            <div className="mt-4 flex flex-col gap-3">
              <GlassCard className="border-neuro/30">
                <p className="text-xs font-semibold text-neuro mb-3">
                  ✅ MNE {result.mne_version} — {result.duration_ms}ms
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Channels: </span>
                    <span className="font-mono">{result.n_channels}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Sample rate: </span>
                    <span className="font-mono">{result.sfreq} Hz</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Clean epochs: </span>
                    <span className="font-mono text-green-400">{result.epochs_count}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Rejected: </span>
                    <span className="font-mono text-amber-400">{result.artifacts_rejected}</span>
                  </div>
                </div>
              </GlassCard>
              <GlassCard>
                <p className="text-xs font-semibold text-muted-foreground mb-3">
                  🧠 Cognitive State
                </p>
                {[
                  ["Attention", result.attention, "#818cf8"],
                  ["Workload", result.workload, "#a78bfa"],
                  ["Arousal", result.arousal, "#f59e0b"],
                ].map(([label, value, color]) => (
                  <div key={label as string} className="mb-3">
                    <div className="mb-1 flex justify-between text-[11px]">
                      <span className="text-muted-foreground">{label}</span>
                      <span className="font-mono font-semibold" style={{ color: color as string }}>
                        {Math.round((value as number) * 100)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-muted/60">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.round((value as number) * 100)}%`,
                          backgroundColor: color as string,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </GlassCard>
              <GlassCard>
                <p className="text-xs font-semibold text-muted-foreground mb-3">📊 Band Powers</p>
                {Object.entries(result.band_powers).map(([band, values]) => {
                  const avg = values.reduce((s, v) => s + v, 0) / Math.max(1, values.length);
                  const maxVal = Math.max(
                    ...Object.values(result.band_powers).map(
                      (v) => v.reduce((s, x) => s + x, 0) / Math.max(1, v.length),
                    ),
                  );
                  return (
                    <div key={band} className="mb-2">
                      <div className="flex justify-between text-[11px] mb-1">
                        <span className="text-muted-foreground capitalize">{band}</span>
                        <span className="font-mono text-neuro">{avg.toExponential(2)}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-muted/60">
                        <div
                          className="h-full rounded-full bg-neuro"
                          style={{
                            width: `${Math.min(100, (avg / Math.max(1e-30, maxVal)) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </GlassCard>
              <GlassCard className="border-neuro/20">
                <p className="text-xs font-semibold text-muted-foreground mb-3">💾 Save Results</p>
                {experiments && experiments.length > 0 && (
                  <div className="mb-3">
                    <label className="text-xs text-muted-foreground">
                      Link to Experiment (optional)
                    </label>
                    <select
                      value={selectedExp}
                      onChange={(e) => setSelectedExp(e.target.value)}
                      className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm"
                    >
                      <option value="">— No experiment —</option>
                      {experiments.map((exp) => (
                        <option key={exp.id} value={exp.id}>
                          {exp.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                {selectedExp && (
                  <div className="mb-3">
                    <label className="text-xs text-muted-foreground">Run name (optional)</label>
                    <input
                      value={runName}
                      onChange={(e) => setRunName(e.target.value)}
                      placeholder={`MNE run — ${file?.name}`}
                      className="mt-1 w-full rounded border border-border bg-background px-3 py-2 text-sm"
                    />
                  </div>
                )}
                {savedId ? (
                  <div className="flex items-center gap-2 text-green-400 text-sm">
                    <CheckCircle className="h-4 w-4" /> Saved ✓
                  </div>
                ) : (
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-2 rounded-lg bg-neuro px-4 py-2 text-sm font-semibold text-white disabled:opacity-40 hover:opacity-90"
                  >
                    {saving ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" /> Saving…
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4" /> Save to My Analyses
                      </>
                    )}
                  </button>
                )}
              </GlassCard>
            </div>
          )}
        </>
      )}

      {state.status === "error" && (
        <GlassCard className="mt-6 border-destructive/40">
          <div className="flex items-center gap-2 text-destructive mb-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-semibold">Failed to load MNE-Python</span>
          </div>
          <p className="text-xs text-muted-foreground">{state.error}</p>
          <button onClick={initialize} className="mt-3 text-xs text-neuro hover:underline">
            Try again
          </button>
        </GlassCard>
      )}
    </DashboardShell>
  );
}
