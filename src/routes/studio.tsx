import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronRight,
  CircuitBoard,
  Cpu,
  Download,
  FileUp,
  Filter,
  FlaskConical,
  Gauge,
  History,
  Layers,
  Loader2,
  Pause,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  Waves,
  Zap,
} from "lucide-react";
import { SiteShell } from "@/components/site-shell";
import { GlassCard, PageHeader, Section, StatPill } from "@/components/ui-bits";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const Route = createFileRoute("/studio")({
  head: () => ({
    meta: [
      { title: "Neural Processing Studio — NeuroWeave" },
      {
        name: "description",
        content:
          "End-to-end EEG signal processing workspace: ingestion, preprocessing, representation learning, and cognitive state analysis.",
      },
      { property: "og:title", content: "Neural Processing Studio — NeuroWeave" },
      {
        property: "og:description",
        content:
          "Ingest EEG, configure preprocessing, generate latent representations, and inspect cognitive state estimates.",
      },
    ],
  }),
  component: StudioPage,
});

// ---------- types ----------
type StageId = "ingest" | "preprocess" | "represent" | "cognitive";
type JobStatus = "queued" | "running" | "done" | "failed";

type EegFile = {
  id: string;
  name: string;
  size: string;
  channels: number;
  sfreq: number;
  duration: string;
  status: "uploaded" | "uploading" | "validated";
  progress: number;
};

type Job = {
  id: string;
  kind: string;
  file: string;
  status: JobStatus;
  duration: string;
  at: string;
};

// ---------- mock seed ----------
const SEED_FILES: EegFile[] = [
  {
    id: "rec_8821",
    name: "subj_021_session_03.edf",
    size: "184.2 MB",
    channels: 64,
    sfreq: 1000,
    duration: "00:42:18",
    status: "validated",
    progress: 100,
  },
  {
    id: "rec_8820",
    name: "subj_017_resting_eo.bdf",
    size: "92.7 MB",
    channels: 32,
    sfreq: 512,
    duration: "00:18:04",
    status: "validated",
    progress: 100,
  },
  {
    id: "rec_8819",
    name: "cohort_b_task_n2.csv",
    size: "41.3 MB",
    channels: 19,
    sfreq: 256,
    duration: "00:11:52",
    status: "validated",
    progress: 100,
  },
];

const SEED_JOBS: Job[] = [
  { id: "job_10241", kind: "Representation · NeuroFormer-L", file: "subj_021_session_03.edf", status: "running", duration: "00:01:42", at: "12s ago" },
  { id: "job_10240", kind: "Preprocess · notch+bandpass+ICA", file: "subj_017_resting_eo.bdf", status: "done", duration: "00:00:38", at: "4m ago" },
  { id: "job_10239", kind: "Cognitive State · v2", file: "cohort_b_task_n2.csv", status: "done", duration: "00:00:21", at: "11m ago" },
  { id: "job_10238", kind: "Ingest · validate", file: "subj_014_oddball.edf", status: "failed", duration: "00:00:04", at: "26m ago" },
  { id: "job_10237", kind: "Export · embeddings.parquet", file: "subj_017_resting_eo.bdf", status: "done", duration: "00:00:09", at: "1h ago" },
];

const STAGES: { id: StageId; label: string; icon: typeof Brain; sub: string }[] = [
  { id: "ingest", label: "Ingestion", icon: Upload, sub: "EDF · BDF · CSV" },
  { id: "preprocess", label: "Preprocessing", icon: Filter, sub: "Filter · Artifact · Norm" },
  { id: "represent", label: "Representation", icon: Layers, sub: "Latent encoder" },
  { id: "cognitive", label: "Cognitive State", icon: Brain, sub: "Decoded metrics" },
];

// ---------- utils ----------
function useTick(ms = 1200) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((x) => x + 1), ms);
    return () => clearInterval(id);
  }, [ms]);
  return t;
}

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// ---------- page ----------
function StudioPage() {
  const [stage, setStage] = useState<StageId>("ingest");
  const [files, setFiles] = useState<EegFile[]>(SEED_FILES);
  const [activeFile, setActiveFile] = useState<string>(SEED_FILES[0].id);
  const [jobs, setJobs] = useState<Job[]>(SEED_JOBS);

  return (
    <SiteShell>
      <Section>
        <PageHeader
          eyebrow="Neural Processing Studio"
          title="EEG ingestion → representation → cognitive state."
          sub="A production workspace for neuroscience teams: validate recordings, configure preprocessing, generate latent representations, and inspect decoded cognitive metrics."
        />

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatPill label="Recordings" value={`${files.length}`} />
          <StatPill label="Active jobs" value={`${jobs.filter((j) => j.status === "running").length}`} />
          <StatPill label="GPU pool · H100" value="4 / 8" />
          <StatPill label="Pipeline" value="nwf-7b · v0.9.3" />
        </div>

        <PipelineTracker stage={stage} setStage={setStage} />

        <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_320px]">
          <div className="min-w-0">
            {stage === "ingest" && (
              <IngestionStage
                files={files}
                setFiles={setFiles}
                activeFile={activeFile}
                setActiveFile={setActiveFile}
                onNext={() => setStage("preprocess")}
              />
            )}
            {stage === "preprocess" && (
              <PreprocessStage
                fileName={files.find((f) => f.id === activeFile)?.name ?? "—"}
                onRun={(summary) => {
                  setJobs((j) => [
                    {
                      id: `job_${10242 + j.length}`,
                      kind: `Preprocess · ${summary}`,
                      file: files.find((f) => f.id === activeFile)?.name ?? "—",
                      status: "running",
                      duration: "00:00:00",
                      at: "now",
                    },
                    ...j,
                  ]);
                  setStage("represent");
                }}
              />
            )}
            {stage === "represent" && (
              <RepresentationStage
                fileName={files.find((f) => f.id === activeFile)?.name ?? "—"}
                onNext={() => setStage("cognitive")}
              />
            )}
            {stage === "cognitive" && (
              <CognitiveStage fileName={files.find((f) => f.id === activeFile)?.name ?? "—"} />
            )}
          </div>

          <aside className="space-y-6">
            <ActivityFeed />
            <SessionManager />
            <ApiEndpoints stage={stage} />
          </aside>
        </div>

        <JobHistory jobs={jobs} />
      </Section>
    </SiteShell>
  );
}

// ---------- pipeline tracker ----------
function PipelineTracker({ stage, setStage }: { stage: StageId; setStage: (s: StageId) => void }) {
  const idx = STAGES.findIndex((s) => s.id === stage);
  return (
    <GlassCard className="mt-6 p-4">
      <div className="flex items-center gap-2 overflow-x-auto">
        {STAGES.map((s, i) => {
          const Icon = s.icon;
          const active = s.id === stage;
          const done = i < idx;
          return (
            <div key={s.id} className="flex flex-1 items-center gap-2 min-w-fit">
              <button
                onClick={() => setStage(s.id)}
                className={`group flex flex-1 items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all ${
                  active
                    ? "border-neuro/50 bg-neuro/5 ring-glow"
                    : done
                      ? "border-border/60 bg-muted/20"
                      : "border-border/60 bg-transparent hover:border-border"
                }`}
              >
                <div
                  className={`grid h-8 w-8 place-items-center rounded-md ${
                    active ? "bg-neuro-gradient text-background" : done ? "bg-muted text-foreground" : "bg-muted/40 text-muted-foreground"
                  }`}
                >
                  {done ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                      Stage {i + 1}
                    </span>
                    {active && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-neuro/10 px-1.5 py-0.5 font-mono text-[9px] text-neuro">
                        <span className="h-1 w-1 rounded-full bg-neuro animate-pulse-glow" />
                        active
                      </span>
                    )}
                  </div>
                  <div className="truncate text-sm font-semibold">{s.label}</div>
                  <div className="truncate font-mono text-[10px] text-muted-foreground">{s.sub}</div>
                </div>
              </button>
              {i < STAGES.length - 1 && (
                <ChevronRight className={`h-4 w-4 shrink-0 ${i < idx ? "text-neuro" : "text-muted-foreground/50"}`} />
              )}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

// ---------- STAGE 1: INGESTION ----------
function IngestionStage({
  files,
  setFiles,
  activeFile,
  setActiveFile,
  onNext,
}: {
  files: EegFile[];
  setFiles: React.Dispatch<React.SetStateAction<EegFile[]>>;
  activeFile: string;
  setActiveFile: (id: string) => void;
  onNext: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const onAdd = (list: FileList | null) => {
    if (!list) return;
    Array.from(list).forEach((f, i) => {
      const id = `rec_${8822 + Math.floor(Math.random() * 999)}_${i}`;
      const entry: EegFile = {
        id,
        name: f.name,
        size: formatBytes(f.size),
        channels: 32 + Math.floor(Math.random() * 96),
        sfreq: [256, 500, 512, 1000][Math.floor(Math.random() * 4)],
        duration: `00:${String(10 + Math.floor(Math.random() * 50)).padStart(2, "0")}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`,
        status: "uploading",
        progress: 0,
      };
      setFiles((prev) => [entry, ...prev]);
      // simulate upload
      let p = 0;
      const t = setInterval(() => {
        p += 8 + Math.random() * 18;
        if (p >= 100) {
          p = 100;
          clearInterval(t);
          setFiles((prev) => prev.map((x) => (x.id === id ? { ...x, progress: 100, status: "validated" } : x)));
        } else {
          setFiles((prev) => prev.map((x) => (x.id === id ? { ...x, progress: p } : x)));
        }
      }, 220);
    });
  };

  return (
    <div className="space-y-4">
      <GlassCard>
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold">EEG Upload Workspace</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Drop EDF, BDF, or CSV recordings. Files are validated for channel layout, sampling frequency, and segment integrity.
            </p>
          </div>
          <span className="hidden font-mono text-[10px] uppercase tracking-wider text-muted-foreground md:inline">
            org · neuroweave-lab / workspace · default
          </span>
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            onAdd(e.dataTransfer.files);
          }}
          className={`mt-4 grid place-items-center rounded-xl border border-dashed px-6 py-10 text-center transition-colors ${
            drag ? "border-neuro/70 bg-neuro/5" : "border-border/70 bg-muted/10"
          }`}
        >
          <FileUp className="h-6 w-6 text-muted-foreground" />
          <div className="mt-3 text-sm font-medium">Drop recordings here</div>
          <div className="mt-1 font-mono text-[11px] text-muted-foreground">
            .edf · .bdf · .csv · max 2 GB · auto-detect channel montage
          </div>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".edf,.bdf,.csv"
            className="hidden"
            onChange={(e) => onAdd(e.target.files)}
          />
          <button
            onClick={() => inputRef.current?.click()}
            className="mt-4 inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow"
          >
            <Upload className="h-3.5 w-3.5" /> Select files
          </button>
        </div>
      </GlassCard>

      <GlassCard className="p-0">
        <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
          <div className="text-sm font-semibold">Recordings</div>
          <span className="font-mono text-[10px] text-muted-foreground">{files.length} file(s)</span>
        </div>
        {files.length === 0 ? (
          <EmptyState icon={Waves} title="No recordings yet" sub="Upload an EDF/BDF/CSV to begin the pipeline." />
        ) : (
          <ul className="divide-y divide-border/60">
            {files.map((f) => {
              const active = f.id === activeFile;
              return (
                <li
                  key={f.id}
                  onClick={() => setActiveFile(f.id)}
                  className={`grid cursor-pointer grid-cols-12 items-center gap-3 px-5 py-3 text-sm transition-colors ${
                    active ? "bg-neuro/5" : "hover:bg-muted/20"
                  }`}
                >
                  <div className="col-span-12 flex items-center gap-2 md:col-span-4">
                    <div className={`h-2 w-2 rounded-full ${active ? "bg-neuro animate-pulse-glow" : "bg-muted-foreground/50"}`} />
                    <div className="min-w-0">
                      <div className="truncate font-mono text-xs">{f.name}</div>
                      <div className="font-mono text-[10px] text-muted-foreground">{f.id}</div>
                    </div>
                  </div>
                  <div className="col-span-3 md:col-span-1 font-mono text-xs text-muted-foreground">{f.size}</div>
                  <div className="col-span-3 md:col-span-1 font-mono text-xs text-muted-foreground">{f.channels} ch</div>
                  <div className="col-span-3 md:col-span-2 font-mono text-xs text-muted-foreground">{f.sfreq} Hz</div>
                  <div className="col-span-3 md:col-span-2 font-mono text-xs text-muted-foreground">{f.duration}</div>
                  <div className="col-span-12 md:col-span-2">
                    {f.status === "uploading" ? (
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <div className="h-full bg-neuro-gradient transition-all" style={{ width: `${f.progress}%` }} />
                        </div>
                        <span className="font-mono text-[10px] text-muted-foreground">{Math.round(f.progress)}%</span>
                      </div>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 rounded-full bg-neuro/10 px-2 py-0.5 font-mono text-[10px] text-neuro">
                        <CheckCircle2 className="h-3 w-3" /> {f.status}
                      </span>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        <div className="flex items-center justify-between border-t border-border/60 px-5 py-3">
          <span className="font-mono text-[10px] text-muted-foreground">Active: {files.find((f) => f.id === activeFile)?.name ?? "—"}</span>
          <button
            onClick={onNext}
            disabled={files.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow disabled:opacity-50"
          >
            Continue to preprocessing <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </GlassCard>
    </div>
  );
}

// ---------- STAGE 2: PREPROCESS ----------
function PreprocessStage({ fileName, onRun }: { fileName: string; onRun: (summary: string) => void }) {
  const [notch, setNotch] = useState<50 | 60 | null>(50);
  const [bandLow, setBandLow] = useState(0.5);
  const [bandHigh, setBandHigh] = useState(45);
  const [ica, setIca] = useState(true);
  const [motion, setMotion] = useState(true);
  const [blink, setBlink] = useState(true);
  const [norm, setNorm] = useState<"zscore" | "minmax">("zscore");
  const [winLen, setWinLen] = useState(4);
  const [overlap, setOverlap] = useState(50);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [quality, setQuality] = useState(0);

  const summary = useMemo(() => {
    const parts: string[] = [];
    if (notch) parts.push(`notch${notch}`);
    parts.push(`bp ${bandLow}–${bandHigh}Hz`);
    if (ica) parts.push("ICA");
    if (motion) parts.push("motion");
    if (blink) parts.push("blink");
    parts.push(norm);
    parts.push(`win ${winLen}s/${overlap}%`);
    return parts.join(" · ");
  }, [notch, bandLow, bandHigh, ica, motion, blink, norm, winLen, overlap]);

  const run = () => {
    setRunning(true);
    setProgress(0);
    setQuality(0);
    let p = 0;
    const t = setInterval(() => {
      p += 4 + Math.random() * 7;
      if (p >= 100) {
        p = 100;
        clearInterval(t);
        setRunning(false);
        setQuality(0.86 + Math.random() * 0.1);
        setTimeout(() => onRun(summary), 600);
      }
      setProgress(p);
    }, 180);
  };

  return (
    <div className="space-y-4">
      <GlassCard>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold">Signal Preprocessing</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Target: <span className="font-mono text-foreground">{fileName}</span>
            </p>
          </div>
          <Settings2 className="h-4 w-4 text-muted-foreground" />
        </div>

        <div className="mt-5 grid gap-5 md:grid-cols-2">
          <Group title="Signal filtering" icon={Filter}>
            <div>
              <Label>Notch filter</Label>
              <SegSwitch
                value={notch ?? "off"}
                onChange={(v) => setNotch(v === "off" ? null : (Number(v) as 50 | 60))}
                opts={[
                  { v: "off", label: "Off" },
                  { v: "50", label: "50 Hz" },
                  { v: "60", label: "60 Hz" },
                ]}
              />
            </div>
            <div>
              <Label>Bandpass (Hz)</Label>
              <div className="grid grid-cols-2 gap-2">
                <NumberField value={bandLow} onChange={setBandLow} step={0.1} min={0.1} max={10} suffix="lo" />
                <NumberField value={bandHigh} onChange={setBandHigh} step={1} min={5} max={120} suffix="hi" />
              </div>
            </div>
          </Group>

          <Group title="Artifact handling" icon={Sparkles}>
            <Toggle label="ICA artifact removal" sub="FastICA, n_components=auto" value={ica} onChange={setIca} />
            <Toggle label="Motion artifact suppression" sub="ASR · cutoff k=5" value={motion} onChange={setMotion} />
            <Toggle label="Eye-blink removal" sub="EOG regression" value={blink} onChange={setBlink} />
          </Group>

          <Group title="Normalization" icon={Gauge}>
            <SegSwitch
              value={norm}
              onChange={(v) => setNorm(v as "zscore" | "minmax")}
              opts={[
                { v: "zscore", label: "Z-score" },
                { v: "minmax", label: "Min–max" },
              ]}
            />
            <p className="text-[11px] text-muted-foreground">Per-channel, computed on training segments only.</p>
          </Group>

          <Group title="Segmentation" icon={Layers}>
            <div>
              <Label>Window length (s)</Label>
              <NumberField value={winLen} onChange={setWinLen} step={0.5} min={0.5} max={30} />
            </div>
            <div>
              <Label>Overlap (%)</Label>
              <NumberField value={overlap} onChange={setOverlap} step={5} min={0} max={90} />
            </div>
          </Group>
        </div>

        <div className="mt-5 rounded-lg border border-border/60 bg-muted/10 p-3">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Pipeline summary</div>
              <div className="mt-1 truncate font-mono text-xs">{summary}</div>
            </div>
            <button
              onClick={run}
              disabled={running}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow disabled:opacity-60"
            >
              {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {running ? "Processing" : "Run preprocessing"}
            </button>
          </div>
          {(running || progress > 0) && (
            <div className="mt-3">
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-neuro-gradient transition-all" style={{ width: `${progress}%` }} />
              </div>
              <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>{progress < 100 ? "estimating noise floor → ICA → segmentation" : "complete"}</span>
                <span>{Math.round(progress)}%</span>
              </div>
            </div>
          )}
          {quality > 0 && (
            <div className="mt-3 flex items-center gap-3 rounded-md border border-border/60 bg-background/40 px-3 py-2">
              <Gauge className="h-4 w-4 text-neuro" />
              <div className="text-xs">
                Signal quality: <span className="font-mono text-neuro">{quality.toFixed(2)}</span> · channels rejected: 2/64 · bad segments: 1.4%
              </div>
            </div>
          )}
        </div>
      </GlassCard>
    </div>
  );
}

function Group({ title, icon: Icon, children }: { title: string; icon: typeof Brain; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border/60 bg-background/30 p-4">
      <div className="mb-3 flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-neuro" />
        <div className="text-xs font-semibold">{title}</div>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{children}</div>;
}

function SegSwitch({
  value,
  onChange,
  opts,
}: {
  value: string | number;
  onChange: (v: string) => void;
  opts: { v: string; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-md border border-border/60 bg-background/40 p-0.5">
      {opts.map((o) => {
        const active = String(value) === o.v;
        return (
          <button
            key={o.v}
            onClick={() => onChange(o.v)}
            className={`rounded px-2.5 py-1 text-xs transition-colors ${
              active ? "bg-neuro-gradient text-background" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function NumberField({
  value,
  onChange,
  step = 1,
  min,
  max,
  suffix,
}: {
  value: number;
  onChange: (n: number) => void;
  step?: number;
  min?: number;
  max?: number;
  suffix?: string;
}) {
  return (
    <label className="flex items-center gap-2 rounded-md border border-border/60 bg-background/40 px-2 py-1">
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-transparent font-mono text-xs outline-none"
      />
      {suffix && <span className="font-mono text-[10px] text-muted-foreground">{suffix}</span>}
    </label>
  );
}

function Toggle({ label, sub, value, onChange }: { label: string; sub?: string; value: boolean; onChange: (b: boolean) => void }) {
  return (
    <button onClick={() => onChange(!value)} className="flex w-full items-center justify-between gap-3 text-left">
      <div className="min-w-0">
        <div className="text-xs">{label}</div>
        {sub && <div className="font-mono text-[10px] text-muted-foreground">{sub}</div>}
      </div>
      <div className={`relative h-5 w-9 shrink-0 rounded-full transition-colors ${value ? "bg-neuro" : "bg-muted"}`}>
        <div
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-background transition-all ${value ? "left-4" : "left-0.5"}`}
        />
      </div>
    </button>
  );
}

// ---------- STAGE 3: REPRESENTATION ----------
function RepresentationStage({ fileName, onNext }: { fileName: string; onNext: () => void }) {
  const tick = useTick(900);
  const [dim, setDim] = useState<256 | 512 | 1024>(512);
  const [running, setRunning] = useState(true);
  const [progress, setProgress] = useState(72);

  useEffect(() => {
    if (!running) return;
    setProgress((p) => Math.min(100, p + 2 + Math.random() * 3));
  }, [tick, running]);

  const vector = useMemo(
    () => Array.from({ length: 24 }, () => (Math.random() - 0.5) * 1.8),
    [tick, dim],
  );

  const timeline = useMemo(
    () =>
      Array.from({ length: 48 }, (_, i) => ({
        t: i,
        a: Math.sin(i / 4 + tick / 3) * 0.5 + 0.5 + Math.random() * 0.1,
        b: Math.cos(i / 5 + tick / 4) * 0.4 + 0.5 + Math.random() * 0.1,
      })),
    [tick],
  );

  return (
    <div className="space-y-4">
      <GlassCard>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold">Neural Representation Engine</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Encodes preprocessed EEG segments into a fixed-dimensional latent space using <span className="font-mono">NeuroFormer-L</span>.
              Source: <span className="font-mono text-foreground">{fileName}</span>
            </p>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-neuro/10 px-2 py-0.5 font-mono text-[10px] text-neuro">
            <span className="h-1.5 w-1.5 rounded-full bg-neuro animate-pulse-glow" /> encoding
          </span>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          <div className="rounded-lg border border-border/60 bg-background/30 p-4">
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Embedding dimension</div>
            <div className="mt-2">
              <SegSwitch
                value={dim}
                onChange={(v) => setDim(Number(v) as 256 | 512 | 1024)}
                opts={[
                  { v: "256", label: "256" },
                  { v: "512", label: "512" },
                  { v: "1024", label: "1024" },
                ]}
              />
            </div>
            <div className="mt-4 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Representation quality</span>
              <span className="font-mono text-neuro">{(0.78 + (dim / 1024) * 0.1).toFixed(3)}</span>
            </div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
              <div className="h-full bg-neuro-gradient" style={{ width: `${78 + (dim / 1024) * 10}%` }} />
            </div>
            <div className="mt-4 text-xs text-muted-foreground">
              Status:{" "}
              <span className="font-mono text-foreground">{running ? "encoding segments" : "ready"}</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
              <div className="h-full bg-neuro-gradient transition-all" style={{ width: `${progress}%` }} />
            </div>
            <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>seg 312 / 432</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <button
              onClick={() => setRunning((r) => !r)}
              className="mt-3 inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-border/60 px-3 py-1.5 text-xs hover:bg-muted/30"
            >
              {running ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              {running ? "Pause encoder" : "Resume encoder"}
            </button>
          </div>

          <div className="rounded-lg border border-border/60 bg-background/30 p-4 lg:col-span-2">
            <div className="flex items-center justify-between">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Latent vector preview · z[0:24]</div>
              <div className="font-mono text-[10px] text-muted-foreground">dim={dim} · L2={(1.0 + Math.random() * 0.05).toFixed(3)}</div>
            </div>
            <div className="mt-3 grid grid-cols-12 gap-1">
              {vector.map((v, i) => (
                <div key={i} className="relative h-12 overflow-hidden rounded bg-muted/40">
                  <div
                    className="absolute inset-x-0 bg-neuro-gradient"
                    style={{
                      height: `${Math.min(100, Math.abs(v) * 60)}%`,
                      bottom: v >= 0 ? "50%" : "auto",
                      top: v < 0 ? "50%" : "auto",
                      opacity: 0.85,
                    }}
                  />
                  <div className="absolute inset-x-0 top-1/2 h-px bg-border/80" />
                </div>
              ))}
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
              <Meta label="model" value="nwf-7b" />
              <Meta label="ckpt" value="2026-04-12" />
              <Meta label="pooling" value="mean+cls" />
              <Meta label="seq_len" value="2048" />
            </div>
          </div>

          <div className="rounded-lg border border-border/60 bg-background/30 p-4 lg:col-span-3">
            <div className="flex items-center justify-between">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Temporal embedding timeline</div>
              <div className="font-mono text-[10px] text-muted-foreground">2 representative components</div>
            </div>
            <div className="mt-2 h-40">
              <ResponsiveContainer>
                <LineChart data={timeline}>
                  <XAxis dataKey="t" hide />
                  <YAxis hide domain={[0, 1.2]} />
                  <Tooltip contentStyle={{ background: "oklch(0.18 0.014 260)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 11 }} />
                  <Line type="monotone" dataKey="a" stroke="oklch(0.85 0.18 195)" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="b" stroke="oklch(0.7 0.22 295)" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="mt-5 flex items-center justify-between">
          <span className="font-mono text-[10px] text-muted-foreground">
            Output: <span className="text-foreground">embeddings_{dim}.parquet</span> · 432 segments · ~{(dim * 432 * 4 / 1024 / 1024).toFixed(1)} MB
          </span>
          <button
            onClick={onNext}
            className="inline-flex items-center gap-1.5 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow"
          >
            Continue to cognitive state <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </GlassCard>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/60 bg-muted/10 px-2 py-1.5">
      <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="font-mono text-xs">{value}</div>
    </div>
  );
}

// ---------- STAGE 4: COGNITIVE STATE ----------
function CognitiveStage({ fileName }: { fileName: string }) {
  const tick = useTick(1400);
  const metrics = useMemo(
    () => [
      { key: "Focus Index", v: 0.72 + (Math.sin(tick / 3) + 1) * 0.08, conf: 0.88 },
      { key: "Cognitive Load", v: 0.54 + (Math.cos(tick / 4) + 1) * 0.07, conf: 0.82 },
      { key: "Mental Fatigue", v: 0.31 + (Math.sin(tick / 5) + 1) * 0.05, conf: 0.79 },
      { key: "Engagement", v: 0.66 + (Math.cos(tick / 3) + 1) * 0.09, conf: 0.85 },
    ],
    [tick],
  );

  const series = useMemo(
    () =>
      Array.from({ length: 60 }, (_, i) => ({
        t: i,
        focus: 0.55 + Math.sin(i / 6 + tick / 4) * 0.18 + Math.random() * 0.03,
        load: 0.45 + Math.cos(i / 7 + tick / 5) * 0.16 + Math.random() * 0.03,
        fatigue: 0.3 + Math.sin(i / 9 + tick / 6) * 0.1 + Math.random() * 0.02,
      })),
    [tick],
  );

  return (
    <div className="space-y-4">
      <GlassCard>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold">Cognitive State Analysis</div>
            <p className="mt-1 text-xs text-muted-foreground">
              Estimated cognitive metrics decoded from latent representations. Source: <span className="font-mono text-foreground">{fileName}</span>
            </p>
          </div>
          <span className="font-mono text-[10px] text-muted-foreground">decoder · cogstate-v2 · cv-mean</span>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map((m) => (
            <CogCard key={m.key} label={m.key} value={m.v} confidence={m.conf} />
          ))}
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-[1.5fr_1fr]">
          <div className="rounded-lg border border-border/60 bg-background/30 p-4">
            <div className="flex items-center justify-between">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Temporal evolution · 60s window</div>
              <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
                <LegendDot color="oklch(0.85 0.18 195)" label="focus" />
                <LegendDot color="oklch(0.7 0.22 295)" label="load" />
                <LegendDot color="oklch(0.75 0.2 50)" label="fatigue" />
              </div>
            </div>
            <div className="mt-2 h-56">
              <ResponsiveContainer>
                <AreaChart data={series}>
                  <defs>
                    <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="t" hide />
                  <YAxis hide domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "oklch(0.18 0.014 260)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 11 }} />
                  <Area type="monotone" dataKey="focus" stroke="oklch(0.85 0.18 195)" fill="url(#g1)" strokeWidth={1.5} />
                  <Area type="monotone" dataKey="load" stroke="oklch(0.7 0.22 295)" fill="transparent" strokeWidth={1.5} />
                  <Area type="monotone" dataKey="fatigue" stroke="oklch(0.75 0.2 50)" fill="transparent" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-lg border border-border/60 bg-background/30 p-4">
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Probability distribution · state class</div>
            <div className="mt-2 h-56">
              <ResponsiveContainer>
                <BarChart data={[
                  { k: "focused", p: 0.46 },
                  { k: "neutral", p: 0.27 },
                  { k: "fatigued", p: 0.11 },
                  { k: "stressed", p: 0.09 },
                  { k: "drowsy", p: 0.07 },
                ]}>
                  <XAxis dataKey="k" stroke="oklch(0.6 0.02 260)" tick={{ fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis hide />
                  <Tooltip contentStyle={{ background: "oklch(0.18 0.014 260)", border: "1px solid oklch(1 0 0 / 0.1)", borderRadius: 8, fontSize: 11 }} />
                  <Bar dataKey="p" fill="oklch(0.78 0.16 200)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-2 text-[11px] text-muted-foreground">
              Class probabilities are aggregated over the last 60 seconds and intended for research use only.
            </p>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <span className="font-mono text-[10px] text-muted-foreground">
            Decoded 432 segments · model nwf-cogstate-v2 · calibration: held-out subject CV
          </span>
          <div className="flex gap-2">
            <button className="inline-flex items-center gap-1.5 rounded-md border border-border/60 px-3 py-1.5 text-xs hover:bg-muted/30">
              <Download className="h-3.5 w-3.5" /> Export results
            </button>
            <button className="inline-flex items-center gap-1.5 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow">
              <FlaskConical className="h-3.5 w-3.5" /> Save as experiment
            </button>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}

function CogCard({ label, value, confidence }: { label: string; value: number; confidence: number }) {
  return (
    <div className="rounded-lg border border-border/60 bg-background/30 p-4">
      <div className="flex items-center justify-between">
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
        <span className="font-mono text-[10px] text-neuro">conf {confidence.toFixed(2)}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <div className="text-2xl font-semibold tabular-nums">{value.toFixed(2)}</div>
        <div className="font-mono text-[10px] text-muted-foreground">/ 1.00</div>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-neuro-gradient transition-all" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

// ---------- side panels ----------
function ActivityFeed() {
  const tick = useTick(2400);
  const events = useMemo(
    () => [
      { i: Cpu, t: "encoder@h100-3 emitted batch 312/432", at: "just now" },
      { i: Filter, t: "ICA removed 3 components on subj_021_session_03", at: "8s ago" },
      { i: Activity, t: "signal quality 0.91 after notch+bandpass", at: "21s ago" },
      { i: Upload, t: "subj_017_resting_eo.bdf validated · 32ch · 512Hz", at: "1m ago" },
      { i: Zap, t: "cogstate-v2 decoder warm cache hit", at: "2m ago" },
    ],
    [tick],
  );
  return (
    <GlassCard className="p-0">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="text-sm font-semibold">Processing activity</div>
        <span className="inline-flex items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-neuro animate-pulse-glow" /> live
        </span>
      </div>
      <ul className="divide-y divide-border/60">
        {events.map((e, i) => {
          const Icon = e.i;
          return (
            <li key={i} className="flex items-start gap-3 px-4 py-2.5">
              <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-neuro" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-xs">{e.t}</div>
                <div className="font-mono text-[10px] text-muted-foreground">{e.at}</div>
              </div>
            </li>
          );
        })}
      </ul>
    </GlassCard>
  );
}

function SessionManager() {
  const [sessions, setSessions] = useState([
    { id: "exp_0421", name: "P300 oddball · cohort B", active: true },
    { id: "exp_0418", name: "Resting-state baseline", active: false },
    { id: "exp_0399", name: "Workload pilot · n=12", active: false },
  ]);
  return (
    <GlassCard className="p-0">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="text-sm font-semibold">Experiment sessions</div>
        <button
          onClick={() =>
            setSessions((s) => [
              { id: `exp_${422 + s.length}`, name: `Untitled experiment ${s.length + 1}`, active: false },
              ...s,
            ])
          }
          className="inline-flex items-center gap-1 rounded border border-border/60 px-2 py-0.5 font-mono text-[10px] text-muted-foreground hover:text-foreground"
        >
          <Plus className="h-3 w-3" /> new
        </button>
      </div>
      <ul className="divide-y divide-border/60">
        {sessions.map((s) => (
          <li key={s.id} className="flex items-center justify-between gap-2 px-4 py-2.5">
            <button
              onClick={() => setSessions((arr) => arr.map((x) => ({ ...x, active: x.id === s.id })))}
              className="flex min-w-0 flex-1 items-center gap-2 text-left"
            >
              <span className={`h-2 w-2 rounded-full ${s.active ? "bg-neuro animate-pulse-glow" : "bg-muted-foreground/40"}`} />
              <div className="min-w-0">
                <div className="truncate text-xs">{s.name}</div>
                <div className="font-mono text-[10px] text-muted-foreground">{s.id}</div>
              </div>
            </button>
            <button
              onClick={() => setSessions((arr) => arr.filter((x) => x.id !== s.id))}
              className="rounded p-1 text-muted-foreground hover:bg-muted/30 hover:text-destructive"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

function ApiEndpoints({ stage }: { stage: StageId }) {
  const map: Record<StageId, { m: string; p: string }[]> = {
    ingest: [
      { m: "POST", p: "/v1/recordings" },
      { m: "GET", p: "/v1/recordings/:id" },
    ],
    preprocess: [
      { m: "POST", p: "/v1/preprocess" },
      { m: "GET", p: "/v1/preprocess/:job_id" },
    ],
    represent: [
      { m: "POST", p: "/v1/embeddings" },
      { m: "GET", p: "/v1/embeddings/:id" },
    ],
    cognitive: [
      { m: "POST", p: "/v1/cogstate/decode" },
      { m: "GET", p: "/v1/cogstate/:job_id" },
    ],
  };
  return (
    <GlassCard className="p-0">
      <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
        <div className="text-sm font-semibold">API endpoints</div>
        <CircuitBoard className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <ul className="space-y-1.5 px-4 py-3">
        {map[stage].map((e) => (
          <li key={e.p} className="flex items-center gap-2 font-mono text-[11px]">
            <span className={`rounded px-1.5 py-0.5 text-[9px] ${e.m === "POST" ? "bg-accent/20 text-accent" : "bg-neuro/10 text-neuro"}`}>
              {e.m}
            </span>
            <span className="truncate text-muted-foreground">{e.p}</span>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

// ---------- job history ----------
function JobHistory({ jobs }: { jobs: Job[] }) {
  return (
    <GlassCard className="mt-6 p-0">
      <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <History className="h-4 w-4 text-muted-foreground" />
          Job history
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">{jobs.length} jobs · last 24h</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="text-left font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <th className="px-5 py-2">Job</th>
              <th className="px-5 py-2">Kind</th>
              <th className="px-5 py-2">File</th>
              <th className="px-5 py-2">Duration</th>
              <th className="px-5 py-2">Status</th>
              <th className="px-5 py-2">When</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {jobs.map((j) => (
              <tr key={j.id}>
                <td className="px-5 py-3 font-mono text-xs">{j.id}</td>
                <td className="px-5 py-3 text-muted-foreground">{j.kind}</td>
                <td className="px-5 py-3 font-mono text-xs text-muted-foreground">{j.file}</td>
                <td className="px-5 py-3 font-mono text-xs text-muted-foreground">{j.duration}</td>
                <td className="px-5 py-3">
                  <StatusPill status={j.status} />
                </td>
                <td className="px-5 py-3 font-mono text-[11px] text-muted-foreground">{j.at}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </GlassCard>
  );
}

function StatusPill({ status }: { status: JobStatus }) {
  const cfg: Record<JobStatus, { label: string; cls: string; dot: string }> = {
    queued: { label: "queued", cls: "bg-muted text-muted-foreground", dot: "bg-muted-foreground" },
    running: { label: "running", cls: "bg-neuro/10 text-neuro", dot: "bg-neuro animate-pulse-glow" },
    done: { label: "done", cls: "bg-muted text-foreground", dot: "bg-foreground/60" },
    failed: { label: "failed", cls: "bg-destructive/15 text-destructive", dot: "bg-destructive" },
  };
  const c = cfg[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] ${c.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${c.dot}`} />
      {c.label}
    </span>
  );
}

function EmptyState({ icon: Icon, title, sub }: { icon: typeof Brain; title: string; sub: string }) {
  return (
    <div className="grid place-items-center px-6 py-12 text-center">
      <Icon className="h-6 w-6 text-muted-foreground" />
      <div className="mt-3 text-sm font-medium">{title}</div>
      <div className="mt-1 font-mono text-[11px] text-muted-foreground">{sub}</div>
    </div>
  );
}