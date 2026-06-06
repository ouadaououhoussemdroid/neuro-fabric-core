import { createFileRoute } from "@tanstack/react-router";
import { useRef, useState } from "react";
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
import { useTick } from "@/components/live-ops";
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
      { name: "description", content: "End-to-end EEG pipeline workspace" },
    ],
  }),
  component: StudioPage,
});

type StageId = "ingest" | "preprocess" | "represent" | "cognitive";
type JobStatus = "queued" | "running" | "done" | "failed";

type EegFile = {
  id: string;
  name: string;
  size: string;
  channels: number;
  sfreq: number;
  duration: string;
  status: "uploading" | "validated";
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

const SEED_FILES: EegFile[] = [
  {
    id: "rec_001",
    name: "subj_017_resting_eo.bdf",
    size: "248 MB",
    channels: 64,
    sfreq: 512,
    duration: "00:08:42",
    status: "validated",
    progress: 100,
  },
];

const SEED_JOBS: Job[] = [
  {
    id: "job_10241",
    kind: "Preprocessing",
    file: "subj_017_resting_eo.bdf",
    status: "done",
    duration: "00:02:14",
    at: "12m ago",
  },
];

const STAGES = [
  { id: "ingest" as StageId, label: "Ingestion", sub: "Upload & Validate", icon: Upload },
  { id: "preprocess" as StageId, label: "Preprocessing", sub: "Clean & Segment", icon: Filter },
  { id: "represent" as StageId, label: "Representation", sub: "Neural Embedding", icon: Brain },
  { id: "cognitive" as StageId, label: "Cognitive State", sub: "Decode Metrics", icon: Gauge },
];

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

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
          sub="A production workspace for neuroscience teams."
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
            {stage === "ingest" && <IngestionStage files={files} setFiles={setFiles} activeFile={activeFile} setActiveFile={setActiveFile} onNext={() => setStage("preprocess")} />}
            {stage === "preprocess" && <PreprocessStage fileName={files.find((f) => f.id === activeFile)?.name ?? "—"} onRun={(summary) => { setJobs((j) => [{ id: `job_${Date.now()}`, kind: `Preprocess · ${summary}`, file: files.find((f) => f.id === activeFile)?.name ?? "—", status: "running", duration: "00:00:00", at: "now" }, ...j]); setStage("represent"); }} />}
            {stage === "represent" && <RepresentationStage fileName={files.find((f) => f.id === activeFile)?.name ?? "—"} onNext={() => setStage("cognitive")} />}
            {stage === "cognitive" && <CognitiveStage fileName={files.find((f) => f.id === activeFile)?.name ?? "—"} />}
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
                  active ? "border-neuro/50 bg-neuro/5 ring-glow" : done ? "border-border/60 bg-muted/20" : "border-border/60 bg-transparent hover:border-border"
                }`}
              >
                <div className={`grid h-8 w-8 place-items-center rounded-md ${active ? "bg-neuro-gradient text-background" : done ? "bg-muted text-foreground" : "bg-muted/40 text-muted-foreground"}`}>
                  {done ? <CheckCircle2 className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Stage {i + 1}</span>
                    {active && <span className="inline-flex items-center gap-1 rounded-full bg-neuro/10 px-1.5 py-0.5 font-mono text-[9px] text-neuro"><span className="h-1 w-1 rounded-full bg-neuro animate-pulse-glow" /> active</span>}
                  </div>
                  <div className="truncate text-sm font-semibold">{s.label}</div>
                  <div className="truncate font-mono text-[10px] text-muted-foreground">{s.sub}</div>
                </div>
              </button>
              {i < STAGES.length - 1 && <ChevronRight className={`h-4 w-4 shrink-0 ${i < idx ? "text-neuro" : "text-muted-foreground/50"}`} />}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function IngestionStage({ files, setFiles, activeFile, setActiveFile, onNext }: any) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const onAdd = (list: FileList | null) => {
    if (!list) return;
    Array.from(list).forEach((f, i) => {
      const id = `rec_\( {8822 + Math.floor(Math.random() * 999)}_ \){i}`;
      const entry: EegFile = {
        id,
        name: f.name,
        size: formatBytes(f.size),
        channels: 32 + Math.floor(Math.random() * 96),
        sfreq: [256, 500, 512, 1000][Math.floor(Math.random() * 4)],
        duration: `00:\( {String(10 + Math.floor(Math.random() * 50)).padStart(2, "0")}: \){String(Math.floor(Math.random() * 60)).padStart(2, "0")}`,
        status: "u
