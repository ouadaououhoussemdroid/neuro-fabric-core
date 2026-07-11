import { useEffect, useMemo, useState } from "react";
import {
  Brain,
  ChevronRight,
  Cpu,
  Eye,
  Image as ImageIcon,
  Layers,
  Sparkles,
  Waves,
} from "lucide-react";
import { GlassCard, Eyebrow } from "@/components/ui-bits";
import { LiveDot } from "@/components/live-ops";

/** A single procedural "reconstructed image" tile. Looks like a denoised latent. */
function ReconTile({
  hue,
  hue2 = 295,
  blur = 0,
  saturate = 1,
  className = "",
  children,
  scanlines = true,
}: {
  hue: number;
  hue2?: number;
  blur?: number;
  saturate?: number;
  className?: string;
  children?: React.ReactNode;
  scanlines?: boolean;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-border bg-card/40 ${className}`}
    >
      <div
        className="absolute inset-0 transition-[filter,opacity] duration-500"
        style={{
          background: `
            radial-gradient(55% 55% at 38% 42%, oklch(0.82 0.2 ${hue} / 0.7), transparent 65%),
            radial-gradient(45% 45% at 70% 70%, oklch(0.72 0.22 ${hue2} / 0.55), transparent 65%),
            radial-gradient(30% 30% at 80% 25%, oklch(0.9 0.16 ${(hue + 60) % 360} / 0.4), transparent 70%),
            linear-gradient(135deg, oklch(0.2 0.04 260), oklch(0.14 0.02 260))
          `,
          filter: `blur(${blur}px) saturate(${saturate})`,
        }}
      />
      {/* Holographic sheen */}
      <div
        className="pointer-events-none absolute inset-0 mix-blend-screen opacity-40"
        style={{
          background:
            "linear-gradient(115deg, transparent 30%, oklch(0.95 0.05 200 / 0.18) 50%, transparent 70%)",
          backgroundSize: "200% 200%",
          animation: "shimmer 4s linear infinite",
        }}
      />
      {scanlines && (
        <div
          className="pointer-events-none absolute inset-0 opacity-25"
          style={{
            backgroundImage:
              "repeating-linear-gradient(0deg, oklch(1 0 0 / 0.06) 0 1px, transparent 1px 3px)",
          }}
        />
      )}
      <div className="pointer-events-none absolute inset-0 grid-bg opacity-30" />
      {children}
    </div>
  );
}

/** EEG-style mini trace SVG, deterministic via seed. */
function MiniTrace({
  seed = 0,
  color = "oklch(0.85 0.18 195)",
}: {
  seed?: number;
  color?: string;
}) {
  const pts = useMemo(() => {
    const n = 96;
    const rng = (i: number) => {
      const x = Math.sin((i + 1) * (seed + 1) * 12.9898) * 43758.5453;
      return x - Math.floor(x);
    };
    return Array.from({ length: n }, (_, i) => {
      const v =
        Math.sin((i + seed) / 5) * 0.6 + Math.sin((i + seed) / 2.3) * 0.3 + (rng(i) - 0.5) * 0.25;
      return { x: (i / (n - 1)) * 100, y: 50 - v * 22 };
    });
  }, [seed]);
  const d = pts
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(" ");
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
      <path d={d} fill="none" stroke={color} strokeWidth={0.6} strokeLinecap="round" />
    </svg>
  );
}

const GALLERY = [
  {
    label: "visual.faces",
    caption: "young woman with red hair, frontal portrait",
    hue: 12,
    conf: 0.82,
    state: "attention.high",
  },
  {
    label: "visual.scenes",
    caption: "snow-covered mountain range at dusk",
    hue: 220,
    conf: 0.74,
    state: "affect.calm",
  },
  {
    label: "visual.animals",
    caption: "golden retriever on green grass",
    hue: 90,
    conf: 0.79,
    state: "affect.positive",
  },
  {
    label: "visual.objects",
    caption: "ceramic teacup on wooden table",
    hue: 35,
    conf: 0.68,
    state: "attention.focused",
  },
  {
    label: "visual.text",
    caption: "neon sign reading 'open'",
    hue: 320,
    conf: 0.71,
    state: "arousal.med",
  },
  {
    label: "visual.abstract",
    caption: "swirling blue and violet light",
    hue: 270,
    conf: 0.66,
    state: "imagery.dream",
  },
] as const;

const PIPELINE = [
  { icon: Waves, k: "signal", t: "EEG · 64ch · 250Hz" },
  { icon: Brain, k: "embedding", t: "nwf-7b-embed · 768d" },
  { icon: Layers, k: "latent", t: "Aligned latent · z₁₂₈" },
  { icon: Cpu, k: "diffusion", t: "nw-vision-v1 · 40 steps" },
  { icon: ImageIcon, k: "image", t: "RGB · 512×512" },
] as const;

export function ReconstructionShowcase() {
  // Featured reconstruction state + diffusion progress
  const [active, setActive] = useState(0);
  const [step, setStep] = useState(40);
  const [running, setRunning] = useState(false);

  const reconstruct = (i: number) => {
    setActive(i);
    setRunning(true);
    setStep(0);
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= 40) {
          clearInterval(id);
          setRunning(false);
          return 40;
        }
        return s + 1;
      });
    }, 55);
    return () => clearInterval(id);
  }, [running]);

  const sample = GALLERY[active];
  const progress = step / 40;
  const blur = (1 - progress) * 22;
  const sat = 0.6 + progress * 0.8;

  return (
    <div className="space-y-12">
      {/* Pipeline diagram */}
      <div>
        <Eyebrow>Latent-to-Image Pipeline</Eyebrow>
        <h3 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">
          From visual cortex to pixels in five stages.
        </h3>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          A multimodal decoder aligns EEG embeddings with a CLIP-conditioned diffusion prior. Each
          stage is observable, versioned, and replayable for clinical-grade reproducibility.
        </p>

        <div className="mt-6 grid items-stretch gap-3 md:grid-cols-[repeat(5,1fr)_auto] md:grid-cols-9">
          {PIPELINE.map((p, i) => (
            <div key={p.k} className="contents">
              <GlassCard className="relative overflow-hidden p-4">
                <div className="flex items-center justify-between">
                  <div className="grid h-7 w-7 place-items-center rounded-md border border-border bg-muted/40">
                    <p.icon className="h-3.5 w-3.5 text-neuro" />
                  </div>
                  <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                    stage {i + 1}
                  </span>
                </div>
                <div className="mt-3 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {p.k}
                </div>
                <div className="mt-1 text-sm font-medium">{p.t}</div>
                <div className="mt-3 h-8 opacity-70">
                  {p.k === "signal" && <MiniTrace seed={1} />}
                  {p.k === "embedding" && <DotGrid n={48} />}
                  {p.k === "latent" && <DotGrid n={64} hue={295} />}
                  {p.k === "diffusion" && <NoiseBar progress={progress} />}
                  {p.k === "image" && <PixelPreview hue={sample.hue} progress={progress} />}
                </div>
              </GlassCard>
              {i < PIPELINE.length - 1 && (
                <div className="hidden items-center justify-center md:flex">
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Featured reconstruction */}
      <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
        <GlassCard className="p-0">
          <div className="flex items-center justify-between border-b border-border/60 px-4 py-3">
            <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <LiveDot /> reconstruction · {sample.label}
            </div>
            <span className="font-mono text-[10px] text-muted-foreground">
              nw-vision-v1 · step {step}/40
            </span>
          </div>
          <div className="grid gap-0 sm:grid-cols-2">
            {/* Before: raw EEG */}
            <div className="relative aspect-square border-r border-border/60 bg-background/40 p-4">
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                Before · raw EEG
              </div>
              <div className="mt-2 h-[calc(100%-1.75rem)]">
                <div className="grid h-full grid-rows-6 gap-1">
                  {[0, 1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="rounded bg-muted/20">
                      <MiniTrace seed={active * 7 + i} color={`oklch(0.85 0.16 ${195 + i * 8})`} />
                    </div>
                  ))}
                </div>
              </div>
              <div className="absolute bottom-3 left-3 rounded border border-border bg-background/70 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                ch · Fp1 · F3 · Cz · P4 · O1 · Oz
              </div>
            </div>

            {/* After: reconstructed */}
            <ReconTile hue={sample.hue} blur={blur} saturate={sat} className="aspect-square">
              <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded border border-border bg-background/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                <Eye className="h-3 w-3 text-neuro" /> after · decoded
              </div>
              <div className="absolute bottom-3 left-3 right-3">
                <div className="h-1 overflow-hidden rounded-full bg-background/60">
                  <div
                    className="h-full bg-neuro-gradient transition-[width] duration-100"
                    style={{ width: `${progress * 100}%` }}
                  />
                </div>
                <div className="mt-1 flex justify-between font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                  <span>denoising</span>
                  <span>{Math.round(progress * 100)}%</span>
                </div>
              </div>
            </ReconTile>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/60 px-4 py-3">
            <p className="text-sm italic text-muted-foreground">"{sample.caption}"</p>
            <button
              onClick={() => reconstruct(active)}
              disabled={running}
              className="inline-flex items-center gap-2 rounded-md bg-neuro-gradient px-3 py-1.5 text-xs font-medium text-background glow disabled:opacity-60"
            >
              <Sparkles className="h-3.5 w-3.5" /> {running ? "Decoding…" : "Replay decode"}
            </button>
          </div>
        </GlassCard>

        {/* Confidence + state */}
        <GlassCard>
          <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            Decoding confidence
          </div>
          <div className="mt-3 flex items-center gap-5">
            <RadialScore value={sample.conf * progress + 0.001} />
            <div className="space-y-2 text-xs">
              <Row k="Alignment" v={0.84 * progress} />
              <Row k="CLIP cos sim" v={0.78 * progress} />
              <Row k="Perceptual SSIM" v={0.62 * progress} />
              <Row k="Caption BLEU" v={0.41 * progress} />
            </div>
          </div>

          <div className="mt-6">
            <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              Brain-state condition
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {[sample.state, "valence.pos", "workload.low", "imagery.visual"].map((t) => (
                <span
                  key={t}
                  className="rounded border border-border bg-muted/30 px-2 py-0.5 font-mono text-[10px] text-muted-foreground"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div className="mt-6 rounded-lg border border-border/60 bg-muted/20 p-3 text-xs text-muted-foreground">
            <span className="font-mono text-[10px] uppercase tracking-wider text-foreground">
              Multimodal decoder ·{" "}
            </span>
            EEG embeddings are projected into a shared CLIP-aligned latent, then denoised by a
            cross-attention diffusion prior conditioned on decoded cognitive state. Output is
            sampled with classifier-free guidance (cfg=4.5) over 40 DDIM steps.
          </div>
        </GlassCard>
      </div>

      {/* Gallery */}
      <div>
        <div className="flex items-end justify-between">
          <div>
            <Eyebrow>Reconstruction Gallery</Eyebrow>
            <h3 className="mt-3 text-2xl font-semibold tracking-tight md:text-3xl">
              Brain-state conditioned outputs.
            </h3>
          </div>
          <span className="hidden font-mono text-[10px] uppercase tracking-wider text-muted-foreground md:inline">
            held-out subjects · n=6
          </span>
        </div>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {GALLERY.map((g, i) => (
            <button
              key={g.label}
              onClick={() => reconstruct(i)}
              className={`group relative text-left transition-transform hover:-translate-y-0.5 ${
                active === i ? "ring-1 ring-neuro/60 rounded-xl" : ""
              }`}
            >
              <ReconTile hue={g.hue} className="aspect-[4/3]">
                <div className="absolute left-2 top-2 rounded border border-border bg-background/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                  {g.label}
                </div>
                <div className="absolute right-2 top-2 rounded border border-border bg-background/60 px-1.5 py-0.5 font-mono text-[9px] text-neuro">
                  {g.conf.toFixed(2)}
                </div>
                <div className="absolute inset-x-2 bottom-2 rounded border border-border/60 bg-background/70 px-2 py-1.5 backdrop-blur">
                  <p className="line-clamp-1 text-[11px] text-foreground">"{g.caption}"</p>
                  <div className="mt-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                    {g.state}
                  </div>
                </div>
              </ReconTile>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: number }) {
  return (
    <div className="w-44">
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>{k}</span>
        <span className="text-foreground tabular-nums">{v.toFixed(2)}</span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted/40">
        <div
          className="h-full bg-neuro-gradient transition-[width] duration-150"
          style={{ width: `${Math.min(1, v) * 100}%` }}
        />
      </div>
    </div>
  );
}

function RadialScore({ value }: { value: number }) {
  const v = Math.min(1, Math.max(0, value));
  const r = 36;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative grid h-28 w-28 place-items-center">
      <svg className="absolute inset-0 -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={r} stroke="oklch(1 0 0 / 0.08)" strokeWidth="6" fill="none" />
        <circle
          cx="50"
          cy="50"
          r={r}
          stroke="url(#rgrad)"
          strokeWidth="6"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - v)}
          style={{ transition: "stroke-dashoffset 150ms linear" }}
        />
        <defs>
          <linearGradient id="rgrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="oklch(0.85 0.18 195)" />
            <stop offset="100%" stopColor="oklch(0.7 0.22 295)" />
          </linearGradient>
        </defs>
      </svg>
      <div className="text-center">
        <div className="font-mono text-xl font-semibold tabular-nums">{(v * 100).toFixed(0)}</div>
        <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
          score
        </div>
      </div>
    </div>
  );
}

function DotGrid({ n = 48, hue = 200 }: { n?: number; hue?: number }) {
  const cells = useMemo(() => Array.from({ length: n }, () => Math.random()), [n]);
  return (
    <div
      className="grid h-full gap-[2px]"
      style={{ gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(n)) * 2}, minmax(0,1fr))` }}
    >
      {cells.map((v, i) => (
        <div
          key={i}
          className="rounded-[1px]"
          style={{ background: `oklch(0.78 0.16 ${hue} / ${0.1 + v * 0.7})` }}
        />
      ))}
    </div>
  );
}

function NoiseBar({ progress }: { progress: number }) {
  return (
    <div className="relative h-full w-full overflow-hidden rounded">
      <div
        className="absolute inset-0"
        style={{
          background:
            "repeating-linear-gradient(90deg, oklch(0.78 0.16 200 / 0.6) 0 2px, transparent 2px 5px)",
          filter: `blur(${(1 - progress) * 4}px)`,
        }}
      />
      <div
        className="absolute inset-y-0 left-0 bg-neuro/20"
        style={{ width: `${progress * 100}%` }}
      />
    </div>
  );
}

function PixelPreview({ hue, progress }: { hue: number; progress: number }) {
  return (
    <div
      className="h-full w-full rounded transition-[filter] duration-150"
      style={{
        background: `radial-gradient(60% 60% at 40% 40%, oklch(0.82 0.2 ${hue} / 0.7), transparent 60%), linear-gradient(135deg, oklch(0.2 0.04 260), oklch(0.16 0.02 260))`,
        filter: `blur(${(1 - progress) * 6}px) saturate(${0.6 + progress})`,
      }}
    />
  );
}
