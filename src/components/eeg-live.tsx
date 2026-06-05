import { useEffect, useRef } from "react";

/**
 * Animated multi-channel EEG visualization rendered on canvas.
 * Each band (delta/theta/alpha/beta) is a sum of sines with characteristic
 * frequency ranges, plus jitter — produces a realistic-looking trace.
 */
const BANDS = [
  { name: "Delta", hz: "1–4 Hz", color: "oklch(0.75 0.18 295)", base: 2, amp: 28 },
  { name: "Theta", hz: "4–8 Hz", color: "oklch(0.78 0.16 260)", base: 6, amp: 22 },
  { name: "Alpha", hz: "8–13 Hz", color: "oklch(0.82 0.18 200)", base: 10, amp: 16 },
  { name: "Beta", hz: "13–30 Hz", color: "oklch(0.85 0.18 175)", base: 20, amp: 11 },
] as const;

export function EEGLive({ height = 260 }: { height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let w = 0, h = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const buffers: number[][] = BANDS.map(() => []);
    const COLS = 600;

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = r.width; h = r.height;
      canvas.width = w * dpr; canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const start = performance.now();
    const draw = () => {
      const t = (performance.now() - start) / 1000;
      ctx.clearRect(0, 0, w, h);

      // grid
      ctx.strokeStyle = "oklch(1 0 0 / 0.05)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      const rowH = h / BANDS.length;
      BANDS.forEach((band, i) => {
        // generate next sample
        const v =
          Math.sin(t * band.base * 2 * Math.PI) * band.amp * 0.5 +
          Math.sin(t * band.base * 0.7 * Math.PI + i) * band.amp * 0.3 +
          (Math.random() - 0.5) * band.amp * 0.4;
        const buf = buffers[i];
        buf.push(v);
        if (buf.length > COLS) buf.shift();

        const cy = rowH * i + rowH / 2;
        ctx.strokeStyle = band.color;
        ctx.lineWidth = 1.4;
        ctx.shadowColor = band.color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        for (let x = 0; x < buf.length; x++) {
          const px = (x / COLS) * w;
          const py = cy + buf[x] * 0.6;
          if (x === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      });

      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, []);

  return (
    <div className="relative w-full overflow-hidden rounded-lg border border-border/60 bg-background/40" style={{ height }}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden />
      <div className="pointer-events-none absolute inset-y-0 left-0 flex flex-col justify-between p-3">
        {BANDS.map((b) => (
          <div key={b.name} className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: b.color, boxShadow: `0 0 8px ${b.color}` }} />
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              {b.name} <span className="text-foreground/70">· {b.hz}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-1.5 rounded-full border border-border/60 bg-background/60 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-neuro" /> live · 256 Hz
      </div>
    </div>
  );
}