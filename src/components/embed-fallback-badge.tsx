import { useEffect, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { EMBED_FALLBACK_EVENT } from "@/lib/ai/embeddings";

/**
 * Listens for {@link EMBED_FALLBACK_EVENT} and renders a sticky, dismissible
 * banner whenever the EEG embedding pipeline degraded to the PCA baseline.
 * Makes the silent-fallback failure mode loudly visible in the UI — paired
 * with the `console.error` and structured log emitted by `embed()`.
 */
type FallbackDetail = {
  requestedModelId: string;
  resolvedModelId: string;
  reason: string;
};

export function EmbedFallbackBadge() {
  const [info, setInfo] = useState<FallbackDetail | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    function onFallback(e: Event) {
      const detail = (e as CustomEvent<FallbackDetail>).detail;
      if (!detail) return;
      setInfo(detail);
      setDismissed(false);
    }
    window.addEventListener(EMBED_FALLBACK_EVENT, onFallback as EventListener);
    return () =>
      window.removeEventListener(EMBED_FALLBACK_EVENT, onFallback as EventListener);
  }, []);

  if (!info || dismissed) return null;

  return (
    <div
      role="alert"
      aria-live="polite"
      className="fixed bottom-4 right-4 z-50 max-w-sm rounded-lg border border-amber-400/40 bg-amber-500/10 p-3 text-amber-200 shadow-lg backdrop-blur"
    >
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="flex-1 text-xs">
          <p className="font-semibold">Degraded mode: PCA baseline</p>
          <p className="mt-1 text-amber-200/80">
            The neural model{" "}
            <span className="font-mono">{info.requestedModelId}</span> was
            unavailable. Results came from{" "}
            <span className="font-mono">{info.resolvedModelId}</span>.
          </p>
          <p
            className="mt-1 truncate font-mono text-[10px] text-amber-200/60"
            title={info.reason}
          >
            {info.reason}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          aria-label="Dismiss"
          className="rounded p-0.5 text-amber-200/60 hover:bg-amber-500/20 hover:text-amber-100"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}