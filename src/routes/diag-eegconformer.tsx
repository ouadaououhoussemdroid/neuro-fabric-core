import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { embedEEG } from "@/lib/ai/inference/embed-eeg";
import { hasModel, getDescriptor, listModels } from "@/lib/ai/models/registry";
import { isONNXRuntimeAvailable } from "@/lib/ai/adapters/onnx-adapter";
import type { ModelInput } from "@/lib/ai/types";

export const Route = createFileRoute("/diag-eegconformer")({
  component: DiagPage,
});

function makeInput(): ModelInput {
  const C = 22, T = 1000, sr = 250;
  const data: number[][] = [];
  for (let c = 0; c < C; c++) {
    const ch = new Array<number>(T);
    for (let t = 0; t < T; t++) {
      ch[t] = 0.5 * Math.sin((2 * Math.PI * (c + 1) * t) / 64) + 0.1 * Math.random();
    }
    data.push(ch);
  }
  return { kind: "windows", windows: [{ data, sampleRate: sr, start: 0, end: T }] };
}

function DiagPage() {
  const [report, setReport] = useState<string>("running…");

  useEffect(() => {
    (async () => {
      const lines: string[] = [];
      const log = (s: string) => {
        lines.push(s);
        console.log("[DIAG]", s);
      };
      try {
        log(`registry ids: ${listModels().map((m) => m.id).join(", ")}`);
        log(`hasModel(braindecode-eegconformer-prod) = ${hasModel("braindecode-eegconformer-prod")}`);
        const d = getDescriptor("braindecode-eegconformer-prod");
        log(`descriptor: ${JSON.stringify(d?.capabilities)}`);
        const ortAvail = await isONNXRuntimeAvailable();
        log(`onnxruntime-web available: ${ortAvail}`);

        // Probe artifact fetch
        const t0 = performance.now();
        const resp = await fetch("/models/eegconformer.onnx", { method: "HEAD" });
        log(`HEAD /models/eegconformer.onnx → ${resp.status} ${resp.headers.get("content-length") ?? "?"} bytes (${(performance.now() - t0).toFixed(1)} ms)`);

        log("calling embedEEG() with default routing…");
        const tEmb = performance.now();
        const res = await embedEEG(makeInput());
        const dt = (performance.now() - tEmb).toFixed(1);
        log(`embedEEG done in ${dt} ms`);
        log(`  modelId       = ${res.modelId}`);
        log(`  dim           = ${res.dim}`);
        log(`  fellBack      = ${res.fellBack}`);
        log(`  fallbackReason= ${res.fallbackReason ?? "(none)"}`);
        log(`  normalized    = ${res.normalized}`);
        log(`  vector[0..3]  = [${res.vector.slice(0, 4).map((v) => v.toFixed(4)).join(", ")}]`);
        const norm = Math.sqrt(res.vector.reduce((s, v) => s + v * v, 0));
        log(`  L2 norm       = ${norm.toFixed(4)}`);

        const verdict =
          !res.fellBack && res.modelId === "braindecode-eegconformer-prod" && res.dim === 32
            ? "PASS — EEGConformer ONNX live, 32-D, no fallback"
            : "FAIL — see above";
        log(`VERDICT: ${verdict}`);
      } catch (e) {
        log(`ERROR: ${(e as Error).message}`);
        log(`STACK: ${(e as Error).stack ?? ""}`);
      }
      setReport(lines.join("\n"));
    })();
  }, []);

  return (
    <pre style={{ padding: 16, fontSize: 12, whiteSpace: "pre-wrap" }} data-diag-report>
      {report}
    </pre>
  );
}