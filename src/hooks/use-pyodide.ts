import { useState, useRef, useCallback } from "react";

export type PyodideStatus =
  | "idle"
  | "loading-pyodide"
  | "loading-packages"
  | "ready"
  | "running"
  | "error";

export interface PyodideState {
  status: PyodideStatus;
  message: string;
  progress: number;
  error: string | null;
}

const PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/";

async function loadPyodideScript(): Promise<void> {
  if (window.loadPyodide) return;
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `${PYODIDE_CDN}pyodide.js`;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Pyodide script"));
    document.head.appendChild(script);
  });
}

export function usePyodide() {
  const [state, setState] = useState<PyodideState>({
    status: "idle", message: "Not started", progress: 0, error: null,
  });
  const pyRef = useRef<PyodideInstance | null>(null);
  const set = (s: Partial<PyodideState>) => setState(prev => ({ ...prev, ...s }));

  const initialize = useCallback(async () => {
    if (pyRef.current) return;
    if (window.pyodideInstance) {
      pyRef.current = window.pyodideInstance;
      set({ status: "ready", message: "Ready", progress: 100 });
      return;
    }
    try {
      set({ status: "loading-pyodide", message: "Loading Pyodide runtime…", progress: 5 });
      await loadPyodideScript();
      set({ message: "Initializing Python…", progress: 15 });
      if (!window.loadPyodide) throw new Error("Pyodide script did not register window.loadPyodide");
      const py = await window.loadPyodide({ indexURL: PYODIDE_CDN });
      window.pyodideInstance = py;
      pyRef.current = py;
      set({ status: "loading-packages", message: "Loading NumPy + SciPy…", progress: 30 });
      await py.loadPackage(["numpy", "scipy"]);
      set({ message: "Loading scikit-learn…", progress: 55 });
      await py.loadPackage(["scikit-learn"]);
      set({ message: "Loading MNE-Python…", progress: 70 });
      await py.loadPackage("micropip");
      await py.runPythonAsync(`
import micropip
await micropip.install("mne")
import mne
mne.set_log_level("WARNING")
print(f"MNE {mne.__version__} loaded")
      `);
      set({ status: "ready", message: "MNE-Python ready ✓", progress: 100, error: null });
    } catch (err) {
      set({ status: "error", message: "Failed to load", error: (err as Error).message, progress: 0 });
    }
  }, []);

  const runPython = useCallback(async (code: string): Promise<unknown> => {
    if (!pyRef.current) throw new Error("Pyodide not loaded");
    set({ status: "running" });
    try {
      const result = await pyRef.current.runPythonAsync(code);
      set({ status: "ready" });
      return result;
    } catch (err) {
      set({ status: "error", error: (err as Error).message });
      throw err;
    }
  }, []);

  const setGlobal = useCallback((key: string, value: unknown) => {
    pyRef.current?.globals.set(key, pyRef.current.toPy(value));
  }, []);

  const getGlobal = useCallback((key: string): unknown => {
    return pyRef.current?.globals.get(key);
  }, []);

  return { state, initialize, runPython, setGlobal, getGlobal, py: pyRef.current };
}
