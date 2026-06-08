import { useState, useCallback } from "react";

export interface MOABBDataset {
  id: string; name: string; description: string;
  n_subjects: number; n_sessions: number; n_runs: number;
  paradigm: string; sfreq: number; n_channels: number; scientific_ref: string;
}

export interface MOABBEpoch {
  subject: string; session: string; label: string;
  data: number[][]; sfreq: number; ch_names: string[];
}

export interface MOABBProgress {
  phase: "idle"|"loading-moabb"|"fetching"|"processing"|"done"|"error";
  message: string; progress: number; error?: string;
}

export const MOABB_DATASETS: MOABBDataset[] = [
  { id: "BNCI2014_001", name: "BCI Competition IV 2a", description: "Motor imagery: left hand, right hand, feet, tongue. 9 subjects × 2 sessions.", n_subjects: 9, n_sessions: 2, n_runs: 6, paradigm: "MotorImagery", sfreq: 250, n_channels: 22, scientific_ref: "Tangermann et al. 2012" },
  { id: "BNCI2014_004", name: "BCI Competition IV 2b", description: "Motor imagery: left hand, right hand. 9 subjects × 5 sessions.", n_subjects: 9, n_sessions: 5, n_runs: 5, paradigm: "MotorImagery", sfreq: 250, n_channels: 3, scientific_ref: "Leeb et al. 2008" },
  { id: "Cho2017", name: "Cho 2017", description: "Motor imagery: left hand, right hand. 52 subjects.", n_subjects: 52, n_sessions: 1, n_runs: 1, paradigm: "MotorImagery", sfreq: 512, n_channels: 64, scientific_ref: "Cho et al. 2017" },
  { id: "PhysionetMI", name: "PhysioNet Motor Imagery", description: "Motor imagery: hands and feet. 109 subjects.", n_subjects: 109, n_sessions: 1, n_runs: 14, paradigm: "MotorImagery", sfreq: 160, n_channels: 64, scientific_ref: "Goldberger et al. 2000" },
];

export function useMOABB() {
  const [progress, setProgress] = useState<MOABBProgress>({ phase: "idle", message: "Not started", progress: 0 });
  const [epochs, setEpochs] = useState<MOABBEpoch[]>([]);
  const set = (p: Partial<MOABBProgress>) => setProgress(prev => ({ ...prev, ...p }));

  const loadDataset = useCallback(async (datasetId: string, subjects: number[] = [1]) => {
    const py = window.pyodideInstance;
    if (!py) throw new Error("Pyodide not loaded — load MNE-Python first");
    setEpochs([]);
    set({ phase: "loading-moabb", message: "Installing MOABB…", progress: 5 });
    try {
      await py.runPythonAsync(`
import micropip
await micropip.install(["moabb", "pooch", "requests"])
import moabb
moabb.set_log_level("WARNING")
      `);
      set({ phase: "fetching", message: `Loading ${datasetId}…`, progress: 20 });
      py.globals.set("_dataset_id", datasetId);
      py.globals.set("_subjects", py.toPy(subjects));
      const result = await py.runPythonAsync(`
import json, numpy as np
dataset_classes = {
    "BNCI2014_001": ("moabb.datasets", "BNCI2014_001"),
    "BNCI2014_004": ("moabb.datasets", "BNCI2014_004"),
    "Cho2017": ("moabb.datasets", "Cho2017"),
    "PhysionetMI": ("moabb.datasets", "PhysionetMI"),
}
mod_name, cls_name = dataset_classes[_dataset_id]
import importlib
mod = importlib.import_module(mod_name)
DatasetClass = getattr(mod, cls_name)
dataset = DatasetClass()
dataset.subject_list = list(_subjects)
from moabb.paradigms import MotorImagery
paradigm = MotorImagery(fmin=1.0, fmax=40.0, tmin=0.0, tmax=2.0)
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=list(_subjects))
epochs_out = []
for i in range(min(50, X.shape[0])):
    epochs_out.append({
        "subject": str(metadata.iloc[i]["subject"]),
        "session": str(metadata.iloc[i]["session"]),
        "label": str(labels[i]),
        "data": X[i].tolist(),
        "sfreq": 250.0,
        "ch_names": [f"ch{j}" for j in range(X.shape[1])],
    })
json.dumps({"epochs": epochs_out, "n_total": X.shape[0], "n_channels": X.shape[1], "n_times": X.shape[2], "labels": list(set(labels.tolist()))})
      `) as string;
      const parsed = JSON.parse(result);
      setEpochs(parsed.epochs as MOABBEpoch[]);
      set({ phase: "done", message: `Loaded ${parsed.n_total} epochs from ${datasetId}`, progress: 100 });
      return parsed;
    } catch (err) {
      set({ phase: "error", message: "Failed", error: (err as Error).message, progress: 0 });
      throw err;
    }
  }, []);

  return { progress, epochs, loadDataset, MOABB_DATASETS };
}
