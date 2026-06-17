import { useState, useCallback, useRef } from "react";

export type TrainerPhase = "idle"|"installing"|"extracting"|"training"|"exporting"|"loading"|"ready"|"inferring"|"error";

export interface TrainerState {
  phase: TrainerPhase; message: string; progress: number;
  error?: string; accuracy?: number; modelSizeKB?: number;
}

export interface InferenceResult {
  label: string; probabilities: Record<string, number>; latencyMs: number;
}

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";

async function loadOrtScript(): Promise<void> {
  if (window.ort) return;
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = ORT_CDN;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load onnxruntime-web"));
    document.head.appendChild(s);
  });
}

export function useOnnxTrainer() {
  const [state, setState] = useState<TrainerState>({ phase: "idle", message: "Not started", progress: 0 });
  const sessionRef = useRef<any>(null);
  const labelsRef = useRef<string[]>([]);
  const onnxBytesRef = useRef<Uint8Array | null>(null);
  const set = (s: Partial<TrainerState>) => setState(prev => ({ ...prev, ...s }));

  const trainAndExport = useCallback(async (
    datasetId = "BNCI2014_001",
    subjects = [1],
    classifier: "svm"|"lda"|"rf" = "lda",
  ) => {
    const py = window.pyodideInstance;
    if (!py) throw new Error("Pyodide not loaded — open /mne first");
    set({ phase: "installing", message: "Installing skl2onnx…", progress: 5 });
    try {
      await py.runPythonAsync(`
import micropip
await micropip.install(["scikit-learn","skl2onnx","onnxconverter-common","moabb","pooch"])
      `);
      set({ phase: "extracting", message: `Loading ${datasetId}…`, progress: 20 });
      py.globals.set("_dataset_id", datasetId);
      py.globals.set("_subjects", py.toPy(subjects));
      py.globals.set("_classifier", classifier);
      const result = await py.runPythonAsync(`
import json, numpy as np, importlib
from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import moabb
moabb.set_log_level("WARNING")

dataset_classes = {
    "BNCI2014_001":("moabb.datasets","BNCI2014_001"),
    "BNCI2014_004":("moabb.datasets","BNCI2014_004"),
    "Cho2017":("moabb.datasets","Cho2017"),
    "PhysionetMI":("moabb.datasets","PhysionetMI"),
}
mod_name,cls_name = dataset_classes[_dataset_id]
DatasetClass = getattr(importlib.import_module(mod_name), cls_name)
dataset = DatasetClass()
dataset.subject_list = list(_subjects)
from moabb.paradigms import MotorImagery
paradigm = MotorImagery(fmin=1.0, fmax=40.0, tmin=0.0, tmax=2.0)
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=list(_subjects))
sfreq = 250.0
bands = [(0.5,4),(4,8),(8,13),(13,30),(30,45)]
def bp(epoch):
    f=[]
    for ch in epoch:
        freqs,psd=sp_signal.welch(ch,fs=sfreq,nperseg=min(256,len(ch)))
        for lo,hi in bands:
            idx=(freqs>=lo)&(freqs<=hi)
            f.append(float(np.mean(psd[idx])))
    return f
features=np.array([bp(ep) for ep in X],dtype=np.float32)
le=LabelEncoder(); y=le.fit_transform(labels); label_names=list(le.classes_)
if _classifier=="lda":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf=LinearDiscriminantAnalysis()
elif _classifier=="svm":
    from sklearn.svm import SVC
    clf=SVC(kernel="rbf",probability=True,C=1.0)
else:
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
pipe=Pipeline([("scaler",StandardScaler()),("clf",clf)])
cv=cross_val_score(pipe,features,y,cv=5,scoring="accuracy")
pipe.fit(features,y)
onnx_model=convert_sklearn(pipe,initial_types=[("float_input",FloatTensorType([None,features.shape[1]]))],target_opset=12)
onnx_bytes=onnx_model.SerializeToString()
json.dumps({"accuracy":round(float(np.mean(cv)),4),"cv_scores":[round(float(s),4) for s in cv],"n_epochs":int(X.shape[0]),"n_features":int(features.shape[1]),"n_classes":len(label_names),"label_names":label_names,"onnx_bytes":list(onnx_bytes),"onnx_size_kb":round(len(onnx_bytes)/1024,1),"classifier":_classifier})
      `) as string;
      const parsed = JSON.parse(result);
      labelsRef.current = parsed.label_names;
      set({ phase: "loading", message: `Loading ONNX (${parsed.onnx_size_kb} KB)…`, progress: 75, accuracy: parsed.accuracy, modelSizeKB: parsed.onnx_size_kb });
      await loadOrtScript();
      const onnxBytes = new Uint8Array(parsed.onnx_bytes);
      onnxBytesRef.current = onnxBytes;
      const session = await window.ort!.InferenceSession.create(onnxBytes.buffer, { executionProviders: ["wasm"] });
      sessionRef.current = session;
      set({ phase: "ready", message: `Ready — ${parsed.classifier.toUpperCase()} · ${(parsed.accuracy*100).toFixed(1)}% CV · ${parsed.onnx_size_kb} KB`, progress: 100, accuracy: parsed.accuracy, modelSizeKB: parsed.onnx_size_kb });
      return parsed;
    } catch (err) {
      set({ phase: "error", message: "Failed", error: (err as Error).message, progress: 0 });
      throw err;
    }
  }, []);

  const infer = useCallback(async (features: number[]): Promise<InferenceResult> => {
    const session = sessionRef.current;
    if (!session || !window.ort) throw new Error("Model not loaded");
    set({ phase: "inferring" });
    const t0 = performance.now();
    const tensor = new window.ort.Tensor("float32", new Float32Array(features), [1, features.length]);
    const output = await session.run({ [session.inputNames[0]]: tensor });
    const latencyMs = +(performance.now()-t0).toFixed(2);
    const labels = labelsRef.current;
    const probKey = session.outputNames.find((n: string) => n.includes("prob")) ?? session.outputNames[1] ?? session.outputNames[0];
    const probData = output[probKey]?.data as Float32Array ?? new Float32Array(labels.length).fill(1/labels.length);
    const labelData = output[session.outputNames[0]]?.data;
    const predictedIdx = labelData ? Number(labelData[0]) : Array.from(probData).indexOf(Math.max(...Array.from(probData)));
    const probabilities: Record<string,number> = {};
    labels.forEach((l,i) => { probabilities[l] = +(probData[i]??0).toFixed(4); });
    set({ phase: "ready" });
    return { label: labels[predictedIdx]??"unknown", probabilities, latencyMs };
  }, []);

  const downloadOnnx = useCallback((filename = "eeg-classifier.onnx") => {
    if (!onnxBytesRef.current) return;
    const blob = new Blob([onnxBytesRef.current as BlobPart], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href=url; a.download=filename; a.click();
    URL.revokeObjectURL(url);
  }, []);

  return { state, trainAndExport, infer, downloadOnnx, labels: labelsRef.current };
                                             }
