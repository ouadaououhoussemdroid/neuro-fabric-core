import { physionet } from "../eeg/loaders/physionet";
import { preprocess } from "../eeg/preprocessing";
import { bandPowerFeatures } from "../embeddings/features";
import type { EEGWindow } from "../eeg/types";

export interface TrainingSample {
  features: number[];
  labels: number[];
  subject: string;
  task: string;
}

export interface TrainedWeights {
  W1: number[][];B1: number[];W2: number[][];B2: number[];W3: number[][];B3: number[];
  meta: { n_samples: number; n_subjects: number; val_mse: number; epochs_trained: number; trained_at: string };
}

export interface TrainingProgress {
  phase: "fetching"|"preprocessing"|"training"|"done"|"error";
  subject?: string; epoch?: number; totalEpochs?: number;
  loss?: number; valLoss?: number; samplesLoaded?: number;
  message?: string; error?: string;
}

function taskLabels(task: string, f: number[]): [number,number,number] {
  const total = f.reduce((s,v)=>s+v,0)+1e-9;
  const [,nt,na,nb,ng] = f.map(v=>v/total);
  if (task==="baseline") {
    return [
      Math.min(0.4,Math.max(0.1, 0.2+0.3*(nb/Math.max(1e-9,na+nt)))),
      Math.min(0.4,Math.max(0.1, 0.1+0.3*(nt/Math.max(1e-9,na)))),
      Math.min(0.4,Math.max(0.1, 0.1+0.4*(nb+ng))),
    ];
  }
  return [
    Math.min(0.95,Math.max(0.4, 0.5+0.4*(nb/Math.max(1e-9,na+nt)))),
    Math.min(0.95,Math.max(0.3, 0.4+0.4*(nt/Math.max(1e-9,na)))),
    Math.min(0.95,Math.max(0.3, 0.3+0.5*(nb+ng))),
  ];
}

function extractMeanBandPower(windows: EEGWindow[]): number[] {
  if (windows.length===0) return [0,0,0,0,0];
  const C = windows[0].data.length;
  const acc = new Array(5).fill(0);
  for (const w of windows) {
    const f = bandPowerFeatures(w);
    for (let c=0;c<C;c++) for (let b=0;b<5;b++) acc[b]+=f[c*5+b];
  }
  const n = windows.length*Math.max(1,C);
  const raw = acc.map(v=>v/n);
  const total = raw.reduce((s,v)=>s+v,0)+1e-9;
  return raw.map(v=>v/total);
}

function randn(): number {
  return Math.sqrt(-2*Math.log(Math.random()+1e-9))*Math.cos(2*Math.PI*Math.random());
}
function initW(rows: number, cols: number, scale: number): number[][] {
  return Array.from({length:rows},()=>Array.from({length:cols},()=>randn()*scale));
}
function relu(x: number) { return Math.max(0,x); }
function reluG(x: number) { return x>0?1:0; }
function sig(x: number) { return 1/(1+Math.exp(-Math.max(-20,Math.min(20,x)))); }
function sigG(y: number) { return y*(1-y); }
function fwd(x:number[],W1:number[][],B1:number[],W2:number[][],B2:number[],W3:number[][],B3:number[]) {
  const z1=W1.map((r,i)=>r.reduce((s,w,j)=>s+w*x[j],B1[i])); const h1=z1.map(relu);
  const z2=W2.map((r,i)=>r.reduce((s,w,j)=>s+w*h1[j],B2[i])); const h2=z2.map(relu);
  const z3=W3.map((r,i)=>r.reduce((s,w,j)=>s+w*h2[j],B3[i])); const out=z3.map(sig);
  return {h1,h2,out};
}

function trainMLP(samples: TrainingSample[], epochs=150, lr=0.001, h1=32, h2=16,
  onEpoch?: (e:number,l:number,v:number)=>void): TrainedWeights {
  const I=5,O=3;
  let W1=initW(h1,I,Math.sqrt(2/I)),B1=new Array(h1).fill(0);
  let W2=initW(h2,h1,Math.sqrt(2/h1)),B2=new Array(h2).fill(0);
  let W3=initW(O,h2,Math.sqrt(2/h2)),B3=new Array(O).fill(0);
  const b1=0.9,b2=0.999,eps=1e-8; let t=0;
  const mW=(W:number[][])=>({m:W.map(r=>r.map(()=>0)),v:W.map(r=>r.map(()=>0))});
  const mV=(B:number[])=>({m:B.map(()=>0),v:B.map(()=>0)});
  let mW1=mW(W1),vW1=mW(W1),mB1=mV(B1),vB1=mV(B1);
  let mW2=mW(W2),vW2=mW(W2),mB2=mV(B2),vB2=mV(B2);
  let mW3=mW(W3),vW3=mW(W3),mB3=mV(B3),vB3=mV(B3);
  const split=Math.floor(samples.length*0.8);
  const sh=[...samples].sort(()=>Math.random()-0.5);
  const tr=sh.slice(0,split),va=sh.slice(split);
  function adam(p:number[][],g:number[][],ms:ReturnType<typeof mW>,vs:ReturnType<typeof mW>,t:number) {
    const lr_t=lr*Math.sqrt(1-b2**t)/(1-b1**t);
    return p.map((r,i)=>r.map((v,j)=>{
      ms.m[i][j]=b1*ms.m[i][j]+(1-b1)*g[i][j];
      vs.v[i][j]=b2*vs.v[i][j]+(1-b2)*g[i][j]**2;
      return v-lr_t*ms.m[i][j]/(Math.sqrt(vs.v[i][j])+eps);
    }));
  }
  function adamV(p:number[],g:number[],ms:ReturnType<typeof mV>,vs:ReturnType<typeof mV>,t:number) {
    const lr_t=lr*Math.sqrt(1-b2**t)/(1-b1**t);
    return p.map((v,i)=>{
      ms.m[i]=b1*ms.m[i]+(1-b1)*g[i]; vs.v[i]=b2*vs.v[i]+(1-b2)*g[i]**2;
      return v-lr_t*ms.m[i]/(Math.sqrt(vs.v[i])+eps);
    });
  }
  let bestVal=Infinity,best={W1,B1,W2,B2,W3,B3},pat=0;
  for (let ep=0;ep<epochs;ep++) {
    t++;
    const batch=[...tr].sort(()=>Math.random()-0.5);
    const dW1=W1.map(r=>r.map(()=>0)),dB1=B1.map(()=>0);
    const dW2=W2.map(r=>r.map(()=>0)),dB2=B2.map(()=>0);
    const dW3=W3.map(r=>r.map(()=>0)),dB3=B3.map(()=>0);
    let tl=0;
    for (const s of batch) {
      const {h1:a1,h2:a2,out}=fwd(s.features,W1,B1,W2,B2,W3,B3);
      const dO=out.map((o,i)=>o-s.labels[i]);
      tl+=dO.reduce((s,d)=>s+d*d,0)/O;
      const dZ3=dO.map((d,i)=>d*sigG(out[i]));
      for(let i=0;i<O;i++){for(let j=0;j<h2;j++)dW3[i][j]+=dZ3[i]*a2[j];dB3[i]+=dZ3[i];}
      const dA2=new Array(h2).fill(0);
      for(let j=0;j<h2;j++)for(let i=0;i<O;i++)dA2[j]+=dZ3[i]*W3[i][j];
      const dZ2=dA2.map((d,j)=>d*reluG(a2[j]));
      for(let i=0;i<h2;i++){for(let j=0;j<h1;j++)dW2[i][j]+=dZ2[i]*a1[j];dB2[i]+=dZ2[i];}
      const dA1=new Array(h1).fill(0);
      for(let j=0;j<h1;j++)for(let i=0;i<h2;i++)dA1[j]+=dZ2[i]*W2[i][j];
      const dZ1=dA1.map((d,j)=>d*reluG(a1[j]));
      for(let i=0;i<h1;i++){for(let j=0;j<I;j++)dW1[i][j]+=dZ1[i]*s.features[j];dB1[i]+=dZ1[i];}
    }
    const N=batch.length;
    W1=adam(W1,dW1.map(r=>r.map(v=>v/N)),mW1,vW1,t);B1=adamV(B1,dB1.map(v=>v/N),mB1,vB1,t);
    W2=adam(W2,dW2.map(r=>r.map(v=>v/N)),mW2,vW2,t);B2=adamV(B2,dB2.map(v=>v/N),mB2,vB2,t);
    W3=adam(W3,dW3.map(r=>r.map(v=>v/N)),mW3,vW3,t);B3=adamV(B3,dB3.map(v=>v/N),mB3,vB3,t);
    const vl=va.reduce((s,sv)=>{const{out}=fwd(sv.features,W1,B1,W2,B2,W3,B3);return s+out.reduce((s2,o,i)=>s2+(o-sv.labels[i])**2,0)/O;},0)/Math.max(1,va.length);
    if(ep%10===0||ep===0)onEpoch?.(ep+1,tl/N,vl);
    if(vl<bestVal){bestVal=vl;best={W1:W1.map(r=>[...r]),B1:[...B1],W2:W2.map(r=>[...r]),B2:[...B2],W3:W3.map(r=>[...r]),B3:[...B3]};pat=0;}
    else if(++pat>=20)break;
  }
  return {...best,meta:{n_samples:samples.length,n_subjects:new Set(samples.map(s=>s.subject)).size,val_mse:bestVal,epochs_trained:t,trained_at:new Date().toISOString()}};
}

export async function runTrainingPipeline(options:{maxSubjects?:number;runsPerSubject?:number;epochs?:number;onProgress?:(p:TrainingProgress)=>void}={}): Promise<TrainedWeights> {
  const {maxSubjects=10,runsPerSubject=4,epochs=150,onProgress}=options;
  const rep=(p:TrainingProgress)=>onProgress?.(p);
  const samples:TrainingSample[]=[];
  const records=await physionet.list();
  const subjects=[...new Set(records.map(r=>r.subject))].slice(0,maxSubjects);
  for (const subject of subjects) {
    const recs=records.filter(r=>r.subject===subject).slice(0,runsPerSubject);
    for (const record of recs) {
      rep({phase:"fetching",subject:record.id,samplesLoaded:samples.length,message:`Fetching ${record.id}…`});
      try {
        const signal=await physionet.load(record);
        rep({phase:"preprocessing",subject:record.id,samplesLoaded:samples.length,message:`Preprocessing ${record.id}…`});
        const {windows}=preprocess(signal,{bandpass:{low:1,high:40},notch:{fc:60},normalize:true,segment:{windowSec:2,overlap:0.5},artifactRejection:{enabled:true}});
        if(windows.length===0)continue;
        const features=extractMeanBandPower(windows);
        const labels=taskLabels(record.task,features);
        for(let aug=0;aug<5;aug++){
          const noisy=features.map(f=>Math.max(0,f+(Math.random()-0.5)*0.02));
          const total=noisy.reduce((s,v)=>s+v,0)+1e-9;
          samples.push({features:noisy.map(v=>v/total),labels:labels.map(l=>Math.min(1,Math.max(0,l+(Math.random()-0.5)*0.05))),subject,task:record.task});
        }
      } catch(err){rep({phase:"fetching",subject:record.id,message:`Skipped ${record.id}: ${(err as Error).message}`});}
    }
  }
  if(samples.length<20)throw new Error(`Not enough samples (got ${samples.length}). Check PhysioNet access.`);
  rep({phase:"training",samplesLoaded:samples.length,message:`Training on ${samples.length} samples…`});
  const weights=trainMLP(samples,epochs,0.001,32,16,(ep,l,vl)=>{
    if(ep%10===0||ep===1)rep({phase:"training",epoch:ep,totalEpochs:epochs,loss:l,valLoss:vl,samplesLoaded:samples.length});
  });
  rep({phase:"done",samplesLoaded:samples.length,message:`Done! val_mse=${weights.meta.val_mse.toFixed(4)}`});
  return weights;
  }
