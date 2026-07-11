declare global {
  interface PyodideInstance {
    loadPackage: (pkgs: string | string[]) => Promise<void>;
    runPythonAsync: (code: string) => Promise<unknown>;
    globals: {
      get: (key: string) => unknown;
      set: (key: string, value: unknown) => void;
    };
    toPy: (obj: unknown) => unknown;
  }
  interface OrtTensor {
    data: unknown;
  }
  interface OrtInferenceSession {
    inputNames: string[];
    outputNames: string[];
    run: (feeds: Record<string, OrtTensor>) => Promise<Record<string, OrtTensor>>;
  }
  interface Window {
    loadPyodide?: (options: { indexURL: string }) => Promise<PyodideInstance>;
    pyodideInstance?: PyodideInstance;
    ort?: {
      InferenceSession: {
        create: (buffer: ArrayBuffer, options?: object) => Promise<OrtInferenceSession>;
      };
      Tensor: new (type: string, data: Float32Array, dims: number[]) => OrtTensor;
    };
  }
}

export {};
