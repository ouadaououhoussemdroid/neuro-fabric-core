interface PyodideInstance {
  loadPackage: (pkgs: string | string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: {
    get: (key: string) => unknown;
    set: (key: string, value: unknown) => void;
  };
  toPy: (obj: unknown) => unknown;
}

declare global {
  interface Window {
    loadPyodide: (options: { indexURL: string }) => Promise<PyodideInstance>;
    pyodideInstance?: PyodideInstance;
    ort?: {
      InferenceSession: {
        create: (buffer: ArrayBuffer, options?: object) => Promise<any>;
      };
      Tensor: new (type: string, data: Float32Array, dims: number[]) => any;
    };
  }
}

export {};
