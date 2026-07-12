import { describe, it, expect } from "vitest";
import { syncModelFromMLflow, type MLflowClient, type MLflowRunInfo } from "../registry-sync";
import type { ArtefactManifest } from "../../artefacts/hashed-artefact";
import { sha256Hex } from "../../artefacts/hashed-artefact";

const ARTEFACT = new Uint8Array([1, 2, 3, 4, 5]);
const HASH = sha256Hex(ARTEFACT);

const manifest: ArtefactManifest = {
  generated: "2026-07-11T00:00:00Z",
  models: {
    eegconformer: {
      id: "eegconformer",
      url: "/models/eegconformer.onnx",
      sha256: HASH,
      size: ARTEFACT.length,
    },
  },
};

function mockMLflow(runs: MLflowRunInfo[] = []): MLflowClient {
  return {
    getRun: async () => null,
    searchRuns: async () => runs,
  };
}

describe("syncModelFromMLflow", () => {
  it("returns no_manifest when the model is not in the manifest", async () => {
    const result = await syncModelFromMLflow("unknown-model", mockMLflow(), manifest, ARTEFACT);
    expect(result.status).toBe("no_manifest");
    expect(result.verified).toBe(false);
  });

  it("returns hash_mismatch when artefact bytes don't match", async () => {
    const wrongBytes = new Uint8Array([9, 9, 9]);
    const result = await syncModelFromMLflow(
      "braindecode-eegconformer-prod",
      mockMLflow(),
      manifest,
      wrongBytes,
    );
    expect(result.status).toBe("hash_mismatch");
    expect(result.verified).toBe(false);
  });

  it("returns synced when Storage hash matches and no MLflow run exists", async () => {
    const result = await syncModelFromMLflow(
      "braindecode-eegconformer-prod",
      mockMLflow([]),
      manifest,
      ARTEFACT,
    );
    expect(result.status).toBe("synced");
    expect(result.verified).toBe(true);
    expect(result.mlflowRunId).toBeNull();
  });

  it("returns synced when MLflow hash tag matches Storage hash", async () => {
    const run: MLflowRunInfo = {
      runId: "run-1",
      experimentName: "eegconformer-bciiv2a",
      status: "finished",
      params: { lr: "0.001" },
      metrics: { val_loss: 0.45 },
      tags: { "sha256_eegconformer.onnx": HASH },
    };
    const result = await syncModelFromMLflow(
      "braindecode-eegconformer-prod",
      mockMLflow([run]),
      manifest,
      ARTEFACT,
    );
    expect(result.status).toBe("synced");
    expect(result.verified).toBe(true);
    expect(result.mlflowRunId).toBe("run-1");
    expect(result.mlflowHash).toBe(HASH);
  });

  it("returns hash_mismatch when MLflow hash tag differs from Storage", async () => {
    const run: MLflowRunInfo = {
      runId: "run-2",
      experimentName: "eegconformer-bciiv2a",
      status: "finished",
      params: {},
      metrics: {},
      tags: { "sha256_eegconformer.onnx": "0".repeat(64) },
    };
    const result = await syncModelFromMLflow(
      "braindecode-eegconformer-prod",
      mockMLflow([run]),
      manifest,
      ARTEFACT,
    );
    expect(result.status).toBe("hash_mismatch");
    expect(result.mlflowHash).toBe("0".repeat(64));
    expect(result.storageHash).toBe(HASH);
  });

  it("returns mlflow_unavailable when searchRuns throws", async () => {
    const client: MLflowClient = {
      getRun: async () => null,
      searchRuns: async () => {
        throw new Error("connection refused");
      },
    };
    const result = await syncModelFromMLflow(
      "braindecode-eegconformer-prod",
      client,
      manifest,
      ARTEFACT,
    );
    expect(result.status).toBe("mlflow_unavailable");
    expect(result.verified).toBe(false);
  });
});
