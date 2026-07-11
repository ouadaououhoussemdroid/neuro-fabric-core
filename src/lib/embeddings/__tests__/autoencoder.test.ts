import { describe, it, expect } from "vitest";
import { fitAutoencoder, encode, decode } from "../autoencoder";
import { fitPCA, transformPCA } from "../pca";

const X = Array.from({ length: 20 }, (_, i) => [
  Math.sin(i * 0.3),
  Math.cos(i * 0.3),
  Math.sin(i * 0.7) + i * 0.05,
]);

describe("fitAutoencoder", () => {
  it("reduces exactly to PCA — encoder equals pca.components, mean matches", () => {
    // Both PCA and the "autoencoder" seed their power iteration from the
    // same fixed seed (NFC-014), so fitting each independently on
    // identical input is deterministic and directly comparable.
    const ae = fitAutoencoder(X, 2);
    const pca = fitPCA(X, 2);
    expect(ae.encoder).toEqual(pca.components);
    expect(ae.mean).toEqual(pca.mean);
    expect(ae.kind).toBe("linear-ae");
    expect(ae.latentDim).toBe(2);
  });

  it("decoder is the transpose of the encoder", () => {
    const ae = fitAutoencoder(X, 2);
    for (let k = 0; k < ae.encoder.length; k++) {
      for (let d = 0; d < ae.encoder[k].length; d++) {
        expect(ae.decoder[d][k]).toBe(ae.encoder[k][d]);
      }
    }
  });
});

describe("encode", () => {
  it("matches transformPCA against an equivalent PCAModel", () => {
    const ae = fitAutoencoder(X, 2);
    const encoded = encode(ae, X[3]);
    const expected = transformPCA(
      { mean: ae.mean, components: ae.encoder, explainedVariance: [] },
      X[3],
    );
    expect(encoded).toEqual(expected);
  });
});

describe("decode", () => {
  it("implements decoder @ z + mean", () => {
    const ae = fitAutoencoder(X, 2);
    const z = encode(ae, X[0]);
    const decoded = decode(ae, z);
    // Hand-compute the same formula independently
    const expected = ae.decoder.map((row, i) => {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += row[j] * z[j];
      return s + ae.mean[i];
    });
    expect(decoded).toEqual(expected);
  });

  it("round-trips (encode then decode) back close to the original at full rank", () => {
    const twoD = X.map((row) => row.slice(0, 2)); // 2-D input
    const ae = fitAutoencoder(twoD, 2); // full-rank latent dim
    const z = encode(ae, twoD[5]);
    const reconstructed = decode(ae, z);
    for (let i = 0; i < 2; i++) {
      expect(reconstructed[i]).toBeCloseTo(twoD[5][i], 3);
    }
  });
});
