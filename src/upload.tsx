import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/upload")({
  component: UploadPage,
});

function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("/api/eeg/upload", {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: 600,
      margin: "40px auto",
      padding: 24,
      fontFamily: "sans-serif"
    }}>
      <h1>رفع ملف EEG</h1>

      <input
        type="file"
        accept=".edf,.csv,.npy"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        style={{ marginBottom: 16, display: "block" }}
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        style={{
          padding: "10px 24px",
          background: loading ? "#ccc" : "#6c47ff",
          color: "white",
          border: "none",
          borderRadius: 8,
          cursor: loading ? "not-allowed" : "pointer",
          fontSize: 16
        }}
      >
        {loading ? "جاري المعالجة..." : "رفع وتحليل"}
      </button>

      {error && (
        <div style={{
          marginTop: 16,
          padding: 12,
          background: "#fee",
          borderRadius: 8,
          color: "red"
        }}>
          خطأ: {error}
        </div>
      )}

      {result && (
        <div style={{
          marginTop: 24,
          padding: 16,
          background: "#f0f7ff",
          borderRadius: 8
        }}>
          <h2>النتائج:</h2>

          <h3>🧠 الحالة الذهنية:</h3>
          <p>التركيز: {result.decoder?.attention?.toFixed(2)}</p>
          <p>العبء الذهني: {result.decoder?.workload?.toFixed(2)}</p>
          <p>اليقظة: {result.decoder?.arousal?.toFixed(2)}</p>

          <h3>📊 الإشارة:</h3>
          <p>القنوات: {result.signal?.channels}</p>
          <p>معدل العينات: {result.signal?.sampleRate} Hz</p>
          <p>العينات: {result.signal?.samples}</p>

          <h3>⏱️ الأوقات:</h3>
          <p>المعالجة: {result.timings?.preprocess_ms} ms</p>
          <p>الـ Embedding: {result.timings?.embed_ms} ms</p>
          <p>الإجمالي: {result.timings?.total_ms} ms</p>
        </div>
      )}
    </div>
  );
      }
