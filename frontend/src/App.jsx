import { useMemo, useState } from "react";
import ReportView from "./ReportView.jsx";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function predictImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/v1/predict?include_gradcam=true`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const onSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setLoading(true);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Adversarial safeguards dashboard</h1>
      <p className="sub">
        Images are resized to 32×32 (CIFAR-style). The API runs the detector, optional input defenses,
        the classifier, Grad-CAM, and returns a transparency report. Generate sample PNGs with{" "}
        <code className="inline-code">python scripts/generate_ui_test_images.py</code> (saved under{" "}
        <code className="inline-code">artifacts/ui_test_images/</code>).
      </p>

      <div className="card">
        <form onSubmit={onSubmit} className="row">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Running pipeline…" : "Run pipeline"}
          </button>
        </form>
        {previewUrl && (
          <div style={{ marginTop: "1rem" }}>
            <img src={previewUrl} alt="preview" style={{ maxWidth: 160, borderRadius: 8 }} />
          </div>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      {result?.transparency && <ReportView transparency={result.transparency} />}
    </div>
  );
}
