import { useEffect, useRef } from "react";

function formatPct(x) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function formatFloat(x, digits = 4) {
  if (typeof x !== "number" || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

/** rgba: H×W×4 in [0,1] */
function GradCamCanvas({ rgba, label }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !rgba?.length) return;
    const h = rgba.length;
    const w = rgba[0]?.length ?? 0;
    if (!h || !w) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const px = rgba[y][x];
        const r = px?.[0] ?? 0;
        const g = px?.[1] ?? 0;
        const b = px?.[2] ?? 0;
        const a = px?.[3] ?? 1;
        const i = (y * w + x) * 4;
        img.data[i] = Math.round(Math.min(255, Math.max(0, r * 255)));
        img.data[i + 1] = Math.round(Math.min(255, Math.max(0, g * 255)));
        img.data[i + 2] = Math.round(Math.min(255, Math.max(0, b * 255)));
        img.data[i + 3] = Math.round(Math.min(255, Math.max(0, a * 255)));
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [rgba]);

  if (!rgba?.length) {
    return <p className="muted small">No Grad-CAM data in this response.</p>;
  }

  return (
    <div className="gradcam-wrap">
      {label && <p className="small muted">{label}</p>}
      <canvas ref={ref} className="gradcam-canvas" title="Grad-CAM saliency (upsampled to input size)" />
    </div>
  );
}

function FlagChips({ flags }) {
  const entries = Object.entries(flags || {});
  if (!entries.length) return <p className="muted small">No flags.</p>;
  return (
    <div className="chip-row">
      {entries.map(([k, v]) => (
        <span key={k} className={`chip ${v ? "chip-on" : "chip-off"}`} title={k}>
          {k.replace(/_/g, " ")}: {v ? "yes" : "no"}
        </span>
      ))}
    </div>
  );
}

function ScoreTable({ scores, calibration }) {
  const rows = [
    ["Max probability (detector pass)", scores?.max_prob, formatPct(scores?.max_prob)],
    ["Entropy", scores?.entropy, formatFloat(scores?.entropy, 3)],
    ["KL to clean prior", scores?.kl_to_ref_mean, formatFloat(scores?.kl_to_ref_mean, 4)],
    ["Confidence Δ under noise", scores?.confidence_delta_noisy, formatFloat(scores?.confidence_delta_noisy, 4)],
  ];
  const calRows = [
    ["Clean cal: conf p05", calibration?.clean_conf_p05, formatPct(calibration?.clean_conf_p05)],
    ["Clean cal: KL p95", calibration?.kl_clean_p95, formatFloat(calibration?.kl_clean_p95, 4)],
  ];
  return (
    <div className="tables">
      <table className="kv">
        <tbody>
          {rows.map(([label, , disp]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{disp}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <table className="kv kv-muted">
        <tbody>
          {calRows.map(([label, , disp]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{disp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ConfidenceMeter({ value }) {
  const pct = Math.min(100, Math.max(0, (value ?? 0) * 100));
  return (
    <div className="meter" aria-label={`Confidence ${pct.toFixed(1)} percent`}>
      <div className="meter-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function ReportView({ transparency }) {
  if (!transparency) return null;

  const pred = transparency.prediction || {};
  const det = transparency.detector || {};
  const mon = transparency.monitoring || {};
  const eth = transparency.ethics || {};
  const cam = mon.grad_cam;

  return (
    <div className="report">
      <section className="card report-hero">
        <div className="report-hero-top">
          <div>
            <h2 className="report-title">Transparency report</h2>
            <p className="mono small muted">
              Request <span className="select-all">{transparency.request_id}</span>
            </p>
            <p className="small muted">
              Model: <strong>{transparency.model_name}</strong> · v{transparency.model_version}
            </p>
          </div>
          <span className={`badge ${transparency.risk_tier}`}>Risk: {transparency.risk_tier}</span>
        </div>

        <div className="pred-block">
          <div className="pred-label">Prediction</div>
          <div className="pred-main">
            <span className="pred-class">{pred.class_name ?? "—"}</span>
            <span className="pred-index muted small">(class #{pred.class_index ?? "—"})</span>
          </div>
          <div className="pred-conf-row">
            <span className="small">Confidence</span>
            <span className="pred-conf-val">{formatPct(pred.confidence)}</span>
          </div>
          <ConfidenceMeter value={pred.confidence} />
        </div>
      </section>

      {transparency.risk_rationale?.length > 0 && (
        <section className="card">
          <h3 className="section-title">Why this risk tier</h3>
          <ul className="readable-list">
            {transparency.risk_rationale.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h3 className="section-title">Detector</h3>
        <p className="small muted section-lead">Statistical signals vs clean calibration (not a certified guarantee).</p>
        <ScoreTable scores={det.scores} calibration={det.calibration} />
        <h4 className="subsection-title">Flags</h4>
        <FlagChips flags={det.flags} />
      </section>

      <section className="card">
        <h3 className="section-title">Monitoring & explainability</h3>
        <ul className="kv-inline">
          <li>
            <span className="muted">Input defense</span>{" "}
            <strong>{mon.input_defense_enabled ? "On" : "Off"}</strong>
          </li>
          {mon.grad_cam?.class_index != null && (
            <li>
              <span className="muted">Grad-CAM target class</span> <strong>#{mon.grad_cam.class_index}</strong>
            </li>
          )}
        </ul>
        {mon.notes?.length > 0 && (
          <ul className="readable-list small">
            {mon.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        )}
        <h4 className="subsection-title">Grad-CAM</h4>
        <GradCamCanvas rgba={cam?.heatmap_rgba} label="Saliency heatmap (RGBA, normalized input size)" />
      </section>

      <section className="card">
        <h3 className="section-title">Ethics & governance</h3>
        <p className="readable-p">
          <strong>Human review recommended:</strong> {eth.human_review_recommended ? "Yes" : "No"}
        </p>
        {eth.purpose && <p className="readable-p muted small">{eth.purpose}</p>}
      </section>

      {transparency.limitations?.length > 0 && (
        <section className="card card-muted">
          <h3 className="section-title">Limitations</h3>
          <ul className="readable-list small">
            {transparency.limitations.map((L) => (
              <li key={L}>{L}</li>
            ))}
          </ul>
        </section>
      )}

      <details className="card details-raw">
        <summary>Raw JSON (for audits / copy-paste)</summary>
        <pre className="pre-compact">{JSON.stringify(transparency, null, 2)}</pre>
      </details>
    </div>
  );
}
