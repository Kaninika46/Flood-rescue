import { useState } from "react";

import { REPORT, STATUS } from "../colors";
import { fmtTime } from "../geo";
import type { FieldReport, LngLat, ReportKind, RoadProps } from "../types";

type Props = {
  selected: { road: RoadProps; at: LngLat } | null;
  reports: FieldReport[];
  onReport: (kind: ReportKind, reporter: string, note: string) => Promise<void>;
  onClose: () => void;
};

export default function ReportPanel({ selected, reports, onReport, onClose }: Props) {
  const [reporter, setReporter] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const road = selected?.road;
  return (
    <section className="panel">
      <h2>Field reports</h2>
      {!road && <p className="muted">Click a road on the map to enter a report phoned in by a team.</p>}
      {road && (
        <div className="selected-road">
          <div className="row spread">
            <b>Road {road.edge_id}</b>
            <button className="link" onClick={onClose}>close</button>
          </div>
          <p>
            Status <span className="tag" style={{ background: STATUS[road.status]?.color }}>
              {road.status}</span>
            {road.max_depth_m != null && <> · depth {road.max_depth_m.toFixed(2)} m</>}
            {road.confidence != null && <> · confidence {road.confidence.toFixed(2)}</>}
            {road.bridge && <> · bridge</>} · source {road.source}
          </p>
          <input placeholder="Reported by (team / person)" value={reporter} maxLength={100}
            onChange={(e) => setReporter(e.target.value)} />
          <input placeholder="Note (optional)" value={note} maxLength={200} onChange={(e) => setNote(e.target.value)} />
          <div className="report-buttons">
            {(Object.keys(REPORT) as ReportKind[]).map((k) => (
              <button key={k} disabled={busy} style={{ borderColor: REPORT[k].color }}
                onClick={async () => {
                  setBusy(true);
                  try {
                    await onReport(k, reporter, note);
                    setNote("");
                  } finally {
                    setBusy(false);
                  }
                }}>
                {REPORT[k].label}
              </button>
            ))}
          </div>
        </div>
      )}
      <ul className="reports">
        {reports.slice(0, 12).map((r) => (
          <li key={r.id}>
            <span className="dot" style={{ background: REPORT[r.report]?.color }} />
            <b>{REPORT[r.report]?.label ?? r.report}</b> <code>{r.edge_id}</code>
            <span className="muted"> {fmtTime(r.created_at)}{r.reporter ? ` · ${r.reporter}` : ""}</span>
            {r.note && <div className="muted">{r.note}</div>}
          </li>
        ))}
      </ul>
    </section>
  );
}
