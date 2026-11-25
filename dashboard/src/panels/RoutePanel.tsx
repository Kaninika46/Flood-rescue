import type { Mode } from "../MapView";
import type { LngLat, RouteResult, Vehicle } from "../types";

type Props = {
  vehicle: Vehicle;
  setVehicle: (v: Vehicle) => void;
  mode: Mode;
  setMode: (m: Mode) => void;
  start: LngLat | null;
  end: LngLat | null;
  route: RouteResult | null;
  error: string | null;
  busy: boolean;
  onClear: () => void;
  onSelectEdge: (edgeId: string) => void;
};

const VEHICLES: { id: Vehicle; label: string }[] = [
  { id: "car", label: "Car (0.15 m)" },
  { id: "truck", label: "Truck (0.30 m)" },
  { id: "foot", label: "On foot (0.15 m)" },
  { id: "boat", label: "Boat (flooded only)" },
];

const fmt = (p: LngLat | null) => (p ? `${p[1].toFixed(4)} N, ${p[0].toFixed(4)} E` : "not set");

export default function RoutePanel(p: Props) {
  return (
    <section className="panel">
      <h2>Safe route</h2>
      <div className="row">
        <label htmlFor="vehicle">Vehicle</label>
        <select id="vehicle" value={p.vehicle} onChange={(e) => p.setVehicle(e.target.value as Vehicle)}>
          {VEHICLES.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
        </select>
      </div>
      <div className="row">
        <button className={p.mode === "start" ? "active" : ""}
          onClick={() => p.setMode(p.mode === "start" ? "browse" : "start")}>
          {p.mode === "start" ? "Click the map…" : "Set start (A)"}
        </button>
        <span className="muted">{fmt(p.start)}</span>
      </div>
      <div className="row">
        <button className={p.mode === "end" ? "active" : ""}
          onClick={() => p.setMode(p.mode === "end" ? "browse" : "end")}>
          {p.mode === "end" ? "Click the map…" : "Set end (B)"}
        </button>
        <span className="muted">{fmt(p.end)}</span>
      </div>
      {p.busy && <p className="muted">Computing route…</p>}
      {p.error && <p className="error">{p.error}</p>}
      {p.route && (
        <div className="route-result">
          <p className="big">{p.route.total_time_min.toFixed(1)} min</p>
          {p.route.verify_segments.length > 0 ? (
            <>
              <p className="warn">
                {p.route.verify_segments.length} segment(s) to verify on the ground:
              </p>
              <ul className="chips">
                {p.route.verify_segments.map((id) => (
                  <li key={id}><button className="chip" onClick={() => p.onSelectEdge(id)}>{id}</button></li>
                ))}
              </ul>
            </>
          ) : <p>No low-confidence segments on this route.</p>}
        </div>
      )}
      {(p.route || p.error) && <p className="advice">{p.route?.advice ?? "Route is advice; the team on the ground decides."}</p>}
      {(p.start || p.end) && <button className="link" onClick={p.onClear}>Clear route</button>}
    </section>
  );
}
