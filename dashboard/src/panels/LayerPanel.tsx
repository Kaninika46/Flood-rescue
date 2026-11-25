import { REPORT, ROUTE_COLOR, STATUS } from "../colors";
import type { Layer, Visible } from "../types";

type Props = { visible: Visible; setVisible: (v: Visible) => void; layers: Layer[] };

const RASTER_LABEL = {
  probability: "Flood probability",
  depth: "Water depth (0 to 3 m)",
  confidence: "Low confidence: verify (hatched)",
} as const;

export default function LayerPanel({ visible, setVisible, layers }: Props) {
  const has = (k: string) => layers.some((l) => l.kind === k);
  const toggle = (k: keyof Visible) => setVisible({ ...visible, [k]: !visible[k] });
  const row = (k: keyof Visible, label: string, swatch: JSX.Element, disabled = false) => (
    <label className={`layer-row${disabled ? " disabled" : ""}`} key={k}>
      <input type="checkbox" checked={visible[k] && !disabled} disabled={disabled}
        onChange={() => toggle(k)} />
      {swatch}
      <span>{label}</span>
    </label>
  );
  return (
    <section className="panel">
      <h2>Layers</h2>
      {(["probability", "depth", "confidence"] as const).map((k) =>
        row(k, RASTER_LABEL[k], <span className={`swatch raster-${k}`} />, !has(k)))}
      {row("roads", "Roads by status", <span className="swatch" />)}
      <ul className="legend">
        {Object.entries(STATUS).map(([k, s]) => (
          <li key={k}>
            <span className={`line${s.dashed ? " dashed" : ""}`} style={{ color: s.color }} />
            {s.label}
          </li>
        ))}
      </ul>
      {row("route", "Route (verify parts dashed orange)",
        <span className="line" style={{ color: ROUTE_COLOR }} />)}
      {row("priorities", "Priority areas (black outline: cut off)",
        <span className="swatch priority" />)}
      {row("reports", "Field reports", <span className="dot" style={{ background: REPORT.road_closed.color }} />)}
    </section>
  );
}
