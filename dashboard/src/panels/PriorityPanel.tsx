import { fmtInt } from "../geo";
import type { Priorities, PriorityFeature } from "../types";

type Props = {
  priorities: Priorities | null;
  onFocus: (f: PriorityFeature) => void;
  onDraft: (priorityId: number) => void;
  drafting: number | null;
};

export default function PriorityPanel({ priorities, onFocus, onDraft, drafting }: Props) {
  const rows = priorities?.features ?? [];
  return (
    <section className="panel">
      <h2>Priority areas</h2>
      {rows.length === 0 && <p className="muted">No ranked areas for this scene.</p>}
      <ol className="priorities">
        {rows.map((f) => {
          const a = f.properties;
          return (
            <li key={a.id} className={a.cut_off ? "cut" : ""}>
              <button className="rank" onClick={() => onFocus(f)} title="Zoom to area">{a.priority}</button>
              <div className="body">
                <button className="name" onClick={() => onFocus(f)}>{a.area_name}</button>
                <div className="facts">
                  <span><b>{fmtInt(a.exposed_people)}</b> people exposed</span>
                  <span>{a.flooded_km2.toFixed(1)} km² flooded</span>
                  {a.cut_off && <span className="tag black">cut off</span>}
                  {a.verify && <span className="tag orange">verify</span>}
                </div>
              </div>
              <button onClick={() => onDraft(a.id)} disabled={drafting === a.id}>
                {drafting === a.id ? "…" : "Draft alert"}
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
