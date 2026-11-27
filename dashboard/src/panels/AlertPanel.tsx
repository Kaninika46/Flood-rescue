import { useState } from "react";

import { fmtTime } from "../geo";
import type { Alert } from "../types";

type Props = {
  alerts: Alert[];
  onApprove: (id: number, officer: string) => Promise<void>;
  onSend: (id: number) => Promise<void>;
};

function AlertCard({ a, onApprove, onSend }: { a: Alert } & Omit<Props, "alerts">) {
  const [officer, setOfficer] = useState("");
  const [busy, setBusy] = useState(false);
  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };
  return (
    <li className={`alert ${a.state}`}>
      <div className="alert-head">
        <span className={`state ${a.state}`}>{a.state.toUpperCase()}</span>
        <code>{a.identifier}</code>
      </div>
      <p className="sms">{a.sms}</p>
      <details>
        <summary>CAP 1.2 XML (status Exercise)</summary>
        <pre>{a.cap_xml}</pre>
      </details>
      {a.state === "draft" && (
        <form className="row" onSubmit={(e) => {
          e.preventDefault();
          if (officer.trim().length >= 2) void run(() => onApprove(a.id, officer.trim()));
        }}>
          <input aria-label="Officer name" placeholder="Officer name" value={officer} maxLength={200}
            onChange={(e) => setOfficer(e.target.value)} />
          <button type="submit" className="primary" disabled={busy || officer.trim().length < 2}>
            Approve
          </button>
        </form>
      )}
      {a.approved_by && (
        <p className="muted">
          Approved by <b>{a.approved_by}</b>
          {a.approved_role ? ` (${a.approved_role} token)` : ""} at {fmtTime(a.approved_at)}
        </p>
      )}
      {a.state === "approved" && (
        <button disabled={busy} onClick={() => void run(() => onSend(a.id))}>
          Send to registered test phones
        </button>
      )}
      {a.state === "sent" && (
        <p className="muted">Sent to {a.sent_count} test phone(s) at {fmtTime(a.sent_at)}</p>
      )}
      {a.state === "cancelled" && (
        <p className="muted">
          Cancelled at {fmtTime(a.cancelled_at ?? null)}: {a.cancel_reason ?? "no reason recorded"}
        </p>
      )}
    </li>
  );
}

export default function AlertPanel({ alerts, onApprove, onSend }: Props) {
  return (
    <section className="panel">
      <h2>Alert drafts</h2>
      <p className="note">
        Drafts need an officer's approval. Sending goes only to the test phones in
        <code> config/test_recipients.yaml</code>; every other number is refused.
      </p>
      {alerts.length === 0 && <p className="muted">No alerts yet. Draft one from a priority area.</p>}
      <ul className="alerts">
        {alerts.map((a) => <AlertCard key={a.id} a={a} onApprove={onApprove} onSend={onSend} />)}
      </ul>
    </section>
  );
}
