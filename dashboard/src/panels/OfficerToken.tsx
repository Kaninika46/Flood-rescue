import { useState } from "react";

import { api, setOfficerToken } from "../api";

type Props = { unlocked: boolean; onChange: (unlocked: boolean) => void };

/**
 * Asks for the officer token once per tab. The token is checked with GET /whoami and kept only in
 * memory (src/api.ts); reloading the page or a 401 from the server asks again.
 */
export default function OfficerToken({ unlocked, onChange }: Props) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  if (unlocked) {
    return (
      <span className="token ok">
        officer ✓{" "}
        <button className="link" onClick={() => { setOfficerToken(null); onChange(false); }}>
          lock
        </button>
      </span>
    );
  }

  const submit = async () => {
    setBusy(true);
    setProblem(null);
    try {
      const role = await api.whoami(value.trim());
      if (role !== "officer") {
        setProblem("that is a team token; approving needs the officer token");
        return;
      }
      setOfficerToken(value);
      setValue("");
      onChange(true);
    } catch {
      setProblem("token not accepted");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="token" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
      <input type="password" aria-label="Officer token" placeholder="Officer token" value={value}
        autoComplete="off" onChange={(e) => setValue(e.target.value)} />
      <button type="submit" disabled={busy || !value.trim()}>Unlock</button>
      {problem && <span className="token-problem">{problem}</span>}
    </form>
  );
}
