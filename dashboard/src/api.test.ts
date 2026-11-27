import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, ApiError, hasOfficerToken, setOfficerToken } from "./api";

type Call = { url: string; init: RequestInit };

let calls: Call[];
let status: number;
const storage = { setItem: vi.fn(), getItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn() };

const authOf = (c: Call) => (c.init.headers as Record<string, string>).authorization;

beforeEach(() => {
  calls = [];
  status = 200;
  setOfficerToken(null);
  vi.stubGlobal("localStorage", storage);
  vi.stubGlobal("sessionStorage", storage);
  vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    const body = status === 401 ? { detail: "officer token required" } : { role: "officer", id: 1 };
    return new Response(JSON.stringify(body), { status });
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("officer token", () => {
  it("is sent as a bearer header on officer actions only", async () => {
    setOfficerToken("  secret  ");
    expect(hasOfficerToken()).toBe(true);
    await api.approveAlert(3, "Officer Sharma");
    await api.draftAlert(1);
    await api.sendAlert(3);
    await api.processScene(2);
    await api.addReport({ edge_id: "A_B_0", report: "road_closed" });
    await api.alerts();
    await api.route("car", [0, 0], [1, 1]);
    expect(calls.slice(0, 5).map(authOf)).toEqual(Array(5).fill("Bearer secret"));
    expect(calls.slice(5).map(authOf)).toEqual([undefined, undefined]);
  });

  it("is kept in memory only, never in browser storage", async () => {
    setOfficerToken("secret");
    await api.approveAlert(3, "Officer Sharma");
    expect(storage.setItem).not.toHaveBeenCalled();
  });

  it("is forgotten after a 401, so the dashboard asks again", async () => {
    setOfficerToken("old");
    status = 401;
    await expect(api.approveAlert(3, "Officer Sharma")).rejects.toMatchObject({ status: 401 });
    expect(hasOfficerToken()).toBe(false);
  });

  it("is not forgotten after other errors", async () => {
    setOfficerToken("secret");
    status = 409;
    await expect(api.sendAlert(3)).rejects.toBeInstanceOf(ApiError);
    expect(hasOfficerToken()).toBe(true);
  });

  it("whoami checks a token without keeping it", async () => {
    expect(await api.whoami("candidate")).toBe("officer");
    expect(authOf(calls[0])).toBe("Bearer candidate");
    expect(calls[0].url).toBe("/api/whoami");
    expect(hasOfficerToken()).toBe(false);
  });

  it("an empty token counts as none", () => {
    setOfficerToken("   ");
    expect(hasOfficerToken()).toBe(false);
  });
});
