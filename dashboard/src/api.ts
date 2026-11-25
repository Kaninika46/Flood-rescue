import type {
  Alert, FieldReport, Layer, LngLat, Priorities, ReportKind, Roads, RouteResult, Scene, Vehicle,
} from "./types";

/** Backend base URL: "/api" goes through the Vite proxy in development. */
export const API: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(API + path, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!r.ok) {
    let detail: unknown = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (await r.json()) as T;
}

const post = <T>(path: string, body: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body) });

const q = (sceneId: number | null) => (sceneId == null ? "" : `?scene_id=${sceneId}`);

export const api = {
  scenes: () => req<Scene[]>("/scenes"),
  processScene: (id: number) => post<{ job_id: string }>(`/scenes/${id}/process`, {}),
  layers: (sceneId: number | null) => req<Layer[]>(`/layers${q(sceneId)}`),
  layerCorners: (id: number) => req<[LngLat, LngLat, LngLat, LngLat]>(`/layers/${id}/corners`),
  layerPngUrl: (id: number) => `${API}/layers/${id}/png`,
  roads: (sceneId: number | null) => req<Roads>(`/roads/status${q(sceneId)}`),
  priorities: (sceneId: number | null) => req<Priorities>(`/priorities${q(sceneId)}`),
  route: (vehicle: Vehicle, start: LngLat, end: LngLat) =>
    post<RouteResult>("/routes", { vehicle, start, end }),
  alerts: () => req<Alert[]>("/alerts"),
  draftAlert: (priorityId: number) => post<Alert>("/alerts", { priority_id: priorityId }),
  approveAlert: (id: number, officer: string) =>
    post<Alert>(`/alerts/${id}/approve`, { officer }),
  sendAlert: (id: number) => post<Alert>(`/alerts/${id}/send`, {}),
  reports: () => req<FieldReport[]>("/field_reports"),
  addReport: (body: {
    edge_id: string; report: ReportKind; lon?: number; lat?: number; note?: string;
    reporter?: string;
  }) => post<{ id: number }>("/field_reports", body),
};
