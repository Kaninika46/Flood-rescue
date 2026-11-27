import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError } from "./api";
import { bbox } from "./geo";
import MapView, { type Mode } from "./MapView";
import AlertPanel from "./panels/AlertPanel";
import LayerPanel from "./panels/LayerPanel";
import PriorityPanel from "./panels/PriorityPanel";
import ReportPanel from "./panels/ReportPanel";
import RoutePanel from "./panels/RoutePanel";
import type {
  Alert, FieldReport, Layer, LngLat, Priorities, PriorityFeature, ReportKind, RoadProps, Roads,
  RouteResult, Scene, Vehicle, Visible,
} from "./types";

type Tab = "priorities" | "route" | "alerts" | "reports";
const POLL_MS = 15_000;
const ADVICE = "Route is advice; the team on the ground decides.";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState<number | null>(null);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [roads, setRoads] = useState<Roads | null>(null);
  const [priorities, setPriorities] = useState<Priorities | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [reports, setReports] = useState<FieldReport[]>([]);
  const [visible, setVisible] = useState<Visible>({
    probability: true, depth: false, confidence: true, roads: true, priorities: true,
    route: true, reports: true,
  });
  const [tab, setTab] = useState<Tab>("priorities");
  const [vehicle, setVehicle] = useState<Vehicle>("car");
  const [mode, setMode] = useState<Mode>("browse");
  const [start, setStart] = useState<LngLat | null>(null);
  const [end, setEnd] = useState<LngLat | null>(null);
  const [route, setRoute] = useState<RouteResult | null>(null);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [routeBusy, setRouteBusy] = useState(false);
  const [selected, setSelected] = useState<{ road: RoadProps; at: LngLat } | null>(null);
  const [focus, setFocus] = useState<{ bounds: [number, number, number, number]; n: number } | null>(null);
  const [drafting, setDrafting] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const reportCount = useRef(0);

  const [showSuperseded, setShowSuperseded] = useState(false);

  const scene = scenes.find((s) => s.id === sceneId) ?? null;
  // superseded scenes stay in the database but are hidden unless asked for; the selected scene is
  // always listed so the dropdown never disagrees with what the map shows
  const listedScenes = showSuperseded
    ? scenes
    : scenes.filter((s) => s.state !== "superseded" || s.id === sceneId);
  const hiddenCount = scenes.length - listedScenes.length;

  const toggleSuperseded = (show: boolean) => {
    setShowSuperseded(show);
    if (!show && scene?.state === "superseded") {
      const current = scenes.filter((s) => s.state !== "superseded");
      setSceneId((current.find((s) => s.state === "done") ?? current[0])?.id ?? null);
    }
  };

  const loadScene = useCallback(async (id: number | null) => {
    const [l, r, p] = await Promise.all([api.layers(id), api.roads(id), api.priorities(id)]);
    setLayers(l);
    setRoads(r);
    setPriorities(p);
  }, []);

  // first load and polling (field reports from the rescuer app, new scenes, alert states)
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const [s, a, f] = await Promise.all([api.scenes(), api.alerts(), api.reports()]);
        if (stop) return;
        setOffline(false);
        setScenes(s);
        setAlerts(a);
        setReports(f);
        setSceneId((cur) => cur ?? s.find((x) => x.state === "done")?.id ?? null);
        if (f.length !== reportCount.current && reportCount.current !== 0) {
          setRoads(await api.roads(null));
        }
        reportCount.current = f.length;
      } catch {
        if (!stop) setOffline(true);
      }
    };
    void tick();
    const t = setInterval(() => void tick(), POLL_MS);
    return () => {
      stop = true;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (sceneId != null) loadScene(sceneId).catch((e) => setError(message(e)));
  }, [sceneId, loadScene]);

  const computeRoute = useCallback(async (v: Vehicle, a: LngLat | null, b: LngLat | null) => {
    if (!a || !b) return;
    setRouteBusy(true);
    setRouteError(null);
    try {
      setRoute(await api.route(v, a, b));
    } catch (e) {
      setRoute(null);
      setRouteError(e instanceof ApiError && e.status === 404
        ? `No safe route for this vehicle. ${ADVICE}` : message(e));
    } finally {
      setRouteBusy(false);
    }
  }, []);

  useEffect(() => {
    void computeRoute(vehicle, start, end);
  }, [vehicle, start, end, computeRoute]);

  const onPick = (p: LngLat) => {
    if (mode === "start") {
      setStart(p);
      setMode(end ? "browse" : "end");
    } else {
      setEnd(p);
      setMode("browse");
    }
    setTab("route");
  };

  const onReport = async (kind: ReportKind, reporter: string, note: string) => {
    if (!selected) return;
    try {
      await api.addReport({
        edge_id: selected.road.edge_id, report: kind, lon: selected.at[0], lat: selected.at[1],
        reporter: reporter ? `control room: ${reporter}` : "control room", note: note || undefined,
      });
      const [r, f] = await Promise.all([api.roads(sceneId), api.reports()]);
      setRoads(r);
      setReports(f);
      reportCount.current = f.length;
      const road = r.features.find((x) => x.properties.edge_id === selected.road.edge_id);
      if (road) setSelected({ road: road.properties, at: selected.at });
      await computeRoute(vehicle, start, end); // a report changes the route
    } catch (e) {
      setError(message(e));
    }
  };

  const onDraft = async (priorityId: number) => {
    setDrafting(priorityId);
    try {
      const a = await api.draftAlert(priorityId);
      setAlerts((cur) => [a, ...cur]);
      setTab("alerts");
    } catch (e) {
      setError(message(e));
    } finally {
      setDrafting(null);
    }
  };

  const replaceAlert = (a: Alert) => setAlerts((cur) => cur.map((x) => (x.id === a.id ? a : x)));
  const onApprove = async (id: number, officer: string) => {
    try {
      replaceAlert(await api.approveAlert(id, officer));
    } catch (e) {
      setError(message(e));
    }
  };
  const onSend = async (id: number) => {
    try {
      replaceAlert(await api.sendAlert(id));
    } catch (e) {
      setError(message(e));
    }
  };

  const onFocus = (f: PriorityFeature) => {
    const b = bbox([f.geometry]);
    if (b) setFocus((cur) => ({ bounds: b, n: (cur?.n ?? 0) + 1 }));
  };

  const selectEdge = (edgeId: string) => {
    const road = roads?.features.find((x) => x.properties.edge_id === edgeId);
    if (!road) return;
    const c = road.geometry.coordinates[Math.floor(road.geometry.coordinates.length / 2)];
    setSelected({ road: road.properties, at: [c[0], c[1]] });
    const b = bbox([road.geometry]);
    if (b) setFocus((cur) => ({ bounds: b, n: (cur?.n ?? 0) + 1 }));
    setTab("reports");
  };

  const total = scene?.minutes ? Object.values(scene.minutes).reduce((a, b) => a + b, 0) : null;
  const drafts = alerts.filter((a) => a.state === "draft").length;

  return (
    <div className="app">
      <header>
        <h1>Flood Rescue AI <span>control room</span></h1>
        <div className="scene">
          <label htmlFor="scene">Scene</label>
          <select id="scene" value={sceneId ?? ""} onChange={(e) => setSceneId(Number(e.target.value))}>
            {listedScenes.length === 0 && (
              <option value="">{scenes.length === 0 ? "no scenes" : "no current scenes"}</option>
            )}
            {listedScenes.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.state})</option>)}
          </select>
          <label className="small" title="Superseded scenes are kept in the database">
            <input type="checkbox" checked={showSuperseded}
              onChange={(e) => toggleSuperseded(e.target.checked)} />
            show superseded{!showSuperseded && hiddenCount > 0 ? ` (${hiddenCount})` : ""}
          </label>
          {scene && (scene.state === "new" || scene.state === "failed") && (
            <button onClick={() => api.processScene(scene.id).catch((e) => setError(message(e)))}>
              Process
            </button>
          )}
          {total != null && <span className="muted">image to alert draft: {total.toFixed(1)} min</span>}
        </div>
        <div className="exercise" role="note">
          EXERCISE · student project · routes are advice · alerts go to test phones only
        </div>
      </header>
      {offline && <div className="banner error">Backend not reachable. Retrying every 15 s.</div>}
      {error && (
        <div className="banner error" role="alert">
          {error} <button className="link" onClick={() => setError(null)}>dismiss</button>
        </div>
      )}
      <main>
        <div className="map-wrap">
          <MapView layers={layers} visible={visible} roads={roads} priorities={priorities}
            reports={reports} route={route} start={start} end={end}
            selectedEdge={selected?.road.edge_id ?? null} focus={focus} mode={mode}
            onPick={onPick}
            onRoadClick={(road, at) => {
              setSelected({ road, at });
              setTab("reports");
            }} />
          <div className="map-legend">
            <LayerPanel visible={visible} setVisible={setVisible} layers={layers} />
          </div>
          {mode !== "browse" && (
            <div className="map-hint">Click the map to set the {mode === "start" ? "start (A)" : "end (B)"}</div>
          )}
        </div>
        <aside>
          <nav className="tabs" role="tablist">
            {([
              ["priorities", "Priorities"],
              ["route", "Route"],
              ["alerts", `Alerts${drafts ? ` (${drafts})` : ""}`],
              ["reports", `Reports (${reports.length})`],
            ] as [Tab, string][]).map(([id, label]) => (
              <button key={id} role="tab" aria-selected={tab === id}
                className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>
            ))}
          </nav>
          <div className="tab-body">
            {tab === "priorities" && (
              <PriorityPanel priorities={priorities} onFocus={onFocus} onDraft={onDraft}
                drafting={drafting} />
            )}
            {tab === "route" && (
              <RoutePanel vehicle={vehicle} setVehicle={setVehicle} mode={mode} setMode={setMode}
                start={start} end={end} route={route} error={routeError} busy={routeBusy}
                onClear={() => {
                  setStart(null); setEnd(null); setRoute(null); setRouteError(null);
                }}
                onSelectEdge={selectEdge} />
            )}
            {tab === "alerts" && <AlertPanel alerts={alerts} onApprove={onApprove} onSend={onSend} />}
            {tab === "reports" && (
              <ReportPanel selected={selected} reports={reports} onReport={onReport}
                onClose={() => setSelected(null)} />
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
