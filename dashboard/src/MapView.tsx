import maplibregl, { type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

import { api } from "./api";
import { PRIORITY_COLOR, REPORT, ROUTE_COLOR, STATUS } from "./colors";
import { bbox, centre } from "./geo";
import type {
  FieldReport, Layer, LngLat, Priorities, RoadProps, Roads, RouteResult, Visible,
} from "./types";

export type Mode = "browse" | "start" | "end";

type Props = {
  layers: Layer[];
  visible: Visible;
  roads: Roads | null;
  priorities: Priorities | null;
  reports: FieldReport[];
  route: RouteResult | null;
  start: LngLat | null;
  end: LngLat | null;
  selectedEdge: string | null;
  focus: { bounds: [number, number, number, number]; n: number } | null;
  mode: Mode;
  onPick: (p: LngLat) => void;
  onRoadClick: (road: RoadProps, at: LngLat) => void;
};

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const RASTERS = ["probability", "depth", "confidence"] as const;
// rasters go above the light priority fill and below every outline and road
const FIRST_VECTOR = "priorities-line";

const statusColor: maplibregl.ExpressionSpecification = [
  "match", ["get", "status"],
  ...Object.entries(STATUS).flatMap(([k, v]) => [k, v.color]),
  "#888",
] as unknown as maplibregl.ExpressionSpecification;

const roadWidth = (base: number): maplibregl.ExpressionSpecification =>
  ["interpolate", ["linear"], ["zoom"], 10, base, 14, base * 2.2, 17, base * 4];

/** Zoom-scaled width; dry roads thinner (zoom must be the outermost expression). */
const statusWidth = (dry: number, other: number): maplibregl.ExpressionSpecification => {
  const at = (f: number): maplibregl.ExpressionSpecification =>
    ["case", ["==", ["get", "status"], "dry"], dry * f, other * f];
  return ["interpolate", ["linear"], ["zoom"], 10, at(1), 14, at(2.2), 17, at(4)];
};

/** OSM raster base map; works without any API key. */
const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#eef0f2" } },
    { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.6 } },
  ],
};

function addVectorLayers(map: MlMap): void {
  for (const id of ["priorities", "roads", "route", "reports", "selected"]) {
    map.addSource(id, { type: "geojson", data: EMPTY });
  }
  map.addLayer({
    id: "priorities-fill", type: "fill", source: "priorities",
    paint: {
      "fill-color": PRIORITY_COLOR,
      "fill-opacity": ["case", ["<=", ["get", "priority"], 3], 0.18, 0.05],
    },
  });
  map.addLayer({
    id: "priorities-line", type: "line", source: "priorities",
    paint: {
      "line-color": ["case", ["get", "cut_off"], "#1a1a1a", PRIORITY_COLOR],
      "line-width": ["case", ["get", "cut_off"], 3.5, ["<=", ["get", "priority"], 3], 2.5, 1],
    },
  });
  map.addLayer({
    id: "roads-solid", type: "line", source: "roads",
    filter: ["in", ["get", "status"], ["literal", ["dry", "flooded", "closed"]]],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-color": statusColor,
      "line-width": statusWidth(1.2, 2.6),
    },
  });
  map.addLayer({
    id: "roads-dashed", type: "line", source: "roads",
    filter: ["in", ["get", "status"], ["literal", ["verify", "unknown"]]],
    layout: { "line-join": "round" },
    paint: { "line-color": statusColor, "line-width": roadWidth(2.6), "line-dasharray": [2, 1.2] },
  });
  map.addLayer({
    id: "route-casing", type: "line", source: "route",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ffffff", "line-width": roadWidth(5.5) },
  });
  map.addLayer({
    id: "route-line", type: "line", source: "route",
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": ROUTE_COLOR, "line-width": roadWidth(3.5) },
  });
  map.addLayer({
    id: "route-verify", type: "line", source: "roads",
    filter: ["in", ["get", "edge_id"], ["literal", []]],
    paint: { "line-color": STATUS.verify.color, "line-width": roadWidth(4.5),
      "line-dasharray": [1, 0.8] },
  });
  map.addLayer({
    id: "selected-edge", type: "line", source: "selected",
    paint: { "line-color": "#00e5ff", "line-width": roadWidth(5), "line-opacity": 0.9 },
  });
  map.addLayer({
    id: "reports", type: "circle", source: "reports",
    paint: {
      "circle-radius": 8,
      "circle-color": [
        "match", ["get", "report"],
        ...Object.entries(REPORT).flatMap(([k, v]) => [k, v.color]),
        "#888",
      ] as unknown as maplibregl.ExpressionSpecification,
      "circle-stroke-color": "#fff",
      "circle-stroke-width": 2.5,
    },
  });
}

function setData(map: MlMap, id: string, data: GeoJSON.GeoJSON): void {
  (map.getSource(id) as GeoJSONSource | undefined)?.setData(data);
}

function setVisible(map: MlMap, ids: string[], on: boolean): void {
  for (const id of ids) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  }
}

function pin(color: string, text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "pin";
  el.style.background = color;
  el.textContent = text;
  return el;
}

export default function MapView(p: Props) {
  const box = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);
  const cb = useRef(p);
  cb.current = p;
  const labels = useRef<maplibregl.Marker[]>([]);
  const ends = useRef<maplibregl.Marker[]>([]);
  const fitted = useRef(false);

  useEffect(() => {
    const map = new maplibregl.Map({
      container: box.current!, style: STYLE, center: [87.2, 26.55], zoom: 10, maxZoom: 18,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    map.on("load", () => {
      addVectorLayers(map);
      setReady(true);
    });
    map.on("click", (e) => {
      const at: LngLat = [e.lngLat.lng, e.lngLat.lat];
      if (cb.current.mode !== "browse") {
        cb.current.onPick(at);
        return;
      }
      const pad = 6;
      const hits = map.queryRenderedFeatures(
        [[e.point.x - pad, e.point.y - pad], [e.point.x + pad, e.point.y + pad]],
        { layers: ["roads-solid", "roads-dashed"] },
      );
      if (hits.length) cb.current.onRoadClick(hits[0].properties as RoadProps, at);
    });
    for (const layer of ["roads-solid", "roads-dashed"]) {
      map.on("mouseenter", layer, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", layer, () => (map.getCanvas().style.cursor = ""));
    }
    mapRef.current = map;
    return () => map.remove();
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (map) map.getCanvas().style.cursor = p.mode === "browse" ? "" : "crosshair";
  }, [p.mode]);

  // flood rasters: one image source per layer of the selected scene
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    let cancelled = false;
    for (const kind of RASTERS) {
      if (map.getLayer(`raster-${kind}`)) map.removeLayer(`raster-${kind}`);
      if (map.getSource(`raster-${kind}`)) map.removeSource(`raster-${kind}`);
    }
    for (const layer of p.layers) {
      api.layerCorners(layer.id).then((coordinates) => {
        if (cancelled || map.getSource(`raster-${layer.kind}`)) return;
        map.addSource(`raster-${layer.kind}`, {
          type: "image", url: api.layerPngUrl(layer.id), coordinates,
        });
        map.addLayer({
          id: `raster-${layer.kind}`, type: "raster", source: `raster-${layer.kind}`,
          paint: { "raster-opacity": layer.kind === "confidence" ? 1 : 0.85,
            "raster-resampling": "nearest" },
          layout: { visibility: cb.current.visible[layer.kind] ? "visible" : "none" },
        }, FIRST_VECTOR);
        // keep the order probability < depth < confidence under the vector layers
        for (const k of RASTERS) {
          if (map.getLayer(`raster-${k}`)) map.moveLayer(`raster-${k}`, FIRST_VECTOR);
        }
      }).catch(() => undefined);
    }
    return () => {
      cancelled = true;
    };
  }, [ready, p.layers]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    for (const kind of RASTERS) setVisible(map, [`raster-${kind}`], p.visible[kind]);
    setVisible(map, ["roads-solid", "roads-dashed"], p.visible.roads);
    setVisible(map, ["priorities-fill", "priorities-line"], p.visible.priorities);
    setVisible(map, ["route-casing", "route-line", "route-verify"], p.visible.route);
    setVisible(map, ["reports"], p.visible.reports);
    labels.current.forEach((m) => (m.getElement().style.display = p.visible.priorities ? "" : "none"));
  }, [ready, p.visible]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    setData(map, "roads", p.roads ?? EMPTY);
    if (p.roads && p.roads.features.length && !fitted.current) {
      const b = bbox(p.roads.features.map((f) => f.geometry));
      if (b) map.fitBounds(b, { padding: 40, duration: 0 });
      fitted.current = true;
    }
  }, [ready, p.roads]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    setData(map, "priorities", p.priorities ?? EMPTY);
    labels.current.forEach((m) => m.remove());
    labels.current = (p.priorities?.features ?? []).slice(0, 10).flatMap((f) => {
      const c = centre(f.geometry);
      if (!c) return [];
      const el = pin(f.properties.cut_off ? "#1a1a1a" : PRIORITY_COLOR, String(f.properties.priority));
      el.title = `${f.properties.area_name}: priority ${f.properties.priority}`;
      return [new maplibregl.Marker({ element: el }).setLngLat(c).addTo(map)];
    });
  }, [ready, p.priorities]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    setData(map, "reports", {
      type: "FeatureCollection",
      features: p.reports.filter((r) => r.lon != null && r.lat != null).map((r) => ({
        type: "Feature", geometry: { type: "Point", coordinates: [r.lon!, r.lat!] },
        properties: { report: r.report, edge_id: r.edge_id },
      })),
    });
  }, [ready, p.reports]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    setData(map, "route", p.route ?? EMPTY);
    map.setFilter("route-verify",
      ["in", ["get", "edge_id"], ["literal", p.route?.verify_segments ?? []]]);
  }, [ready, p.route]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    ends.current.forEach((m) => m.remove());
    ends.current = [];
    if (p.start) ends.current.push(new maplibregl.Marker({ element: pin("#009E73", "A") })
      .setLngLat(p.start).addTo(map));
    if (p.end) ends.current.push(new maplibregl.Marker({ element: pin("#56B4E9", "B") })
      .setLngLat(p.end).addTo(map));
  }, [ready, p.start, p.end]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const f = p.roads?.features.filter((x) => x.properties.edge_id === p.selectedEdge) ?? [];
    setData(map, "selected", { type: "FeatureCollection", features: f });
  }, [ready, p.selectedEdge, p.roads]);

  useEffect(() => {
    const map = mapRef.current;
    if (map && ready && p.focus) map.fitBounds(p.focus.bounds, { padding: 80, maxZoom: 14 });
  }, [ready, p.focus]);

  return <div ref={box} className="map" aria-label="Flood map" />;
}
