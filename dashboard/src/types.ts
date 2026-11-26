import type { Feature, FeatureCollection, LineString } from "geojson";

export type Scene = {
  id: number;
  name: string;
  state: string;
  minutes: Record<string, number> | null;
  created_at: string;
};

export type LayerKind = "probability" | "confidence" | "depth";

export type Layer = { id: number; scene_id: number; kind: LayerKind; png: string };

export type RoadStatus = "dry" | "flooded" | "verify" | "unknown" | "closed";

export type RoadProps = {
  edge_id: string;
  status: RoadStatus;
  confidence: number | null;
  max_depth_m: number | null;
  source: string;
  bridge: boolean;
};

export type PriorityProps = {
  id: number;
  priority: number;
  area_name: string;
  exposed_people: number;
  flooded_km2: number;
  cut_off: boolean;
  verify: boolean;
  mean_confidence: number | null;
};

export type Alert = {
  id: number;
  identifier: string;
  state: "draft" | "approved" | "sent" | "cancelled";
  sms: string;
  approved_by: string | null;
  approved_at: string | null;
  sent_at: string | null;
  sent_count: number | null;
  priority_id: number | null;
  cap_xml: string;
  cancel_reason?: string | null;
  cancelled_at?: string | null;
};

export type ReportKind = "road_open" | "road_closed" | "bridge_gone" | "flooded";

export type FieldReport = {
  id: number;
  edge_id: string;
  report: ReportKind;
  note: string | null;
  reporter: string | null;
  created_at: string;
  lon: number | null;
  lat: number | null;
};

export type RouteResult = FeatureCollection<LineString> & {
  route_id: number;
  advice: string;
  verify_segments: string[];
  total_time_min: number;
};

export type Vehicle = "foot" | "car" | "truck" | "boat";

export type LngLat = [number, number];

export type Roads = FeatureCollection<LineString, RoadProps>;
export type Priorities = FeatureCollection<GeoJSON.Geometry, PriorityProps>;
export type PriorityFeature = Feature<GeoJSON.Geometry, PriorityProps>;

export type Visible = Record<"probability" | "confidence" | "depth" | "roads" | "priorities" |
  "route" | "reports", boolean>;
