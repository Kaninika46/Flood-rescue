import type { ReportKind, RoadStatus } from "./types";

/** Road status colours (Okabe-Ito based; "verify" and "unknown" are also dashed on the map). */
export const STATUS: Record<RoadStatus, { color: string; label: string; dashed: boolean }> = {
  flooded: { color: "#0072B2", label: "Flooded", dashed: false },
  closed: { color: "#1a1a1a", label: "Closed", dashed: false },
  verify: { color: "#E69F00", label: "Verify (low confidence)", dashed: true },
  unknown: { color: "#8e44ad", label: "Unknown (no data)", dashed: true },
  dry: { color: "#8a9099", label: "Dry", dashed: false },
};

export const REPORT: Record<ReportKind, { color: string; label: string }> = {
  road_open: { color: "#009E73", label: "Road open" },
  road_closed: { color: "#1a1a1a", label: "Road closed" },
  bridge_gone: { color: "#D55E00", label: "Bridge gone" },
  flooded: { color: "#0072B2", label: "Flooded" },
};

export const ROUTE_COLOR = "#CC79A7";
export const PRIORITY_COLOR = "#D55E00";
