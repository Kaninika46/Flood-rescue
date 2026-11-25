import type { Geometry } from "geojson";

/** [west, south, east, north] of any GeoJSON geometry (null when empty). */
export function bbox(geoms: (Geometry | null)[]): [number, number, number, number] | null {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  const visit = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number") {
      const [x, y] = c as number[];
      w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y);
    } else if (Array.isArray(c)) c.forEach(visit);
  };
  for (const g of geoms) {
    if (!g) continue;
    if (g.type === "GeometryCollection") g.geometries.forEach((x) => visit(coords(x)));
    else visit(coords(g));
  }
  return Number.isFinite(w) ? [w, s, e, n] : null;
}

function coords(g: Geometry): unknown {
  return g.type === "GeometryCollection" ? [] : g.coordinates;
}

/** Centre of a geometry's bounding box. */
export function centre(g: Geometry | null): [number, number] | null {
  const b = bbox([g]);
  return b ? [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] : null;
}

export const fmtInt = (x: number) => Math.round(x).toLocaleString("en-GB");
export const fmtTime = (iso: string | null) =>
  iso ? new Date(iso).toLocaleString("en-GB", { dateStyle: "short", timeStyle: "short" }) : "";
