# dashboard/

Control-room dashboard (prompt D3): React + TypeScript + MapLibre GL, built with Vite.

What it shows:

| Map layer | How |
|---|---|
| Flood probability | blue, opacity rises with the probability (PNG rendered by the backend, `GET /layers/{id}/png`) |
| Water depth | light to dark blue, 0 to 3 m (off by default) |
| Confidence | orange hatching where the ensemble std is above `status.max_std` in `configs/routing.yaml`: "verify" |
| Roads | coloured by status: flooded (blue), closed (black), verify (orange, dashed), unknown (purple, dashed), dry (grey) |
| Route | pink; segments to verify are drawn dashed orange on top and listed in the Route tab |
| Priority areas | outlined, numbered by rank; black outline = cut off from every facility |
| Field reports | dots coloured by report type |

Side panel tabs: **Priorities** (zoom to an area, "Draft alert"), **Route** (vehicle, click start and
end on the map; every answer shows the advice text), **Alerts** (CAP draft, officer name +
**Approve**, then "Send to registered test phones"), **Reports** (click a road on the map, then
road open / road closed / bridge gone / flooded; the route is recomputed at once). The page polls
the backend every 15 s, so reports from the rescuer app appear on their own.

## Run

Start the backend first (see `backend/README.md`), then:

```bash
cd dashboard && npm install
```
```bash
npm run dev
```

Open http://localhost:5173. In development `/api/...` is forwarded to http://localhost:8000
(set `API_TARGET` to change it).

**For demos, serve the production build, not the dev server** (the dev server is for editing
code only):

```bash
npm run demo
```

builds into `dist/` and serves it on http://localhost:4173 (localhost only), with the same `/api`
forwarding. For a static build served elsewhere:

```bash
VITE_API_URL=http://localhost:8000 npm run build
```

and add the page's origin to the backend's `CORS_ORIGINS`.

## Try it without the real data

`scripts/demo_seed.py` writes a **synthetic** event (made-up river, street grid, wards and
population) into a new, empty `DATA_ROOT` and stores it as the scene "SYNTHETIC demo (not real
data)". It runs the real road-status, priority and alert code on it:

```bash
export DATA_ROOT=~/flood-demo-data
```
```bash
python scripts/demo_seed.py
```

The map tiles come from openstreetmap.org (needs internet; light interactive use only, as the
OSM tile policy asks). Check before a demo: `npm run build` must pass with no type errors.
