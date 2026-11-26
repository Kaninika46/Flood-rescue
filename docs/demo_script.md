# Demo script: 10-minute Koshi replay (prompt D6)

The story: a Sentinel-1 scene of 27 September 2024 arrives → flood map → depth → safe routes →
priority list → alert approved by an officer → a field report from a rescue team changes a route.
Everything runs on one laptop; the rescuer app runs on your own Android phone.

Say at the start and keep visible: **this is an exercise.** Routes are advice, alerts are CAP
"Exercise" messages sent only to your own registered test phone, nothing reaches the public.

## Before the day (once)

| What | Command / action | Check |
|---|---|---|
| Event processed | `python scripts/run_event.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif` (add `--maxar-tiles ...` if the Level 2 evaluation was run) | prints minutes per step; the dashboard header shows "image to alert draft: N min" |
| Timing table | `python scripts/time_pipeline.py --scene $DATA_ROOT/nepal/s1/S1_20240927.tif` | `docs/system_metrics.md` |
| Results tables | `python scripts/make_report_tables.py` | `docs/results_tables.md` |
| Offline map | `python scripts/make_mbtiles.py --scene-prob <the scene's flood_prob.tif>`, then in the app: Settings → Download offline map | app map works in airplane mode |
| Test phone | `config/test_recipients.yaml` with **your own** number (git-ignored); `TWILIO_*` variables only if you send real SMS, otherwise the console gateway prints the message | `python scripts/alert_demo.py` sends to your phone only |
| Rehearse | run the whole script once with a stopwatch | under 10 minutes |

## On the day (5 minutes before)

```bash
docker compose -f backend/docker-compose.yml up -d db redis
```
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
```bash
rq worker --url redis://localhost:6379/0 flood
```
```bash
cd dashboard && npm run dev
```

Open http://localhost:5173 on the projector (browser zoom 110 to 125 %). Phone on the same
Wi-Fi, app open, server address set to the laptop. Pick the Koshi scene in the header.

## The 10 minutes

| Time | Show | Say |
|---|---|---|
| 0:00 | Dashboard, whole Koshi area | "27 September 2024: the Koshi barrage released record flows; UNOSAT estimated about 17,000 people exposed that day (Analysis: UNOSAT). This is a replay of that day with open data." |
| 0:45 | Header: scene name and "image to alert draft: N min" | "A new Sentinel-1 radar scene arrives (radar sees through clouds). One command runs everything; each step is timed." Name the slowest step. |
| 1:30 | **Flood probability** layer on, confidence on | "Level 1: an ensemble of U-Nets trained on Sen1Floods11 plus HAND terrain. Blue is water. Orange hatching is where the ensemble members disagree: we never hide that, we mark it **verify**." Quote the Level 1 test IoU from `docs/results_tables.md`. |
| 3:00 | Probability off, **water depth** on | "Depth from the flood edge and the elevation model (a NumPy port of FwDET). Darker is deeper; it decides which vehicle can pass." |
| 4:00 | Route tab → car, click start (Biratnagar) and end (Inaruwa) | "Every road is flooded, dry, verify or closed. A car may cross at most 15 cm; bridges in the flood zone stay closed until a team confirms them." Point at the advice box: "**Route is advice; the team on the ground decides.**" Point at the orange dashed parts: "segments to verify". |
| 5:00 | Switch vehicle to truck, then foot | "Same request, different limits: the truck (30 cm) gets a shorter route." |
| 5:45 | Priorities tab; click area 1 | "Areas ranked by people in flood water (WorldPop) and whether every road to a hospital or town is cut. Black outline = cut off." |
| 6:45 | "Draft alert" on area 1 → Alerts tab → open the CAP XML | "A CAP 1.2 draft: status Exercise, scope Restricted, Nepal time. It cannot be sent as a draft." |
| 7:15 | Type the officer name → **Approve** → "Send to registered test phones" | "An officer approves; the record keeps their name and time. The gateway sends only to numbers in our test list and refuses every other number." Show your phone receiving it (or the console output). |
| 8:00 | Phone: Priorities → Route to area 1 → Map; press **Report road → Road closed** on a road of the route | "The rescuer app works offline: the map, the route and the priority list are on the phone; reports queue until there is signal." |
| 8:45 | Dashboard (refreshes within 15 s): report dot and black road; Route tab shows the new route and time | "The control room sees the report, the road is closed for everyone, and the next route avoids it." |
| 9:15 | `docs/results_tables.md` (or a slide) | Level 1 table (E1 → E7), Level 2 APLS and flooded-road recall, minutes from image to alert draft, and **0 m** of any route through water above its vehicle limit. One limitation: Maxar post-event images exist only for the Kathmandu Valley, so Level 2 was tested there, not on the Koshi plains. |

## If something fails

| Problem | Fallback |
|---|---|
| Real data not ready | `export DATA_ROOT=~/flood-demo-data` (new, empty folder), `python scripts/demo_seed.py`, restart the API. The scene is called "SYNTHETIC demo (not real data)": say so out loud and do not quote numbers from it. |
| No internet | Dashboard base map tiles need internet; the flood layers, roads and panels still work. The app uses its offline map. |
| Phone cannot reach the laptop | Make the field report from the dashboard instead (click a road → Road closed): same effect on the route. |
| SMS does not arrive | Use the console gateway (default): the message is printed in the API terminal. |
| No safe route | That is a valid answer ("No safe route for this vehicle"); switch to truck or foot. |

## After the demo

- `python scripts/make_report_tables.py` and paste `docs/results_tables.md` into the report.
- Screenshots: dashboard with the route and verify segments, the approved alert, the phone queue.
