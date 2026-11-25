# app/: rescuer app (Flutter, Android)

Prompt D4. One screen per job, readable outside on a phone:

| Tab | What |
|---|---|
| **Map** | offline map (MBTiles) or OpenStreetMap when online; the assigned route (pink) with "verify" segments dashed orange; flooded (blue), closed (black), verify/unknown (dashed) roads; numbered priority areas; your GPS position; your reports. The route card always shows "Route is advice; the team on the ground decides." |
| **Priorities** | ranked areas from the control room; **Route** asks the server for a route from your position with your vehicle |
| **Reports** | the offline queue: waiting / sent / refused, with **Sync now** |

**Report road** (button on the map): *Road open*, *Road closed*, *Bridge gone*, at your GPS
position (or the map centre without a fix). Tap a road first to report that exact road; long-press
the map to report at a point. Reports are saved on the phone at once and sent oldest first when
the server is reachable (every 30 s and on "Sync now"). Without a road id the server snaps the
report to the nearest road within 60 m, or refuses it (shown as "refused" with the reason). After
a report the app asks for the route again, because the report can change it.

Cached on the phone for offline use: settings, priority list, the last route, road statuses
around the route (or around you) and the report queue.

## Build and run

Flutter 3.50 (stable) and the Android SDK (Android Studio, or the command-line tools):

```bash
cd app && flutter pub get
```
```bash
flutter test
```
```bash
flutter run            # phone connected with USB debugging, or an emulator
```
```bash
flutter build apk --release      # build/app/outputs/flutter-apk/app-release.apk
```

The release APK is signed with the debug key (fine for your own test phones; do not publish it).

## Connect to the backend

1. Start the backend on the laptop so the phone can reach it:
   `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`.
2. Phone and laptop on the same Wi-Fi. In the app: **Settings → Server address**
   `http://<laptop IP>:8000` (the emulator reaches the laptop at `http://10.0.2.2:8000`, the default).
3. Set the team name and vehicle.

## Offline map

```bash
python scripts/make_mbtiles.py --scene-prob $DATA_ROOT/nepal/level1/S1_20240927_flood_prob.tif
```

draws tiles (zoom 10 to 15) from the OSM road graph and the flood map into
`$DATA_ROOT/tiles/koshi.mbtiles` (openstreetmap.org tiles may not be bulk-downloaded). Then, on
the phone, at the base: **Settings → Download offline map**. It is served by the backend at
`/offline/tiles.mbtiles`.

## Safety

The app is an exercise tool: it shows the route as advice, never hides "verify" segments, and has
no way to send alerts or messages to the public. Field reports go only to your own backend.
