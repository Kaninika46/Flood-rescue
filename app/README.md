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

1. Start the backend on the laptop so the phone can reach it (it listens on 127.0.0.1 unless you
   ask for the phone demo):
   ```bash
   export OFFICER_TOKEN=... TEAM_TOKEN=...      # never commit them
   python -m backend.app --phone-demo
   ```
2. Phone and laptop on the same Wi-Fi. In the app: **Settings → Server address**
   `http://<laptop IP>:8000` (the emulator reaches the laptop at `http://10.0.2.2:8000`, the default).
   The laptop IP is the Windows Wi-Fi address (`ipconfig` → "Wireless LAN adapter Wi-Fi").
3. Set the team name, the **team token** (from the control room: the backend's `TEAM_TOKEN`) and
   the vehicle. Road reports need the token. Without it, or with a wrong one, reports stay in the
   queue as "waiting" and go out once the token is fixed; they are never thrown away.

### Windows firewall: let in only the phone (WSL2)

The API has no other network protection, so open port 8000 to the phone's IP only (phone:
Settings → Wi-Fi → your network → IP address). In an **administrator** PowerShell:

**Mirrored networking** (recommended; survives restarts):

```powershell
$phone = "192.168.1.23"   # the phone's IP
Add-Content "$env:USERPROFILE\.wslconfig" "[wsl2]`nnetworkingMode=mirrored"
New-NetFirewallHyperVRule -Name FloodRescue8000 -DisplayName "Flood Rescue backend (phone only)" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8000 -RemoteAddresses $phone -Action Allow
wsl --shutdown
```

**Or a port-forward** (NAT mode; the WSL IP changes after each restart, so rerun it then):

```powershell
$phone = "192.168.1.23"
$wsl = (wsl hostname -I).Trim().Split()[0]
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wsl
New-NetFirewallRule -DisplayName "Flood Rescue backend (phone only)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress $phone -Profile Any
```

If the phone gets a new IP, update the rule (`Set-NetFirewallHyperVRule -Name FloodRescue8000
-RemoteAddresses <new IP>`, or `Set-NetFirewallRule -DisplayName "Flood Rescue backend (phone
only)" -RemoteAddress <new IP>`). After the demo, remove it (`Remove-NetFirewallHyperVRule -Name
FloodRescue8000`, or `Remove-NetFirewallRule` and `netsh interface portproxy delete v4tov4
listenport=8000 listenaddress=0.0.0.0`). Check from the phone's browser:
`http://<laptop IP>:8000/health`. Shared or campus Wi-Fi often blocks phone-to-laptop traffic;
use the laptop's hotspot or a home router then.

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
