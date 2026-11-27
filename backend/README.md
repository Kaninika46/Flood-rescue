# backend/

FastAPI + PostGIS + Redis/RQ backend (prompt D1). It calls the code in `src/`; no model is
reimplemented here.

| Endpoint | What |
|---|---|
| `GET /health` | liveness |
| `GET/POST /scenes`, `POST /scenes/{id}/process` | register a Sentinel-1 scene (under `$DATA_ROOT`), queue the event pipeline (RQ) |
| `GET /layers` | flood probability, confidence and depth GeoTIFF paths |
| `GET /roads/status` | road edges with status (GeoJSON), latest scene plus field reports |
| `POST /routes` | `{vehicle, start: [lon, lat], end}` -> route GeoJSON with the advice text and "verify" segments |
| `GET /priorities` | ranked areas (GeoJSON) |
| `GET/POST /alerts`, `POST /alerts/{id}/approve`, `/cancel`, `/send` | CAP draft -> officer approval -> test phones only (officer token) |
| `GET/POST /field_reports` | road open / closed / bridge gone; the next route request uses it (team or officer token) |

## Tokens

Actions that change things need a bearer token (`Authorization: Bearer <token>`); reads and
`POST /routes` do not. Two secrets, from environment variables only (never commit them):

| Variable | Allows |
|---|---|
| `OFFICER_TOKEN` | `POST /alerts`, `/alerts/{id}/approve`, `/cancel`, `/send`; `POST /scenes`, `/scenes/{id}/process`; field reports |
| `TEAM_TOKEN` | `POST /field_reports` (the rescuer app) |

A missing or wrong token gets 401. If a variable is not set, its actions are refused for everyone.
Approvals record the role of the token (`approved_role`) next to the officer's name.
`GET /whoami` returns the role of a token. Make a token with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

The dashboard asks for the officer token once per tab and keeps it in memory only; the rescuer
app has a "Team token" field in Settings.

## Rate limits

`POST /routes` and `POST /field_reports` are limited per client IP (sliding minute, in memory):
`RATE_LIMIT_ROUTES` (default 30 per minute) and `RATE_LIMIT_REPORTS` (default 60). Over the limit
the answer is 429 with `Retry-After`; the rescuer app keeps the report queued and tries again.
Behind a Windows port-forward all phones share one address, so they share the limit.

Run:

```bash
docker compose -f backend/docker-compose.yml up -d db redis
```
```bash
python -m backend.app --reload
```

This listens on 127.0.0.1:8000 only. `--phone-demo` listens on all interfaces for phones on the
same Wi-Fi; it refuses to start unless `OFFICER_TOKEN` and `TEAM_TOKEN` are set. See
`app/README.md` for the firewall rule that lets in only the phone.
```bash
rq worker --url redis://localhost:6379/0 flood
```

Interactive API docs: http://localhost:8000/docs. Tests: `tests/test_backend_api.py`
(set `TEST_DATABASE_URL` to an empty PostGIS database).
