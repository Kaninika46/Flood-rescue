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
| `GET/POST /alerts`, `POST /alerts/{id}/approve`, `POST /alerts/{id}/send` | CAP draft -> officer approval -> test phones only |
| `GET/POST /field_reports` | road open / closed / bridge gone; the next route request uses it |

Run:

```bash
docker compose -f backend/docker-compose.yml up -d db redis
```
```bash
uvicorn backend.app.main:app --reload
```
```bash
rq worker --url redis://localhost:6379/0 flood
```

Interactive API docs: http://localhost:8000/docs. Tests: `tests/test_backend_api.py`
(set `TEST_DATABASE_URL` to an empty PostGIS database).
