# backend/

FastAPI + PostGIS + Redis/RQ backend (prompt D1). It calls the code in `src/`; no model is
reimplemented here.

| Endpoint | What |
|---|---|
| `GET /health` | liveness |
| `GET/POST /scenes`, `POST /scenes/{id}/process` | register a Sentinel-1 scene (under `$DATA_ROOT`), queue the event pipeline (RQ) |
| `GET /layers` | flood probability, confidence and depth layers (id, kind, PNG URL; no server file paths) |
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

## Secrets: `backend/.env`

The database and Redis passwords and the two API tokens live in `backend/.env`, which git
ignores. Nothing secret is written in `docker-compose.yml`.

```bash
cp backend/.env.example backend/.env
```
```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"   # once per value in backend/.env
```

Docker Compose reads `backend/.env` on its own (it sits next to `docker-compose.yml`) and stops
with "set POSTGRES_PASSWORD in backend/.env" if a password is missing. Redis requires its password
(`--requirepass`). Every published port is bound to 127.0.0.1. Postgres sets the password only
when it creates the `pgdata` volume; for an older volume made with the former password, change it
once with `docker compose -f backend/docker-compose.yml exec db psql -U flood -c "ALTER USER flood
PASSWORD '<new>'"`.

For the API and worker outside Docker, load the same file into your shell:

```bash
set -a && source backend/.env && set +a
```
```bash
export DATABASE_URL="postgresql+psycopg://flood:$POSTGRES_PASSWORD@localhost:5432/flood" REDIS_URL="redis://:$REDIS_PASSWORD@localhost:6379/0"
```

(With the local PostgreSQL over its Unix socket instead of Docker, `DATABASE_URL` is
`postgresql+psycopg://kaninika@/flood?host=/var/run/postgresql` and needs no password.)

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
rq worker --url "$REDIS_URL" flood
```

Interactive API docs: http://localhost:8000/docs. Tests: `tests/test_backend_api.py`
(set `TEST_DATABASE_URL` to an empty PostGIS database).
