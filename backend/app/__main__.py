"""Start the API: localhost only, unless asked for the phone demo.

    python -m backend.app                    # http://127.0.0.1:8000 (this laptop only)
    python -m backend.app --phone-demo       # 0.0.0.0:8000, for phones on the same Wi-Fi

``--phone-demo`` needs ``OFFICER_TOKEN`` and ``TEAM_TOKEN`` set and a Windows firewall rule that
lets in only the phone's IP (app/README.md, "Connect to the backend").
"""

from __future__ import annotations

import os

import typer

LOCAL_HOST = "127.0.0.1"
ALL_HOSTS = "0.0.0.0"  # noqa: S104 - only with --phone-demo

app = typer.Typer(add_completion=False, help=__doc__)


def choose_host(phone_demo: bool) -> str:
    """127.0.0.1 by default; every interface only for the phone demo."""
    return ALL_HOSTS if phone_demo else LOCAL_HOST


def missing_tokens() -> list[str]:
    """Token variables that are not set (the phone demo refuses to start without them)."""
    return [name for name in ("OFFICER_TOKEN", "TEAM_TOKEN") if not os.environ.get(name)]


@app.command()
def main(
    phone_demo: bool = typer.Option(
        False, "--phone-demo", help="Listen on all interfaces so phones on the Wi-Fi can connect."
    ),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Restart on code changes (development)."),
) -> None:
    """Run uvicorn with a safe host."""
    import uvicorn

    if phone_demo:
        missing = missing_tokens()
        if missing:
            typer.echo(f"--phone-demo needs {', '.join(missing)} set; refusing to start.", err=True)
            raise typer.Exit(2)
        typer.echo(
            f"Listening on all interfaces, port {port}. Allow only your phone's IP through the "
            "Windows firewall (app/README.md)."
        )
    uvicorn.run("backend.app.main:app", host=choose_host(phone_demo), port=port, reload=reload)


if __name__ == "__main__":
    app()
