"""Draft -> approve -> "send" one alert to the console mock (prompt C4 "done when").

    python scripts/alert_demo.py --officer "Your Name"
    python scripts/alert_demo.py --officer "Your Name" \
        --polygon $DATA_ROOT/nepal/unosat/flood.geojson

Builds a CAP 1.2 Exercise alert for the largest flood polygon (UNOSAT file or, by default, a
small box on the Koshi floodplain), saves the XML to ``outputs/alerts/``, approves it in the
officer's name and sends its SMS text through the console gateway to the numbers in
``config/test_recipients.yaml``. With no such file it stops: alerts go to test phones only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.alerts.approval import AlertRecord, approve
from src.alerts.cap import AlertContent, build_alert, polygon_from_geometry, sms_text
from src.alerts.gateway import ConsoleGateway, RecipientRefused, load_test_recipients, send_alert
from src.common.config import repo_root

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    officer: str = typer.Option(..., help="Name recorded as approved_by."),
    polygon: Path = typer.Option(None, help="Flood polygons (GeoJSON/shapefile)."),
    district: str = typer.Option("Saptari"),
    shelter: str = typer.Option("the high school at Rajbiraj"),
) -> None:
    """Run the three steps and print the record."""
    from shapely.geometry import box

    if polygon is not None:
        import geopandas as gpd

        geom = gpd.read_file(polygon).to_crs("EPSG:4326").union_all()
    else:
        geom = box(86.75, 26.55, 86.95, 26.70)
    content = AlertContent(
        river="Koshi",
        district=district,
        shelter=shelter,
        depth_m=1.0,
        area_desc=f"{district}, Koshi floodplain",
        polygon=polygon_from_geometry(geom),
    )
    xml = build_alert(content)
    record = AlertRecord(alert_id="demo-1", cap_xml=xml, sms=sms_text(content))
    out = repo_root() / "outputs" / "alerts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo-1.xml").write_text(xml, encoding="utf-8")
    print(f"1. draft written to {out / 'demo-1.xml'} (state {record.state})")
    approve(record, officer)
    print(f"2. approved by {record.approved_by} at {record.approved_at.isoformat()}")
    recipients = sorted(load_test_recipients())
    try:
        send_alert(record, recipients, ConsoleGateway())
    except RecipientRefused as err:
        raise typer.Exit(
            f"3. not sent: {err}. Create config/test_recipients.yaml from the "
            "example with your own test phone."
        ) from err
    print(
        f"3. sent to {len(record.sent_to)} test phone(s) via the console mock; state {record.state}"
    )


if __name__ == "__main__":
    app()
