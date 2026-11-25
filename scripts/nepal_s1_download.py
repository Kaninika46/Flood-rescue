"""Sentinel-1 scenes over Koshi and Madhesh provinces from Google Earth Engine (prompt A10).

``COPERNICUS/S1_GRD`` is already calibrated, terrain-corrected and in dB, the same processing
as the Sen1Floods11 chips. Exports go to your Google Drive (GEE splits big images into several
GeoTIFFs; ``mosaic`` joins them).

One-time setup on the laptop (needs a Google account with Earth Engine access and a Cloud
project with the Earth Engine API enabled):

    earthengine authenticate
    export EE_PROJECT=<your-cloud-project-id>

Then:

    python scripts/nepal_s1_download.py list                  # acquisitions, orbits, passes
    python scripts/nepal_s1_download.py export --dry-run      # what would be exported, sizes
    python scripts/nepal_s1_download.py export --date 2024-09-27  # one post date + its pre scene
    python scripts/nepal_s1_download.py export                # every post date + pre scenes + JRC
    #   ... wait for the tasks (https://code.earthengine.google.com/tasks), download the files
    #   from the Drive folder into $DATA_ROOT/nepal/s1/raw/, then:
    python scripts/nepal_s1_download.py mosaic                # -> $DATA_ROOT/nepal/s1/S1_<date>.tif

Each post date gets its own pre-flood scene: the latest acquisition in ``dates.pre_window`` on
the same relative orbit, saved as ``S1_<post date>_pre`` (for example ``S1_20240927_pre`` is the
15 Sep 2024 orbit-121 pass). The area is the province outline intersected with
``aoi.clip_bbox``.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger

log = get_logger("nepal_s1")
app = typer.Typer(add_completion=False, help=__doc__)


def _cfg() -> dict[str, Any]:
    return load_config(repo_root() / "configs" / "nepal.yaml")


def init_ee(cfg: dict[str, Any]) -> Any:
    """Initialise Earth Engine with the configured / environment project."""
    import ee

    project = os.environ.get("EE_PROJECT") or cfg["gee"].get("project")
    if not project:
        raise typer.Exit("set EE_PROJECT (or gee.project in configs/nepal.yaml)")
    ee.Initialize(project=project)
    return ee


def aoi_geometry(ee: Any, cfg: dict[str, Any]) -> Any:
    """Koshi + Madhesh: local file, else geoBoundaries names, else the bbox; then clip_bbox."""
    a = cfg["aoi"]
    clip = a.get("clip_bbox")
    local = data_path(a["file"])
    if local.exists():
        import geopandas as gpd
        from shapely.geometry import box

        shape = gpd.read_file(local).to_crs("EPSG:4326").union_all()
        if clip:
            shape = shape.intersection(box(*clip))
        log.info("AOI: %s%s", local, " clipped to clip_bbox" if clip else "")
        return ee.Geometry(shape.__geo_interface__)
    provinces = ee.FeatureCollection(a["gee_collection"]).filter(
        ee.Filter.eq("shapeGroup", a["country_iso"])
    )
    chosen = provinces.filter(ee.Filter.inList("shapeName", a["names"]))
    if chosen.size().getInfo() >= 2:
        log.info("AOI: %s", chosen.aggregate_array("shapeName").getInfo())
        geom = chosen.geometry()
    else:
        log.warning("province names not found in %s; using the bbox", a["gee_collection"])
        geom = ee.Geometry.Rectangle(a["bbox"], None, False)
    if clip:
        log.info("AOI clipped to %s", clip)
        geom = geom.intersection(ee.Geometry.Rectangle(clip, None, False), ee.ErrorMargin(1))
    return geom


def aoi_bounds(ee: Any, aoi: Any) -> tuple[float, float, float, float]:
    """Lon/lat bounding box (xmin, ymin, xmax, ymax) of an Earth Engine geometry."""
    ring = aoi.bounds(ee.ErrorMargin(1), "EPSG:4326").coordinates().get(0).getInfo()
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def s1_collection(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """IW scenes with both VV and VH over the AOI between two dates (end exclusive)."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def describe(ee: Any, collection: Any) -> list[dict[str, Any]]:
    """Date, orbit, pass and id of each image."""
    props = [
        "system:time_start",
        "relativeOrbitNumber_start",
        "orbitProperties_pass",
        "system:index",
    ]
    rows = collection.reduceColumns(ee.Reducer.toList(len(props)), props).get("list").getInfo()
    out = []
    for t, orbit, direction, index in rows:
        out.append(
            {
                "date": ee.Date(t).format("YYYY-MM-dd").getInfo(),
                "orbit": orbit,
                "pass": direction,
                "id": index,
            }
        )
    return sorted(out, key=lambda r: r["date"])


# GEE turns a scale in metres into degrees with the equatorial length of one degree.
METRES_PER_DEGREE = 111_319.490793


def pair_pre_scenes(
    post_rows: dict[str, list[dict[str, Any]]], pre_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any] | None]:
    """For each post date, the latest earlier pre-flood acquisition on one of its orbits."""
    out: dict[str, dict[str, Any] | None] = {}
    for day, rows in post_rows.items():
        orbits = {r["orbit"] for r in rows}
        matches = [r for r in pre_rows if r["orbit"] in orbits and r["date"] < day]
        out[day] = max(matches, key=lambda r: r["date"]) if matches else None
    return out


def select_post_dates(post_dates: list[str], date: str | None) -> list[str]:
    """All configured post dates, or only ``date`` (which must be one of them)."""
    if date is None:
        return list(post_dates)
    if date not in post_dates:
        raise typer.BadParameter(f"--date must be one of {post_dates}, got {date!r}")
    return [date]


def export_shape(
    bounds: tuple[float, float, float, float], scale_m: float, crs: str
) -> tuple[int, int]:
    """Width and height in pixels of an export over lon/lat ``bounds`` at ``scale_m``."""
    from pyproj import CRS, Transformer

    target = CRS.from_user_input(crs)
    if target.is_geographic:
        step = scale_m / METRES_PER_DEGREE
        xmin, ymin, xmax, ymax = bounds
    else:
        to_target = Transformer.from_crs("EPSG:4326", target, always_xy=True)
        xmin, ymin, xmax, ymax = to_target.transform_bounds(*bounds)
        step = scale_m
    # Round first so float noise (50.0000001 px) does not add a whole extra pixel.
    return math.ceil(round((xmax - xmin) / step, 6)), math.ceil(round((ymax - ymin) / step, 6))


def estimate_bytes(width: int, height: int, bands: int) -> int:
    """Uncompressed float32 size: width x height x bands x 4 bytes."""
    return width * height * bands * 4


def human_size(n_bytes: int) -> str:
    """Bytes as GB (10^9) or MB (10^6), one decimal."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.1f} GB"
    return f"{n_bytes / 1e6:.1f} MB"


def _next_day(day: str) -> str:
    from datetime import date, timedelta

    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


@app.command("list")
def list_scenes() -> None:
    """Print every acquisition in the post dates and the pre-flood window."""
    cfg = _cfg()
    ee = init_ee(cfg)
    aoi = aoi_geometry(ee, cfg)
    windows = [(d, _next_day(d)) for d in cfg["dates"]["post"]] + [
        tuple(cfg["dates"]["pre_window"])
    ]
    for start, end in windows:
        print(f"\n{start} .. {end}")
        for row in describe(ee, s1_collection(ee, aoi, start, end)):
            print(f"  {row['date']} orbit {row['orbit']:3d} {row['pass']:10s} {row['id']}")


@app.command()
def export(
    date: str | None = typer.Option(
        None, "--date", help="Only this post date (YYYY-MM-DD) and its pre-flood scene."
    ),
    jrc: bool = typer.Option(True, "--jrc/--no-jrc", help="Also export JRC water occurrence."),
    dry_run: bool = typer.Option(False, help="Only print what would be exported and its size."),
) -> None:
    """Export VV/VH mosaics of each post date and its own pre-flood scene on the same orbit."""
    cfg = _cfg()
    days = select_post_dates(cfg["dates"]["post"], date)
    ee = init_ee(cfg)
    aoi = aoi_geometry(ee, cfg)
    s1_bands = ["VV", "VH"]

    post_rows: dict[str, list[dict[str, Any]]] = {}
    post_colls: dict[str, Any] = {}
    for day in days:
        coll = s1_collection(ee, aoi, day, _next_day(day))
        rows = describe(ee, coll)
        if not rows:
            log.warning("no Sentinel-1 scene on %s", day)
            continue
        if len({r["orbit"] for r in rows}) > 1:
            log.warning("%s has several orbits; the post mosaic mixes them", day)
        post_rows[day] = rows
        post_colls[day] = coll
    pre_rows = describe(ee, s1_collection(ee, aoi, *cfg["dates"]["pre_window"]))
    pairs = pair_pre_scenes(post_rows, pre_rows)

    # (file prefix, image, bands, what it is)
    jobs: list[tuple[str, Any, list[str], str]] = []
    for day, rows in post_rows.items():
        stem = f"S1_{day.replace('-', '')}"
        orbits = sorted({r["orbit"] for r in rows})
        what = f"{day}, orbit {orbits}, {len(rows)} scenes"
        jobs.append((stem, post_colls[day].mosaic(), s1_bands, what))
        pre = pairs[day]
        if pre is None:
            log.warning("no pre-flood scene for %s on orbits %s", day, orbits)
            continue
        pre_coll = s1_collection(ee, aoi, pre["date"], _next_day(pre["date"])).filter(
            ee.Filter.eq("relativeOrbitNumber_start", pre["orbit"])
        )
        what = f"{pre['date']}, orbit {pre['orbit']}, {pre_coll.size().getInfo()} scenes"
        jobs.append((f"{stem}_pre", pre_coll.mosaic(), s1_bands, what))
    if jrc:
        image = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
        jobs.append(("JRC_occurrence", image, ["occurrence"], "JRC Global Surface Water 1.4"))

    scale, crs = cfg["gee"]["scale_m"], cfg["gee"]["crs"]
    bounds = aoi_bounds(ee, aoi)
    width, height = export_shape(bounds, scale, crs)
    print(f"region {[round(b, 4) for b in bounds]}: {width} x {height} px at {scale} m, {crs}")
    total = 0
    for prefix, image, bands, what in jobs:
        size = estimate_bytes(width, height, len(bands))
        total += size
        print(f"export {prefix} ({what}): bands {bands}, about {human_size(size)}")
        if dry_run:
            continue
        task = ee.batch.Export.image.toDrive(
            image=image.select(bands).toFloat().clip(aoi),
            description=prefix,
            folder=cfg["gee"]["drive_folder"],
            fileNamePrefix=prefix,
            region=aoi,
            scale=scale,
            crs=crs,
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        task.start()
        print(f"  started task {task.id}")
    print(f"total about {human_size(total)} (uncompressed float32 upper bound)")
    if dry_run:
        print("dry run: nothing exported")


@app.command()
def mosaic(
    raw_dir: str = typer.Option("nepal/s1/raw", help="Downloaded GeoTIFFs (under DATA_ROOT)"),
) -> None:
    """Join the pieces GEE wrote for each export into one Cloud-Optimised GeoTIFF."""
    from osgeo import gdal

    gdal.UseExceptions()
    cfg = _cfg()
    raw = data_path(raw_dir)
    groups: dict[str, list[str]] = defaultdict(list)
    for tif in sorted(raw.glob("*.tif")):
        key = tif.stem.split("-")[0]  # GEE pieces: <prefix>-0000000000-0000000000.tif
        groups[key].append(str(tif))
    out_dir = data_path(cfg["paths"]["s1_dir"])
    for key, files in groups.items():
        vrt = str(out_dir / f"{key}.vrt")
        gdal.BuildVRT(vrt, files)
        out = out_dir / f"{key}.tif"
        gdal.Translate(
            str(out),
            vrt,
            format="COG",
            creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"],
        )
        Path(vrt).unlink()
        print(f"{out} from {len(files)} piece(s)")


if __name__ == "__main__":
    app()
