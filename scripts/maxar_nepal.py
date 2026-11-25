"""Maxar Open Data for the Nepal floods of September 2024 (prompt B7). Run in the "flood" env.

    python scripts/maxar_nepal.py list        # every item: date, cloud cover, GSD, footprint
    python scripts/maxar_nepal.py pairs       # pre/post overlaps, vs UNOSAT flood + Koshi/Madhesh
    python scripts/maxar_nepal.py tiles --max-pairs 60
    python scripts/maxar_nepal.py osm         # OSM roads and buildings per tile, SpaceNet 8 schema

Imagery: CC BY-NC 4.0, academic use only. Credit: "Maxar Open Data Program".

* ``list`` walks the static STAC catalog (collection -> acquisition collections -> items) and
  writes ``$DATA_ROOT/nepal/maxar/items.geojson`` plus ``docs/results/maxar_nepal_items.csv``.
* ``pairs`` intersects pre-event items (before 2024-09-27) with post-event items
  (2024-09-27 or later), keeps overlaps inside ``maxar.area_bbox`` of configs/nepal.yaml and,
  when flood polygons for that area are present in ``nepal/unosat``,
  are present, scores each overlap by flooded area so pairs that cover flooded roads come first.
* ``tiles`` cuts the overlaps into 1300x1300 pre-event tiles on the pre image's own grid (as
  SpaceNet 8 did) and the matching post-event window at its native resolution (the dataset code
  warps post onto pre). Files: ``nepal/maxar/tiles/{PRE-event,POST-event}/<catid>_<row>_<col>.tif``.
* ``osm`` (needs Overpass, run on the laptop) writes ``tiles/labels/<tile id>.geojson``: roads and
  buildings in each tile with the
  SpaceNet 8 attribute names and an empty ``flooded`` field, ready for labelling in QGIS
  (``docs/labelling_guide.md``).
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger

ROOT = "https://maxar-opendata.s3.amazonaws.com/events/Nepal-Floods-Sept-2024/collection.json"
EVENT_DATE = "2024-09-27"
TILE = 1300
log = get_logger("maxar")
app = typer.Typer(add_completion=False, help=__doc__)


def fetch_json(url: str) -> dict[str, Any]:
    """GET a JSON document."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def walk_items(url: str) -> list[dict[str, Any]]:
    """All STAC items below a catalog/collection URL (static catalog, relative links)."""
    doc = fetch_json(url)
    if doc.get("type") == "Feature":
        doc["_href"] = url
        return [doc]
    items = []
    for link in doc.get("links", []):
        if link["rel"] in ("child", "item"):
            items += walk_items(urllib.parse.urljoin(url, link["href"]))
    return items


def summarise(item: dict[str, Any]) -> dict[str, Any]:
    """Flat row for one item."""
    p = item["properties"]
    visual = item.get("assets", {}).get("visual", {}).get("href", "")
    return {
        "id": item["id"],
        "datetime": p.get("datetime", "")[:19],
        "catalog_id": p.get("catalog_id"),
        "platform": p.get("platform"),
        "gsd": p.get("gsd"),
        "cloud_cover": p.get("eo:cloud_cover"),
        "off_nadir": p.get("view:off_nadir"),
        "utm_epsg": p.get("proj:epsg"),
        "visual": urllib.parse.urljoin(item["_href"], visual) if visual else "",
        "phase": "post" if p.get("datetime", "")[:10] >= EVENT_DATE else "pre",
    }


@app.command("list")
def list_items() -> None:
    """Catalogue every item of the Nepal-Floods-Sept-2024 collection."""
    import geopandas as gpd
    from shapely.geometry import shape

    items = walk_items(ROOT)
    rows = [summarise(i) for i in items]
    gdf = gpd.GeoDataFrame(rows, geometry=[shape(i["geometry"]) for i in items], crs="EPSG:4326")
    out = data_path("nepal", "maxar", "items.geojson")
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    csv = repo_root() / "docs" / "results" / "maxar_nepal_items.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").assign(
        bounds=[",".join(f"{v:.4f}" for v in g.bounds) for g in gdf.geometry]
    ).sort_values("datetime").to_csv(csv, index=False)
    log.info(
        "%d items (%d post-event) -> %s and %s",
        len(gdf),
        int((gdf["phase"] == "post").sum()),
        out,
        csv,
    )


def analysis_area() -> Any:
    """``maxar.area_bbox`` of configs/nepal.yaml (Kathmandu Valley), else the Level 1 AOI."""
    import geopandas as gpd
    from shapely.geometry import box

    full = load_config(repo_root() / "configs" / "nepal.yaml")
    if full.get("maxar", {}).get("area_bbox"):
        return box(*full["maxar"]["area_bbox"])
    cfg = full["aoi"]
    path = data_path(cfg["file"])
    if path.exists():
        return gpd.read_file(path).to_crs("EPSG:4326").union_all()
    return box(*cfg["bbox"])


@app.command()
def pairs(min_overlap_km2: float = typer.Option(1.0)) -> None:
    """Pre/post overlaps inside the analysis area, ranked by UNOSAT flooded area."""
    import geopandas as gpd

    from src.common.geo import read_vectors

    items = gpd.read_file(data_path("nepal", "maxar", "items.geojson"))
    area = analysis_area()
    pre, post = items[items["phase"] == "pre"], items[items["phase"] == "post"]
    flood = None
    unosat = data_path("nepal", "unosat")
    if unosat.is_dir() and any(unosat.iterdir()):
        flood = read_vectors(unosat).union_all()
    rows = []
    for _, a in post.iterrows():
        for _, b in pre.iterrows():
            overlap = a.geometry.intersection(b.geometry).intersection(area)
            if overlap.is_empty:
                continue
            km2 = gpd.GeoSeries([overlap], crs="EPSG:4326").to_crs(6933).area.iloc[0] / 1e6
            if km2 < min_overlap_km2:
                continue
            flooded = 0.0
            if flood is not None:
                hit = overlap.intersection(flood)
                flooded = gpd.GeoSeries([hit], crs="EPSG:4326").to_crs(6933).area.iloc[0] / 1e6
            rows.append(
                {
                    "post_id": a["id"],
                    "post_date": a["datetime"],
                    "pre_id": b["id"],
                    "pre_date": b["datetime"],
                    "post_cloud": a["cloud_cover"],
                    "overlap_km2": round(km2, 2),
                    "unosat_flood_km2": round(flooded, 2),
                    "geometry": overlap,
                }
            )
    if not rows:
        log.warning("no pre/post overlap inside the analysis area; check India-Floods-Oct-2023")
        return
    df = gpd.GeoDataFrame(rows, crs="EPSG:4326").sort_values(
        ["unosat_flood_km2", "overlap_km2"], ascending=False
    )
    df.to_file(data_path("nepal", "maxar", "pairs.geojson"), driver="GeoJSON")
    print(df.drop(columns="geometry").head(20).to_string(index=False))


@app.command()
def tiles(max_pairs: int = typer.Option(60, help="Stop after this many tile pairs.")) -> None:
    """Cut 1300x1300 pre tiles and matching post windows over the ranked overlaps."""
    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform
    from shapely.geometry import box

    ranked = gpd.read_file(data_path("nepal", "maxar", "pairs.geojson"))
    items = gpd.read_file(data_path("nepal", "maxar", "items.geojson")).set_index("id")
    out = data_path("nepal", "maxar", "tiles")
    made = 0
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"}
    with rasterio.Env(**env):
        for pair in ranked.itertuples():
            pre_url, post_url = items.loc[pair.pre_id, "visual"], items.loc[pair.post_id, "visual"]
            with (
                rasterio.open("/vsicurl/" + pre_url) as pre,
                rasterio.open("/vsicurl/" + post_url) as post,
            ):
                zone = gpd.GeoSeries([pair.geometry], crs="EPSG:4326").to_crs(pre.crs).iloc[0]
                for row in range(0, pre.height - TILE + 1, TILE):
                    for col in range(0, pre.width - TILE + 1, TILE):
                        win = Window(col, row, TILE, TILE)
                        bounds = window_bounds(win, pre.transform)
                        if not zone.contains(box(*bounds)):
                            continue
                        # SpaceNet 8 style <catalog id>_<tile id>;
                        # item ids are <zone>/<quadkey>/<catid>
                        _, quadkey, catid = pair.pre_id.split("/")
                        name = f"{catid}_{quadkey}_{row // TILE}_{col // TILE}"
                        pre_arr = pre.read(window=win)
                        if (pre_arr.max(axis=0) == 0).mean() > 0.05:
                            continue  # outside the image
                        post_bounds = transform_bounds(pre.crs, post.crs, *bounds)
                        post_win = from_bounds(*post_bounds, post.transform).round_offsets()
                        post_arr = post.read(window=post_win, boundless=True, fill_value=0)
                        for kind, arr, src, w in (
                            ("PRE-event", pre_arr, pre, win),
                            ("POST-event", post_arr, post, post_win),
                        ):
                            path = out / kind / f"{name}.tif"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            profile = {  # plain RGB (Maxar COGs are JPEG/YCbCr)
                                "driver": "GTiff",
                                "dtype": arr.dtype.name,
                                "count": arr.shape[0],
                                "width": arr.shape[2],
                                "height": arr.shape[1],
                                "crs": src.crs,
                                "transform": window_transform(w, src.transform),
                                "photometric": "RGB",
                                "compress": "deflate",
                            }
                            with rasterio.open(path, "w", **profile) as dst:
                                dst.write(arr)
                        made += 1
                        if made >= max_pairs:
                            log.info("%d tile pairs in %s", made, out)
                            return
    log.info("%d tile pairs in %s", made, out)


@app.command()
def osm(buffer_m: float = typer.Option(0.0)) -> None:
    """OSM roads and buildings per tile, SpaceNet 8 attributes, ``flooded`` empty (laptop)."""
    import geopandas as gpd
    import osmnx as ox
    import pandas as pd
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    pre_dir = data_path("nepal", "maxar", "tiles", "PRE-event")
    out_dir = data_path("nepal", "maxar", "tiles", "labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["highway", "lanes", "surface", "oneway", "bridge", "layer", "tunnel", "building"]
    for tif in sorted(pre_dir.glob("*.tif")):
        with rasterio.open(tif) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        poly = box(west, south, east, north)
        frames = []
        for tags in ({"highway": True}, {"building": True}):
            try:
                gdf = ox.features_from_polygon(poly, tags)
            except Exception:
                continue
            gdf = gdf[gdf.geometry.intersects(poly)].copy()
            gdf["geometry"] = gdf.geometry.intersection(poly)
            frames.append(gdf)
        if not frames:
            continue
        gdf = gpd.GeoDataFrame(pd.concat(frames), crs="EPSG:4326").reset_index()
        out = gpd.GeoDataFrame(
            {
                "id": [
                    f"{t}/{i}"
                    for t, i in zip(
                        gdf.get("element", gdf.get("element_type", "")),
                        gdf.get("id", gdf.get("osmid", "")),
                        strict=False,
                    )
                ],
                **{k: gdf.get(k, None) for k in keep},
                "flooded": None,
                "type": None,
                "image_id": tif.stem.split("_")[0],
                "location": "Nepal_2024",
            },
            geometry=gdf.geometry,
            crs="EPSG:4326",
        )
        out.to_file(out_dir / f"{'_'.join(tif.stem.split('_')[1:])}.geojson", driver="GeoJSON")
    log.info("annotations in %s; label them following docs/labelling_guide.md", out_dir)


if __name__ == "__main__":
    app()
