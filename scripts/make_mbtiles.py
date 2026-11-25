"""Offline map tiles (MBTiles) for the rescuer app, drawn from our own data (prompt D4).

    python scripts/make_mbtiles.py                       # bbox from configs/routing.yaml
    python scripts/make_mbtiles.py --max-zoom 14 --scene-prob $DATA_ROOT/events/.../prob.tif

Draws 256 px raster tiles from the OSM road graph (``configs/routing.yaml`` graph.path) and,
when given, the flood probability map, and writes ``$DATA_ROOT/tiles/koshi.mbtiles``. Copy that
file to the phone (see app/README.md). The tiles are drawn here because openstreetmap.org's tile
servers do not allow bulk downloads for offline use. Map data © OpenStreetMap contributors (ODbL).
"""

from __future__ import annotations

import io
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import typer

from src.common.config import data_path, load_config, repo_root
from src.common.logging import get_logger

log = get_logger("mbtiles")
app = typer.Typer(add_completion=False, help=__doc__)

TILE = 256
ROAD_STYLE = {  # highway type -> (width px at z14, colour)
    "motorway": (5, (232, 146, 162)),
    "trunk": (5, (249, 178, 156)),
    "primary": (4, (252, 214, 164)),
    "secondary": (3.5, (247, 250, 191)),
    "tertiary": (3, (255, 255, 255)),
}
DEFAULT_ROAD = (2, (255, 255, 255))
BACKGROUND = (242, 239, 233)
CASING = (180, 175, 165)


def lonlat_to_pixel(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Global Web Mercator pixel at zoom ``z``."""
    n = TILE * 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_range(bbox: tuple[float, float, float, float], z: int) -> tuple[range, range]:
    """Tile columns and rows (XYZ) covering a lon/lat bbox."""
    w, s, e, n = bbox
    x0, y0 = lonlat_to_pixel(w, n, z)
    x1, y1 = lonlat_to_pixel(e, s, z)
    return range(int(x0 // TILE), int(x1 // TILE) + 1), range(int(y0 // TILE), int(y1 // TILE) + 1)


def _highway(data: dict) -> str:
    h = data.get("highway", "")
    return h[0] if isinstance(h, list) else str(h)


def draw_tile(lines: list, flood: tuple | None, z: int, tx: int, ty: int) -> bytes:
    """PNG of one tile: background, optional flood tint, roads with casing."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (TILE, TILE), BACKGROUND)
    if flood is not None:
        img = _flood_overlay(img, flood, z, tx, ty)
    draw = ImageDraw.Draw(img)
    scale = 2 ** (z - 14)
    ox, oy = tx * TILE, ty * TILE
    for casing in (True, False):
        for coords, (width, colour) in lines:
            pts = [(x * 2**z - ox, y * 2**z - oy) for x, y in coords]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) < -10 or min(xs) > TILE + 10 or max(ys) < -10 or min(ys) > TILE + 10:
                continue
            w = max(1, round(width * scale))
            draw.line(
                pts, fill=CASING if casing else colour, width=w + 2 if casing else w, joint="curve"
            )
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _flood_overlay(img, flood: tuple, z: int, tx: int, ty: int):
    """Blend the flood probability (>= 0.5) in blue over the tile."""
    from PIL import Image

    prob, inv_transform = flood
    cols, rows = np.meshgrid(np.arange(TILE) + 0.5, np.arange(TILE) + 0.5)
    n = TILE * 2**z
    lon = (tx * TILE + cols) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (ty * TILE + rows) / n))))
    c, r = inv_transform * (lon, lat)
    c, r = np.floor(c).astype(int), np.floor(r).astype(int)
    inside = (r >= 0) & (r < prob.shape[0]) & (c >= 0) & (c < prob.shape[1])
    wet = np.zeros((TILE, TILE), bool)
    wet[inside] = np.nan_to_num(prob[r[inside], c[inside]]) >= 0.5
    if not wet.any():
        return img
    over = np.zeros((TILE, TILE, 4), np.uint8)
    over[wet] = (90, 150, 220, 150)
    return Image.alpha_composite(img.convert("RGBA"), Image.fromarray(over, "RGBA")).convert("RGB")


def graph_lines(G) -> list:
    """Road polylines in normalised Web Mercator (0-1 x zoom-0 pixels) with style."""
    from src.routing.graph import edge_geometry

    lines, seen = [], set()
    for u, v, _k, data in G.edges(keys=True, data=True):
        if (v, u) in seen:
            continue
        seen.add((u, v))
        coords = [lonlat_to_pixel(x, y, 0) for x, y in edge_geometry(G, u, v, data).coords]
        lines.append((coords, ROAD_STYLE.get(_highway(data), DEFAULT_ROAD)))
    order = {s: i for i, s in enumerate(ROAD_STYLE.values())}
    lines.sort(key=lambda ln: -order.get(ln[1], -1))  # major roads drawn last
    return lines


def write_mbtiles(path: Path, tiles: dict, bbox: tuple, min_zoom: int, max_zoom: int) -> None:
    """MBTiles 1.3 (TMS row order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE metadata (name text, value text);"
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, "
        "tile_row integer, tile_data blob);"
        "CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row);"
    )
    w, s, e, n = bbox
    meta = {
        "name": "Koshi offline roads",
        "format": "png",
        "type": "baselayer",
        "version": "1",
        "bounds": f"{w},{s},{e},{n}",
        "center": f"{(w + e) / 2},{(s + n) / 2},{min_zoom + 2}",
        "minzoom": str(min_zoom),
        "maxzoom": str(max_zoom),
        "attribution": "© OpenStreetMap contributors",
    }
    con.executemany("INSERT INTO metadata VALUES (?, ?)", meta.items())
    con.executemany(
        "INSERT INTO tiles VALUES (?, ?, ?, ?)",
        [(z, x, (2**z - 1) - y, blob) for (z, x, y), blob in tiles.items()],
    )
    con.commit()
    con.close()


@app.command()
def main(
    min_zoom: int = 10,
    max_zoom: int = 15,
    scene_prob: Path = typer.Option(None, help="Flood probability GeoTIFF to tint (optional)."),
    out: Path = typer.Option(None, help="Default: $DATA_ROOT/tiles/koshi.mbtiles"),
) -> None:
    """Render and write the MBTiles file."""
    from src.routing.graph import load_graph

    cfg = load_config(repo_root() / "configs" / "routing.yaml")
    bbox = tuple(cfg["graph"]["bbox"])
    G = load_graph(data_path(cfg["graph"]["path"]))
    lines = graph_lines(G)
    flood = None
    if scene_prob is not None:
        import rasterio
        from rasterio.warp import Resampling, reproject

        with rasterio.open(scene_prob) as src:
            # resample to lon/lat so the per-pixel lookup is a plain affine inverse
            from rasterio.transform import from_bounds

            w, s, e, n = bbox
            width = 2048
            height = int(width * (n - s) / (e - w))
            transform = from_bounds(w, s, e, n, width, height)
            prob = np.full((height, width), np.nan, np.float32)
            reproject(
                rasterio.band(src, 1),
                prob,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        flood = (prob, ~transform)
    tiles = {}
    for z in range(min_zoom, max_zoom + 1):
        xs, ys = tile_range(bbox, z)
        for x in xs:
            for y in ys:
                tiles[z, x, y] = draw_tile(lines, flood, z, x, y)
        log.info("zoom %d: %d tiles", z, len(xs) * len(ys))
    out = out or data_path("tiles", "koshi.mbtiles")
    write_mbtiles(out, tiles, bbox, min_zoom, max_zoom)
    log.info("wrote %s (%d tiles, %.1f MB)", out, len(tiles), out.stat().st_size / 1e6)


if __name__ == "__main__":
    app()
