"""UNOSAT flood products: pick the flood-extent and analysis-extent layers of one date.

UNOSAT delivers one shapefile per layer and date, for example
``S1_20240927_FloodExtent_Koshi_Madhesh.shp`` and ``S1_20240927_AnalysisExtent_Koshi_Madhesh.shp``
(product FL20240928NPL). The date is taken from the **file name**, not from the attribute table:
in FL20240928NPL the 1 October flood layer stores ``Sensor_Dat`` 2024-01-10 (day and month
swapped), and the 29 September layer carries another event's code.

Credit every use: "Analysis: UNOSAT".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CREDIT = "Analysis: UNOSAT"
VECTOR_SUFFIXES = (".shp", ".geojson", ".gpkg")
_DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


@dataclass(frozen=True)
class UnosatLayers:
    """Vector files of one acquisition date (None when that layer is missing)."""

    day: str
    flood: Path | None
    extent: Path | None


def file_date(path: Path) -> str | None:
    """``YYYY-MM-DD`` from the first 8-digit date in the file name, or None."""
    match = _DATE.search(path.stem)
    if not match:
        return None
    d = match.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def layer_kind(path: Path) -> str | None:
    """``extent`` for analysis-extent files, ``flood`` for flood/water-extent files, else None."""
    name = path.stem.lower()
    if "analysis" in name or "aoi" in name:
        return "extent"
    if "flood" in name or "water" in name:
        return "flood"
    return None


def find_layers(folder: Path, day: str) -> UnosatLayers:
    """The flood and analysis-extent files for ``day`` (``YYYY-MM-DD``), searched recursively.

    Raises ``ValueError`` if a date has more than one file of the same kind, so two products
    are never silently merged.
    """
    found: dict[str, list[Path]] = {"flood": [], "extent": []}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in VECTOR_SUFFIXES or file_date(path) != day:
            continue
        kind = layer_kind(path)
        if kind:
            found[kind].append(path)
    for kind, paths in found.items():
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            raise ValueError(f"several UNOSAT {kind} layers for {day}: {names}")
    return UnosatLayers(
        day,
        found["flood"][0] if found["flood"] else None,
        found["extent"][0] if found["extent"] else None,
    )
