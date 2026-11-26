"""Report tables from result files and from tracker runs (prompt D6)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.make_report_tables import (
    Cell,
    level1_from_files,
    level1_from_runs,
    level1_table,
    level2_from_runs,
    level2_table,
    markdown,
    system_table,
)


def test_cell_mean_std() -> None:
    c = Cell.of([0.6, 0.7, float("nan")])
    assert c.n == 2 and math.isclose(c.mean, 0.65) and c.text() == "0.650 ± 0.071"
    assert Cell.of([0.5]).text() == "0.500" and Cell.of([]).text() == "n/a"


def test_level1_from_runs_latest_per_seed_and_variants() -> None:
    runs = [
        {"experiment": "E4", "seed": 0, "start": 1, "metrics": {"s11_test/iou": 0.1}},  # old
        {"experiment": "E4", "seed": 0, "start": 2, "metrics": {"s11_test/iou": 0.70}},
        {
            "experiment": "E4",
            "seed": 1,
            "start": 3,
            "metrics": {"s11_test/iou": 0.72, "s11_test/event/Ghana/iou": 0.1},
        },
        {"experiment": "E1", "seed": 0, "start": 1, "metrics": {"s11_test/vh/iou": 0.5}},
        {"experiment": "E4", "seed": 2, "start": 9, "smoke": True, "metrics": {"s11_test/iou": 0}},
        {"experiment": "S0", "seed": 0, "start": 1, "metrics": {"s11_test/iou": 0.9}},
    ]
    data = level1_from_runs(runs)
    assert set(data) == {("E4", "default"), ("E1", "vh")}
    cell = data["E4", "default"]["s11_test", "iou"]
    assert cell.n == 2 and math.isclose(cell.mean, 0.71)
    table = level1_table(data)
    assert [r[0] for r in table[1:]] == ["E1", "E4"]
    assert table[2][3] == "0.710 ± 0.014" and table[2][-1] == "2"


def test_level1_from_files(tmp_path: Path) -> None:
    summary = {
        "default": {
            "s11_test": {
                m: {"mean": 0.6, "std": 0.01, "n_seeds": 3}
                for m in ("iou", "f1", "precision", "recall")
            }
        }
    }
    (tmp_path / "E2.json").write_text(json.dumps({"experiment": "E2", "summary": summary}))
    (tmp_path / "E9.json").write_text(
        json.dumps({"experiment": "E9", "smoke": True, "summary": summary})
    )
    data = level1_from_files(tmp_path)
    assert list(data) == [("E2", "default")]
    row = level1_table(data)[1]
    assert row[:4] == ["E2", "Random forest", "", "0.600 ± 0.010"] and row[4] == "n/a"


def test_level2_and_system(tmp_path: Path) -> None:
    runs = [
        {
            "experiment": "S3",
            "seed": 0,
            "start": 1,
            "metrics": {
                "sn8_val/apls": 0.6,
                "sn8_val/flooded_road_len/recall": 0.8,
                "sn8_val/road/iou": 0.5,
            },
        }
    ]
    table = level2_table(level2_from_runs(runs))
    assert table[1][:3] == ["S3", "S2 + cloud mask + ensemble confidence", "sn8_val"]
    assert table[1][table[0].index("APLS")] == "0.600"
    path = tmp_path / "system_metrics.json"
    path.write_text(
        json.dumps(
            {
                "scene": "S1_20240927.tif",
                "gpu": "RTX 4060",
                "minutes": {"level1": 3.0, "roads": 1.0},
                "total_minutes": 4.0,
                "route_minutes": {"car": 41.8, "boat": None},
            }
        )
    )
    rows = system_table(path)
    assert ["Minutes: image to alert draft", "4.00"] in rows
    assert ["Demo route time (boat), min", "no safe route"] in rows
    assert markdown(rows).startswith("| Metric | Value |")


def test_level1_files_with_duplicate_experiment_names(tmp_path: Path) -> None:
    """Two files with experiment "E4": the final run keeps row E4, the other gets its own row."""

    def summary(iou: float) -> dict:
        return {"default": {"s11_test": {"iou": {"mean": iou, "std": 0.0, "n_seeds": 3}}}}

    (tmp_path / "E4.json").write_text(json.dumps({"experiment": "E4", "summary": summary(0.658)}))
    (tmp_path / "E4_patience10.json").write_text(
        json.dumps({"experiment": "E4", "summary": summary(0.649)})
    )
    data = level1_from_files(tmp_path)
    assert set(data) == {("E4", "default"), ("E4_patience10", "default")}
    assert math.isclose(data["E4", "default"]["s11_test", "iou"].mean, 0.658)
    rows = {r[0]: r for r in level1_table(data)[1:]}
    assert rows["E4"][3] == "0.658 ± 0.000"
    assert rows["E4_patience10"][3] == "0.649 ± 0.000"
    assert "superseded" in rows["E4_patience10"][1]
    assert [r[0] for r in level1_table(data)[1:]] == ["E4", "E4_patience10"]
