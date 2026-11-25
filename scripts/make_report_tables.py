"""Results tables for the Word report (prompt D6).

    python scripts/make_report_tables.py                    # from docs/results/*.json (default)
    python scripts/make_report_tables.py --source mlflow    # final eval runs in mlruns/
    python scripts/make_report_tables.py --source wandb --entity <your W&B user or team>

Writes ``docs/results/tables/{level1,level2,system}.{csv,md}`` and ``docs/results_tables.md``
(all three Markdown tables, ready to paste). Numbers are mean ± standard deviation over seeds.

Sources: the JSON files are written by ``scripts/evaluate.py`` (Level 1), ``scripts/score_sn8.py``
(Level 2) and ``scripts/time_pipeline.py`` (system); they are the default because they are the
only place final numbers come from. ``--source mlflow|wandb`` rebuilds Level 1 and Level 2 from
the trackers' ``eval`` runs instead (the latest eval run per experiment and seed; smoke runs
excluded), which is a useful cross-check.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import repo_root

app = typer.Typer(add_completion=False, help=__doc__)

L1_METRICS = ("iou", "f1", "precision", "recall")
L1_COLUMNS = [  # (header, split, metric)
    ("IoU test", "s11_test", "iou"),
    ("IoU Bolivia", "s11_bolivia", "iou"),
    ("IoU South Asia", "s11_southasia_holdout_test", "iou"),
    ("F1 test", "s11_test", "f1"),
    ("Precision test", "s11_test", "precision"),
    ("Recall test", "s11_test", "recall"),
]
L2_COLUMNS = [  # (header, key in the flat Level 2 record)
    ("Building IoU", "building/iou"),
    ("Flooded building IoU", "flooded_building/iou"),
    ("Road IoU", "road/iou"),
    ("Flooded road IoU", "flooded_road/iou"),
    ("Flooded road pixel recall", "flooded_road/pixel_recall"),
    ("Flooded road length precision", "flooded_road_len/precision"),
    ("Flooded road length recall", "flooded_road_len/recall"),
    ("APLS", "apls"),
]
EXPERIMENT_DESCRIPTIONS = {
    "E0": "ETCI released weights on ETCI validation",
    "E1": "Otsu threshold",
    "E2": "Random forest",
    "E3": "ETCI U-Net, no retraining",
    "E4": "Fine-tuned on hand labels",
    "E5": "E4 + pseudo-labels",
    "E6": "E5 + HAND channel",
    "E7": "Fusion of E1, E2, E6",
    "S0": "SpaceNet 8 baseline, 85/15 split",
    "S1": "Train Louisiana-East, test Germany",
    "S2": "S0 + SpaceNet 2/3/5 pre-training",
    "S3": "S2 + cloud mask + ensemble confidence",
    "S4": "Best model on Nepal 2024 tiles",
}


@dataclass
class Cell:
    """Mean, standard deviation and number of seeds of one number."""

    mean: float
    std: float
    n: int

    @classmethod
    def of(cls, values: Iterable[float]) -> Cell:
        vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
        if not vals:
            return cls(float("nan"), float("nan"), 0)
        return cls(
            statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals)
        )

    def text(self, digits: int = 3) -> str:
        if self.n == 0:
            return "n/a"
        if self.n == 1:
            return f"{self.mean:.{digits}f}"
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"


def exp_key(name: str) -> tuple[str, int, str]:
    """Sort E0 < E1 < E2 < E2sa < E10 and S0 < S4."""
    m = re.match(r"([A-Z])(\d+)(.*)", name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (name, 0, "")


# --------------------------------------------------------------------------- level 1


def level1_from_files(folder: Path) -> dict[tuple[str, str], dict[tuple[str, str], Cell]]:
    """{(experiment, variant): {(split, metric): Cell}} from evaluate.py result files."""
    out: dict[tuple[str, str], dict[tuple[str, str], Cell]] = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke"):
            continue
        for variant, splits in data.get("summary", {}).items():
            row = out.setdefault((data["experiment"], variant), {})
            for split, metrics in splits.items():
                for metric, s in metrics.items():
                    row[split, metric] = Cell(s["mean"], s["std"], s["n_seeds"])
    return out


def level1_from_runs(
    runs: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[tuple[str, str], Cell]]:
    """Same structure from tracker runs: ``{"experiment", "seed", "metrics"}`` (latest first)."""
    values: dict[tuple[str, str], dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in _latest_per_seed(runs, prefix="E"):
        for key, value in run["metrics"].items():
            parts = key.split("/")
            if "event" in parts or parts[-1] not in L1_METRICS or len(parts) not in (2, 3):
                continue
            split, metric = parts[0], parts[-1]
            variant = parts[1] if len(parts) == 3 else "default"
            values[run["experiment"], variant][split, metric].append(value)
    return {k: {m: Cell.of(v) for m, v in row.items()} for k, row in values.items()}


def level1_table(data: Mapping[tuple[str, str], Mapping[tuple[str, str], Cell]]) -> list[list[str]]:
    """Header + one row per experiment and variant."""
    header = ["Experiment", "What", "Variant", *[c[0] for c in L1_COLUMNS], "Seeds"]
    rows = [header]
    for exp, variant in sorted(data, key=lambda k: (exp_key(k[0]), k[1])):
        cells = data[exp, variant]
        seeds = max((c.n for c in cells.values()), default=0)
        what = EXPERIMENT_DESCRIPTIONS.get(re.sub(r"[a-z]+$", "", exp), "")
        if exp.endswith("sa"):
            what += " (South Asia hold-out training)"
        rows.append(
            [
                exp,
                what,
                "" if variant == "default" else variant,
                *[
                    cells.get((split, metric), Cell.of([])).text()
                    for _, split, metric in L1_COLUMNS
                ],
                str(seeds),
            ]
        )
    return rows


# --------------------------------------------------------------------------- level 2


def _flatten_level2(data: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in data.get("pixel", {}).items():
        flat[f"{k}/iou"] = v["overall"]["iou"]
        if k == "flooded_road":
            flat["flooded_road/pixel_recall"] = v["overall"]["recall"]
    for k, v in data.get("flooded_road_by_length", {}).items():
        flat[f"flooded_road_len/{k}"] = v
    if "apls" in data:
        flat["apls"] = data["apls"]["overall"]
    return flat


def level2_from_files(folder: Path) -> dict[tuple[str, str], dict[str, Cell]]:
    """{(experiment, split): {key: Cell}} from score_sn8.py result files."""
    out: dict[tuple[str, str], dict[str, Cell]] = {}
    for path in sorted(folder.glob("*__*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["experiment"], data["split"]] = {
            k: Cell.of([v]) for k, v in _flatten_level2(data).items()
        }
    return out


def level2_from_runs(runs: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Cell]]:
    """Same structure from tracker runs (keys ``<split>/<metric path>``)."""
    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wanted = {key for _, key in L2_COLUMNS}
    for run in _latest_per_seed(runs, prefix="S"):
        for key, value in run["metrics"].items():
            split, _, rest = key.partition("/")
            if rest in wanted:
                values[run["experiment"], split][rest].append(value)
    return {k: {m: Cell.of(v) for m, v in row.items()} for k, row in values.items()}


def level2_table(data: Mapping[tuple[str, str], Mapping[str, Cell]]) -> list[list[str]]:
    """Header + one row per experiment and test split."""
    rows = [["Experiment", "What", "Test split", *[c[0] for c in L2_COLUMNS], "Seeds"]]
    for exp, split in sorted(data, key=lambda k: (exp_key(k[0]), k[1])):
        cells = data[exp, split]
        rows.append(
            [
                exp,
                EXPERIMENT_DESCRIPTIONS.get(exp, ""),
                split,
                *[cells.get(key, Cell.of([])).text() for _, key in L2_COLUMNS],
                str(max((c.n for c in cells.values()), default=0)),
            ]
        )
    return rows


# --------------------------------------------------------------------------- system


def system_table(path: Path) -> list[list[str]]:
    """Minutes per step and per route from time_pipeline.py (``system_metrics.json``)."""
    rows = [["Metric", "Value"]]
    if not path.exists():
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(["Scene", str(data.get("scene", ""))])
    rows.append(["GPU", str(data.get("gpu", ""))])
    for step, minutes in data.get("minutes", {}).items():
        rows.append([f"Minutes: {step}", f"{minutes:.2f}"])
    rows.append(["Minutes: image to alert draft", f"{data.get('total_minutes', float('nan')):.2f}"])
    for vehicle, minutes in (data.get("route_minutes") or {}).items():
        rows.append(
            [
                f"Demo route time ({vehicle}), min",
                "no safe route" if minutes is None else f"{minutes:.1f}",
            ]
        )
    for vehicle, metres in (data.get("unsafe_length_m") or {}).items():
        rows.append([f"Route length through water above the {vehicle} limit, m", f"{metres:.0f}"])
    for status, n in (data.get("edge_status_counts") or {}).items():
        rows.append([f"Road edges: {status}", str(n)])
    return rows


# --------------------------------------------------------------------------- trackers


def _latest_per_seed(runs: Iterable[Mapping[str, Any]], prefix: str) -> list[Mapping[str, Any]]:
    """Latest eval run per (experiment, seed); runs come with ``start`` timestamps."""
    best: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        exp = run["experiment"]
        if not exp.startswith(prefix) or run.get("smoke"):
            continue
        key = (exp, run["seed"])
        if key not in best or run["start"] > best[key]["start"]:
            best[key] = run
    return list(best.values())


def _seed(name: str) -> int:
    m = re.search(r"-s(\d+)(?:-smoke)?$", name or "")
    return int(m.group(1)) if m else 0


def runs_from_mlflow() -> list[dict[str, Any]]:
    """Eval runs from the local MLflow store (mlruns/)."""
    import mlflow

    from src.common.logging import MLFLOW_EXPERIMENT, mlflow_uri

    mlflow.set_tracking_uri(mlflow_uri(repo_root() / "mlruns"))
    found = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        filter_string="tags.job_type = 'eval'",
        output_format="list",
    )
    return [
        {
            "experiment": r.data.tags.get("experiment", ""),
            "seed": _seed(r.info.run_name),
            "start": r.info.start_time or 0,
            "smoke": (r.info.run_name or "").endswith("-smoke"),
            "metrics": dict(r.data.metrics),
        }
        for r in found
    ]


def runs_from_wandb(entity: str) -> list[dict[str, Any]]:
    """Eval runs from the W&B project ``flood-rescue``."""
    import wandb

    from src.common.logging import WANDB_PROJECT

    out = []
    for r in wandb.Api().runs(f"{entity}/{WANDB_PROJECT}", filters={"jobType": "eval"}):
        metrics = {k: v for k, v in r.summary.items() if isinstance(v, int | float)}
        out.append(
            {
                "experiment": r.group or r.name.rsplit("-s", 1)[0],
                "seed": _seed(r.name),
                "start": r.created_at or "",
                "smoke": r.name.endswith("-smoke"),
                "metrics": metrics,
            }
        )
    return out


# --------------------------------------------------------------------------- output


def markdown(rows: list[list[str]]) -> str:
    """GitHub/Word-friendly Markdown table."""
    if len(rows) < 2:
        return "_No results yet._\n"
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines) + "\n"


def write(rows: list[list[str]], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    stem.with_suffix(".md").write_text(markdown(rows), encoding="utf-8")


@app.command()
def main(
    source: str = typer.Option("files", help="files | mlflow | wandb"),
    entity: str = typer.Option(None, help="W&B user or team (with --source wandb)."),
    results: Path = typer.Option(None, help="Results folder (default docs/results)."),
) -> None:
    """Build the three tables."""
    results = results or repo_root() / "docs" / "results"
    if source == "files":
        l1 = level1_from_files(results / "level1")
        l2 = level2_from_files(results / "level2")
    elif source in ("mlflow", "wandb"):
        if source == "wandb" and not entity:
            raise typer.BadParameter("--entity is needed with --source wandb")
        runs = runs_from_mlflow() if source == "mlflow" else runs_from_wandb(entity)
        l1, l2 = level1_from_runs(runs), level2_from_runs(runs)
    else:
        raise typer.BadParameter("source must be files, mlflow or wandb")
    tables = {
        "level1": level1_table(l1),
        "level2": level2_table(l2),
        "system": system_table(results / "system_metrics.json"),
    }
    for name, rows in tables.items():
        write(rows, results / "tables" / name)
    doc = [
        "# Results tables",
        "",
        f"Built by `scripts/make_report_tables.py --source {source}`. Mean ± standard deviation "
        "over seeds; one number when there is one seed; n/a when not run yet.",
        "",
        "## Level 1: radar flood map (Sen1Floods11; flood-class IoU, TP/FP/FN summed per split)",
        "",
        markdown(tables["level1"]),
        "## Level 2: road-level flood map (SpaceNet 8 scoring)",
        "",
        markdown(tables["level2"]),
        "## System (Koshi replay)",
        "",
        markdown(tables["system"]),
    ]
    out = (
        repo_root() / "docs" / "results_tables.md"
        if results == repo_root() / "docs" / "results"
        else results / "results_tables.md"
    )
    out.write_text("\n".join(doc), encoding="utf-8")
    print("\n".join(doc))


if __name__ == "__main__":
    app()
