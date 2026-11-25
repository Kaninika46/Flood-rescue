"""Final test numbers for Level 1 experiments (the only source of reported results).

    python scripts/evaluate.py --config configs/E1.yaml
    python scripts/evaluate.py --config configs/E4.yaml --split s11_test --split s11_bolivia

For every variant (E1), seed and split it scores the predictions with ``src/common/metrics.py``
(TP/FP/FN summed over the split, plus per event), logs them to the tracker as an ``eval`` run,
and writes ``docs/results/level1/<experiment>.json`` with per-seed numbers and mean / standard
deviation over seeds. ``scripts/make_report_tables.py`` builds the report tables from these files.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.common.config import load_config, repo_root
from src.common.logging import get_logger, git_info, start_run
from src.common.metrics import METRIC_NAMES, SplitScorer, binarize
from src.common.seed import seed_everything
from src.level1_radar.datasets import load_label
from src.level1_radar.methods import make_predictor
from src.level1_radar.s11 import read_split

app = typer.Typer(add_completion=False, help=__doc__)
log = get_logger("evaluate")


def split_rows(split: str) -> list[dict[str, Any]]:
    """Rows of a committed split file, or ETCI tiles for ``etci_<split>`` (E0)."""
    if split.startswith("etci_"):
        from src.level1_radar.etci_reproduce import etci_rows

        return etci_rows(split.removeprefix("etci_")).to_dict("records")
    return read_split(split).to_dict("records")


def reference(row: dict[str, Any]) -> Any:
    """Reference labels of a row (-1 ignore, 0, 1)."""
    if "label" in row:
        return load_label(row["label"])
    from src.level1_radar.etci_reproduce import etci_row_arrays

    return etci_row_arrays(row)[2]


def score(
    predictor: Any, split: str, threshold: float, limit: int = 0, skip_blank: bool = False
) -> SplitScorer:
    """Run a predictor over every chip of a split and pool the counts."""
    rows = split_rows(split)
    if limit:
        rows = rows[:limit]
    scorer = SplitScorer()
    for row in rows:
        if skip_blank:
            from src.level1_radar.etci_reproduce import etci_row_arrays, is_blank

            if is_blank(etci_row_arrays(row)[0]):
                continue
        prob = predictor(row)
        scorer.update(binarize(prob, threshold), reference(row), event=row["event"])
    return scorer


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation over seeds for each (variant, split, metric)."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    keys = {(r["variant"], r["split"]) for r in results}
    for variant, split in sorted(keys):
        runs = [r for r in results if r["variant"] == variant and r["split"] == split]
        entry = {}
        for metric in METRIC_NAMES:
            values = [r["overall"][metric] for r in runs]
            entry[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
            }
        table.setdefault(variant, {})[split] = entry
    return table


@app.command()
def main(
    config: Path = typer.Option(..., help="Experiment config, e.g. configs/E2.yaml"),
    split: list[str] = typer.Option([], help="Splits to score (default: eval.splits)."),
    seed: list[int] = typer.Option([], help="Seeds (default: the config's seeds)."),
    set_: list[str] = typer.Option([], "--set", help="Config override key=value."),
    smoke: bool = typer.Option(False, help="Score 2 chips per split, console logging only."),
    out_dir: Path = typer.Option(None, help="Default: docs/results/level1"),
) -> None:
    """Score an experiment and write its results file."""
    cfg = load_config(config, set_)
    splits = split or cfg["eval"]["splits"]
    seeds = seed or cfg.get("seeds", [0])
    variants = list(cfg.get("variants", {"default": None}))
    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    results = []
    for s in seeds:
        seed_everything(s)
        with start_run(cfg, s, job_type="eval", smoke=smoke) as run:
            for variant in variants:
                predictor = make_predictor(cfg, s, None if variant == "default" else variant)
                for sp in splits:
                    start = time.perf_counter()
                    scorer = score(
                        predictor,
                        sp,
                        threshold,
                        limit=2 if smoke else 0,
                        skip_blank=cfg.get("eval", {}).get("skip_blank", False),
                    )
                    prefix = f"{sp}/" if variant == "default" else f"{sp}/{variant}/"
                    run.log_summary(scorer.flat(prefix=prefix))
                    summary = scorer.summary()
                    results.append({"variant": variant, "seed": s, "split": sp, **summary})
                    o = summary["overall"]
                    log.info(
                        "%s %s seed %d %s: IoU %.4f F1 %.4f P %.4f R %.4f (%.0f s)",
                        cfg["experiment"],
                        variant,
                        s,
                        sp,
                        o["iou"],
                        o["f1"],
                        o["precision"],
                        o["recall"],
                        time.perf_counter() - start,
                    )
    payload = {
        "experiment": cfg["experiment"],
        "description": cfg.get("description", ""),
        "method": cfg["method"],
        "threshold": threshold,
        "git_commit": git_info()["git_commit"],
        "smoke": smoke,
        "summary": summarise(results),
        "results": results,
    }
    if out_dir is None:  # smoke results never land in docs/
        out_dir = repo_root() / ("outputs/smoke" if smoke else "docs/results/level1")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["experiment"] + ("-smoke" if smoke else "")
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    app()
