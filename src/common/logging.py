"""Experiment tracking: Weights & Biases when online, MLflow in ``./mlruns`` when offline.

Usage in every run (after ``seed_everything``)::

    with start_run(cfg, seed=seed, job_type="train") as run:
        run.log_metrics({"train/loss": loss}, step=epoch)
        run.log_summary(scorer.flat(prefix="val/"))
        run.log_file("outputs/feature_importance.csv")

Every run records the resolved config, experiment ID, seed, run name (``E4-s1``), git commit
(and whether the tree had uncommitted changes), GPU name and memory, effective batch size;
on exit it records the runtime in minutes, the peak GPU memory and the final status.

Backend choice: the ``backend`` argument, else env ``FLOOD_TRACKER`` (``wandb``, ``mlflow`` or
``console``), else automatic: W&B if installed, logged in and reachable, otherwise MLflow.
``--smoke`` runs use the console backend unless a backend is given explicitly.

MLflow 3 refuses the old plain-folder store, so offline runs go to an SQLite database plus
artifacts inside ``mlruns/`` at the repo root. View them with::

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

Note: this module is ``src.common.logging``; inside the project, ``import logging`` still gives
the standard library. Never run a file in ``src/common`` directly as a script.
"""

from __future__ import annotations

import logging as std_logging
import math
import netrc
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

from src.common import gpu
from src.common.config import effective_batch_size, repo_root, run_name, save_config

WANDB_PROJECT = "flood-rescue"
MLFLOW_EXPERIMENT = "flood-rescue"
TRACKER_ENV = "FLOOD_TRACKER"
BACKENDS = ("wandb", "mlflow", "console")

_log = std_logging.getLogger("flood")


def get_logger(name: str = "flood") -> std_logging.Logger:
    """Return a console logger with timestamps (configures the root handler once)."""
    root = std_logging.getLogger()
    if not root.handlers:
        std_logging.basicConfig(
            level=std_logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return std_logging.getLogger(name)


# --------------------------------------------------------------------------- metadata


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Return ``git_commit``, ``git_branch`` and ``git_dirty`` (``unknown``/None without git)."""
    repo = repo or repo_root()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": None if status is None else bool(status),
    }


def flatten(data: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts: ``{"train": {"lr": 1e-3}}`` -> ``{"train.lr": 1e-3}``."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, full, sep))
        else:
            out[full] = value
    return out


def run_metadata(cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Everything a run must record besides its config and metrics."""
    experiment = str(cfg.get("experiment", "unnamed"))
    meta: dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "run_name": run_name(experiment, seed),
        "effective_batch_size": effective_batch_size(cfg),
        "python_version": platform.python_version(),
        "platform": platform.platform(terse=True),
    }
    meta.update(git_info())
    meta.update(gpu.gpu_info())
    return meta


# --------------------------------------------------------------------------- runs


class Run:
    """Base class for a tracked run; use it as a context manager.

    Subclasses implement ``_log_metrics``, ``_log_summary``, ``_log_file`` and ``_close``.
    """

    backend = "console"

    def __init__(self, name: str, meta: dict[str, Any]) -> None:
        self.name = name
        self.meta = meta
        self.id: str = name
        self._start = time.perf_counter()
        self._closed = False
        gpu.reset_peak_memory()

    # public API --------------------------------------------------------------
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log scalar metrics, e.g. per epoch with ``step=epoch``."""
        self._log_metrics(_scalars(metrics), step)

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        """Record final values (e.g. best validation IoU, test scores)."""
        self._log_summary(_scalars(metrics))

    def log_file(self, path: str | os.PathLike[str]) -> None:
        """Attach a small file (CSV, PNG, YAML) to the run."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._log_file(path)

    def resume_info(self) -> dict[str, str]:
        """Store this in the checkpoint; pass it to ``start_run(resume=...)`` after ``--resume``."""
        return {"backend": self.backend, "id": self.id}

    def finish(self, status: str = "ok") -> None:
        """Log runtime, peak GPU memory and status, then close the run. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        end = {
            "runtime_min": round((time.perf_counter() - self._start) / 60, 3),
            "gpu_peak_memory_gb": gpu.peak_memory_gb(),
        }
        self._log_summary(_scalars(end))
        _log.info("run %s finished (%s): %s", self.name, status, end)
        self._close(status)

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            status = "ok"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        self.finish(status)

    # backend hooks -------------------------------------------------------------
    def _log_metrics(self, metrics: dict[str, float], step: int | None) -> None:
        at = "" if step is None else f" step {step}"
        _log.info("[%s]%s %s", self.name, at, _fmt(metrics))

    def _log_summary(self, metrics: dict[str, float]) -> None:
        _log.info("[%s] summary %s", self.name, _fmt(metrics))

    def _log_file(self, path: Path) -> None:
        _log.info("[%s] file %s", self.name, path)

    def _close(self, status: str) -> None:
        pass


class ConsoleRun(Run):
    """Prints everything to the terminal; nothing is stored. Used for ``--smoke`` and tests."""

    backend = "console"


class WandbRun(Run):
    """Weights & Biases run in project ``flood-rescue``; seeds of one experiment share a group."""

    backend = "wandb"

    def __init__(
        self,
        name: str,
        meta: dict[str, Any],
        cfg: Mapping[str, Any],
        job_type: str,
        tags: Iterable[str],
        resume_id: str | None,
    ) -> None:
        import wandb

        super().__init__(name, meta)
        self._wandb = wandb
        self._run = wandb.init(
            project=WANDB_PROJECT,
            name=name,
            group=meta["experiment"],
            job_type=job_type,
            tags=list(tags),
            config={**dict(cfg), "meta": meta},
            id=resume_id,
            resume="allow" if resume_id else None,
            settings=wandb.Settings(init_timeout=60),
        )
        self.id = self._run.id

    def _log_metrics(self, metrics: dict[str, float], step: int | None) -> None:
        self._run.log(metrics, step=step)

    def _log_summary(self, metrics: dict[str, float]) -> None:
        self._run.summary.update(metrics)

    def _log_file(self, path: Path) -> None:
        self._run.save(str(path), base_path=str(path.parent), policy="now")

    def _close(self, status: str) -> None:
        self._run.summary["status"] = status
        self._run.finish(exit_code=0 if status == "ok" else 1)


class MlflowRun(Run):
    """MLflow run stored in ``mlruns/`` (SQLite database + artifacts folder) at the repo root."""

    backend = "mlflow"

    def __init__(
        self,
        name: str,
        meta: dict[str, Any],
        cfg: Mapping[str, Any],
        job_type: str,
        tags: Iterable[str],
        resume_id: str | None,
        root: Path | None = None,
    ) -> None:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        super().__init__(name, meta)
        self._mlflow = mlflow
        root = (root or repo_root() / "mlruns").resolve()
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or mlflow_uri(root))
        if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT, artifact_location=(root / "artifacts").as_uri()
            )
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_tags = {"experiment": meta["experiment"], "job_type": job_type}
        if tags:
            run_tags["tags"] = ",".join(tags)
        self._run = mlflow.start_run(
            run_id=resume_id, run_name=None if resume_id else name, tags=run_tags
        )
        self.id = self._run.info.run_id
        if resume_id is None:
            params = {k: _param(v) for k, v in flatten({**dict(cfg), "meta": meta}).items()}
            items = list(params.items())
            for i in range(0, len(items), 100):  # MLflow accepts at most 100 params per call
                mlflow.log_params(dict(items[i : i + 100]))
            mlflow.log_dict(dict(cfg), "config.yaml")

    def _log_metrics(self, metrics: dict[str, float], step: int | None) -> None:
        self._mlflow.log_metrics(_mlflow_keys(metrics), step=step)

    def _log_summary(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(_mlflow_keys(metrics))

    def _log_file(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def _close(self, status: str) -> None:
        self._mlflow.set_tag("status", status)
        mlflow_status = {"ok": "FINISHED", "interrupted": "KILLED"}.get(status, "FAILED")
        self._mlflow.end_run(status=mlflow_status)


def mlflow_uri(root: Path | None = None) -> str:
    """Tracking URI of the local offline store: ``sqlite:///<repo>/mlruns/mlflow.db``."""
    root = (root or repo_root() / "mlruns").resolve()
    return f"sqlite:///{root / 'mlflow.db'}"


# --------------------------------------------------------------------------- backend choice


def _wandb_has_key() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = urllib.parse.urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        return netrc.netrc().authenticators(host or "api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
    except urllib.error.HTTPError:
        return True  # the server answered
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


def wandb_status() -> tuple[bool, str]:
    """Return ``(usable, reason)`` for Weights & Biases on this machine right now."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False, "wandb is not installed"
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in {"offline", "disabled", "dryrun"}:
        return False, f"WANDB_MODE={mode}"
    if not _wandb_has_key():
        return False, "not logged in to W&B (run `wandb login` once)"
    url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    if not _reachable(url):
        return False, f"no internet ({url} unreachable)"
    return True, "ok"


def choose_backend(backend: str | None = None, smoke: bool = False) -> str:
    """Resolve the backend name: argument > ``$FLOOD_TRACKER`` > smoke/console > auto."""
    choice = (backend or os.environ.get(TRACKER_ENV, "") or "").strip().lower()
    if choice:
        if choice not in BACKENDS:
            raise ValueError(f"unknown tracker {choice!r}; choose one of {BACKENDS}")
        return choice
    if smoke:
        return "console"
    ok, reason = wandb_status()
    if ok:
        return "wandb"
    _log.warning("W&B not used (%s); logging to MLflow in mlruns/", reason)
    return "mlflow"


def start_run(
    cfg: Mapping[str, Any],
    seed: int,
    *,
    job_type: str = "train",
    tags: Iterable[str] = (),
    backend: str | None = None,
    smoke: bool = False,
    resume: Mapping[str, str] | None = None,
    save_config_to: str | os.PathLike[str] | None = None,
) -> Run:
    """Start a tracked run named ``<experiment>-s<seed>``.

    Args:
        cfg: The resolved config from ``load_config`` (must contain ``experiment``).
        seed: The run seed.
        job_type: ``train``, ``eval``, ``predict``... (W&B job type, MLflow tag).
        tags: Extra free-form tags.
        backend: Force ``wandb``, ``mlflow`` or ``console``.
        smoke: True for ``--smoke`` runs (console unless ``backend`` is given).
        resume: ``run.resume_info()`` saved in the checkpoint, to continue the same run.
        save_config_to: Also write the resolved config YAML here (e.g. the checkpoint folder).
    """
    get_logger()
    meta = run_metadata(cfg, seed)
    name = meta["run_name"] + ("-smoke" if smoke else "")
    if save_config_to is not None:
        save_config({**dict(cfg), "meta": meta}, save_config_to)
    explicit = backend or os.environ.get(TRACKER_ENV)
    chosen = choose_backend(backend, smoke)
    resume_id = None
    if resume and resume.get("backend") == chosen:
        resume_id = resume.get("id")
    elif resume:
        _log.warning(
            "cannot resume a %s run with %s; starting a new run", resume.get("backend"), chosen
        )

    _log.info(
        "run %s: tracker=%s gpu=%s commit=%s%s",
        name,
        chosen,
        gpu.describe(meta),
        meta["git_commit"][:10],
        " (uncommitted changes!)" if meta["git_dirty"] else "",
    )
    if chosen == "wandb":
        try:
            return WandbRun(name, meta, cfg, job_type, tags, resume_id)
        except Exception as err:  # network drop or login problem: fall back, never lose a run
            if explicit:
                raise
            _log.warning("W&B start failed (%s); falling back to MLflow in mlruns/", err)
            chosen, resume_id = "mlflow", None
    if chosen == "mlflow":
        return MlflowRun(name, meta, cfg, job_type, tags, resume_id)
    run = ConsoleRun(name, meta)
    _log.info("[%s] config %s", name, flatten(dict(cfg)))
    return run


# --------------------------------------------------------------------------- helpers


def _scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep numeric values only (bools and ints become floats; None is dropped)."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, str):
            continue
        if hasattr(value, "item"):
            value = value.item()
        out[str(key)] = float(value)
    return out


def _fmt(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{k}={'nan' if math.isnan(v) else f'{v:.4g}'}" for k, v in metrics.items())


_MLFLOW_BAD_CHARS = re.compile(r"[^\w\-./ :]")


def _mlflow_keys(metrics: Mapping[str, float]) -> dict[str, float]:
    return {_MLFLOW_BAD_CHARS.sub("_", k): v for k, v in metrics.items()}


def _param(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 6000 else text[:5997] + "..."
