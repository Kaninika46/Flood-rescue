"""Long jobs (prediction, routing updates) run by an RQ worker.

    rq worker --url $REDIS_URL flood          # from the repo root, in the "flood" env

With ``JOB_MODE=sync`` (tests, laptops without Redis) jobs run in the request instead.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from backend.app.db import get_engine
from backend.app.models import Scene
from backend.app.settings import get_settings

QUEUE = "flood"


def process_scene(scene_id: int) -> str:
    """Run the whole event pipeline for a scene and store the results."""
    from backend.app.services import store_event
    from src.common.config import data_path
    from src.event_pipeline import run_event

    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    with factory() as session:
        scene = session.get(Scene, scene_id)
        scene.state = "processing"
        session.commit()
        try:
            run = run_event(Path(scene.s1_path), data_path("events", f"scene_{scene_id}"))
            store_event(session, scene, run)
        except Exception as err:  # store the failure for the dashboard, then re-raise
            scene.state, scene.error = "failed", f"{err}\n{traceback.format_exc()[-2000:]}"
            session.commit()
            raise
    return "done"


def enqueue_scene(scene_id: int) -> str:
    """Queue ``process_scene`` (RQ) or run it now (``JOB_MODE=sync``); returns a job id."""
    settings = get_settings()
    if settings.job_mode == "sync":
        process_scene(scene_id)
        return f"sync-{scene_id}"
    from redis import Redis
    from rq import Queue

    queue = Queue(QUEUE, connection=Redis.from_url(settings.redis_url))
    job = queue.enqueue(process_scene, scene_id, job_timeout=6 * 3600)
    return job.id
