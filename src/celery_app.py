import importlib
import pkgutil

from celery import Celery
from celery.schedules import crontab, timedelta
from celery.signals import (
    after_setup_logger,
    after_setup_task_logger,
    worker_process_init,
)

from src.config import settings


@worker_process_init.connect
def _init_worker(**kwargs):
    """Initialise logging + Sentry in each forked worker process."""
    from src.logging_setup import setup_logging
    setup_logging()


# Celery overrides handlers after worker boot. These signals run AFTER
# Celery's setup, giving us a chance to (re)attach our stdout + Logtail
# handlers so task logger.info() calls from child workers are visible.
@after_setup_logger.connect
def _setup_root_logger(logger, **kwargs):
    from src.logging_setup import attach_handlers_to
    attach_handlers_to(logger)


@after_setup_task_logger.connect
def _setup_task_logger(logger, **kwargs):
    from src.logging_setup import attach_handlers_to
    attach_handlers_to(logger)


app = Celery("clear_pipeline", broker=settings.celery_broker_url)

app.conf.update(
    result_backend=settings.celery_broker_url,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Broker resilience. Hosted Redis (Render, Upstash) periodically
    # drops idle connections; the default config raises ConnectionError
    # straight out of the consumer loop and the worker dies until a
    # human restarts it (observed 2026-06-17). These knobs let kombu
    # retry on startup AND mid-run with exponential backoff, heartbeat
    # the connection so a half-open socket is detected before it's
    # used, and cancel any task that was in flight when the connection
    # dropped so it gets redelivered to a healthy consumer instead of
    # ack'd into the void.
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_heartbeat=30,
    worker_cancel_long_running_tasks_on_connection_loss=True,
)

app.conf.beat_schedule = {
    "poll-dataminr": {
        "task": "src.tasks.poll.poll_dataminr",
        "schedule": timedelta(seconds=settings.poll_interval_seconds),
    },
    "poll-gdacs": {
        "task": "src.tasks.poll_gdacs.poll_gdacs",
        "schedule": timedelta(minutes=settings.gdacs_poll_interval_minutes),
    },
    "poll-acled": {
        "task": "src.tasks.poll_acled.poll_acled",
        "schedule": timedelta(minutes=settings.acled_poll_interval_minutes),
    },
    "poll-darfur24": {
        "task": "src.tasks.poll_darfur24.poll_darfur24",
        "schedule": timedelta(minutes=settings.darfur24_poll_interval_minutes),
    },
    # Daily digest — every day at 07:00 UTC
    "daily-alert-digest": {
        "task": "src.tasks.notify.send_daily_digest",
        "schedule": crontab(hour=7, minute=0),
    },
    # Weekly digest — every Monday at 07:00 UTC
    "weekly-alert-digest": {
        "task": "src.tasks.notify.send_weekly_digest",
        "schedule": crontab(hour=7, minute=0, day_of_week=1),
    },
    # Monthly digest — 1st of each month at 07:00 UTC
    "monthly-alert-digest": {
        "task": "src.tasks.notify.send_monthly_digest",
        "schedule": crontab(hour=7, minute=0, day_of_month=1),
    },
    # Daily archival — 03:00 UTC, archive alerts whose event last saw
    # a signal more than 7 days ago (matches the v2 grouping window so an
    # event stops accepting new signals and gets archived on the same cadence).
    "archive-stale-alerts": {
        "task": "src.tasks.archive.archive_stale_alerts",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"older_than_days": 7},
    },
    # Weekly IOM DTM backfill — Mondays at 02:00 UTC. Refreshes
    # locationMetadata(type="iom_dtm_displacement") per admin-2.
    "backfill-dtm-displacement": {
        "task": "src.tasks.dtm.backfill_dtm_displacement",
        "schedule": crontab(hour=2, minute=0, day_of_week=1),
    },
    # Monthly LogIE roads & bridges refresh — 1st of month, 02:30 UTC.
    # Refreshes locationMetadata(type="logie_roads" / "logie_bridges") per
    # A0 country for settings.logistics_iso3 (SDN, AFG, VEN).
    "backfill-logistics-infrastructure": {
        "task": "src.tasks.logistics.backfill_logistics_infrastructure",
        "schedule": crontab(hour=2, minute=30, day_of_month=1),
    },
}

# ─── Task autodiscovery ─────────────────────────────────────────────────────
# Walk src/tasks/*.py and import each so every @app.task decorator
# registers with the global app at celery_app load time. Dropping a new
# file under src/tasks/ is now sufficient — no celery_app.py edit needed
# and no risk of unregistered tasks dropping messages on the floor
# (see the translate_entity_task incident on 2026-06-16: the new module
# was missing from the previous explicit include list, and clear-api's
# lazy-on-read enqueue queued messages the worker silently discarded).
#
# Why pkgutil instead of `app.autodiscover_tasks()` — Celery's built-in
# autodiscover expects each package to have a `tasks` submodule
# (`<pkg>.tasks`); our layout has tasks as direct siblings inside the
# `src.tasks` package, which doesn't fit that convention.
from src import tasks as _tasks_pkg  # noqa: E402  (import after app config)

for _module_info in pkgutil.iter_modules(_tasks_pkg.__path__):
    importlib.import_module(f"{_tasks_pkg.__name__}.{_module_info.name}")
