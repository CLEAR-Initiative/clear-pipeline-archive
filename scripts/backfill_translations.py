"""
One-shot translation backfill.

Walks every (or up to --limit) row of the given entity type and runs
translate_and_upsert on each — the staleness diff inside it means
re-runs on already-current rows are free (no Claude call, no DB write).

Designed for:
  - Initial cutover ("translate every existing crisis / location once").
  - Catch-up after a TARGET_LOCALES expansion ("we just turned on fa,
    backfill so existing rows don't show fallback English forever").
  - Periodic safety net via cron, if you don't trust the per-entity
    hooks to cover edge cases.

Usage:
    python scripts/backfill_translations.py --entity-type location
    python scripts/backfill_translations.py --entity-type crisis --limit 50
    python scripts/backfill_translations.py --entity-type event --limit 200 --dry-run
    python scripts/backfill_translations.py --entity-type event --async
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clients import graphql  # noqa: E402
from src.services.translate import (  # noqa: E402
    configured_target_locales,
    translate_and_upsert,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Iterators per entity type ────────────────────────────────────────────────
# Each returns a list of {id, ...canonical_fields} for translate_and_upsert.
# We use list queries that exist today; if the volume ever grows past
# what a single GraphQL response can return, swap to a paginated query.


def _iter_locations(limit: int | None) -> list[dict]:
    """Every admin location at every level the API exposes. Iterates
    coarsest-first (country → state → district → sub-district → landmark)
    so user-facing translations appear earliest for coarse-grained views
    while a long backfill drains; L3/L4 still get translated because the
    lazy-on-read loader keeps re-enqueueing them otherwise (every
    /detection list view references locations at every level via event
    associations)."""
    rows: list[dict] = []
    for level in (0, 1, 2, 3, 4):
        for loc in graphql.get_locations_by_level(level):
            rows.append({"id": loc["id"], "name": loc.get("name")})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def _iter_events(limit: int | None) -> list[dict]:
    """Events list from clear-api. The list endpoint already excludes
    dummies and respects team scoping for the pipeline user."""
    events = graphql.get_events()
    rows = [
        {
            "id": e["id"],
            "title": e.get("title"),
            "description": e.get("description"),
        }
        for e in events
    ]
    return rows if limit is None else rows[:limit]


def _iter_crises(limit: int | None) -> list[dict]:
    """Crises don't have a list endpoint in the pipeline client yet, so
    fetch via raw GraphQL. Skipped via a no-op if you call this without
    target locales configured."""
    query = """
      query CrisesForBackfill {
        crises { id title summary scenarios needs }
      }
    """
    data = graphql._execute(query)  # type: ignore[attr-defined]
    crises = data.get("crises") or []
    rows = [
        {
            "id": c["id"],
            "title": c.get("title"),
            "summary": c.get("summary"),
            "scenarios": c.get("scenarios"),
            "needs": c.get("needs"),
        }
        for c in crises
    ]
    return rows if limit is None else rows[:limit]


ITERATORS = {
    "location": _iter_locations,
    "event": _iter_events,
    "crisis": _iter_crises,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill translations for every entity of one type.",
    )
    parser.add_argument(
        "--entity-type",
        required=True,
        choices=sorted(ITERATORS),
        help="Which entity type to backfill.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many rows to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the entity list and report counts, no Claude calls.",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Enqueue translate_entity_task per row instead of running "
             "synchronously. Useful for large event backfills so the script "
             "returns quickly and Celery workers spread the load.",
    )
    args = parser.parse_args()

    target_locales = configured_target_locales()
    if not target_locales:
        logger.error(
            "TARGET_LOCALES is empty — translation is disabled. Set it in "
            ".env before running the backfill.",
        )
        sys.exit(2)

    logger.info(
        "Backfill starting: entity_type=%s target_locales=%s limit=%s dry_run=%s async=%s",
        args.entity_type, target_locales, args.limit, args.dry_run, args.async_mode,
    )

    iterator = ITERATORS[args.entity_type]
    rows = iterator(args.limit)
    total = len(rows)
    logger.info("Found %d %s row(s) to process.", total, args.entity_type)

    # Prune entities that already have rows for every target locale.
    # Without this, the worker would still run translate_and_upsert on
    # each one, immediately hit the staleness diff, log "all current",
    # and return — pure broker + DB-read overhead. Union the per-locale
    # missing sets so an entity with `ar` but missing `fr` still gets
    # through.
    missing_union: set[str] = set()
    for locale in target_locales:
        missing_union.update(
            graphql.get_entities_missing_translation(args.entity_type, locale)
        )
    pre_filter_total = total
    rows = [r for r in rows if r["id"] in missing_union]
    skipped_by_filter = pre_filter_total - len(rows)
    logger.info(
        "Pre-filter: %d %s row(s) already covered for all target locale(s) — %d remain.",
        skipped_by_filter, args.entity_type, len(rows),
    )
    total = len(rows)

    if args.dry_run:
        logger.info("[DRY-RUN] First 5 ids: %s", [r["id"] for r in rows[:5]])
        return

    if args.async_mode:
        # Local import — only the async path needs Celery imported.
        from src.tasks.translate import translate_entity_task

        for r in rows:
            translate_entity_task.delay(args.entity_type, r["id"])
        logger.info("Enqueued %d translate_entity_task(s). Check Celery worker for progress.", total)
        return

    # Synchronous path — runs Claude inline. Slower but easier to observe
    # for small entity sets (locations, crises).
    written = 0
    skipped = 0
    failed = 0
    started = time.monotonic()
    for i, row in enumerate(rows, start=1):
        try:
            summary = translate_and_upsert(args.entity_type, row["id"], row)
            if summary is None:
                skipped += 1
            else:
                written += 1
        except Exception as exc:
            failed += 1
            logger.error(
                "[%d/%d] %s %s failed: %s",
                i, total, args.entity_type, row["id"], exc,
            )
            continue

        if i % 25 == 0 or i == total:
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] processed (rate=%.1f/s written=%d skipped=%d failed=%d)",
                i, total, rate, written, skipped, failed,
            )

    logger.info(
        "Done: total=%d written=%d skipped=%d failed=%d",
        total, written, skipped, failed,
    )


if __name__ == "__main__":
    main()
