"""
Celery task that translates a single entity by id. Triggered from:
  - clear-api's resolvers via the lazy-on-read enqueue path
    (src/context.ts pushes the task directly onto the Redis broker)
  - scripts/backfill_translations.py (one-shot backfill, optional --async)

The task is intentionally idempotent: the staleness diff inside
translate_and_upsert means re-running it on a current entity is a no-op,
so duplicate enqueues are cheap.
"""

from __future__ import annotations

import logging

import anthropic

from src.celery_app import app
from src.clients import graphql
from src.clients.claude import ClaudeRateLimited
from src.services.translate import translate_and_upsert

logger = logging.getLogger(__name__)

_CANONICAL_GETTERS = {
    "event":    graphql.get_event_canonical,
    "crisis":   graphql.get_crisis_canonical,
    "location": graphql.get_location_canonical,
}


@app.task(
    name="src.tasks.translate.translate_entities_batch_task",
    bind=True,
    acks_late=True,
    # Hard ceiling on wasted work when a Claude call spins in the SDK's
    # exponential-backoff retry loop. Soft raises SoftTimeLimitExceeded
    # into the task at 4 min; hard SIGKILLs at 5 min. The per-entity
    # dedup lock in translate_and_upsert has a 6-min TTL so a killed
    # worker's stale lock expires shortly after the SIGKILL.
    soft_time_limit=240,
    time_limit=300,
)
def translate_entities_batch_task(self, items: list[dict]) -> dict:
    """Translate a batch of entities in one task call.

    Replaces N independent translate_entity_task enqueues from clear-api's
    lazy-on-read path. clear-api buffers misses for ~500ms and pushes the
    accumulated list as a single broker message; this task fans them out
    in-process so a 200-miss /detection load hits the broker once
    instead of 200 times.

    Per-item failures are isolated: a missing canonical row or a Claude
    rejection on one item doesn't drop the rest. Returns a summary
    that's visible in worker logs for diagnosing partial failures.
    """
    succeeded: list[str] = []
    failed: list[dict] = []
    for item in items or []:
        try:
            entity_type = str(item.get("entity_type", "")).lower()
            entity_id = item.get("entity_id")
            if not entity_type or not entity_id:
                failed.append({
                    "entity_id": entity_id,
                    "reason": "missing_entity_type_or_id",
                })
                continue
            getter = _CANONICAL_GETTERS.get(entity_type)
            if getter is None:
                failed.append({
                    "entity_id": entity_id,
                    "reason": f"unknown_entity_type:{entity_type}",
                })
                continue
            try:
                canonical = getter(entity_id)
            except graphql.GraphQLClientError as exc:
                failed.append({
                    "entity_id": entity_id,
                    "reason": f"canonical_fetch_failed:{exc}",
                })
                continue
            if not canonical:
                failed.append({
                    "entity_id": entity_id,
                    "reason": "canonical_not_found",
                })
                continue
            try:
                translate_and_upsert(entity_type, entity_id, canonical)
                succeeded.append(entity_id)
            except (ClaudeRateLimited, anthropic.APIStatusError) as exc:
                # Transient Claude errors aren't retried at the batch
                # level — re-enqueueing the whole batch would over-
                # translate the items that did succeed. The lazy-on-read
                # path will re-surface the still-missing item on its
                # next user request.
                failed.append({
                    "entity_id": entity_id,
                    "reason": f"claude_transient:{type(exc).__name__}",
                })
            except Exception as exc:
                logger.error(
                    "[TRANSLATE-BATCH] %s %s failed: %s",
                    entity_type, entity_id, exc, exc_info=True,
                )
                failed.append({
                    "entity_id": entity_id,
                    "reason": f"error:{exc}",
                })
        except Exception as exc:
            # Defensive — never let one malformed item crash the batch.
            logger.error(
                "[TRANSLATE-BATCH] unexpected error on item %r: %s",
                item, exc, exc_info=True,
            )
            failed.append({"reason": f"unexpected:{exc}"})

    logger.info(
        "[TRANSLATE-BATCH] processed %d item(s): %d ok, %d failed",
        len(items or []), len(succeeded), len(failed),
    )
    return {"succeeded": succeeded, "failed": failed}


@app.task(
    name="src.tasks.translate.translate_entity_task",
    bind=True,
    max_retries=2,
    acks_late=True,
    # Same rationale as translate_entities_batch_task — a single-entity
    # task shouldn't spin longer than ~5 min through SDK retries.
    soft_time_limit=240,
    time_limit=300,
)
def translate_entity_task(self, entity_type: str, entity_id: str) -> dict | None:
    """Translate one entity by id. Fire-and-forget — caller doesn't
    await the result.

    Transient Anthropic errors retry with the same backoff the crisis
    task uses (~120s for overload, retry_after for rate-limits). Any
    other failure is logged and dropped — translation is non-critical
    and the next enrichment pass will pick it up.
    """
    entity_type = entity_type.lower()
    getter = _CANONICAL_GETTERS.get(entity_type)
    if getter is None:
        logger.error(
            "[TRANSLATE-TASK] unknown entity_type=%r (expected event|crisis|location)",
            entity_type,
        )
        return None

    try:
        canonical = getter(entity_id)
    except graphql.GraphQLClientError as exc:
        logger.error(
            "[TRANSLATE-TASK] failed to fetch canonical %s %s: %s",
            entity_type, entity_id, exc,
        )
        return None

    if not canonical:
        logger.warning(
            "[TRANSLATE-TASK] canonical %s %s not found — dropping",
            entity_type, entity_id,
        )
        return None

    try:
        return translate_and_upsert(entity_type, entity_id, canonical)
    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] translate %s %s backing off %.0fs",
            entity_type, entity_id, exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except anthropic.APIStatusError as exc:
        logger.warning(
            "[TRANSLATE-TASK] Anthropic %s on %s %s — retrying in 120s",
            type(exc).__name__, entity_type, entity_id,
        )
        raise self.retry(exc=exc, countdown=120)
    except Exception as exc:
        logger.error(
            "[TRANSLATE-TASK] %s %s failed: %s",
            entity_type, entity_id, exc, exc_info=True,
        )
        # Don't retry on unknown errors — would just rack up the same
        # failure. The next periodic enrichment will get another shot.
        return None
