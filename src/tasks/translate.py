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
    name="src.tasks.translate.translate_entity_task",
    bind=True,
    max_retries=2,
    acks_late=True,
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
