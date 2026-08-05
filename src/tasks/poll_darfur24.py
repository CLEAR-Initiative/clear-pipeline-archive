"""Celery task: poll darfur24.com for new articles and ingest them as signals."""

import logging
from datetime import UTC, datetime

from src.celery_app import app
from src.clients.darfur24 import fetch_darfur24_articles, get_last_synced, mark_seen
from src.clients.graphql import (
    GraphQLClientError,
    create_signal,
    get_data_sources,
    get_locations_by_level,
)

logger = logging.getLogger(__name__)

_darfur24_source_id: str | None = None
_darfur24_location_id: str | None = None


def _get_darfur24_source_id() -> str:
    """Get the darfur24 data source ID from the CLEAR API (cached)."""
    global _darfur24_source_id
    if _darfur24_source_id is not None:
        return _darfur24_source_id

    from src.config import settings
    sources = get_data_sources()
    for src in sources:
        if src["name"] == settings.darfur24_source_name:
            _darfur24_source_id = src["id"]
            return _darfur24_source_id
    raise RuntimeError(
        f"Data source '{settings.darfur24_source_name}' not found in CLEAR API. "
        "Ensure it exists in the data_sources table."
    )


def _get_darfur24_location_id() -> str | None:
    """Resolve `darfur24_default_country`'s L0 location id (cached).

    Best-effort by design: returns None — after logging a warning — when the
    lookup fails or the country has no level-0 row. Signal creation must not
    fail because the location could not be resolved; a location-less signal
    is still recoverable via updateSignalLocation, a dropped article is not
    (expo-385). Only a successful resolution is cached, so a transient API
    error is retried on the next poll round.
    """
    global _darfur24_location_id
    if _darfur24_location_id is not None:
        return _darfur24_location_id

    from src.config import settings
    try:
        for loc in get_locations_by_level(0):
            if loc["name"] == settings.darfur24_default_country:
                _darfur24_location_id = loc["id"]
                return _darfur24_location_id
        logger.warning(
            "[DARFUR24] No level-0 location named %r in CLEAR API; "
            "creating signals without location",
            settings.darfur24_default_country,
        )
    except Exception as e:
        logger.warning(
            "[DARFUR24] Failed to resolve level-0 location for %r: %s; "
            "creating signals without location",
            settings.darfur24_default_country,
            e,
        )
    return None


def _build_signal_input(article: dict, source_id: str, location_id: str | None) -> dict:
    """Convert a parsed darfur24 article into a CLEAR CreateSignalInput dict.

    News articles carry no structured casualty or coordinate data — unlike
    ACLED/GDACS we deliberately set none of those. Location is the country
    L0 (see `darfur24_default_country`); finer-grained resolution is the
    classification follow-up, not this tracer.
    """
    input_data = {
        "sourceId": source_id,
        # Dedup key — (sourceId, externalId) is unique in the CLEAR API, so
        # re-ingesting the same article (Redis seen-set expired, feed replay)
        # returns the existing row instead of creating a duplicate.
        "externalId": f"darfur24:{article['darfur24_id']}",
        "rawData": article["raw"],
        "publishedAt": article.get("published_at") or datetime.now(UTC).isoformat(),
        "url": article["url"],
        "title": article["title"],
        "description": article.get("description"),
        # Documented informational default for news-source signals: 1 is the
        # scale floor ("informational"), not an estimated threat level. A
        # null severity makes the signal vanish from any severity-filtered
        # view (the API's gte/lte range filter drops null rows) — expo-385.
        "severity": 1,
    }
    if location_id is not None:
        input_data["locationId"] = location_id
    return input_data


@app.task(name="src.tasks.poll_darfur24.poll_darfur24", bind=True, max_retries=3)
def poll_darfur24(self):
    """
    Poll darfur24.com's RSS feed for new articles.

    - Fetches the configured feeds (default: English edition)
    - Creates a CLEAR signal per new article with source attribution

    Phase-0 tracer: signals are created but NOT dispatched into the
    classify/group pipeline — ACLED/GDACS build their classifications from
    structured metadata that news articles don't have, and there is no
    generic news-classification path yet. Wiring darfur24 signals into
    classification/grouping is a follow-up.
    """
    try:
        last_synced = get_last_synced()
        if last_synced:
            logger.info("[DARFUR24] Polling; last_synced=%s", last_synced.isoformat())
        else:
            logger.info("[DARFUR24] Polling; no last_synced yet (first run)")

        articles = fetch_darfur24_articles()

        if not articles:
            logger.info("[DARFUR24] No new articles to ingest")
            return {"articles_found": 0, "signals_created": 0}

        source_id = _get_darfur24_source_id()
        location_id = _get_darfur24_location_id()  # None → no location (best-effort)
        logger.info(
            "[DARFUR24] Creating signals using source_id=%s location_id=%s",
            source_id,
            location_id,
        )

        created_count = 0
        failed_count = 0

        for article in articles:
            try:
                input_data = _build_signal_input(article, source_id, location_id)
                created = create_signal(input_data)
                logger.info(
                    "[DARFUR24] Signal created: id=%s title=%s",
                    created["id"],
                    article.get("title", "")[:80],
                )
                created_count += 1
                # Only now — after the API confirmed the signal (created, or
                # returned the existing row for a duplicate externalId) — is
                # the article marked seen in Redis. A failed creation leaves
                # it unmarked so the next poll retries it (expo-383).
                mark_seen(article["darfur24_id"])

            except Exception as e:
                failed_count += 1
                logger.error(
                    "[DARFUR24] Failed to ingest article %s: %s",
                    article.get("darfur24_id"),
                    e,
                    exc_info=True,
                )

        logger.info(
            "[DARFUR24] Poll complete: %d articles found → %d signals created (%d failed)",
            len(articles), created_count, failed_count,
        )
        return {
            "articles_found": len(articles),
            "signals_created": created_count,
            "failed": failed_count,
        }

    except GraphQLClientError as exc:
        logger.error("[DARFUR24] poll_darfur24 permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("[DARFUR24] poll_darfur24 failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=60)
