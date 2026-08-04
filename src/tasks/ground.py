"""Celery task: classify a ground source's WhatsApp messages.

Part of the WhatsApp Signal Pipeline (V1 seeded shared core). clear-api
imports a chat export into its groundMessages staging tier and enqueues
this task; the pipeline owns the intelligence and writes results back
over GraphQL, mirroring the Dataminr split.

PIPELINE CONTRACT (consumed by clear-api's celery service):
  Task name:  "classify_ground_messages"
  Signature:  (ground_source_id: str)

  The task processes ALL unclassified groundMessages for that source:
  each is classified as field_report | news_digest | operational |
  chatter, and contributor uncertainty markers ("unconfirmed", "rumour",
  …) are detected and preserved.
"""

import logging

from src.celery_app import app
from src.clients.claude import ClaudeRateLimited
from src.clients.graphql import (
    GraphQLClientError,
    ground_messages_for_classification,
    upsert_ground_message_classifications,
)
from src.services import ground_intel

logger = logging.getLogger(__name__)

# Upper bound on messages fetched per run. A full seed import (the HSS DAO
# export is ~620 messages) drains over a handful of runs; live capture
# volumes sit far below this.
DEFAULT_MESSAGE_LIMIT = 500


@app.task(name="classify_ground_messages", bind=True, max_retries=3)
def classify_ground_messages(self, ground_source_id: str, limit: int = DEFAULT_MESSAGE_LIMIT):
    """Classify unclassified ground messages for one ground source.

    Registered under the bare name "classify_ground_messages" (not the
    module-path convention) because clear-api enqueues it by that exact
    name — see the PIPELINE CONTRACT in this module's docstring.
    """
    try:
        messages = ground_messages_for_classification(ground_source_id, limit)
        logger.info(
            "[GROUND] Source %s: fetched %d messages for classification",
            ground_source_id, len(messages),
        )

        unclassified = [m for m in messages if not m.get("classification")]

        classified_count = 0
        if unclassified:
            labels = ground_intel.classify_messages(unclassified)
            if labels:
                upsert_ground_message_classifications(labels)
            classified_count = len(labels)
            logger.info(
                "[GROUND] Source %s: classified %d/%d messages",
                ground_source_id, classified_count, len(unclassified),
            )

        return {
            "ground_source_id": ground_source_id,
            "messages_fetched": len(messages),
            "messages_classified": classified_count,
        }

    except GraphQLClientError as exc:
        logger.error(
            "[GROUND] classify_ground_messages permanently failed (non-retryable): %s", exc,
        )
        raise
    except ClaudeRateLimited as exc:
        logger.warning(
            "[GROUND] Rate-limited; retrying source %s in %.0fs",
            ground_source_id, exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except Exception as exc:
        logger.error(
            "[GROUND] classify_ground_messages failed for source %s: %s",
            ground_source_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=60)
