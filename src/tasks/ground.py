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
  …) are detected and preserved. Field reports are then clustered into
  incident threads with lifecycle states (reported | updated | confirmed
  | corrected | retracted) via upsertGroundThreads. Threading is
  cross-run aware: the source's existing threads (groundThreadsForSource)
  are offered as append targets, and a continuation — e.g. a retraction
  arriving a run after its incident — is upserted with threadId set so
  the server appends to the original thread.
"""

import logging

from src.celery_app import app
from src.clients.claude import ClaudeRateLimited
from src.clients.graphql import (
    GraphQLClientError,
    ground_messages_for_classification,
    ground_threads_for_source,
    upsert_ground_message_classifications,
    upsert_ground_threads,
)
from src.services import ground_intel

logger = logging.getLogger(__name__)

# Upper bound on messages fetched per run. A typical seed export runs
# ~500-700 messages and drains over a handful of runs; live capture
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
        labels: list[dict] = []
        if unclassified:
            labels = ground_intel.classify_messages(unclassified)
            if labels:
                upsert_ground_message_classifications(labels)
            classified_count = len(labels)
            logger.info(
                "[GROUND] Source %s: classified %d/%d messages",
                ground_source_id, classified_count, len(unclassified),
            )

        # Overlay the labels we just wrote so threading sees the up-to-date
        # classification without a second fetch.
        label_by_id = {row["messageId"]: row["classification"] for row in labels}
        merged = [
            {**m, "classification": m.get("classification") or label_by_id.get(m["id"])}
            for m in messages
        ]

        # Cross-run threading: existing threads are offered as append
        # targets so a correction/retraction landing in a later run (or the
        # tail of an incident straddling two fetch windows) joins its
        # original thread instead of minting an orphan. Only fetched when
        # there is actually something to thread.
        has_candidates = any(
            m.get("classification") == "field_report" and not m.get("threadId")
            for m in merged
        )
        existing_threads = (
            ground_threads_for_source(ground_source_id) if has_candidates else []
        )

        thread_inputs = ground_intel.build_threads(
            ground_source_id, merged, existing_threads
        )
        thread_rows = upsert_ground_threads(thread_inputs) if thread_inputs else []
        if thread_inputs:
            appended = sum(1 for t in thread_inputs if t.get("threadId"))
            logger.info(
                "[GROUND] Source %s: upserted %d incident thread(s) (%d appended to existing)",
                ground_source_id, len(thread_inputs), appended,
            )

        return {
            "ground_source_id": ground_source_id,
            "messages_fetched": len(messages),
            "messages_classified": classified_count,
            "threads_upserted": len(thread_rows),
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
