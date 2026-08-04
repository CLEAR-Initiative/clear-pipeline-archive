"""Ground-intel intelligence: classify WhatsApp ground messages.

Part of the WhatsApp Signal Pipeline. clear-api owns the data (the
groundSources / groundThreads / groundMessages staging tier); this module
owns the intelligence, mirroring the split used for Dataminr signals.

Design follows the event_grouping_v2 precedent: deterministic where
possible, LLM only where semantics are genuinely needed.

- Uncertainty markers ("unconfirmed", "rumour", …) are detected with
  regexes — contributors' own uncertainty tags must survive ingestion
  verbatim, so no model judgement is involved.
- Classification (field_report / news_digest / operational / chatter) is
  semantic → one batched Claude call per chunk of messages.
"""

from __future__ import annotations

import logging
import re

from src.clients.claude import call_claude
from src.models.ground import GROUND_CLASSIFICATIONS, GroundClassificationResponse
from src.prompts.ground import (
    GROUND_CLASSIFY_PROMPT_VERSION,
    GROUND_CLASSIFY_SYSTEM,
    build_ground_classify_prompt,
)

logger = logging.getLogger(__name__)

# Messages per Claude classification call. Output is ~1 short JSON row per
# message, so 50 messages stay comfortably inside max_tokens=4096.
CLASSIFY_CHUNK_SIZE = 50

# ─── Uncertainty markers (deterministic) ───────────────────────────────────
# Ordered: the first pattern that matches wins. The canonical marker string
# (left) is what gets persisted — the PRD requires the contributor's own
# uncertainty tag to survive ingestion, normalised to a stable vocabulary.
_UNCERTAINTY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "unconfirmed",
        re.compile(
            r"\bunconfirmed\b|\bnot\s+(?:yet\s+)?confirmed\b|\bno\s+confirmation\b",
            re.IGNORECASE,
        ),
    ),
    ("rumour", re.compile(r"\brumou?rs?\b", re.IGNORECASE)),
    ("unverified", re.compile(r"\bunverified\b", re.IGNORECASE)),
    ("alleged", re.compile(r"\ballegedl?y?\b", re.IGNORECASE)),
]


def detect_uncertainty_marker(text: str | None) -> str | None:
    """Return the canonical uncertainty marker carried by `text`, or None.

    Deterministic by design: a contributor tagging their own report as
    "unconfirmed" or "rumour" is ground truth about the report's status,
    not something to re-derive with a model.
    """
    if not text:
        return None
    for marker, pattern in _UNCERTAINTY_PATTERNS:
        if pattern.search(text):
            return marker
    return None


# ─── Classification (LLM) ──────────────────────────────────────────────────


def classify_messages(messages: list[dict]) -> list[dict]:
    """Classify ground messages via Claude, in chunks.

    `messages` are groundMessage dicts ({id, text, sentAt, senderRef,
    hasMedia, …}). Returns upsertGroundMessageClassifications inputs:
    [{messageId, classification, uncertaintyMarker}].

    Messages for which Claude returned no label or an invalid label are
    OMITTED from the result — they stay unclassified in clear-api and get
    retried on the next run, rather than being written back with a guessed
    class (misfiling a field report as chatter loses intelligence).

    Raises whatever call_claude raises (ClaudeRateLimited, APIStatusError,
    JSONDecodeError) — the Celery task owns retry semantics.
    """
    results: list[dict] = []
    for start in range(0, len(messages), CLASSIFY_CHUNK_SIZE):
        chunk = messages[start : start + CLASSIFY_CHUNK_SIZE]
        parsed = call_claude(
            GROUND_CLASSIFY_SYSTEM,
            build_ground_classify_prompt(chunk),
            stage="ground_classify",
            prompt_version=GROUND_CLASSIFY_PROMPT_VERSION,
            max_tokens=4096,
        )
        response = GroundClassificationResponse.model_validate(parsed)
        label_by_id = {row.id: row.classification for row in response.classifications}

        for message in chunk:
            label = label_by_id.get(message["id"])
            if label not in GROUND_CLASSIFICATIONS:
                logger.warning(
                    "[GROUND] Message %s got unusable classification %r — "
                    "leaving unclassified for a later run.",
                    message["id"], label,
                )
                continue
            results.append(
                {
                    "messageId": message["id"],
                    "classification": label,
                    "uncertaintyMarker": detect_uncertainty_marker(message.get("text")),
                }
            )
    return results
