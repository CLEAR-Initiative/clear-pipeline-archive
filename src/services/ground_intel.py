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
- Threading (which messages describe the same incident) is semantic → one
  Claude call over the un-threaded field reports, but every proposal is
  validated deterministically (known ids only, each message in exactly one
  thread) and the lifecycle state is derived by rules that the model can
  inform but not overrule.
"""

from __future__ import annotations

import logging
import re

from src.clients.claude import call_claude
from src.models.ground import (
    GROUND_CLASSIFICATIONS,
    GROUND_LIFECYCLE_STATES,
    GroundClassificationResponse,
    GroundThreadingResponse,
)
from src.prompts.ground import (
    GROUND_CLASSIFY_PROMPT_VERSION,
    GROUND_CLASSIFY_SYSTEM,
    GROUND_THREAD_PROMPT_VERSION,
    GROUND_THREAD_SYSTEM,
    build_ground_classify_prompt,
    build_ground_thread_prompt,
)

logger = logging.getLogger(__name__)

# Messages per Claude classification call. Output is ~1 short JSON row per
# message, so 50 messages stay comfortably inside max_tokens=4096.
CLASSIFY_CHUNK_SIZE = 50

# Cap on field reports considered for threading in one run. Threading needs
# every candidate in a single prompt (clusters cross chunk boundaries); the
# oldest N are threaded first and the rest wait for the next run.
THREAD_BATCH_LIMIT = 150

# ─── Uncertainty markers (deterministic) ───────────────────────────────────
# Ordered weakest-credibility first; the first pattern that matches wins, so
# a message carrying several markers ("rumour only, no confirmation") keeps
# the most cautious one. The canonical marker string (left) is what gets
# persisted — the PRD requires the contributor's own uncertainty tag to
# survive ingestion, normalised to a stable vocabulary.
_UNCERTAINTY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rumour", re.compile(r"\brumou?rs?\b", re.IGNORECASE)),
    ("unverified", re.compile(r"\bunverified\b", re.IGNORECASE)),
    (
        "unconfirmed",
        re.compile(
            r"\bunconfirmed\b|\bnot\s+(?:yet\s+)?confirmed\b|\bno\s+confirmation\b",
            re.IGNORECASE,
        ),
    ),
    ("alleged", re.compile(r"\ballegedl?y?\b", re.IGNORECASE)),
]

# ─── Retractions (deterministic override) ──────────────────────────────────
# "This turned out to be misreporting — no strikes on Galaxy" must flip its
# thread to `retracted` even if the model calls it something milder. The
# withdrawal of a report is too consequential to leave to model judgement
# alone; these phrases are the explicit ways contributors withdraw reports.
_RETRACTION_PATTERN = re.compile(
    r"\bmisreport(?:ing|ed)?\b"
    r"|\bturned\s+out\s+to\s+be\s+(?:false|wrong|untrue|incorrect)\b"
    r"|\bretract(?:ed|ing|ion)?\b"
    r"|\bfalse\s+alarm\b"
    r"|\bdid\s+not\s+(?:happen|take\s+place|occur)\b",
    re.IGNORECASE,
)


def is_retraction(text: str | None) -> bool:
    """True when `text` explicitly withdraws an earlier report."""
    return bool(text and _RETRACTION_PATTERN.search(text))


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


# ─── Incident threading (LLM proposes, rules validate) ─────────────────────


def derive_lifecycle_state(llm_state: str | None, messages: list[dict]) -> str:
    """Deterministic lifecycle derivation for a proposed thread.

    The model's judgement is used where the semantics genuinely need it
    (confirmed vs corrected vs updated), but rules set the floor:
      - a thread containing an explicit retraction is "retracted",
        whatever the model proposed
      - a single-message thread can only ever be "reported"
      - an out-of-vocabulary state falls back to reported/updated by size
    """
    if any(is_retraction(m.get("text")) for m in messages):
        return "retracted"
    if len(messages) <= 1:
        return "reported"
    if llm_state in GROUND_LIFECYCLE_STATES:
        return llm_state
    return "updated"


def _fallback_title(messages: list[dict]) -> str:
    first_text = (messages[0].get("text") or "").strip() if messages else ""
    return first_text[:80] or "Untitled incident"


def build_threads(ground_source_id: str, messages: list[dict]) -> list[dict]:
    """Cluster un-threaded field reports into incident threads via Claude.

    `messages` is the full fetched set; candidates are the field_report
    messages not yet attached to a thread. Returns upsertGroundThreads
    inputs: [{groundSourceId, title, lifecycleState, messageIds}].

    Model proposals are validated deterministically:
      - unknown message ids are dropped
      - a message claimed by several threads stays with the first
      - candidate messages the model did not place are left un-threaded
        (they get another chance next run)
      - lifecycle state goes through derive_lifecycle_state()

    Raises whatever call_claude raises — the Celery task owns retries.
    """
    candidates = [
        m
        for m in messages
        if m.get("classification") == "field_report" and not m.get("threadId")
    ]
    candidates.sort(key=lambda m: m.get("sentAt") or "")
    candidates = candidates[:THREAD_BATCH_LIMIT]
    if not candidates:
        return []

    parsed = call_claude(
        GROUND_THREAD_SYSTEM,
        build_ground_thread_prompt(candidates),
        stage="ground_thread",
        prompt_version=GROUND_THREAD_PROMPT_VERSION,
        max_tokens=4096,
    )
    response = GroundThreadingResponse.model_validate(parsed)

    by_id = {m["id"]: m for m in candidates}
    claimed: set[str] = set()
    threads: list[dict] = []

    for proposal in response.threads:
        member_ids = []
        for message_id in proposal.message_ids:
            if message_id not in by_id:
                logger.warning(
                    "[GROUND] Threading proposal referenced unknown message %r — dropped.",
                    message_id,
                )
                continue
            if message_id in claimed:
                logger.warning(
                    "[GROUND] Message %s claimed by more than one thread — "
                    "keeping first assignment.",
                    message_id,
                )
                continue
            member_ids.append(message_id)

        if not member_ids:
            continue
        claimed.update(member_ids)

        members = [by_id[i] for i in member_ids]
        threads.append(
            {
                "groundSourceId": ground_source_id,
                "title": proposal.title.strip() or _fallback_title(members),
                "lifecycleState": derive_lifecycle_state(proposal.lifecycle_state, members),
                "messageIds": member_ids,
            }
        )

    unplaced = len(candidates) - len(claimed)
    if unplaced:
        logger.info(
            "[GROUND] %d field report(s) not placed in any thread this run.", unplaced
        )
    return threads
