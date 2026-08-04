"""Pydantic models for the ground-intel (WhatsApp signal pipeline) stages."""

from pydantic import BaseModel

# The four ground-message classes. Only field_report messages feed incident
# threading; the rest are triage output for the review UI.
GROUND_CLASSIFICATIONS = frozenset(
    {"field_report", "news_digest", "operational", "chatter"}
)


# Incident-thread lifecycle. Observed in the field groups the PRD profiled:
# reported → updated → confirmed / corrected / retracted.
GROUND_LIFECYCLE_STATES = frozenset(
    {"reported", "updated", "confirmed", "corrected", "retracted"}
)


class GroundMessageLabel(BaseModel):
    """One message's classification, as returned by Claude."""

    id: str
    classification: str


class GroundClassificationResponse(BaseModel):
    """Output from the ground_classify Claude stage."""

    classifications: list[GroundMessageLabel]


class GroundThreadProposal(BaseModel):
    """One proposed incident thread, as returned by Claude. Validated and
    lifecycle-checked deterministically before anything is written back.

    `thread_id` set means the proposal APPENDS its message_ids to that
    existing thread (cross-run threading: a correction or retraction that
    arrives after its incident was threaded in an earlier run) instead of
    creating a new one. It must match an offered existing thread id —
    unknown ids are ignored and the proposal becomes a new thread."""

    title: str = ""
    lifecycle_state: str = ""
    message_ids: list[str]
    thread_id: str | None = None


class GroundThreadingResponse(BaseModel):
    """Output from the ground_thread Claude stage."""

    threads: list[GroundThreadProposal]
