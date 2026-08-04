"""Pydantic models for the ground-intel (WhatsApp signal pipeline) stages."""

from pydantic import BaseModel

# The four ground-message classes. Only field_report messages feed incident
# threading; the rest are triage output for the review UI.
GROUND_CLASSIFICATIONS = frozenset(
    {"field_report", "news_digest", "operational", "chatter"}
)


class GroundMessageLabel(BaseModel):
    """One message's classification, as returned by Claude."""

    id: str
    classification: str


class GroundClassificationResponse(BaseModel):
    """Output from the ground_classify Claude stage."""

    classifications: list[GroundMessageLabel]
