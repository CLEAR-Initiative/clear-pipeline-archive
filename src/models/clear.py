"""Pydantic models for CLEAR API GraphQL mutation inputs and responses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


# Canonical sector list for `crises.needs.sector`. Source of truth — the
# prompt template references this same tuple so the LLM produces matching
# keys, and the validator below rejects anything else.
NEEDS_SECTORS: tuple[str, ...] = (
    "Shelter",
    "WASH",
    "Protection",
    "Health",
    "Food Security",
    "Education",
)

# SAF (NRC Situation Analysis Framework) Dimension 6 — Humanitarian
# Conditions severity scale. Ordered low → high. Used as the `severity`
# field on each SectorAnalysis entry.
SAF_SEVERITY_LEVELS = Literal[
    "Minimal",
    "Stressed",
    "Severe",
    "Extreme",
    "Catastrophic",
]


class CreateSignalInput(BaseModel):
    sourceId: str
    rawData: dict
    publishedAt: str  # ISO-8601
    collectedAt: str | None = None
    url: str | None = None
    title: str | None = None
    description: str | None = None
    originId: str | None = None
    destinationId: str | None = None
    locationId: str | None = None
    lat: float | None = None  # For server-side PostGIS geo-resolution
    lng: float | None = None


class CreateEventInput(BaseModel):
    signalIds: list[str]
    title: str | None = None
    description: str | None = None
    descriptionSignals: dict | None = None
    validFrom: str  # ISO-8601
    validTo: str  # ISO-8601
    firstSignalCreatedAt: str  # ISO-8601
    lastSignalCreatedAt: str  # ISO-8601
    originId: str | None = None
    destinationId: str | None = None
    locationId: str | None = None
    types: list[str]
    populationAffected: str | None = None
    rank: float
    lat: float | None = None  # For server-side PostGIS geo-resolution
    lng: float | None = None


class CreateAlertInput(BaseModel):
    eventId: str
    status: str | None = "published"


class SignalClassification(BaseModel):
    """Output from Claude signal classification."""

    disaster_types: list[str]  # glide numbers e.g. ["fl", "ff"]
    relevance: float  # 0.0-1.0
    severity: int  # 1-5
    summary: str


class EventGroupingResult(BaseModel):
    """Output from Claude event grouping."""

    action: str  # "create_new" or "add_to_existing"
    event_id: str | None = None  # if add_to_existing
    title: str | None = None  # for both actions (updated title when adding to existing)
    description: str | None = None  # for both actions (updated description when adding to existing)
    types: list[str] | None = None  # if create_new
    population_affected: int | None = None  # extracted from signal text


class AlertAssessment(BaseModel):
    """Output from Claude alert assessment."""

    should_alert: bool
    status: str = "published"  # "draft" or "published"


class CrisisNarrative(BaseModel):
    """Output from Claude crisis narrative generation.

    `summary` on the crisis row is the JSON-serialised form of
    `{description, tldr}` so frontends can render the prose and the bullet
    list independently without re-deriving one from the other.
    """

    title: str
    description: str
    tldr: list[str]


class CrisisScenarios(BaseModel):
    """Output from Claude crisis-scenarios generation.

    Stored verbatim on `crises.scenarios` (JSONB). Each field is a paragraph
    of prose — `description` covers the scenario variables (political,
    economic, environmental, etc.), the others are the forward trajectories.
    """

    most_likely: str
    best_case: str
    worst_case: str
    description: str


class SectorAnalysis(BaseModel):
    """One per-sector entry inside `crises.needs.sector`.

    Required fields are the ones every UI consumer expects to render
    uniformly across sectors. Anything beyond them (indicator percentages,
    recommended response type, cluster actors, etc.) is allowed via
    `extra='allow'` so the schema can grow without a Pydantic change.

    Fields:
      - description: prose explanation (2-3 sentences).
      - severity: SAF Dimension 6 classification — one of the five levels.
      - responseGap: True when no cluster actor is present in 3W data for
        this sector (a gap that NRC or others may need to fill). False when
        the cluster is covered.
      - nrcRelevant: True when NRC has a core competency for this sector
        (Shelter, WASH, Education, ICLA, LFS — the org's mandate).
    """

    model_config = ConfigDict(extra="allow")

    description: str
    severity: SAF_SEVERITY_LEVELS
    responseGap: bool
    nrcRelevant: bool


class CrisisNeedsAnalysis(BaseModel):
    """Output from Claude needs-analysis generation (NRC SAF framework).

    Top-level `generalSummary` (overall narrative) + `sector` (per-sector
    breakdown keyed by canonical NRC sector names — see `NEEDS_SECTORS`).
    Stored under `crises.needs.{generalSummary, sector}` via a JSONB merge
    so other keys the user supplied at creation time stay intact.

    Sector keys are validated against `NEEDS_SECTORS` — unknown keys are
    rejected (no hallucinated sectors), but the LLM may legitimately omit
    sectors that are clearly Minimal or out-of-scope for a given crisis.
    """

    generalSummary: str
    sector: dict[str, SectorAnalysis]

    @field_validator("sector")
    @classmethod
    def _validate_sector_keys(
        cls, v: dict[str, SectorAnalysis],
    ) -> dict[str, SectorAnalysis]:
        unknown = set(v.keys()) - set(NEEDS_SECTORS)
        if unknown:
            raise ValueError(
                f"Unknown sector keys: {sorted(unknown)}; expected subset of {NEEDS_SECTORS}"
            )
        return v


class EventRewrite(BaseModel):
    """Output from Claude event rewrite. Used by the new district+type grouping
    algorithm, where Claude no longer makes clustering decisions — only polishes
    the human-facing text and provides severity / displacement fallbacks."""

    title: str
    description: str
    """Severity fallback (1-5). Only consulted when at least one signal lacks
    a source-provided severity value. Null means Claude couldn't judge."""
    severity: int | None = None
    """Population displaced (primary source: signal text). Null means no signal
    mentioned a displacement count."""
    population_displaced: int | None = None
