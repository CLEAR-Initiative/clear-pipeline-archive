"""
Crisis enrichment Celery task.

Runs after a crisis is created or an event is added to it. Populates:
  - populationInArea (sum of admin-level-2 populations for the event districts)
  - title + summary (Claude-generated narrative from the linked events)

The `summary` field stored on the crisis is the JSON-serialised form of
`{description, tldr}` — see `CrisisNarrative` for the schema. The column on
the database stays a plain string; consumers JSON.parse it.

Both outputs are written back in a single updateCrisisPopulation mutation
so the crisis record is always consistent.
"""

import json
import logging

from src.celery_app import app
from src.clients import graphql
from src.clients.claude import ClaudeRateLimited, call_claude
from src.models.clear import (
    CrisisNarrative,
    CrisisNeedsClarification,
    CrisisScenarios,
)
from src.prompts.crisis import (
    CLARIFICATION_PROMPT_VERSION,
    CLARIFICATION_SYSTEM_PROMPT,
    CRISIS_PROMPT_VERSION,
    SCENARIOS_PROMPT_VERSION,
    SCENARIOS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_clarification_prompt,
    build_crisis_prompt,
    build_scenarios_prompt,
)
from src.services.population import estimate_population_for_districts

logger = logging.getLogger(__name__)


def _geometry_is_areal(geometry: dict | None) -> bool:
    """Only Polygon/MultiPolygon geometries can be raster-masked meaningfully.
    Point locations (level 4) produce near-zero population and should fall back."""
    if not geometry:
        return False
    return geometry.get("type") in ("Polygon", "MultiPolygon")


def _resolve_location_for_population(loc: dict) -> dict | None:
    """Return a location dict that has either a cached population OR an areal
    geometry. If the given location is a point (or has no geometry and no
    cached population), walk up to its parent. Returns None if no usable
    ancestor is found."""
    current = loc
    while current is not None:
        has_cached = current.get("population") is not None
        has_areal = _geometry_is_areal(current.get("geometry"))
        if has_cached or has_areal:
            return current

        parent_stub = current.get("parent")
        if not parent_stub:
            return None
        logger.info(
            "[CRISIS] Location %s (%s, level=%s) has no cached population or "
            "areal geometry — falling back to parent %s",
            current.get("name"), current.get("id"), current.get("level"),
            parent_stub.get("name"),
        )
        current = graphql.get_location_with_geometry(parent_stub["id"])
    return None


def _compute_population_in_area(district_ids: list[str]) -> int | None:
    """Sum cached location.population; fall back to raster for missing areals,
    and fall back to parent location when a district is a point or has no
    usable geometry.

    De-duplicates by resolved location ID so shared parents aren't summed twice.
    """
    if not district_ids:
        return None

    resolved_by_id: dict[str, dict] = {}
    for did in district_ids:
        loc = graphql.get_location_with_geometry(did)
        if not loc:
            logger.warning("[CRISIS] District %s not found", did)
            continue

        resolved = _resolve_location_for_population(loc)
        if not resolved:
            logger.warning(
                "[CRISIS] No usable ancestor for district %s (%s)",
                loc.get("name"), did,
            )
            continue

        # De-duplicate: if two districts resolved to the same state, only count once
        resolved_by_id[resolved["id"]] = resolved

    if not resolved_by_id:
        logger.warning("[CRISIS] No usable locations resolved")
        return None

    cached_total = 0
    missing_geometries: list[dict] = []
    for loc in resolved_by_id.values():
        pop_str = loc.get("population")
        if pop_str is not None:
            cached_total += int(pop_str)
        elif _geometry_is_areal(loc.get("geometry")):
            missing_geometries.append(loc["geometry"])

    if not missing_geometries:
        logger.info(
            "[CRISIS] All %d resolved locations cached: populationInArea=%d",
            len(resolved_by_id), cached_total,
        )
        return cached_total

    raster_pop = estimate_population_for_districts(missing_geometries) or 0
    total = cached_total + raster_pop
    logger.info(
        "[CRISIS] Mixed (%d resolved): cached=%d raster=%d → populationInArea=%d",
        len(resolved_by_id), cached_total, raster_pop, total,
    )
    return total


def _collect_location_names(events: list[dict]) -> list[str]:
    """Distinct origin/destination/general location names across the events.
    Order-stable so the prompt is reproducible across runs with the same input."""
    locations: list[str] = []
    seen: set[str] = set()
    for e in events:
        for key in ("originLocation", "destinationLocation", "generalLocation"):
            loc = e.get(key)
            if loc and loc.get("name") and loc["name"] not in seen:
                locations.append(loc["name"])
                seen.add(loc["name"])
    return locations


def _generate_needs_clarification(events: list[dict]) -> str | None:
    """Generate an NRC SAF-framework clarification for a crisis via Claude.

    Returns the clarification as a single string of four dash-prefixed
    bullet points (Severity / Drivers / Response gaps / Priority action).
    Stored under `crises.needs.clarification` (merged into the existing
    `needs` JSONB by the GraphQL mutation, leaving other keys untouched).
    Returns None on any failure — clarification is best-effort enrichment.
    """
    if not events:
        return None

    locations = _collect_location_names(events)
    prompt = build_clarification_prompt(events, locations)

    try:
        result_data = call_claude(
            CLARIFICATION_SYSTEM_PROMPT,
            prompt,
            stage="crisis-clarification",
            prompt_version=CLARIFICATION_PROMPT_VERSION,
        )
        parsed = CrisisNeedsClarification.model_validate(result_data)
        return parsed.clarification
    except Exception as e:
        logger.error("[CRISIS] Clarification generation failed: %s", e, exc_info=True)
        return None


def _generate_scenarios(events: list[dict]) -> dict | None:
    """Generate forward-looking scenarios for a crisis via Claude.

    Returns a dict with keys {most_likely, best_case, worst_case, description},
    suitable for storing directly on `crises.scenarios` (JSONB). Returns None
    on any failure — scenarios are best-effort enrichment.
    """
    if not events:
        return None

    locations = _collect_location_names(events)
    prompt = build_scenarios_prompt(events, locations)

    try:
        result_data = call_claude(
            SCENARIOS_SYSTEM_PROMPT,
            prompt,
            stage="crisis-scenarios",
            prompt_version=SCENARIOS_PROMPT_VERSION,
        )
        scenarios = CrisisScenarios.model_validate(result_data)
        return scenarios.model_dump()
    except Exception as e:
        logger.error("[CRISIS] Scenarios generation failed: %s", e, exc_info=True)
        return None


def _generate_narrative(events: list[dict]) -> tuple[str, str] | None:
    """Generate (title, summary) for a crisis via Claude.

    Returns:
      - title: short headline
      - summary: JSON-serialised `{description, tldr}` — the database column
        stays a string; UI consumers JSON.parse on read.
    """
    if not events:
        return None

    locations = _collect_location_names(events)
    prompt = build_crisis_prompt(events, locations)

    try:
        result_data = call_claude(
            SYSTEM_PROMPT,
            prompt,
            stage="crisis",
            prompt_version=CRISIS_PROMPT_VERSION,
        )
        narrative = CrisisNarrative.model_validate(result_data)
        summary_json = json.dumps(
            {"description": narrative.description, "tldr": narrative.tldr},
            ensure_ascii=False,
        )
        return narrative.title, summary_json
    except Exception as e:
        logger.error("[CRISIS] Narrative generation failed: %s", e, exc_info=True)
        return None


@app.task(
    name="src.tasks.crisis.enrich_crisis",
    bind=True,
    max_retries=2,
    acks_late=True,
)
def enrich_crisis(
    self,
    crisis_id: str,
    event_ids: list[str],
    district_ids: list[str],
    generate_narrative: bool = True,
) -> dict:
    """Compute populationInArea + (optional) title/summary, write back in one mutation."""
    logger.info(
        "[CRISIS] enrich_crisis: crisis=%s events=%d districts=%d narrative=%s",
        crisis_id, len(event_ids), len(district_ids), generate_narrative,
    )

    try:
        population_in_area = _compute_population_in_area(district_ids)

        title: str | None = None
        summary: str | None = None
        scenarios: dict | None = None
        clarification: str | None = None
        if generate_narrative and event_ids:
            # Fetch full event details once; all three Claude calls reuse them.
            events: list[dict] = []
            for eid in event_ids:
                e = graphql.get_event_for_crisis(eid)
                if e:
                    events.append(e)

            result = _generate_narrative(events)
            if result:
                title, summary = result
                logger.info("[CRISIS] Narrative: title=%r", title)

            scenarios = _generate_scenarios(events)
            if scenarios:
                logger.info(
                    "[CRISIS] Scenarios generated (most_likely=%d chars)",
                    len(scenarios.get("most_likely", "")),
                )

            clarification = _generate_needs_clarification(events)
            if clarification:
                logger.info(
                    "[CRISIS] Clarification generated (%d chars)",
                    len(clarification),
                )

        # Population + narrative + scenarios in one mutation; clarification
        # merges into `needs` via a dedicated mutation so we don't clobber
        # other keys already on the JSONB object.
        graphql.update_crisis_population(
            crisis_id,
            population_in_area=population_in_area,
            title=title,
            summary=summary,
            scenarios=scenarios,
        )
        if clarification:
            graphql.update_crisis_needs_clarification(crisis_id, clarification)

        return {
            "crisis_id": crisis_id,
            "population_in_area": population_in_area,
            "title": title,
            "summary": summary,
            "scenarios": scenarios,
            "clarification": clarification,
        }

    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] enrich_crisis backing off %.0fs",
            exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except graphql.GraphQLClientError as exc:
        logger.error(
            "[CRISIS] enrich_crisis %s permanently failed (non-retryable): %s",
            crisis_id, exc,
        )
        raise
    except Exception as exc:
        logger.error(
            "[CRISIS] enrich_crisis failed for %s: %s",
            crisis_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=30)
