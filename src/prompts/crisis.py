"""Crisis narrative prompt: generate a coherent title + summary across events."""

# Bump whenever the prompt text changes (see CLASSIFY_PROMPT_VERSION for rationale).
# v2: replaced flat `summary` with structured `description` + `tldr[3]`. The
# pipeline stringifies `{description, tldr}` into the crises.summary column.
CRISIS_PROMPT_VERSION = "crisis-v2"

SYSTEM_PROMPT = """\
You are a humanitarian intelligence analyst for the CLEAR early warning system.

You write concise, actionable narratives for humanitarian workers and NGOs
operating in crisis zones. Your summaries connect multiple events into a
single coherent crisis so responders can act quickly.

You MUST respond with valid JSON only — no markdown, no explanation before or after."""


USER_PROMPT_TEMPLATE = """\
Generate a title, description, and tldr for a humanitarian crisis linking the events below.

Events ({event_count}):
{events_block}

Locations affected: {locations}

Guidelines:
- Title: <=70 chars, human-readable, no emojis, no brackets/quotes. Lead with the
  dominant disaster type(s) and location (e.g. "Floods in North Darfur and Kassala").
- Description: 2-3 sentences (paragraph form). Describe what is happening, where,
  scale (population affected if known), and the humanitarian implication
  (displacement, food security, health risk, etc.). Avoid generic filler.
- TLDR: exactly three one-liner bullet points that together summarise the full
  description. Each bullet is a single short sentence (<=20 words), no leading
  dashes or bullets, no markdown. The three together should cover: (1) what
  happened, (2) where and at what scale, (3) the humanitarian implication.

Respond with this exact JSON structure:
{{
  "title": "<short descriptive title>",
  "description": "<2-3 sentence narrative paragraph>",
  "tldr": [
    "<bullet 1: what happened>",
    "<bullet 2: where and scale>",
    "<bullet 3: humanitarian implication>"
  ]
}}
"""


def build_crisis_prompt(events: list[dict], locations: list[str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        event_count=len(events),
        events_block=_format_events_block(events),
        locations=", ".join(locations) if locations else "unknown",
    )


def _format_events_block(events: list[dict]) -> str:
    """Render the events list into the prose chunk both prompts embed."""
    lines = []
    for i, e in enumerate(events, 1):
        title = e.get("title") or "(untitled)"
        desc = (e.get("description") or "").strip()
        types = ", ".join(e.get("types") or []) or "unknown"
        severity = e.get("severity") if e.get("severity") is not None else "?"
        pop = e.get("populationAffected")
        pop_str = f" pop_affected={pop}" if pop else ""
        lines.append(
            f"{i}. [{types}] severity={severity}{pop_str}\n"
            f"   title: {title}\n"
            f"   description: {desc[:300]}"
        )
    return "\n".join(lines) if lines else "(no events)"


# ─── Scenarios ────────────────────────────────────────────────────────────
# Forward-looking analysis. Stored verbatim on `crises.scenarios` (JSONB).
# Bump this version whenever the prompt text or output schema changes.

SCENARIOS_PROMPT_VERSION = "crisis-scenarios-v1"

SCENARIOS_SYSTEM_PROMPT = """\
You are a humanitarian intelligence analyst for the CLEAR early warning system.

You produce forward-looking scenario analyses for ongoing crises — the kind
NGOs and field operators read to anticipate how a situation may evolve over
the next weeks and months. Be specific. Tie each scenario to the events
provided.

You MUST respond with valid JSON only — no markdown, no explanation before or after."""


SCENARIOS_USER_PROMPT_TEMPLATE = """\
Develop scenarios for how the situation may evolve.

Events ({event_count}):
{events_block}

Locations affected: {locations}

Most Likely Scenario:
- What is the most probable trajectory for the crisis?
- How are humanitarian conditions likely to evolve?
- What factors support this scenario?

Alternative Scenarios:
- What other plausible scenarios exist (best case, worst case)?
- What would trigger these alternative scenarios?
- What is the likelihood of each scenario?

Scenario Variables — consider how these factors may change:
- Political and security dynamics
- Economic conditions
- Environmental/seasonal factors (harvest, rainy season, etc.)
- Disease outbreaks or public health events
- Policy changes
- Humanitarian access
- Response capacity and funding
- Population movements

Each scenario must be 2-4 sentences of concrete prose grounded in the events
above — no generic filler, no markdown bullets inside the strings, no emojis.

Respond with this exact JSON structure:
{{
  "most_likely": "<2-4 sentences on the most probable trajectory, with the factors that support it>",
  "best_case":   "<2-4 sentences on the best plausible outcome and what would trigger it>",
  "worst_case":  "<2-4 sentences on the worst plausible outcome and what would trigger it>",
  "description": "<2-4 sentences summarising the scenario variables (political, economic, environmental, public-health, policy, access, response capacity, population movement) and which way each is currently trending>"
}}
"""


def build_scenarios_prompt(events: list[dict], locations: list[str]) -> str:
    """Render the user-prompt for the scenarios call. Shares the same event
    formatter as the narrative prompt so context is consistent across both."""
    return SCENARIOS_USER_PROMPT_TEMPLATE.format(
        event_count=len(events),
        events_block=_format_events_block(events),
        locations=", ".join(locations) if locations else "unknown",
    )


# ─── Needs clarification (NRC SAF) ────────────────────────────────────────
# The LLM applies the NRC Situation Analysis Framework (Dimensions 6 + 7)
# to MSNA + OCHA 3W data and produces four labelled bullet points. Stored
# inside the existing `crises.needs` JSONB under the `clarification` key.

CLARIFICATION_PROMPT_VERSION = "crisis-clarification-v1"

CLARIFICATION_SYSTEM_PROMPT = """\
You are an emergency response analyst applying the NRC Situation Analysis Framework (SAF). You are given MSNA indicator data (household survey, 8 months old) and OCHA 3W partner presence data for a specific locality in Sudan. Your task is to produce a structured assessment for an Emergency Response Manager deciding whether to deploy a response, run a Rapid Needs Assessment, or monitor.

Apply the SAF Humanitarian Conditions framework (Dimension 6) to assess sector severity, and the SAF Priority Needs framework (Dimension 7) to synthesise across sectors.

You MUST respond with valid JSON only — no markdown, no explanation before or after."""


CLARIFICATION_USER_PROMPT_TEMPLATE = """\
Produce exactly 4 bullet points. Each bullet must be one sentence. Use these labels in order:

- Severity — Classify overall humanitarian conditions using the SAF five-level scale (Minimal / Stressed / Severe / Extreme / Catastrophic). State which sectors drive the classification and your confidence level (High / Medium / Low) based on data age and completeness. If confidence is Low, say so explicitly.
- Drivers — Identify 1–2 causal factors linking the crisis context to these conditions. Do not just restate the indicators. Explain why conditions are what they are (e.g. "displacement has severed access to markets, compounding food insecurity that pre-dates the conflict").
- Response gaps — Identify sectors where conditions are Severe or above AND no cluster actor is present in 3W data. State clearly whether this represents an unmet need or whether 3W absence may reflect reporting lag. Note if NRC has a relevant core competency.
- Priority action — Using SAF Dimension 7 criteria, state whether the evidence supports: immediate life-saving response, stabilisation response, assessment-first (RNA), or monitoring. Be explicit about what would change this recommendation (e.g. "RNA to verify school closure type and current water quality would shift this from assessment to response").

Important:
- Do not reference composite scores or numeric indices.
- Use actual indicator percentages from the data when available.
- Distinguish between what the data shows (observed) and what you are inferring (analytical judgment).
- Flag if data age (8 months) materially limits your confidence in any dimension.
- Return only 4 bullet points, each starting with a dash (-).

Context — events ({event_count}):
{events_block}

Locations affected: {locations}

Locality data (MSNA indicators, OCHA 3W partner presence, other available
location metadata; may be sparse — flag this in your confidence rating):
{locality_data_block}

Respond with this exact JSON structure:
{{
  "clarification": "- Severity — ...\\n- Drivers — ...\\n- Response gaps — ...\\n- Priority action — ..."
}}
"""


def _format_locality_data_block(events: list[dict]) -> str:
    """Format whatever location metadata the events carry into a single
    text block the LLM can read. Falls back to an explicit 'no metadata'
    marker when nothing is available so the LLM downgrades confidence
    instead of fabricating indicator values.
    """
    chunks: list[str] = []
    seen_locations: set[str] = set()
    for event in events:
        for key in ("generalLocation", "originLocation", "destinationLocation"):
            loc = event.get(key)
            if not loc:
                continue
            loc_name = loc.get("name")
            if not loc_name or loc_name in seen_locations:
                continue
            seen_locations.add(loc_name)
            metadata = loc.get("metadata") or []
            if not metadata:
                continue
            chunks.append(f"Location: {loc_name}")
            for entry in metadata:
                meta_type = entry.get("type", "unknown")
                meta_data = entry.get("data") or {}
                chunks.append(f"  [{meta_type}] {meta_data}")
    return "\n".join(chunks) if chunks else "(no MSNA / 3W / locality metadata available for these locations)"


def build_clarification_prompt(events: list[dict], locations: list[str]) -> str:
    """Render the user-prompt for the SAF clarification call. Pulls
    locality metadata off the events themselves — the caller is expected
    to have fetched events with their location.metadata."""
    return CLARIFICATION_USER_PROMPT_TEMPLATE.format(
        event_count=len(events),
        events_block=_format_events_block(events),
        locations=", ".join(locations) if locations else "unknown",
        locality_data_block=_format_locality_data_block(events),
    )
