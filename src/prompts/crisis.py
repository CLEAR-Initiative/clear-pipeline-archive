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
    events_block = "\n".join(lines) if lines else "(no events)"
    locations_str = ", ".join(locations) if locations else "unknown"

    return USER_PROMPT_TEMPLATE.format(
        event_count=len(events),
        events_block=events_block,
        locations=locations_str,
    )
