"""
Single-call translator. Asks Claude for every target locale at once so
an entity with N translatable fields × M target locales costs one API
call, not N×M. The model is told the canonical shape and instructed to
return JSON keyed by locale, mirroring the same shape per locale.

The caller picks which fields to translate via `fields_to_translate` —
the staleness check in tasks/crisis.py uses this to skip fields whose
canonical hash matches the stored source hash from the last run.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import anthropic

from src.clients import graphql
from src.clients.claude import ClaudeRateLimited, call_claude
from src.config import settings
from src.services.translation_hash import compute_source_hashes, stale_fields

logger = logging.getLogger(__name__)

# Built once per process — locale code → human-readable name used in the
# Claude prompt so the model knows what to translate into. We use the
# human name rather than the BCP-47 code to nudge the model toward the
# right variety (e.g. "Arabic — Modern Standard" instead of risking
# dialect drift).
LOCALE_LABELS: dict[str, str] = {
    "ar": "Arabic (Modern Standard, MSA)",
    "fr": "French",
}

TRANSLATION_PROMPT_VERSION = "v1"


def _system_prompt() -> str:
    return (
        "You are a professional translator for humanitarian crisis content "
        "produced by the Norwegian Refugee Council (NRC). You will be given "
        "a JSON object describing one entity (an event, a crisis, or an "
        "admin location) and asked to translate selected fields into one or "
        "more target languages.\n"
        "\n"
        "Rules:\n"
        "- Preserve every JSON key exactly. Only translate string values.\n"
        "- When a value is itself a JSON object or array, recurse: keep its "
        "  shape exactly and translate the string leaves.\n"
        "- Preserve technical terminology, NRC SAF sector names "
        "  (Shelter, WASH, Protection, Health, Food Security, Education), "
        "  glide codes, proper nouns, place names, dates, numbers, and "
        "  acronyms unchanged unless the locale has an established "
        "  convention (e.g. WHO → منظمة الصحة العالمية is acceptable).\n"
        "- Output VALID JSON only. No commentary, no markdown fences.\n"
        "- For each target locale, return an object whose keys are the "
        "  same field names you were asked to translate, with the "
        "  translated values (matching the canonical shape per field).\n"
        "- Top-level shape: {\"<locale>\": {<field>: <translated_value>, ...}, ...}"
    )


def _build_user_prompt(
    entity_type: str,
    canonical: dict[str, Any],
    target_locales: list[str],
    fields_to_translate: list[str],
) -> str:
    # Pick out only the fields we're asking the model to translate to keep
    # the input compact — the rest of the canonical row would be wasted
    # tokens. The model still gets the entity_type so it has context for
    # technical-term decisions.
    fields_payload = {f: canonical.get(f) for f in fields_to_translate}
    locale_descriptions = "\n".join(
        f"  - {code}: {LOCALE_LABELS.get(code, code)}" for code in target_locales
    )
    return (
        f"Entity type: {entity_type}\n"
        f"Target locales:\n{locale_descriptions}\n"
        f"Fields to translate (canonical English, JSON):\n"
        f"{json.dumps(fields_payload, ensure_ascii=False, indent=2)}\n"
        "\n"
        "Return JSON with one key per target locale code. Each value is an "
        "object containing the translated fields, with the SAME keys and "
        "the SAME nested shape as the canonical input above."
    )


def translate_entity(
    entity_type: str,
    canonical: dict[str, Any],
    target_locales: Iterable[str],
    fields_to_translate: Iterable[str],
    *,
    entity_id: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Translate the requested fields of a canonical entity into every
    target locale in a single Claude call.

    Returns:
        {locale: {field: translated_value}} on success — exactly the
        shape the upsertTranslations mutation expects for each row's
        `data` payload.
        None when there's nothing to translate, or Claude returns
        unparseable output (logged but non-fatal so a translation
        miss doesn't break the canonical write).

    Raises:
        ClaudeRateLimited / anthropic.APIStatusError — bubble up so
        the surrounding Celery task can reschedule.
    """
    target_locales = [loc for loc in target_locales if loc and loc != "en"]
    fields_to_translate = list(fields_to_translate)

    if not target_locales or not fields_to_translate:
        return None

    try:
        result = call_claude(
            _system_prompt(),
            _build_user_prompt(
                entity_type, canonical, target_locales, fields_to_translate,
            ),
            stage="translate",
            prompt_version=TRANSLATION_PROMPT_VERSION,
            event_id=entity_id,
            # Translations of nested JSON (crisis.needs, crisis.scenarios)
            # blow past the 1024 default — bump generously so we don't
            # truncate mid-object.
            max_tokens=4096,
        )
    except (anthropic.APIStatusError, ClaudeRateLimited):
        # Transient — let the Celery task reschedule.
        raise
    except Exception as exc:
        logger.error(
            "[TRANSLATE] Claude call failed for %s %s (%d locales × %d fields): %s",
            entity_type, entity_id, len(target_locales), len(fields_to_translate), exc,
        )
        return None

    # Validate the shape Claude returned. A missing locale or a non-dict
    # entry is a silent quality issue — log it but keep what we got so a
    # single bad locale doesn't drop the others.
    out: dict[str, dict[str, Any]] = {}
    for locale in target_locales:
        locale_data = result.get(locale)
        if not isinstance(locale_data, dict):
            logger.warning(
                "[TRANSLATE] %s %s: locale %s missing or non-object in Claude output — skipping",
                entity_type, entity_id, locale,
            )
            continue
        out[locale] = locale_data

    return out or None


def configured_target_locales() -> list[str]:
    """Read `target_locales` from settings, returning a clean list with
    'en' stripped (canonical is never a target) and entries lowercased."""
    raw = settings.target_locales or ""
    parsed = [code.strip().lower() for code in raw.split(",") if code.strip()]
    return [code for code in parsed if code != "en"]


def translate_and_upsert(
    entity_type: str,
    entity_id: str,
    canonical: dict[str, Any],
) -> dict | None:
    """Translate `canonical` into every configured target locale and
    upsert the result through clear-api's upsertTranslations mutation.

    Shared by every per-entity hook (crisis enrichment, event Celery
    task, backfill scripts, FastAPI `/translate` endpoint) so the
    staleness-diff and merge-with-previous logic lives in exactly one
    place.

    Skips Claude entirely when:
      - translation is disabled (TARGET_LOCALES is empty),
      - every locale's stored source_hashes match the freshly-computed
        canonical hashes, or
      - the union of stale fields is empty after the diff.

    Returns a summary dict for logging, or None when there's nothing to
    do. Raises (ClaudeRateLimited / anthropic.APIStatusError) so the
    caller's retry path can fire.
    """
    target_locales = configured_target_locales()
    if not target_locales:
        return None

    fresh_hashes = compute_source_hashes(entity_type, canonical)

    # One read returns every locale's stored row; map by locale.
    stored = {
        row["locale"]: row
        for row in graphql.get_translations(entity_type, entity_id)
    }

    # Per-locale stale set. A locale with no stored row is fully stale
    # (every field needs translating); a fully-current locale contributes
    # nothing and we skip the Claude payload for it.
    per_locale_stale: dict[str, list[str]] = {}
    for locale in target_locales:
        stored_hashes = (stored.get(locale) or {}).get("sourceHashes")
        fields = stale_fields(fresh_hashes, stored_hashes)
        if fields:
            per_locale_stale[locale] = fields

    if not per_locale_stale:
        logger.info(
            "[TRANSLATE] %s %s: all %d locale(s) current — skipping Claude",
            entity_type, entity_id, len(target_locales),
        )
        return None

    # Union the stale field set so one Claude call covers every locale.
    # The model returns per-locale objects; we then layer each onto its
    # locale's existing data so non-stale fields keep their previous
    # translations instead of being dropped.
    union_fields = sorted({f for fields in per_locale_stale.values() for f in fields})
    translated = translate_entity(
        entity_type,
        canonical,
        target_locales=list(per_locale_stale.keys()),
        fields_to_translate=union_fields,
        entity_id=entity_id,
    )
    if not translated:
        return None

    # Build the upsert payload per locale: merge fresh translations over
    # the stored data so we don't lose previously-translated fields the
    # current pass didn't refresh. source_hashes are always overwritten
    # with the freshly-computed canonical hashes — they describe what
    # was translated FROM, not what was translated TO.
    upsert_rows: list[dict] = []
    for locale, new_fields in translated.items():
        previous_data = (stored.get(locale) or {}).get("data") or {}
        merged_data = {**previous_data, **new_fields}
        upsert_rows.append({
            "locale": locale,
            "data": merged_data,
            "sourceHashes": fresh_hashes,
        })

    if not upsert_rows:
        return None

    graphql.upsert_translations(entity_type, entity_id, upsert_rows)
    logger.info(
        "[TRANSLATE] %s %s: wrote %d locale(s), %d field(s) per locale max",
        entity_type, entity_id, len(upsert_rows), len(union_fields),
    )
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "locales_written": list(translated.keys()),
        "fields_translated": union_fields,
    }
