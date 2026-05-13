"""IOM DTM backfill Celery task.

Populates `locationMetadata(type="iom_dtm_displacement")` for CLEAR locations
at admin levels 0 (country), 1 (state/province) and 2 (district) using the
latest round of data from the IOM DTM API.

Scheduled weekly via Celery Beat. Also callable manually (or from the sync
script in scripts/backfill_iom_dtm.py).
"""

from __future__ import annotations

import logging

from src.celery_app import app
from src.clients import graphql, iom_dtm
from src.config import settings

logger = logging.getLogger(__name__)

METADATA_TYPE = "iom_dtm_displacement"

# Per-admin-level config: which fetch function to call and the CLEAR level
# we upsert into.
_LEVEL_FETCH = {
    0: iom_dtm.fetch_admin0_displacement,
    1: iom_dtm.fetch_admin1_displacement,
    2: iom_dtm.fetch_admin2_displacement,
}


def _normalise_name(name: str | None) -> str:
    """Lowercase + strip punctuation for name-based match fallback."""
    import re

    s = (name or "").strip().lower()
    for prefix in ("republic of ", "the ", "el-", "el ", "al-", "al "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def _process_level(
    admin_level: int,
    country_name: str,
    admin0_pcode: str,
    operation: str | None,
    from_round: int | None,
    assessment_type_filter: str | list[str] | None = "BA",
) -> dict:
    """Fetch DTM rows at `admin_level`, sum IDPs per destination across
    origins, and bulk-upsert into location_metadata. Returns per-level stats.

    `assessment_type_filter` accepts a single assessmentType, a priority list
    (e.g. ["BA", "FM"] — BA fills first, FM fills the gaps), or None to pool
    all types. Each upserted row records which type produced it.
    """
    stats = {
        "fetched": 0,
        "distinct": 0,
        "matched": 0,
        "upserted": 0,
        "skipped_no_value": 0,
        "unmatched_pcode": 0,
        "failed": 0,
    }

    fetch = _LEVEL_FETCH[admin_level]
    records = fetch(
        country_name=country_name,
        admin0_pcode=admin0_pcode,
        operation=operation,
        from_round=from_round,
    )
    stats["fetched"] = len(records)

    # Aggregate: sum across (origin admin1, displacement reason) per
    # (destination admin{N}, round), keep latest round per destination.
    latest = iom_dtm.aggregate_displacement_by_destination(
        records,
        admin_level=admin_level,
        assessment_type_filter=assessment_type_filter,
    )
    stats["distinct"] = len(latest)
    logger.info(
        "[IOM DTM L%d] aggregated %d destinations from %d raw rows (assessment=%s)",
        admin_level, len(latest), len(records), assessment_type_filter,
    )

    # CLEAR locations at this level — build pCode and name lookup maps.
    # ISO2 map (admin0 only) handles "SD" ↔ "SDN" drift in either direction.
    # Name map handles everything else.
    clear_rows = graphql.get_locations_by_level(admin_level)
    pcode_to_id: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    iso2_to_id: dict[str, str] = {}
    for loc in clear_rows:
        if loc.get("pCode"):
            pcode_to_id[loc["pCode"]] = loc["id"]
            if admin_level == 0:
                iso2_to_id[loc["pCode"][:2].upper()] = loc["id"]
        if loc.get("name"):
            name_to_id[_normalise_name(loc["name"])] = loc["id"]
    logger.info(
        "[IOM DTM L%d] %d CLEAR locations with pCodes (%d with names)",
        admin_level, len(pcode_to_id), len(name_to_id),
    )

    batch: list[dict] = []
    for pcode, agg in latest.items():
        clear_id = pcode_to_id.get(pcode)
        if not clear_id and admin_level == 0:
            # ISO2/ISO3 drift: IOM "SDN" ↔ CLEAR "SD" (or vice versa).
            clear_id = iso2_to_id.get(pcode[:2].upper())
            if clear_id:
                logger.info(
                    "[IOM DTM L%d ISO2-MATCH] pCode=%s → %s (matched on '%s')",
                    admin_level, pcode, clear_id, pcode[:2].upper(),
                )
        if not clear_id:
            # Name-based fallback for non-admin0 pCode drift.
            rec_name = agg["admin_name"]
            if rec_name:
                clear_id = name_to_id.get(_normalise_name(rec_name))
                if clear_id:
                    logger.info(
                        "[IOM DTM L%d NAME-MATCH] pCode=%s name=%r → %s",
                        admin_level, pcode, rec_name, clear_id,
                    )

        if not clear_id:
            stats["unmatched_pcode"] += 1
            logger.debug(
                "[IOM DTM L%d] No CLEAR location for pCode=%s name=%s",
                admin_level, pcode, agg["admin_name"],
            )
            continue

        if agg["population_displaced"] <= 0:
            stats["skipped_no_value"] += 1
            continue

        payload = {
            "population_displaced": agg["population_displaced"],
            # Per-origin breakdown — sorted desc by count. Empty list when
            # the operation doesn't expose origin info (e.g. older datasets).
            "origin_breakdown": agg["origin_breakdown"],
            "round_number": agg["round_number"],
            "reporting_date": agg["reporting_date"],
            "operation": agg["operation"],
            "admin_level": admin_level,
            "admin_name": agg["admin_name"],
            "admin_pcode": pcode,
            "assessment_type": agg["assessment_type"],
            "source": "iom_dtm_v3",
        }
        stats["matched"] += 1
        batch.append({
            "locationId": clear_id,
            "type": METADATA_TYPE,
            "data": payload,
        })

    if batch:
        try:
            written = graphql.upsert_location_metadata_batch(batch)
            stats["upserted"] = len(written)
            logger.info(
                "[IOM DTM L%d] Bulk-upserted %d rows",
                admin_level, len(written),
            )
        except Exception as e:
            stats["failed"] = len(batch)
            logger.error(
                "[IOM DTM L%d] Bulk upsert failed: %s",
                admin_level, e, exc_info=True,
            )

    return stats


@app.task(
    name="src.tasks.dtm.backfill_dtm_displacement",
    bind=True,
    max_retries=1,
    acks_late=True,
)
def backfill_dtm_displacement(
    self,
    country_name: str | None = None,
    admin0_pcode: str | None = None,
    operation: str | None = None,
    from_round: int | None = None,
    assessment_type: str | None = None,
    levels: list[int] | None = None,
) -> dict:
    """Fetch IOM DTM displacement data at admin levels 0, 1, and 2 and upsert
    into locationMetadata for every matching CLEAR location.

    `levels` lets callers scope to specific levels (default: all three).
    `operation` overrides settings.iom_dtm_operation.
    `from_round` overrides settings.iom_dtm_from_round.
    `assessment_type` overrides settings.iom_dtm_assessment_type. Accepts a
        single type ("BA") or a comma-separated priority list ("BA,FM"): the
        first type fills each pcode, later types fill remaining gaps.
        Empty string pools all types (only safe when BA/FM don't overlap).
    Returns per-level stats plus a top-level "total_upserted" convenience count.
    """
    country = country_name or settings.iom_dtm_country_name
    admin0 = admin0_pcode or settings.iom_dtm_admin0_pcode
    op = operation if operation is not None else (settings.iom_dtm_operation or None)
    # 0 in settings means "no lower bound" — treat as None for the API call.
    fr_setting = from_round if from_round is not None else settings.iom_dtm_from_round
    fr = fr_setting if fr_setting and fr_setting > 0 else None
    at_setting = assessment_type if assessment_type is not None else settings.iom_dtm_assessment_type
    # Setting accepts a comma-separated priority list ("BA,FM"). Single value
    # stays a string; multiple become a list so the aggregator runs the BA→FM
    # fallback. Empty string disables filtering entirely.
    at: str | list[str] | None
    if not at_setting:
        at = None
    else:
        parts = [t.strip() for t in at_setting.split(",") if t.strip()]
        if len(parts) == 0:
            at = None
        elif len(parts) == 1:
            at = parts[0]
        else:
            at = parts
    target_levels = levels or [0, 1, 2]

    if not settings.iom_dtm_subscription_key:
        logger.warning("[IOM DTM] Subscription key not configured — skipping")
        return {"skipped": "no_subscription_key"}

    logger.info(
        "[IOM DTM] backfill country=%s admin0=%s operation=%r from_round=%s assessment=%s levels=%s",
        country, admin0, op, fr, at, target_levels,
    )

    all_stats: dict = {}
    try:
        for lvl in target_levels:
            if lvl not in _LEVEL_FETCH:
                logger.warning("[IOM DTM] Unsupported admin level %s — skipping", lvl)
                continue
            all_stats[f"admin{lvl}"] = _process_level(lvl, country, admin0, op, fr, at)

        all_stats["total_upserted"] = sum(
            s.get("upserted", 0) for s in all_stats.values() if isinstance(s, dict)
        )
        logger.info("[IOM DTM] Done: %s", all_stats)
        return all_stats

    except graphql.GraphQLClientError as exc:
        logger.error("[IOM DTM] permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("[IOM DTM] backfill_dtm_displacement failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=300)
