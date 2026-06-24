"""
Standalone runner for IOM DTM displacement backfill.

Fetches the latest displaced-persons data from the IOM DTM API
(https://dtmapi.iom.int/v3/) at admin levels 0 (country), 1 (state), and 2
(district), then upserts into locationMetadata (type = "iom_dtm_displacement")
on clear-api. Writes are bulk — one network call per admin level.

Runs synchronously in-process — no Celery worker required.

Usage:
    python scripts/backfill_iom_dtm.py                      # all 3 levels (default country)
    python scripts/backfill_iom_dtm.py --levels 1,2         # skip country
    python scripts/backfill_iom_dtm.py --levels 2           # districts only
    python scripts/backfill_iom_dtm.py --dry-run            # fetch, match, print — no writes
    python scripts/backfill_iom_dtm.py --country "Sudan"
    python scripts/backfill_iom_dtm.py --admin0-pcode SDN

    # Afghanistan — there's no canonical AFG operation in the DTM API,
    # so override the operation default to "" to fetch across all operations.
    python scripts/backfill_iom_dtm.py --iso3 AFG --operation ""

    # --iso3 is a shortcut that sets --country and --admin0-pcode together;
    # either explicit flag still wins if you also pass it.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clients import graphql, iom_dtm  # noqa: E402
from src.config import settings  # noqa: E402

METADATA_TYPE = "iom_dtm_displacement"

# --iso3 shortcut: maps to (DTM CountryName filter, DTM Admin0Pcode filter).
# The DTM API treats these as independent filters but in practice they have
# to agree, so we set both from one flag for ergonomics. Either flag can
# still be overridden explicitly on the command line.
ISO3_TO_DTM_FILTERS: dict[str, tuple[str, str]] = {
    "SDN": ("Sudan", "SDN"),
    "AFG": ("Afghanistan", "AFG"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_LEVEL_FETCH = {
    0: iom_dtm.fetch_admin0_displacement,
    1: iom_dtm.fetch_admin1_displacement,
    2: iom_dtm.fetch_admin2_displacement,
}


def _normalise_name(name: str) -> str:
    """Lowercase + strip punctuation for lenient name matching
    (e.g. "Sudan" vs "Republic of Sudan", "El Gezira" vs "Al Jazirah")."""
    import re

    s = (name or "").strip().lower()
    for prefix in ("republic of ", "the ", "el-", "el ", "al-", "al "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def parse_levels(arg: str | None) -> list[int]:
    if not arg:
        return [0, 1, 2]
    try:
        out = [int(x.strip()) for x in arg.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"Invalid --levels value: {arg!r}")
    for lvl in out:
        if lvl not in (0, 1, 2):
            raise SystemExit(f"Unsupported admin level {lvl} — only 0, 1, 2 are valid.")
    return out


def run_level(
    admin_level: int,
    country_name: str,
    admin0_pcode: str,
    operation: str | None,
    from_round: int | None,
    assessment_type_filter: str | list[str] | None,
    dry_run: bool,
) -> dict:
    stats = {
        "fetched": 0,
        "records_without_pcode": 0,
        "distinct": 0,
        "matched": 0,
        "upserted": 0,
        "skipped_no_value": 0,
        "unmatched_pcode": 0,
        "clear_total": 0,
        "clear_missing_pcode": 0,
        "clear_with_no_dtm_row": 0,
        "failed": 0,
    }

    fetch = _LEVEL_FETCH[admin_level]
    logger.info("=" * 60)
    logger.info(
        "Level %d — fetching IOM DTM admin%d for %s (operation=%r from_round=%s assessment=%s)…",
        admin_level, admin_level, country_name, operation, from_round, assessment_type_filter,
    )
    records = fetch(
        country_name=country_name,
        admin0_pcode=admin0_pcode,
        operation=operation,
        from_round=from_round,
    )
    stats["fetched"] = len(records)

    stats["records_without_pcode"] = sum(
        1 for r in records if not iom_dtm.record_pcode(r, admin_level)
    )

    # Aggregate: sum across (origin admin1, displacement reason) per
    # (destination, round); keep latest round per destination. Filtered to
    # one assessmentType so BA + FM don't double-count.
    latest = iom_dtm.aggregate_displacement_by_destination(
        records,
        admin_level=admin_level,
        assessment_type_filter=assessment_type_filter,
    )
    stats["distinct"] = len(latest)
    logger.info(
        "Aggregated: %d destinations from %d raw rows", len(latest), len(records),
    )

    clear_rows = graphql.get_locations_by_level(admin_level)
    stats["clear_total"] = len(clear_rows)
    pcode_to_id: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    # ISO2 prefix → id, populated only at admin0. CLEAR stores Sudan as "SD"
    # while IOM DTM returns "SDN" (and historically the inverse). Keying by
    # the first 2 chars handles both directions cleanly.
    iso2_to_id: dict[str, str] = {}
    clear_missing_pcode: list[str] = []
    for loc in clear_rows:
        if loc.get("pCode"):
            pcode_to_id[loc["pCode"]] = loc["id"]
            if admin_level == 0:
                iso2_to_id[loc["pCode"][:2].upper()] = loc["id"]
        else:
            clear_missing_pcode.append(loc.get("name") or loc["id"])
        if loc.get("name"):
            name_to_id[_normalise_name(loc["name"])] = loc["id"]
    stats["clear_missing_pcode"] = len(clear_missing_pcode)
    logger.info(
        "CLEAR level-%d: %d total (%d with pCode, %d without)",
        admin_level, len(clear_rows), len(pcode_to_id), len(clear_missing_pcode),
    )
    if clear_missing_pcode:
        logger.warning(
            "CLEAR level-%d without pCode (first 10): %s",
            admin_level, clear_missing_pcode[:10],
        )

    dtm_pcodes = set(latest.keys())
    clear_no_dtm: list[str] = []
    for loc in clear_rows:
        pcode = loc.get("pCode")
        if not pcode or pcode in dtm_pcodes:
            continue
        # At admin0, treat "SD" and "SDN" as the same row before warning.
        if admin_level == 0 and any(p[:2].upper() == pcode[:2].upper() for p in dtm_pcodes):
            continue
        clear_no_dtm.append(f"{loc.get('name')} ({pcode})")
    stats["clear_with_no_dtm_row"] = len(clear_no_dtm)
    if clear_no_dtm:
        logger.warning(
            "%d CLEAR level-%d locations have NO IOM DTM row (first 20): %s",
            len(clear_no_dtm), admin_level, clear_no_dtm[:20],
        )

    batch: list[dict] = []
    for pcode, agg in latest.items():
        clear_id = pcode_to_id.get(pcode)
        match_mode = "pcode" if clear_id else None

        if not clear_id and admin_level == 0:
            # ISO2/ISO3 drift: IOM "SDN" ↔ CLEAR "SD" (or vice versa).
            candidate = iso2_to_id.get(pcode[:2].upper())
            if candidate:
                clear_id = candidate
                match_mode = "iso2"
                logger.info(
                    "[ISO2-MATCH L%d] pCode=%s → CLEAR id=%s (matched on '%s')",
                    admin_level, pcode, clear_id, pcode[:2].upper(),
                )

        if not clear_id:
            # Fallback: match by normalised name (handles non-admin0 pCode drift).
            rec_name = agg["admin_name"]
            if rec_name:
                candidate = name_to_id.get(_normalise_name(rec_name))
                if candidate:
                    clear_id = candidate
                    match_mode = "name"
                    logger.info(
                        "[NAME-MATCH L%d] pCode=%s name=%r → CLEAR id=%s",
                        admin_level, pcode, rec_name, clear_id,
                    )

        if not clear_id:
            stats["unmatched_pcode"] += 1
            logger.warning(
                "[UNMATCHED L%d] pCode=%s name=%s",
                admin_level, pcode, agg["admin_name"],
            )
            continue

        value = agg["population_displaced"]
        if value <= 0:
            stats["skipped_no_value"] += 1
            logger.warning(
                "[NO VALUE L%d] pCode=%s round=%s",
                admin_level, pcode, agg["round_number"],
            )
            continue

        payload = {
            "population_displaced": value,
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

        if dry_run:
            top_origin = (
                f", top origin: {agg['origin_breakdown'][0]['origin_admin1_name']}"
                f"={agg['origin_breakdown'][0]['count']}"
                if agg["origin_breakdown"]
                else ""
            )
            logger.info(
                "[DRY-RUN L%d] pCode=%s → %d displaced (round %s, %s, %d origins%s)",
                admin_level, pcode, value,
                agg["round_number"], agg["reporting_date"],
                len(agg["origin_breakdown"]), top_origin,
            )
            continue

        batch.append({
            "locationId": clear_id,
            "type": METADATA_TYPE,
            "data": payload,
        })

    if batch and not dry_run:
        try:
            written = graphql.upsert_location_metadata_batch(batch)
            stats["upserted"] = len(written)
            logger.info("[OK L%d] Bulk-upserted %d rows", admin_level, len(written))
        except Exception as e:
            stats["failed"] = len(batch)
            logger.error("[FAILED L%d] Bulk upsert: %s", admin_level, e, exc_info=True)

    return stats


def run(
    levels: list[int],
    country_name: str,
    admin0_pcode: str,
    operation: str | None,
    from_round: int | None,
    assessment_type_filter: str | list[str] | None,
    dry_run: bool,
) -> dict:
    if not settings.iom_dtm_subscription_key:
        raise SystemExit(
            "IOM_DTM_SUBSCRIPTION_KEY is not set. "
            "Get one from https://dtm-apim.developer.iom.int/ and add to .env."
        )

    all_stats: dict = {}
    for lvl in levels:
        all_stats[f"admin{lvl}"] = run_level(
            lvl, country_name, admin0_pcode, operation, from_round,
            assessment_type_filter, dry_run,
        )

    all_stats["total_upserted"] = sum(
        s.get("upserted", 0) for s in all_stats.values() if isinstance(s, dict)
    )
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill IOM DTM displacement into locationMetadata.")
    parser.add_argument(
        "--iso3",
        default=None,
        help=(
            "Country shortcut — sets --country and --admin0-pcode together. "
            f"Supported: {sorted(ISO3_TO_DTM_FILTERS)}. Explicit --country / "
            "--admin0-pcode flags still win if also supplied."
        ),
    )
    parser.add_argument(
        "--country",
        default=None,
        help=(
            "CountryName query filter. If omitted, derived from --iso3 (or "
            f"settings.iom_dtm_country_name = {settings.iom_dtm_country_name!r})."
        ),
    )
    parser.add_argument(
        "--admin0-pcode",
        default=None,
        help=(
            "Admin0Pcode filter. If omitted, derived from --iso3 (or "
            f"settings.iom_dtm_admin0_pcode = {settings.iom_dtm_admin0_pcode!r})."
        ),
    )
    parser.add_argument(
        "--operation",
        default=settings.iom_dtm_operation,
        help=(
            "DTM Operation (data-gathering project) name. Pass an empty string "
            f"to fetch across all operations. Default: {settings.iom_dtm_operation!r}."
        ),
    )
    parser.add_argument(
        "--from-round",
        type=int,
        default=settings.iom_dtm_from_round,
        help=(
            "Lower bound on FromRoundNumber. 0 (default) means no lower bound — "
            "all rounds are fetched and the latest per pCode wins."
        ),
    )
    parser.add_argument(
        "--assessment-type",
        default=settings.iom_dtm_assessment_type,
        help=(
            "Filter rows by assessmentType before summing. Accepts a single "
            "value ('BA') or a comma-separated priority list ('BA,FM'): the "
            "first type fills each pcode; later types only fill pcodes the "
            "earlier types had no rows for. Default: "
            f"{settings.iom_dtm_assessment_type!r}. Pass an empty string to "
            "disable filtering and pool all types — only safe when the data "
            "has no BA/FM overlap (will double-count otherwise)."
        ),
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated admin levels to process (default: 0,1,2).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no writes.")
    args = parser.parse_args()

    levels = parse_levels(args.levels)

    # Precedence: explicit --country / --admin0-pcode > --iso3 > settings.*.
    iso3_country: str | None = None
    iso3_admin0: str | None = None
    if args.iso3:
        iso3 = args.iso3.upper()
        mapping = ISO3_TO_DTM_FILTERS.get(iso3)
        if not mapping:
            raise SystemExit(
                f"--iso3 {iso3!r} is not recognised. Add it to "
                f"ISO3_TO_DTM_FILTERS or pass --country and --admin0-pcode "
                "directly."
            )
        iso3_country, iso3_admin0 = mapping
    country = args.country or iso3_country or settings.iom_dtm_country_name
    admin0_pcode = args.admin0_pcode or iso3_admin0 or settings.iom_dtm_admin0_pcode

    operation = args.operation or None
    from_round = args.from_round if args.from_round and args.from_round > 0 else None
    # "" → None (filter disabled); "BA" → "BA"; "BA,FM" → ["BA", "FM"] (priority fallback)
    assessment_type: str | list[str] | None
    if not args.assessment_type:
        assessment_type = None
    else:
        parts = [t.strip() for t in args.assessment_type.split(",") if t.strip()]
        if len(parts) == 0:
            assessment_type = None
        elif len(parts) == 1:
            assessment_type = parts[0]
        else:
            assessment_type = parts

    logger.info(
        "Starting IOM DTM backfill: country=%s admin0=%s operation=%r from_round=%s assessment=%s levels=%s dry_run=%s",
        country, admin0_pcode, operation, from_round, assessment_type, levels, args.dry_run,
    )

    stats = run(levels, country, admin0_pcode, operation, from_round, assessment_type, args.dry_run)
    logger.info("Done: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
