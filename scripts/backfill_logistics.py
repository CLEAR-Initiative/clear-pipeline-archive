"""
Standalone runner for the LogIE roads & bridges backfill.

Pulls road (polyline) + bridge (point) situational features from the LogIE /
LogCluster ArcGIS FeatureServers and upserts them into clear-api
locationMetadata (type = "logie_roads" / "logie_bridges"), one domain-decoded
GeoJSON FeatureCollection per country attached to that country's A0 location.

Reuses the Celery task's helpers (src/tasks/logistics.py) so this runner and the
scheduled monthly job share one code path. Runs synchronously in-process — no
Celery worker or Redis required.

Writes go to whatever CLEAR_API_URL / CLEAR_API_KEY point at in .env. Check that
first — the upsert is bitemporal (close-old-then-insert), so re-running is safe
and idempotent (each run just supersedes the prior version).

Usage:
    python scripts/backfill_logistics.py --iso3 SDN --dry-run   # fetch + preview, NO writes
    python scripts/backfill_logistics.py --iso3 SDN             # write SDN only
    python scripts/backfill_logistics.py                        # write all (settings.logistics_iso3)
    python scripts/backfill_logistics.py --iso3 SDN,AFG,VEN     # explicit multi-country
"""

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.clients import graphql  # noqa: E402
from src.config import settings  # noqa: E402
from src.tasks.logistics import (  # noqa: E402
    BRIDGE_FIELDS,
    BRIDGES_TYPE,
    ROAD_FIELDS,
    ROADS_TYPE,
    _a0_by_iso3,
    _fetch,
    _wrap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _feature_type_summary(fc: dict) -> str:
    feats = fc.get("features") or []
    geom = feats[0].get("geometry", {}).get("type") if feats else "n/a"
    return f"{len(feats)} features (geom: {geom})"


def run(iso3_list: list[str], dry_run: bool) -> dict:
    loc_by_iso3 = _a0_by_iso3(iso3_list)
    if not loc_by_iso3:
        logger.warning("No matching A0 locations for %s — nothing to do.", iso3_list)
        return {"skipped": "no_locations", "requested": iso3_list}

    pulled_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict] = []
    stats: dict[str, dict] = {}

    for iso3, loc_id in loc_by_iso3.items():
        where = f"iso3='{iso3}'"
        logger.info("=" * 60)
        logger.info("%s → A0 locationId=%s (where %s)", iso3, loc_id, where)

        roads = _fetch(settings.logie_roads_url, where, ROAD_FIELDS, iso3, "roads")
        bridges = _fetch(settings.logie_bridges_url, where, BRIDGE_FIELDS, iso3, "bridges")

        if roads is not None:
            logger.info("  roads   : %s", _feature_type_summary(roads))
            rows.append({"locationId": loc_id, "type": ROADS_TYPE, "data": _wrap(roads, pulled_at)})
        if bridges is not None:
            logger.info("  bridges : %s", _feature_type_summary(bridges))
            rows.append({"locationId": loc_id, "type": BRIDGES_TYPE, "data": _wrap(bridges, pulled_at)})

        stats[iso3] = {
            "roads": len((roads or {}).get("features") or []),
            "bridges": len((bridges or {}).get("features") or []),
        }

    if dry_run:
        logger.info("=" * 60)
        logger.info("[DRY-RUN] would upsert %d metadata rows → %s", len(rows), settings.clear_api_url)
        for r in rows:
            fc = r["data"]
            logger.info(
                "  %s / %s: %s, pulled_at=%s",
                r["locationId"], r["type"], _feature_type_summary(fc), fc.get("pulled_at"),
            )
        return {"dry_run": True, "would_upsert": len(rows), "per_country": stats, "pulled_at": pulled_at}

    written = graphql.upsert_location_metadata_batch(rows)
    logger.info("=" * 60)
    logger.info("[OK] upserted %d metadata rows → %s", len(written), settings.clear_api_url)
    return {"upserted": len(written), "per_country": stats, "pulled_at": pulled_at}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LogIE roads & bridges into locationMetadata.")
    parser.add_argument(
        "--iso3",
        default=settings.logistics_iso3,
        help=(
            "Comma-separated ISO3 country codes to sync. "
            f"Default: settings.logistics_iso3 = {settings.logistics_iso3!r}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch + preview, no writes.")
    args = parser.parse_args()

    iso3_list = [c.strip().upper() for c in args.iso3.split(",") if c.strip()]
    if not iso3_list:
        raise SystemExit("No countries given (--iso3 was empty).")

    logger.info(
        "Starting LogIE backfill: iso3=%s dry_run=%s target=%s",
        iso3_list, args.dry_run, settings.clear_api_url,
    )
    stats = run(iso3_list, args.dry_run)
    logger.info("Done: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
