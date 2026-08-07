"""LogIE roads & bridges backfill Celery task.

Pulls road (polyline) and bridge (point) situational features from the
LogCluster/LogIE ArcGIS FeatureServers and persists them as clear-api
``locationMetadata(type="logie_roads" / "logie_bridges")`` — one trimmed,
domain-decoded GeoJSON FeatureCollection per country, attached to that
country's A0 location. clear-api's bitemporal upsert gives free refresh
history; a downstream clear-api resolver slims this into the map-ready
Blockages payload (ticket #317, phase 2).

Scheduled **monthly** via Celery Beat (mirrors ``dtm.py``). Also callable
manually: ``backfill_logistics_infrastructure("SDN")``.
"""

from __future__ import annotations

import datetime as dt
import logging

from src.celery_app import app
from src.clients import arcgis, graphql
from src.config import settings

logger = logging.getLogger(__name__)

ROADS_TYPE = "logie_roads"
BRIDGES_TYPE = "logie_bridges"

# Slim attribute sets (source field names). Coded fields additionally get a
# decoded "<field>_label" from the layer's domains (see arcgis.fetch_geojson).
ROAD_FIELDS = [
    "osmid", "roadnameen", "routenameen", "fclass", "currstatus_physical",
    "currstatusremarken", "currsourcename", "currinforely",
    "currleadtime_days", "leadtime_days_season1", "leadtime_days_season2",
    "distance_km", "location_origin", "location_destination",
    "currasofdate", "iso3",
]
BRIDGE_FIELDS = [
    "osmid", "name", "bridgenameloc", "currstatus", "currstatusremarken",
    "currsourcename", "currinforely", "bridgetype", "basematerial",
    "currsurfacecond", "isseasonal", "seasonalrecomen", "currasofdate", "iso3",
]


def _a0_by_iso3(iso3_list: list[str]) -> dict[str, str]:
    """Map each iso3 → CLEAR A0 ``locationId``.

    CLEAR A0 pCodes are ISO2 (e.g. ``SD``); the ArcGIS ``iso3`` is ``SDN`` —
    match on the leading 2 chars (same ISO2/ISO3 drift ``dtm.py`` handles).
    """
    a0 = graphql.get_locations_by_level(0)
    by_iso2: dict[str, str] = {}
    for loc in a0:
        pcode = (loc.get("pCode") or "").upper()
        if pcode:
            by_iso2[pcode[:2]] = loc["id"]

    resolved: dict[str, str] = {}
    for iso3 in iso3_list:
        loc_id = by_iso2.get(iso3[:2].upper())
        if loc_id:
            resolved[iso3] = loc_id
        else:
            logger.warning(
                "[LogIE] no A0 location for iso3=%s (pCode %s) — skipping",
                iso3, iso3[:2],
            )
    return resolved


@app.task(name="src.tasks.logistics.backfill_logistics_infrastructure")
def backfill_logistics_infrastructure(iso3_csv: str | None = None) -> dict:
    """Sync LogIE roads + bridges into ``locationMetadata`` for the configured
    countries. ``iso3_csv`` overrides ``settings.logistics_iso3`` (comma-sep)."""
    iso3_list = [
        c.strip().upper()
        for c in (iso3_csv or settings.logistics_iso3).split(",")
        if c.strip()
    ]
    loc_by_iso3 = _a0_by_iso3(iso3_list)
    if not loc_by_iso3:
        logger.warning("[LogIE] no matching A0 locations for %s — nothing to do", iso3_list)
        return {"skipped": "no_locations", "requested": iso3_list}

    pulled_at = dt.datetime.now(dt.timezone.utc).isoformat()
    rows: list[dict] = []
    stats: dict[str, dict] = {}

    for iso3, loc_id in loc_by_iso3.items():
        where = f"iso3='{iso3}'"
        roads = _fetch(settings.logie_roads_url, where, ROAD_FIELDS, iso3, "roads")
        bridges = _fetch(settings.logie_bridges_url, where, BRIDGE_FIELDS, iso3, "bridges")

        if roads is not None:
            rows.append({"locationId": loc_id, "type": ROADS_TYPE,
                         "data": _wrap(roads, pulled_at)})
        if bridges is not None:
            rows.append({"locationId": loc_id, "type": BRIDGES_TYPE,
                         "data": _wrap(bridges, pulled_at)})
        stats[iso3] = {
            "roads": len((roads or {}).get("features") or []),
            "bridges": len((bridges or {}).get("features") or []),
        }

    written = graphql.upsert_location_metadata_batch(rows)
    logger.info("[LogIE] upserted %d metadata rows: %s", len(written), stats)
    return {"upserted": len(written), "per_country": stats, "pulled_at": pulled_at}


def _fetch(url: str, where: str, keep: list[str], iso3: str, kind: str) -> dict | None:
    """Fetch one layer; isolate failures so one country/layer can't abort the run."""
    try:
        return arcgis.fetch_geojson(url, where=where, keep_fields=keep)
    except Exception as exc:  # noqa: BLE001 — log + continue is intentional
        logger.error("[LogIE] %s fetch failed for %s: %s", kind, iso3, exc)
        return None


def _wrap(fc: dict, pulled_at: str) -> dict:
    """Attach provenance to the FeatureCollection payload."""
    fc["pulled_at"] = pulled_at
    fc["source"] = "LogIE"
    return fc
