"""Generic ArcGIS REST FeatureServer client.

Fetches features as **GeoJSON** (`f=geojson`, WGS84), paging past the layer's
`maxRecordCount`, and decodes **coded-value domains** (integer codes → labels)
from the layer metadata so downstream consumers get human-readable
status/type fields instead of opaque ints. Works against any ArcGIS
FeatureServer layer (both the Esri-hosted roads view and LogCluster's
self-hosted bridges layer share this interface).
"""

from __future__ import annotations

import datetime as dt
import logging

import requests

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 60
PAGE_SIZE = 2000  # typical ArcGIS maxRecordCount; server may return fewer


def _field_meta(
    layer_url: str, session: requests.Session,
) -> tuple[dict[str, dict], set[str]]:
    """Return (coded_domains, date_field_names) from the layer's `?f=json`.

    - coded_domains: {field: {code: label}} for coded-value domains.
    - date_field_names: fields typed `esriFieldTypeDate` — GeoJSON returns
      these as epoch-milliseconds, which we convert to ISO-8601.
    """
    resp = session.get(layer_url, params={"f": "json"}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    meta = resp.json()
    domains: dict[str, dict] = {}
    date_fields: set[str] = set()
    for field in meta.get("fields") or []:
        if field.get("type") == "esriFieldTypeDate":
            date_fields.add(field["name"])
        dom = field.get("domain")
        if dom and dom.get("type") == "codedValue":
            domains[field["name"]] = {
                cv["code"]: cv["name"] for cv in dom.get("codedValues") or []
            }
    return domains, date_fields


def _epoch_ms_to_iso(value: object) -> object:
    """Convert an ArcGIS epoch-milliseconds timestamp to ISO-8601 (UTC).
    Non-numeric / out-of-range values pass through unchanged."""
    if not isinstance(value, (int, float)):
        return value
    try:
        return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError):
        return value


def fetch_geojson(
    layer_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    keep_fields: list[str] | None = None,
    decode_domains: bool = True,
) -> dict:
    """Fetch a GeoJSON FeatureCollection for `where`, paginated.

    Args:
      layer_url: FeatureServer layer URL (…/FeatureServer/<n>).
      where: SQL-ish filter, e.g. ``iso3='SDN'``.
      out_fields: fields to request from the server (``*`` = all).
      keep_fields: if given, prune each feature's properties to these keys
        (plus any decoded ``<field>_label``) — keeps the blob slim.
      decode_domains: add a ``<field>_label`` for every coded-value field.

    Returns a ``{"type": "FeatureCollection", "features": [...]}`` dict in
    WGS84 (EPSG:4326).
    """
    session = requests.Session()
    domains, date_fields = _field_meta(layer_url, session)

    features: list[dict] = []
    offset = 0
    while True:
        resp = session.get(
            f"{layer_url}/query",
            params={
                "where": where,
                "outFields": out_fields,
                "f": "geojson",
                "outSR": 4326,
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        batch = resp.json().get("features") or []

        for feat in batch:
            props = feat.get("properties") or {}
            # Epoch-ms date fields → ISO (always).
            for field in date_fields:
                if field in props:
                    props[field] = _epoch_ms_to_iso(props[field])
            # Decode coded fields in place (adds "<field>_label").
            if decode_domains:
                for field, code_map in domains.items():
                    if props.get(field) in code_map:
                        props[f"{field}_label"] = code_map[props[field]]
            if keep_fields is not None:
                slim = {k: props[k] for k in keep_fields if k in props}
                for k in keep_fields:
                    label = f"{k}_label"
                    if label in props:
                        slim[label] = props[label]
                feat["properties"] = slim
            else:
                feat["properties"] = props

        features.extend(batch)
        logger.info(
            "[arcgis] %s where=%r fetched %d (offset %d)",
            layer_url.rsplit("/services/", 1)[-1], where, len(features), offset,
        )
        if len(batch) < PAGE_SIZE:
            break
        offset += len(batch)

    return {"type": "FeatureCollection", "features": features}
