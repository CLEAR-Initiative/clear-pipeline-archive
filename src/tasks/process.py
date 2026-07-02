"""Celery task: process a single signal through classification → grouping → escalation."""

import json
import logging

import redis

from src.celery_app import app
from src.clients.claude import ClaudeRateLimited, call_claude
from src.clients.graphql import (
    GraphQLClientError,
    escalate_event,
    find_or_create_landmark_l4,
    get_dataminr_source_id,
    get_disaster_types,
    get_signal,
    update_signal_geoparsed_data,
    update_signal_location,
    update_signal_severity,
)
from src.config import settings
from src.models.clear import SignalClassification
from src.models.dataminr import DataminrSignal
from src.prompts.classify import (
    CLASSIFY_PROMPT_VERSION,
    SYSTEM_PROMPT as CLASSIFY_SYSTEM,
    build_classify_prompt,
)
from src.services.alert import is_stale_signal, maybe_escalate
from src.services.event import dispatch_group_signal
from src.services.local_classify import classify_locally
from src.services.geoparser import geoparse_signal
from src.services.redis_lock import redis_lock
from src.services.signal import geoparse_to_dict, ingest_signal

logger = logging.getLogger(__name__)

_redis = redis.from_url(settings.redis_url, decode_responses=True)

# Per-signal dedup lock TTL. Covers worst-case task duration (classify +
# geoparser + grouping + optional escalation ≈ 3-4 min for slow Claude
# calls) with a buffer for a SIGKILL'd worker to expire the lock cleanly
# after the task's `time_limit`. If we bump these limits, bump this too.
_SIGNAL_LOCK_TTL_SECONDS = 360  # 6 min

# Cache data source ID and disaster types to avoid repeated lookups
_source_id_cache: str | None = None
_disaster_types_cache: list[dict] | None = None


def _get_source_id() -> str:
    global _source_id_cache
    if _source_id_cache is None:
        _source_id_cache = get_dataminr_source_id()
    return _source_id_cache


def _get_disaster_types() -> list[dict]:
    global _disaster_types_cache
    if _disaster_types_cache is None:
        _disaster_types_cache = get_disaster_types()
    return _disaster_types_cache


@app.task(
    name="src.tasks.process.process_signal",
    bind=True,
    max_retries=2,
    acks_late=True,
    # Hard ceiling on wasted work when a Claude call spins in the SDK's
    # retry loop. Combined with the per-signal dedup lock below, this
    # stops the "same signal, both workers, both create an event"
    # duplication pattern that broker heartbeat drops trigger under
    # `task_acks_late=True` + `worker_cancel_long_running_tasks_on_connection_loss`.
    soft_time_limit=240,
    time_limit=300,
)
def process_signal(self, signal_data: dict):
    """
    Process a single Dataminr signal through the full pipeline:
    1. Parse & ingest as CLEAR signal
    2. Classify via Claude (disaster type, relevance, severity)
    3. If relevant: group into event (new or existing)
    4. If high severity: assess for alert escalation
    """
    signal = DataminrSignal.model_validate(signal_data)
    source_id = _get_source_id()

    # Per-signal dedup — key on Dataminr's stable alertId so a
    # redelivered message can't start a parallel copy of this task.
    # `wait_seconds=0` means "if a peer is already processing this
    # signal, return immediately" — the peer will succeed or fail
    # on its own; we don't need to pile on with duplicate Claude
    # calls or duplicate event creation.
    lock_key = f"signal:dataminr:{signal.alertId}"
    with redis_lock(
        lock_key,
        ttl_seconds=_SIGNAL_LOCK_TTL_SECONDS,
        wait_seconds=0,
    ) as acquired:
        if not acquired:
            logger.info(
                "[DATAMINR] alertId=%s already being processed — skipping duplicate delivery",
                signal.alertId,
            )
            return {"alert_id": signal.alertId, "skipped": True}
        return _process_signal_locked(self, signal, source_id, signal_data)


def _process_signal_locked(self, signal: DataminrSignal, source_id: str, signal_data: dict):
    """Body of `process_signal`, executed while holding the per-signal
    dedup lock. Extracted so the lock scope is obvious to readers and the
    (already long) core logic stays flat rather than nested inside a
    `with` block. The original `try` / `except ClaudeRateLimited /
    GraphQLClientError / Exception` structure is preserved here so Celery
    retry semantics are unchanged."""
    try:

        # ─── Stage 1: Ingest signal ──────────────────────────────────────────
        created = ingest_signal(signal, source_id)
        signal_id = created["id"]
        logger.info("Signal ingested: %s", signal_id)

        # ─── Stage 2: Classify ───────────────────────────────────────────────
        # In v2 mode we use the local EventClassifier (no Claude call), which
        # drops ~3k tokens/signal off our Anthropic budget. v1 still relies
        # on Claude for the SignalClassification contract.
        cache_key = f"classification:{signal_id}"
        cached = _redis.get(cache_key)

        if cached:
            classification = SignalClassification.model_validate_json(cached)
        elif settings.grouping_algo == "v2":
            classification = classify_locally(
                title=signal.headline,
                description=created.get("title"),
                source_severity=created.get("severity"),
            )
            _redis.setex(cache_key, 24 * 3600, classification.model_dump_json())
        else:
            # Build context from raw data
            raw_context_parts = []
            if signal.publicPost and signal.publicPost.text:
                raw_context_parts.append(f"Post text: {signal.publicPost.text}")
            if signal.publicPost and signal.publicPost.translatedText:
                raw_context_parts.append(f"Translated: {signal.publicPost.translatedText}")
            if signal.intelAgents:
                for agent in signal.intelAgents:
                    if agent.summary:
                        for s in agent.summary:
                            if s.content:
                                raw_context_parts.append(f"Intel: {s.content[0]}")
            if signal.eventCorroboration and signal.eventCorroboration.summary:
                for s in signal.eventCorroboration.summary:
                    if s.content:
                        raw_context_parts.append(f"Corroboration: {s.content}")
            if signal.liveBrief:
                for lb in signal.liveBrief:
                    if lb.summary:
                        raw_context_parts.append(f"Brief: {lb.summary}")

            raw_context = "\n".join(raw_context_parts) if raw_context_parts else "(no additional context)"

            location_name = None
            if signal.estimatedEventLocation and signal.estimatedEventLocation.name:
                location_name = signal.estimatedEventLocation.name

            disaster_types = _get_disaster_types()

            prompt = build_classify_prompt(
                title=signal.headline,
                description=created.get("title"),
                location_name=location_name,
                url=signal.publicPost.href if signal.publicPost else None,
                timestamp=signal.alertTimestamp,
                raw_context=raw_context,
                disaster_types=disaster_types,
            )

            result_data = call_claude(
                CLASSIFY_SYSTEM,
                prompt,
                stage="classify",
                prompt_version=CLASSIFY_PROMPT_VERSION,
                signal_id=signal_id,
            )
            classification = SignalClassification.model_validate(result_data)

            # Cache classification
            _redis.setex(cache_key, 24 * 3600, classification.model_dump_json())

        logger.info(
            "Signal %s classified: types=%s relevance=%.2f severity=%d",
            signal_id,
            classification.disaster_types,
            classification.relevance,
            classification.severity,
        )

        # Signal severity policy differs between grouping algorithms:
        #   v1 — Claude's classifier score wins (overrides the Dataminr estimate).
        #   v2 — source-provided severity is authoritative; Claude's value is
        #        only used as an event-level fallback and must NOT overwrite
        #        the signal row. A signal with no source severity stays null.
        existing_severity = created.get("severity")
        if settings.grouping_algo == "v1" and existing_severity != classification.severity:
            update_signal_severity(signal_id, classification.severity)
            logger.info(
                "Signal %s severity updated (v1): %s → %d",
                signal_id, existing_severity, classification.severity,
            )

        # ─── Stage 3: Event grouping (if relevant) ──────────────────────────
        if classification.relevance < settings.relevance_threshold:
            logger.info(
                "Signal %s below relevance threshold (%.2f < %.2f), skipping event grouping",
                signal_id,
                classification.relevance,
                settings.relevance_threshold,
            )
            return {
                "signal_id": signal_id,
                "classification": classification.model_dump(),
                "event": None,
                "alert": None,
            }

        location_name = None
        signal_lat = None
        signal_lng = None
        probability_radius_km = None
        if signal.estimatedEventLocation:
            location_name = signal.estimatedEventLocation.name
            if signal.estimatedEventLocation.coordinates and len(signal.estimatedEventLocation.coordinates) >= 2:
                signal_lat = signal.estimatedEventLocation.coordinates[0]
                signal_lng = signal.estimatedEventLocation.coordinates[1]
            if signal.estimatedEventLocation.probabilityRadius is not None:
                probability_radius_km = signal.estimatedEventLocation.probabilityRadius

        # Use location resolved by Claude/API during signal creation
        origin_loc = created.get("originLocation")
        general_loc = created.get("generalLocation")
        dest_loc = created.get("destinationLocation")
        # For event grouping, prefer the most specific resolved location name
        resolved_loc = origin_loc or general_loc or dest_loc
        if resolved_loc:
            location_name = resolved_loc.get("name") or location_name
        origin_id = origin_loc["id"] if origin_loc else (general_loc["id"] if general_loc else None)

        # Best-effort populationAffected from raw text (Dataminr has no
        # structured field for this). The grouping layer prefers this over
        # the per-event-type stats lookup when set.
        from src.services.signal import extract_population_affected_from_text
        actual_pop = extract_population_affected_from_text(
            signal.headline, created.get("title"), created.get("description"),
        )

        event = dispatch_group_signal(
            signal_id=signal_id,
            signal_title=signal.headline,
            signal_description=created.get("title"),
            signal_location_name=location_name,
            signal_origin_id=origin_id,
            signal_timestamp=signal.alertTimestamp,
            classification=classification,
            signal_lat=signal_lat,
            signal_lng=signal_lng,
            probability_radius_km=probability_radius_km,
            created_signal=created,
            signal_actual_population_affected=actual_pop,
        )

        # ─── Stage 4: Alert escalation (if high severity) ───────────────────
        alert = None
        if event and classification.severity >= 4:
            alert = maybe_escalate(
                event=event,
                signal_summaries=[classification.summary],
                max_severity=classification.severity,
                signal_published_at=signal.alertTimestamp,
            )

        return {
            "signal_id": signal_id,
            "classification": classification.model_dump(),
            "event_id": event["id"] if event else None,
            "alert_id": alert["id"] if alert else None,
        }

    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] process_signal backing off %.0fs",
            exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except GraphQLClientError as exc:
        # 4xx from clear-api = bug in our request. Fail loudly and stop;
        # retrying would only create duplicate rows (see populationDisplaced
        # incident that spawned the Wave-4 fix).
        logger.error("process_signal permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("process_signal failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=10)


# Trusted source names that get auto-escalated to alerts
TRUSTED_SOURCE_NAMES = {"field_officer", "partner", "government"}


@app.task(
    name="src.tasks.process.process_manual_signal",
    bind=True,
    max_retries=2,
    acks_late=True,
    soft_time_limit=240,
    time_limit=300,
)
def process_manual_signal(
    self,
    signal_id: str,
    source_type: str,
    title: str,
    description: str,
    severity: int | None = None,
    user_id: str = "",
    signal_published_at: str | None = None,
):
    """
    Process a manually created signal from a trusted source:
    1. Classify via Claude (disaster type, severity)
    2. Group into event (new or existing)
    3. If source is trusted (field_officer/partner/government): auto-escalate the event to alert
       and record the user escalation in eventEscaladedByUsers
    """
    # Per-signal dedup — signal_id is the natural key. If a broker
    # heartbeat drop redelivered the message while the first worker is
    # still classifying / geoparsing / grouping, the second worker
    # returns immediately instead of running the whole pipeline in
    # parallel and creating a duplicate event.
    lock_key = f"signal:manual:{signal_id}"
    with redis_lock(
        lock_key,
        ttl_seconds=_SIGNAL_LOCK_TTL_SECONDS,
        wait_seconds=0,
    ) as acquired:
        if not acquired:
            logger.info(
                "[MANUAL] signal_id=%s already being processed — skipping duplicate delivery",
                signal_id,
            )
            return {"signal_id": signal_id, "skipped": True}
        return _process_manual_signal_locked(
            self, signal_id, source_type, title, description, severity, user_id, signal_published_at,
        )


def _process_manual_signal_locked(
    self,
    signal_id: str,
    source_type: str,
    title: str,
    description: str,
    severity: int | None,
    user_id: str,
    signal_published_at: str | None,
):
    """Body of `process_manual_signal`, executed while holding the
    per-signal dedup lock. Original try / except structure preserved
    below so Celery retry semantics on ClaudeRateLimited /
    GraphQLClientError / unhandled exceptions are unchanged."""
    try:
        # ─── Stage 1: Classify ────────────────────────────────────────────────
        if settings.grouping_algo == "v2":
            classification = classify_locally(
                title=title,
                description=description,
                source_severity=severity,
            )
        else:
            disaster_types = _get_disaster_types()
            prompt = build_classify_prompt(
                title=title,
                description=description,
                location_name=None,
                url=None,
                timestamp=None,
                raw_context=f"Manual signal from {source_type} source. Description: {description}",
                disaster_types=disaster_types,
            )
            result_data = call_claude(
                CLASSIFY_SYSTEM,
                prompt,
                stage="classify",
                prompt_version=CLASSIFY_PROMPT_VERSION,
                signal_id=signal_id,
            )
            classification = SignalClassification.model_validate(result_data)

        logger.info(
            "Manual signal %s classified: types=%s severity=%d",
            signal_id,
            classification.disaster_types,
            classification.severity,
        )

        # ─── Fetch the signal state up front ─────────────────────────────────
        # We need to know whether the user picked a location before the
        # geoparser runs, so the geoparser stage can decide whether to
        # promote its candidate into `signal.locationId` (falls back only
        # when no user pick — see the geoparser stage below). The same
        # `created_signal` handle also feeds event grouping after any
        # promotion so grouping sees the up-to-date location.
        try:
            created_signal = get_signal(signal_id) or {}
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "[manual:%s] failed to fetch signal for grouping (falling back to isolated-event behaviour): %s",
                signal_id, exc,
            )
            created_signal = {}

        has_user_location = bool(
            (created_signal.get("generalLocation") or {}).get("id")
        )

        # ─── Geoparser enrichment ─────────────────────────────────────────────
        # Manual signals are created in clear-api before the pipeline runs, so
        # the geoparser fires here as post-hoc enrichment.
        #
        # Two effects when a candidate resolves:
        #   1. Structured result → `signals.geoparsed_data` (always).
        #   2. Promoted L4 → `signals.location_id` (ONLY when the user
        #      didn't pick a location). This is the fix that stops
        #      geoparser-resolvable signals from becoming isolated events
        #      just because the human left the location blank.
        # We deliberately do NOT override a user-picked location — the UI
        # picker is the source of truth for manual signals; the geoparser is
        # a fallback for the empty-picker case.
        logger.info(
            "[manual:%s] Running geoparser (user_location=%s)",
            signal_id, has_user_location,
        )
        try:
            geo_result = geoparse_signal(title, description)
            if geo_result is not None:
                update_signal_geoparsed_data(signal_id, geoparse_to_dict(geo_result))
                logger.info(
                    "[manual:%s] Geoparsed: candidate=%r kind=%s importance=%.2f",
                    signal_id, geo_result.candidate, geo_result.kind, geo_result.importance,
                )
                if not has_user_location:
                    try:
                        promo = find_or_create_landmark_l4(
                            name=geo_result.candidate,
                            lat=geo_result.lat,
                            lng=geo_result.lng,
                            kind=geo_result.kind,
                        )
                        if promo.get("locationId") and not promo.get("abortedReason"):
                            update_signal_location(signal_id, promo["locationId"])
                            # Re-fetch so `created_signal` sees the new
                            # location before event grouping runs. Best-
                            # effort: on lookup failure we fall through with
                            # the pre-promotion snapshot (grouping still
                            # gets the geoparser's raw data, just not the
                            # newly-set location).
                            refreshed = get_signal(signal_id)
                            if refreshed:
                                created_signal = refreshed
                            logger.info(
                                "[manual:%s] Promoted geoparser candidate to L4 %s → set as signal.locationId (reused=%s, point_type=%s)",
                                signal_id, promo["locationId"],
                                promo.get("reused"), promo.get("pointType"),
                            )
                        elif promo.get("abortedReason"):
                            logger.info(
                                "[manual:%s] L4 promotion aborted (%s) — signal stays unlocated",
                                signal_id, promo["abortedReason"],
                            )
                    except Exception as exc:  # noqa: BLE001 — promotion best-effort
                        logger.warning(
                            "[manual:%s] L4 promotion failed: %s",
                            signal_id, exc,
                        )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("[manual:%s] Geoparser enrichment failed: %s", signal_id, exc)

        # Update severity. v1: Claude's classifier value is a valid fallback.
        # v2: only write if the caller provided a source severity — otherwise
        # the signal stays null and the event-level calculator handles it.
        if severity is not None:
            update_signal_severity(signal_id, severity)
        elif settings.grouping_algo == "v1":
            update_signal_severity(signal_id, classification.severity)

        # ─── Stage 2: Event grouping ──────────────────────────────────────────
        # `created_signal` was fetched above (and refreshed after promotion,
        # when the geoparser gave us a location the human didn't). Grouping
        # walks `signal.generalLocation.ancestorIds` to find admin-2, so an
        # L4 that resolves to a real A2 will now cluster instead of going
        # isolated.
        from src.services.signal import extract_population_affected_from_text
        actual_pop = extract_population_affected_from_text(title, description)

        event = dispatch_group_signal(
            signal_id=signal_id,
            signal_title=title,
            signal_description=description,
            signal_location_name=None,
            signal_origin_id=None,
            signal_timestamp=None,
            classification=classification,
            created_signal=created_signal,
            signal_actual_population_affected=actual_pop,
        )

        if not event:
            logger.warning("Manual signal %s: event grouping failed", signal_id)
            return {
                "signal_id": signal_id,
                "classification": classification.model_dump(),
                "event_id": None,
                "alert_id": None,
                "escalated": False,
            }

        # ─── Stage 3: Auto-escalate for trusted sources ──────────────────────
        # Two gates protect against runaway notifications:
        #   1. Severity floor — matches the >=4 gate the auto-poll paths
        #      (Dataminr/GDACS/ACLED) apply. A low-severity manual report
        #      shouldn't email every subscriber.
        #   2. Staleness gate — a manual report about a week-old incident
        #      shouldn't fan out as a live alert; same threshold the
        #      maybe_escalate path uses for auto-polled signals.
        escalated = False
        if source_type not in TRUSTED_SOURCE_NAMES:
            pass  # only trusted sources auto-escalate at all
        elif classification.severity < 4:
            logger.info(
                "[ALERT] Manual signal %s: skipping auto-escalation (severity=%d < 4)",
                signal_id, classification.severity,
            )
        elif is_stale_signal(signal_published_at):
            logger.info(
                "[ALERT] Manual signal %s: skipping auto-escalation — "
                "publishedAt=%s is older than %dh",
                signal_id, signal_published_at, settings.alert_max_signal_age_hours,
            )
        else:
            logger.info(
                "Trusted source (%s) — auto-escalating event %s to alert",
                source_type, event["id"],
            )
            try:
                escalation = escalate_event(event["id"], user_id)
                escalated = True
                logger.info(
                    "Event %s escalated: escalation_id=%s",
                    event["id"], escalation["id"],
                )
            except Exception as e:
                logger.error("Failed to escalate event %s: %s", event["id"], e)

        return {
            "signal_id": signal_id,
            "classification": classification.model_dump(),
            "event_id": event["id"],
            "escalated": escalated,
        }

    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] process_manual_signal backing off %.0fs",
            exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except GraphQLClientError as exc:
        logger.error("process_manual_signal permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("process_manual_signal failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=10)


@app.task(
    name="src.tasks.process.process_gdacs_signal",
    bind=True,
    max_retries=2,
    acks_late=True,
    soft_time_limit=240,
    time_limit=300,
)
def process_gdacs_signal(
    self,
    signal_id: str,
    gdacs_event: dict,
    created_signal: dict | None = None,
):
    """
    Process a GDACS-sourced signal.

    GDACS events are already structured disaster data, so we:
    1. Build classification directly from GDACS metadata (skip Claude classification)
    2. Group into event (new or existing) via Claude
    3. Assess for alert escalation if high severity
    """
    # Per-signal dedup — same rationale as process_manual_signal above.
    lock_key = f"signal:gdacs:{signal_id}"
    with redis_lock(
        lock_key,
        ttl_seconds=_SIGNAL_LOCK_TTL_SECONDS,
        wait_seconds=0,
    ) as acquired:
        if not acquired:
            logger.info(
                "[GDACS] signal_id=%s already being processed — skipping duplicate delivery",
                signal_id,
            )
            return {"signal_id": signal_id, "skipped": True}
        return _process_gdacs_signal_locked(self, signal_id, gdacs_event, created_signal)


def _process_gdacs_signal_locked(
    self,
    signal_id: str,
    gdacs_event: dict,
    created_signal: dict | None,
):
    """Body of `process_gdacs_signal`, executed while holding the per-signal
    dedup lock."""
    try:
        glide_type = gdacs_event.get("glide_type", "ot")
        severity = gdacs_event.get("severity", 3)
        alert_level = gdacs_event.get("alert_level", "Green")
        title = gdacs_event.get("title", "GDACS event")
        description = gdacs_event.get("description") or ""
        location_name = gdacs_event.get("country")
        population_affected = gdacs_event.get("population_affected")

        # Enrich description with population data if available
        if population_affected:
            description = f"{description} Approximately {population_affected:,} people affected."

        # Build classification from GDACS metadata (no Claude needed)
        classification = SignalClassification(
            disaster_types=[glide_type],
            relevance=1.0 if alert_level in ("Red", "Orange") else 0.7,
            severity=severity,
            summary=f"GDACS {alert_level} alert: {title}" + (f" ({population_affected:,} affected)" if population_affected else ""),
        )

        logger.info(
            "GDACS signal %s: type=%s severity=%d alert=%s",
            signal_id, glide_type, severity, alert_level,
        )

        # Update signal severity
        update_signal_severity(signal_id, severity)

        # Skip low-relevance events
        if classification.relevance < settings.relevance_threshold:
            logger.info("GDACS signal %s below relevance threshold, skipping", signal_id)
            return {"signal_id": signal_id, "event_id": None, "alert_id": None}

        # Group into event (no probabilityRadius for GDACS — uses default 1km)
        event = dispatch_group_signal(
            signal_id=signal_id,
            signal_title=title,
            signal_description=description,
            signal_location_name=location_name,
            signal_origin_id=None,
            signal_timestamp=gdacs_event.get("from_date"),
            classification=classification,
            signal_lat=gdacs_event.get("lat"),
            signal_lng=gdacs_event.get("lng"),
            created_signal=created_signal or {},
            # GDACS exposure data ships an actual affected-population estimate.
            signal_actual_population_affected=population_affected,
        )

        # Assess for alert if high severity (Red/Orange)
        alert = None
        if event and severity >= 4:
            alert = maybe_escalate(
                event=event,
                signal_summaries=[classification.summary],
                max_severity=severity,
                signal_published_at=gdacs_event.get("from_date"),
            )

        return {
            "signal_id": signal_id,
            "event_id": event["id"] if event else None,
            "alert_id": alert["id"] if alert else None,
        }

    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] process_gdacs_signal backing off %.0fs",
            exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except GraphQLClientError as exc:
        logger.error("process_gdacs_signal permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("process_gdacs_signal failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=10)


@app.task(
    name="src.tasks.process.process_acled_signal",
    bind=True,
    max_retries=2,
    acks_late=True,
    soft_time_limit=240,
    time_limit=300,
)
def process_acled_signal(
    self,
    signal_id: str,
    acled_event: dict,
    created_signal: dict | None = None,
):
    """
    Process an ACLED-sourced signal.

    ACLED events are structured conflict data, so we:
    1. Build classification directly from ACLED metadata (skip Claude classification)
    2. Group into event (new or existing) via Claude
    3. Assess for alert escalation if high severity (fatalities / event type)
    """
    # Per-signal dedup — same rationale as process_manual_signal above.
    lock_key = f"signal:acled:{signal_id}"
    with redis_lock(
        lock_key,
        ttl_seconds=_SIGNAL_LOCK_TTL_SECONDS,
        wait_seconds=0,
    ) as acquired:
        if not acquired:
            logger.info(
                "[ACLED] signal_id=%s already being processed — skipping duplicate delivery",
                signal_id,
            )
            return {"signal_id": signal_id, "skipped": True}
        return _process_acled_signal_locked(self, signal_id, acled_event, created_signal)


def _process_acled_signal_locked(
    self,
    signal_id: str,
    acled_event: dict,
    created_signal: dict | None,
):
    """Body of `process_acled_signal`, executed while holding the per-signal
    dedup lock."""
    try:
        glide_type = acled_event.get("glide_type", "ot")
        severity = acled_event.get("severity", 2)
        fatalities = acled_event.get("fatalities", 0)
        title = acled_event.get("title", "ACLED event")
        description = acled_event.get("description") or ""
        location_name = (
            acled_event.get("location")
            or acled_event.get("admin2")
            or acled_event.get("admin1")
            or acled_event.get("country")
        )

        # Build classification from ACLED metadata
        summary = f"ACLED {acled_event.get('event_type', 'conflict')} event"
        if fatalities:
            summary += f" ({fatalities} fatalities)"

        classification = SignalClassification(
            disaster_types=[glide_type],
            relevance=1.0 if fatalities > 0 or severity >= 4 else 0.8,
            severity=severity,
            summary=summary,
        )

        logger.info(
            "ACLED signal %s: type=%s severity=%d fatalities=%d",
            signal_id, glide_type, severity, fatalities,
        )

        # Update signal severity
        update_signal_severity(signal_id, severity)

        # Skip low-relevance events
        if classification.relevance < settings.relevance_threshold:
            logger.info("ACLED signal %s below relevance threshold, skipping", signal_id)
            return {"signal_id": signal_id, "event_id": None, "alert_id": None}

        # Group into event
        event = dispatch_group_signal(
            signal_id=signal_id,
            signal_title=title,
            signal_description=description,
            signal_location_name=location_name,
            signal_origin_id=None,
            signal_timestamp=acled_event.get("event_date"),
            classification=classification,
            signal_lat=acled_event.get("lat"),
            signal_lng=acled_event.get("lng"),
            created_signal=created_signal or {},
        )

        # Assess for alert if high severity
        alert = None
        if event and severity >= 4:
            alert = maybe_escalate(
                event=event,
                signal_summaries=[classification.summary],
                max_severity=severity,
                signal_published_at=acled_event.get("event_date"),
            )

        return {
            "signal_id": signal_id,
            "event_id": event["id"] if event else None,
            "alert_id": alert["id"] if alert else None,
        }

    except ClaudeRateLimited as exc:
        logger.warning(
            "[CLAUDE RATE-LIMIT] process_acled_signal backing off %.0fs",
            exc.retry_after,
        )
        raise self.retry(exc=exc, countdown=int(exc.retry_after))
    except GraphQLClientError as exc:
        logger.error("process_acled_signal permanently failed (non-retryable): %s", exc)
        raise
    except Exception as exc:
        logger.error("process_acled_signal failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=10)
