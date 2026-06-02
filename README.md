# CLEAR Pipeline

> **Status: In Development** — This project is under active development and not yet production-ready.

Data ingestion + enrichment pipeline that ingests signals from multiple
external sources (Dataminr, ACLED, GDACS, and manual entries), classifies and
groups them into events, raises alerts, and enriches user-curated crises with
LLM-generated narrative, scenarios, and an NRC SAF needs analysis.

## Architecture

```
┌───────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────────┐
│ Dataminr poll │   │ ACLED    │   │ GDACS     │   │ Manual signal    │
│ (15s)         │   │ poll     │   │ poll      │   │ (createManual…)  │
└───────┬───────┘   └────┬─────┘   └────┬──────┘   └────────┬─────────┘
        │                │              │                    │
        └────────────────┴──────┬───────┴────────────────────┘
                                ▼
                       ┌─────────────────────┐
                       │ build_signal_input  │  ← geoparser (Nominatim cache)
                       │ + create_signal     │    ↳ promote landmark → L4
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ process_signal /    │
                       │ process_manual…     │
                       └──────────┬──────────┘
                                  ▼
                  ┌───────────────┴──────────────────┐
                  │  EventClassifier (local model)   │
                  │  → dispatch_group_signal_v2      │
                  └───────────────┬──────────────────┘
                                  ▼
                  ┌───────────────┴──────────────────┐
                  │  Event grouping → maybe_escalate │
                  │  (alerts ≥ severity 4)           │
                  └──────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────┐
   │ Crisis enrichment (triggered by clear-api on create / add / │
   │ remove event): narrative + scenarios + SAF needs analysis,  │
   │ written back via setCrisisNeedsAnalysis / updateCrisisPopulation │
   └─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Install dependencies
pip install uv
uv pip install --system .

# Copy env and fill in values
cp .env.example .env

# Run with Docker
docker compose up -d

# Or run locally (requires Redis)
celery -A src.celery_app worker --beat --loglevel=info
```

## Configuration

See `.env.example` for the full list. Key variables:

| Variable                       | Description                                                              |
| ------------------------------ | ------------------------------------------------------------------------ |
| `CLEAR_API_URL`                | CLEAR GraphQL API endpoint                                               |
| `CLEAR_API_KEY`                | Service account API key (`sk_live_…`)                                    |
| `REDIS_URL`                    | Redis connection URL (Celery broker + Nominatim rate-limit / circuit)    |
| `CELERY_BROKER_URL`            | Celery broker (typically same Redis instance)                            |
| **Dataminr**                   |                                                                          |
| `DATAMINR_CLIENT_ID`           | Dataminr API client ID                                                   |
| `DATAMINR_CLIENT_SECRET`       | Dataminr API client secret                                               |
| **ACLED**                      |                                                                          |
| `ACLED_USERNAME`               | ACLED account                                                            |
| `ACLED_PASSWORD`               | ACLED password                                                           |
| `ACLED_COUNTRIES`              | Country filter (default: `Sudan`)                                        |
| `ACLED_POLL_INTERVAL_MINUTES`  | Poll cadence (default: 60)                                               |
| **GDACS**                      |                                                                          |
| `GDACS_COUNTRIES`              | Country filter (default: `Sudan,South Sudan,Ethiopia`)                   |
| `GDACS_POLL_INTERVAL_MINUTES`  | Poll cadence (default: 30)                                               |
| **LLM**                        |                                                                          |
| `ANTHROPIC_API_KEY`            | Claude API key (classification, grouping, crisis enrichment)             |
| `CLAUDE_MODEL`                 | Default model (e.g. `claude-sonnet-4-6`)                                 |
| **Geocoder (LocationIQ)**      |                                                                          |
| `LOCATIONIQ_API_KEY`           | LocationIQ key. Empty disables the geoparser entirely — signals fall back to source coords. |
| `LOCATIONIQ_BASE_URL`          | LocationIQ endpoint (default `https://us1.locationiq.com/v1`)            |
| **IOM DTM**                    |                                                                          |
| `IOM_DTM_SUBSCRIPTION_KEY`     | IOM DTM API key. Empty disables DTM backfill.                            |
| **Pipeline tuning**            |                                                                          |
| `POLL_INTERVAL_SECONDS`        | Dataminr poll cadence (default: 15)                                      |
| `INITIAL_LOOKBACK_DAYS`        | First-run lookback window                                                |
| `RELEVANCE_THRESHOLD`          | Min relevance for event creation                                         |
| `GROUPING_ALGO`                | `v1` (Claude-driven) or `v2` (local EventClassifier; default in prod)    |
| `DEFAULT_POPULATION_AFFECTED`  | Last-resort default (`33_000`) when source / per-event-type lookup yield nothing |
| `DEFAULT_POPULATION_DISPLACED` | Last-resort default for `population_displaced` (`1670`)                  |
| `ALERT_MAX_SIGNAL_AGE_HOURS`   | Suppress alert escalation for backdated signals (default: 48)            |

## Pipeline Flow

**Signal ingestion** (every source converges on the same pipeline):

1. **Poll** — Celery beat triggers `poll_dataminr` / `poll_acled` / `poll_gdacs` on per-source cadences. Manual signals come in via `createManualSignal` → Celery task `process_manual_signal`.
2. **Build input** — Source payload → `CreateSignalInput`. Casualties extracted from text (digits and English number words: `"fourteen killed"`); population_affected extracted from text.
3. **Geoparser** — Title + body fed to a regex + LocationIQ-backed geoparser. On a landmark hit, calls `findOrCreateLandmarkL4` on clear-api to promote the candidate into a reusable `point_type = 'landmark-geocoded'` L4. Same-A2 safety check prevents misattribution.
4. **Create signal** — `createSignal` mutation. Pipeline ships `rawData`, `geoparsedData`, and either an explicit `locationId` (landmark hit) or `lat/lng` (source coords).
5. **Classify** — `EventClassifier` (local sentence-transformers model) returns level_1/level_2/level_3 + glide code. Confidence threshold filters irrelevant signals.
6. **Group** — `group_signal_v2` clusters into existing or new event keyed on `(admin_2, level_2)`. Uses per-event-type stats for casualties / populationAffected fallback.
7. **Escalate** — Severity ≥ 4 signals run through `maybe_escalate`. Trusted manual sources auto-escalate the event to a `draft` alert.

**Crisis enrichment** (triggered on `createCrisisFromEvents` / `addEventToCrisis` / `removeEventFromCrisis`):

1. `enrich_crisis` Celery task receives `crisis_id` + `event_ids`.
2. Computes `populationInArea` from district-level WorldPop / cached `locations.population`.
3. Three sequential Claude calls (with `max_tokens=4096` for needs analysis, default for the others):
   - **Narrative** — title + JSON `{description, tldr}` for `crises.summary`.
   - **Scenarios** — `{most_likely, best_case, worst_case, description}` stored on `crises.scenarios`.
   - **Needs analysis** — NRC SAF framework: `generalSummary[]` (4 bullets) + `sector{}` (6 canonical sectors with `description`, `severity`, `responseGap`, `nrcRelevant`). Merged into `crises.needs` JSONB.
4. Transient Anthropic errors (`OverloadedError`, rate limits) bubble out of the generators → Celery retries the whole task with 120s backoff. Title regeneration is skipped if the crisis has any `userFeedbacks` row with a `[title-edit]` marker (= user manually edited the title).

## Project Structure

```
src/
├── config.py                 # Settings from .env (pydantic-settings)
├── celery_app.py             # Celery app + beat schedule
│
├── tasks/
│   ├── poll.py               # poll_dataminr
│   ├── poll_acled.py         # poll_acled
│   ├── poll_gdacs.py         # poll_gdacs
│   ├── process.py            # process_signal, process_manual_signal
│   ├── crisis.py             # enrich_crisis (LLM narrative + scenarios + SAF)
│   ├── dtm.py                # IOM DTM displacement backfill
│   └── population.py         # WorldPop population estimation
│
├── clients/
│   ├── dataminr.py           # Dataminr First Alert API
│   ├── acled.py              # ACLED API
│   ├── gdacs.py              # GDACS feed
│   ├── iom_dtm.py            # IOM DTM API
│   ├── nominatim.py          # LocationIQ-backed geocoder (cache + rate-limit + circuit)
│   ├── graphql.py            # All CLEAR API mutations + queries
│   └── claude.py             # Anthropic SDK wrapper with JSON extraction
│
├── models/
│   ├── dataminr.py           # Dataminr response schemas
│   └── clear.py              # GraphQL input + LLM output schemas
│                             # (CrisisNarrative, CrisisScenarios, CrisisNeedsAnalysis…)
│
├── services/
│   ├── signal.py             # Signal field mapping + casualties/pop extraction
│   ├── geoparser.py          # Regex extraction → Nominatim → L4 promotion
│   ├── event.py              # Event grouping entry point
│   ├── event_grouping_v2.py  # v2 district+type grouping with local classifier
│   ├── event_classifier.py   # Local sentence-transformers classifier
│   ├── event_type_stats.py   # Per-event-type stats lookup (acled_event_type_stats.json)
│   ├── classifier_singleton.py
│   ├── local_classify.py     # Bridge: classifier → SignalClassification contract
│   ├── alert.py              # Alert escalation
│   ├── location.py           # Claude-based location resolution (fallback)
│   ├── admin_resolver.py     # Resolve signal → admin-2 via PostGIS
│   ├── population.py         # WorldPop raster-based population estimation
│   └── geo.py                # Legacy haversine resolver
│
└── prompts/
    ├── classify.py           # Signal classification prompt
    ├── group.py              # v1 event grouping prompt
    ├── rewrite.py            # v2 event rewrite prompt
    ├── assess.py             # Alert assessment prompt
    └── crisis.py             # Crisis narrative + scenarios + SAF needs prompts

tests/
├── test_geoparser.py
├── test_signal_geoparser.py
├── test_event_classifier.py
├── test_event_grouping_casualties.py
├── test_casualty_extraction.py
├── test_crisis_narrative.py
├── test_process_manual_signal.py
└── test_alert.py
```

## Testing

```bash
# Fast suite (skips sentence-transformers loading)
pytest tests/ --ignore=tests/test_event_classifier.py

# Full suite including the slow classifier tests (~40s for model load)
pytest tests/
```

## License

Copyright (C) 2026 Norwegian Refugee Council.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See [`LICENSE`](./LICENSE) for the full text.
