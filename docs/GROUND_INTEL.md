# Ground intel: classification + incident threading

Part of the **WhatsApp Signal Pipeline** (V1 seeded shared core). clear-api owns the
data — the `groundSources` / `groundThreads` / `groundMessages` staging tier sitting in
front of the `signals → events → alerts` graph — and this pipeline owns the
intelligence, mirroring the Dataminr split. This work is deliberately a **Celery task on
clear-pipeline** (not a Dagster asset in clear-context-pipeline): it is operational
signal triage, not Layer-3 knowledge-base work.

## Task contract

Consumed by clear-api's celery service, which enqueues the task after a chat-export
import (the clear-api side of the wiring ships with expo-367):

```
Task name:  classify_ground_messages     (bare name, NOT the module-path convention)
Signature:  (ground_source_id: str)
Module:     src/tasks/ground.py
```

One run processes all unclassified `groundMessages` for the source:

1. **Classify** each message as `field_report` | `news_digest` | `operational` |
   `chatter` (batched Claude calls, stage `ground_classify`, Haiku by default).
2. **Preserve uncertainty markers** — contributor tags like "unconfirmed" / "rumour" are
   detected with regexes (never model judgement) and written back with the
   classification.
3. **Thread** un-threaded field reports into incident threads (one Claude call, stage
   `ground_thread`): messages describing the same real-world incident — including
   corrections and retractions of an earlier report — form one thread.
4. **Set lifecycle states** `reported` | `updated` | `confirmed` | `corrected` |
   `retracted`. Deterministic rules floor the model's proposal: an explicit withdrawal
   ("this turned out to be misreporting") always yields `retracted`; a single-message
   thread is always `reported`.

Messages the model labels outside the vocabulary (or fails to label) are left
unclassified and retried on a later run — never written back with a guessed class.

## GraphQL surface consumed (clear-api)

```graphql
query groundMessagesForClassification(groundSourceId: String!, limit: Int)
  # → [{id, text, sentAt, senderRef, hasMedia, classification, threadId}]

mutation upsertGroundMessageClassifications(
  inputs: [{messageId, classification, uncertaintyMarker}]
)

mutation upsertGroundThreads(
  inputs: [{groundSourceId, title, lifecycleState, messageIds}]
)  # → thread ids
```

Auth: the same pipeline bearer token every other call in
[src/clients/graphql.py](../src/clients/graphql.py) uses. Message text arrives already
redacted (phone numbers stripped at persistence, pseudonymous `senderRef`) — nothing in
this pipeline sees or stores personal identifiers.

## Retry semantics

Standard for this repo: `GraphQLClientError` (4xx) raises without retry; Claude
rate-limits reschedule with the suggested `retry_after`; anything else retries up to 3
times with a 60s countdown. The task is idempotent per source — a re-run re-fetches and
only touches still-unclassified / un-threaded rows.

## Code map

- [src/tasks/ground.py](../src/tasks/ground.py) — the Celery task + contract docstring
- [src/services/ground_intel.py](../src/services/ground_intel.py) — classification,
  uncertainty/retraction detection, thread validation
- [src/prompts/ground.py](../src/prompts/ground.py) — `ground_classify` /
  `ground_thread` prompts
- [src/models/ground.py](../src/models/ground.py) — response models + vocabularies
- [tests/test_ground_intel.py](../tests/test_ground_intel.py) — hermetic suite over a
  fully synthetic fixture
