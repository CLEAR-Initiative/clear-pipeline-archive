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
   corrections and retractions of an earlier report — form one thread. The source's
   existing threads (fetched via `groundThreadsForSource`) are offered to the model as
   append targets, so a message that continues an incident threaded in an EARLIER run
   is appended to that thread (upsert with `threadId`) rather than starting a new one.
   Threads already promoted for review are never append targets.
4. **Set lifecycle states** `reported` | `updated` | `confirmed` | `corrected` |
   `retracted`. Deterministic rules floor the model's proposal: an explicit withdrawal
   ("this turned out to be misreporting") always yields `retracted` — including a
   single-message thread that is itself a lone retraction, which must never read as a
   fresh report; any other single-message thread is `reported`. For appends, the target
   thread's earlier messages count towards its size, so a follow-up never degrades an
   existing multi-message thread to `reported`.

Messages the model labels outside the vocabulary (or fails to label) are left
unclassified and retried on a later run — never written back with a guessed class.

## GraphQL surface consumed (clear-api)

Contract v2 — the pipeline side of clear-api's ground-intel schema:

```graphql
query groundMessagesForClassification(groundSourceId: String!, limit: Int)
  # → [{id, text, sentAt, senderRef, hasMedia, classification, threadId}]

query groundThreadsForSource(groundSourceId: String!, states: [String!])
  # → [{id, title, lifecycleState, reviewState, messageIds}]
  # fetched before the threading stage so later runs can append to
  # threads created by earlier runs

mutation upsertGroundMessageClassifications(
  inputs: [GroundMessageClassificationInput!]!   # {messageId, classification, uncertaintyMarker}
): Int!   # scalar count of upserted rows — no selection set

mutation upsertGroundThreads(
  inputs: [GroundThreadUpsertInput!]!
  # {groundSourceId, title, lifecycleState, messageIds, threadId?}
  # threadId set → the server APPENDS messageIds to that existing
  # (non-promoted) thread and updates its lifecycleState/title,
  # instead of creating a new thread
): [String]!  # thread ids, index-aligned with inputs (entries nullable)
```

Auth: the same pipeline bearer token every other call in
[src/clients/graphql.py](../src/clients/graphql.py) uses. Message text arrives already
redacted (phone numbers stripped at persistence, pseudonymous `senderRef`) — nothing in
this pipeline sees or stores personal identifiers.

## Retry semantics

Standard for this repo: `GraphQLClientError` (4xx) raises without retry; Claude
rate-limits reschedule with the suggested `retry_after`; anything else retries up to 3
times with a 60s countdown.

**Idempotency & cross-run behaviour.** A re-run re-fetches and only touches
still-unclassified / un-threaded rows, so repeating a run is safe. Threading is
additionally cross-run aware: each run fetches the source's existing threads and offers
them as append targets, so a correction or retraction that lands in a LATER run than
its incident — or the tail of an incident that straddles two fetch windows — appends
to the original thread (updating its lifecycle, e.g. flipping it to `retracted`)
instead of minting an orphan single-message thread. The honest limits: the model only
sees messages inside the current fetch window, so an existing thread is summarised by
its title/lifecycle plus whichever of its member texts are still in view; appending to
threads already promoted for review is not possible (the server refuses, and the
pipeline never offers them); and if the model fails to recognise a continuation it
still becomes a separate thread for a reviewer to merge — cross-run threading improves
convergence, it does not guarantee it.

## Code map

- [src/tasks/ground.py](../src/tasks/ground.py) — the Celery task + contract docstring
- [src/services/ground_intel.py](../src/services/ground_intel.py) — classification,
  uncertainty/retraction detection, thread validation
- [src/prompts/ground.py](../src/prompts/ground.py) — `ground_classify` /
  `ground_thread` prompts
- [src/models/ground.py](../src/models/ground.py) — response models + vocabularies
- [tests/test_ground_intel.py](../tests/test_ground_intel.py) — hermetic suite over a
  fully synthetic fixture
