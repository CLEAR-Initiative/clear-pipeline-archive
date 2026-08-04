"""Ground-intel prompts: WhatsApp message classification + incident threading.

Part of the WhatsApp Signal Pipeline (see docs/GROUND_INTEL.md). Messages
arrive from clear-api's ground-intel staging tier already redacted (phone
numbers stripped, pseudonymous sender refs) — these prompts never see or
produce personal identifiers.
"""

# Bump whenever the prompt text changes — the insights dashboard groups calls
# by version to track quality across iterations.
GROUND_CLASSIFY_PROMPT_VERSION = "ground-classify-v1"

GROUND_CLASSIFY_SYSTEM = """\
You are a humanitarian intelligence analyst for the CLEAR early warning system focused on Sudan.

You triage messages captured from WhatsApp field-reporting groups. Roughly one message in
three carries intelligence value; your classification decides which messages feed incident
tracking.

You MUST respond with valid JSON only — no markdown, no explanation."""

GROUND_CLASSIFY_USER_TEMPLATE = """\
Classify each of the following group messages.

Messages (id | sent at | sender ref | text):
{messages_block}

Respond with this exact JSON structure:
{{
  "classifications": [
    {{"id": "<message id>", "classification": "<label>"}},
    ...
  ]
}}

Labels (choose exactly one per message):
- field_report: a first-hand or relayed report of a specific incident on the ground — attack,
  shelling, drone strike, movement incident, checkpoint, displacement, casualties. Includes
  corrections or retractions of earlier incident reports, and media-only messages whose
  caption or context reports an incident.
- news_digest: a summary or roundup of published news — items sourced from media outlets,
  agencies, or radio rather than direct observation.
- operational: internal coordination — staff movement plans, meeting logistics,
  administrative requests, security-procedure reminders.
- chatter: greetings, thanks, social conversation, anything with no intelligence or
  operational content.

Rules:
- Return one entry per message id, and no ids that were not listed.
- When a message mixes content, pick the dominant purpose (a greeting attached to an
  incident report is still field_report).
- Do not guess beyond the text given; a vague message with no incident content is chatter."""

# ─── Incident threading ────────────────────────────────────────────────────

GROUND_THREAD_PROMPT_VERSION = "ground-thread-v1"

GROUND_THREAD_SYSTEM = """\
You are a humanitarian intelligence analyst for the CLEAR early warning system focused on Sudan.

You group WhatsApp field reports into incident threads: all messages describing the SAME
real-world incident belong to one thread. A signal is an incident thread, not a message —
follow-ups, confirmations, corrections, and retractions of an earlier report belong to the
thread of the incident they refer to.

You MUST respond with valid JSON only — no markdown, no explanation."""

GROUND_THREAD_USER_TEMPLATE = """\
Group these field-report messages into incident threads.

Messages, in the order they were sent (id | sent at | sender ref | text):
{messages_block}

Respond with this exact JSON structure:
{{
  "threads": [
    {{
      "title": "<short factual title for the incident>",
      "lifecycle_state": "<state>",
      "message_ids": ["<id>", ...]
    }},
    ...
  ]
}}

Lifecycle states (pick the one that describes where the incident stands after its LAST
message):
- reported: a single initial report, nothing more yet.
- updated: follow-up detail arrived (more strikes, movement, media) without changing the
  original claim.
- confirmed: a later message independently confirms the original report.
- corrected: a later message corrects a detail of the original report (for example the
  location) while the incident itself stands.
- retracted: a later message withdraws the report as misreporting — the incident did not
  happen as reported.

Rules:
- Every message id appears in exactly one thread; never invent ids.
- Messages about different incidents (different place, different day, unrelated subject) go
  in different threads, even if close in time.
- Related strikes in the same area within the same day are ONE incident thread.
- A correction or retraction always joins the thread of the report it corrects — never its
  own thread.
- Titles are neutral and factual; never name or characterise individuals."""


def _messages_block(messages: list[dict], max_text_chars: int = 500) -> str:
    lines = []
    for m in messages:
        text = (m.get("text") or "").replace("\n", " ").strip()[:max_text_chars]
        media = " [media attached]" if m.get("hasMedia") else ""
        lines.append(
            f"[{m['id']}] {m.get('sentAt', '?')} | {m.get('senderRef', '?')} |{media} {text}"
        )
    return "\n".join(lines)


def build_ground_classify_prompt(messages: list[dict]) -> str:
    """Build the user prompt for ground-message classification."""
    return GROUND_CLASSIFY_USER_TEMPLATE.format(messages_block=_messages_block(messages))


def build_ground_thread_prompt(messages: list[dict]) -> str:
    """Build the user prompt for incident threading over field reports."""
    return GROUND_THREAD_USER_TEMPLATE.format(messages_block=_messages_block(messages))
