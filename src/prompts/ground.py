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
